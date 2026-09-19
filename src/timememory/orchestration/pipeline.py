"""编排控制器：访谈 ↔ 写作协作循环。

    interview → material → assess → draft → END
                    ↑_________|（有缺口且预算未尽 → 再补访一轮）

路由是确定性控制流（ready / 轮次预算 / 有无进展），判断仍在各 Agent 的 LLM 里。

控制器编译时不带 checkpointer：子图（素材/评估/成稿）state 里放着 store/llm
对象，嵌套 checkpoint 会序列化失败。跨天续跑靠访谈 Agent 的 save_session
与整档 store 落盘（MySQL / SQLite 文件），控制器本身无状态可重入。
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig

from langgraph.graph import END, START, StateGraph

from ..assessment import get_assessment_llm, render_brief, run_assessment
from ..drafting import get_drafting_llm, run_drafting
from ..interview.agent import InterviewAgent
from ..interview.llm import get_llm as get_interview_llm
from ..interview.models import AgentConfig, ElderProfile
from ..interview.topics import TOPIC_MAP
from ..material import (
    MaterialStore,
    get_db,
    get_embedder,
    get_material_llm,
    run_material_pipeline,
)
from .models import (
    STATUS_COMPLETE,
    STATUS_COMPLETE_WITH_FLAGS,
    STATUS_MAX_ROUNDS,
    MemoirState,
    RoundReport,
)

_DEPS: dict[str, dict] = {}  # run_id → 运行依赖（store/answer_fn/各 LLM）


def _deps(config: RunnableConfig) -> dict:
    return _DEPS[config["configurable"]["deps_id"]]


SCRIPT_PER_ITEM = 2  # 每个缺口最多采用 2 个追问
SCRIPT_MAX = 6  # 一轮补访最多 6 个指定问题
SUPPLEMENT_PAD = 2  # 补访轮次预算 = 指定问题数 + 余量


def _build_script(plan_items: list[dict]) -> list[dict]:
    """写作 Agent 的提纲 → 访谈 Agent 的提问脚本。"""
    script: list[dict] = []
    for it in plan_items or []:
        tid = it.get("topic_id", "")
        if tid not in TOPIC_MAP:
            continue
        for q in (it.get("questions") or [])[:SCRIPT_PER_ITEM]:
            if q and q.strip():
                script.append({"topic_id": tid, "question": q.strip()})
            if len(script) >= SCRIPT_MAX:
                return script
    return script


def interview_node(state: MemoirState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    elder_d = state["elder"]
    elder = ElderProfile(name=elder_d.get("name", "老人家"),
                         age=elder_d.get("age"), hometown=elder_d.get("hometown", ""))
    round_index = state["round_index"]
    if round_index == 0:
        agent_config, first_topic, script = AgentConfig(), None, []
    else:
        script = _build_script(state.get("plan_items", []))
        first_topic = script[0]["topic_id"] if script else None
        budget = len(script) + SUPPLEMENT_PAD if script else 6
        agent_config = AgentConfig(max_total_turns=budget,
                                   first_topic_id=first_topic or "childhood")
    agent = InterviewAgent(elder=elder, config=agent_config, llm=deps["interview_llm"])
    agent.session_id = f"{state['archive_id']}-r{round_index}"
    agent._thread = {"configurable": {"thread_id": agent.session_id}}

    answer_fn: Callable[[str, int, int], str] = deps["answer_fn"]
    reply = agent.start(first_topic_id=first_topic,
                        resume_state={"script": script} if script else None)
    turns = 0
    while not reply.session_ended:
        reply = agent.step(answer_fn(reply.text, round_index, turns) or "")
        turns += 1
        if turns > agent_config.max_total_turns + 5:  # 保险丝，正常不会触发
            break
    frags = agent.fragments_json()
    asked = [m["text"] for m in agent.state.get("messages", []) if m.get("role") == "ai"]
    report = RoundReport(round=round_index, session_id=agent.session_id,
                         turns=agent.state.get("turn_count", 0), fragments=len(frags),
                         focus_topics=sorted({s["topic_id"] for s in script}),
                         script_used=len(script), questions_asked=asked)
    return {"session_ids": state.get("session_ids", []) + [agent.session_id],
            "round_reports": state.get("round_reports", []) + [dict(report)],
            "round_fragments": frags,
            "total_fragments": state.get("total_fragments", 0) + len(frags),
            "round_index": round_index + 1}


def material_node(state: MemoirState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    result = run_material_pipeline(state["session_ids"][-1], state["round_fragments"],
                                   deps["material_llm"], deps["embedder"], deps["store"])
    return {"material_stats": result["stats"]}


def assess_node(state: MemoirState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    out = run_assessment(deps["store"], None, deps["assess_llm"])  # 整档评估
    a, p = out["assessment"], out["plan"]
    return {"assessment": a.model_dump(),
            "plan_items": [i.model_dump() for i in p.items],
            "ready": a.ready, "brief": render_brief(a, p)}


def route_after_assess(state: MemoirState) -> str:
    if state.get("ready"):
        return "draft"
    if state["round_index"] >= state["max_rounds"]:
        return "draft"
    if state["round_index"] >= 1 and not state.get("round_fragments"):
        return "draft"  # 一轮零产出，再访也无意义
    return "interview"


def draft_node(state: MemoirState, config: RunnableConfig) -> dict:
    deps = _deps(config)
    a = state.get("assessment", {})
    gaps = "; ".join(g.get("title", "") for g in a.get("gaps", [])[:6])
    note = (f"{a.get('overall', '')} 缺口：{gaps}" if gaps else a.get("overall", ""))
    out = run_drafting(deps["store"], None, state["elder"],
                       state.get("birth_year"), note, deps["draft_llm"])
    if state.get("ready") and out["stats"]["flags"] == 0:
        status = STATUS_COMPLETE
    elif state.get("ready"):
        status = STATUS_COMPLETE_WITH_FLAGS
    else:
        status = STATUS_MAX_ROUNDS
    return {"manuscript": out["manuscript"],
            "review": [r.model_dump() for r in out["review"]],
            "draft_stats": out["stats"], "outline_title": out["outline"].title,
            "status": status}


def build_conductor_graph():
    """构建编排图（无 checkpointer，见模块 docstring）。"""
    g = StateGraph(MemoirState)
    g.add_node("interview", interview_node)
    g.add_node("material", material_node)
    g.add_node("assess", assess_node)
    g.add_node("draft", draft_node)
    g.add_edge(START, "interview")
    g.add_edge("interview", "material")
    g.add_edge("material", "assess")
    g.add_conditional_edges("assess", route_after_assess,
                            {"interview": "interview", "draft": "draft"})
    g.add_edge("draft", END)
    return g.compile()


def _elder_dict(elder: Any) -> dict:
    if elder is None:
        return {"name": "老人家"}
    if is_dataclass(elder):
        return asdict(elder)
    if isinstance(elder, dict):
        return dict(elder)
    return {"name": str(elder)}


def run_memoir(elder: Any = None, answer_fn: Callable[[str, int, int], str] | None = None,
               birth_year: int | None = None, max_rounds: int = 3,
               archive_id: str | None = None, store: MaterialStore | None = None,
               interview_llm=None, material_llm=None, embedder=None,
               assess_llm=None, draft_llm=None) -> dict[str, Any]:
    """一键跑完整本回忆录工程：访谈 ↔ 写作协作到就绪，然后成稿。

    answer_fn(question, round_index, turn) -> 老人回答；剧本用尽请返回告别语。
    """
    if answer_fn is None:
        raise ValueError("run_memoir 需要 answer_fn（真人输入或模拟剧本）")
    archive = archive_id or f"memoir-{uuid.uuid4().hex[:8]}"
    store = store or MaterialStore(get_db())
    deps_id = uuid.uuid4().hex
    _DEPS[deps_id] = {"store": store, "answer_fn": answer_fn,
                      "interview_llm": interview_llm or get_interview_llm(),
                      "material_llm": material_llm or get_material_llm(),
                      "embedder": embedder or get_embedder(),
                      "assess_llm": assess_llm or get_assessment_llm(),
                      "draft_llm": draft_llm or get_drafting_llm()}
    cfg = {"configurable": {"thread_id": archive, "deps_id": deps_id}}
    try:
        out = build_conductor_graph().invoke({
            "archive_id": archive, "elder": _elder_dict(elder),
            "birth_year": birth_year, "max_rounds": max(1, max_rounds), "round_index": 0,
            "session_ids": [], "round_reports": [], "total_fragments": 0}, cfg)
    finally:
        _DEPS.pop(deps_id, None)
    return {"archive_id": archive, "status": out["status"], "ready": out["ready"],
            "rounds": out["round_reports"], "brief": out["brief"],
            "manuscript": out["manuscript"], "review": out["review"],
            "material_stats": out.get("material_stats", {}),
            "draft_stats": out.get("draft_stats", {}), "store": store}


_STATUS_CN = {STATUS_COMPLETE: "✅ 素材就绪，成稿完毕",
              STATUS_COMPLETE_WITH_FLAGS: "⚠️ 成稿完毕，有存疑待人工审核",
              STATUS_MAX_ROUNDS: "⚠️ 轮次用尽，带缺口成稿"}


def render_memoir_report(result: dict) -> str:
    lines = [f"# 回忆录工程报告（{result['archive_id']}）", "",
             f"**状态**：{_STATUS_CN.get(result['status'], result['status'])}",
             f"**轮次**：{len(result['rounds'])}｜"
             f"**成稿章节**：{result['draft_stats'].get('chapters', 0)}｜"
             f"**存疑**：{result['draft_stats'].get('flags', 0)}", "",
             "## 各轮访谈"]
    for r in result["rounds"]:
        focus = "、".join(r["focus_topics"]) if r["focus_topics"] else "首轮自由访谈"
        lines.append(f"- 第 {r['round'] + 1} 轮 `{r['session_id']}`："
                     f"{r['turns']} 问 {r['fragments']} 片段｜{focus}"
                     + (f"（采用提纲 {r['script_used']} 问）" if r["script_used"] else ""))
    lines += ["", "## 写作评估（最后一轮）", "", result["brief"]]
    if result["review"]:
        lines += ["## 复核清单", ""]
        lines += [f"- 【{r['chapter']}】“{r['quote']}”——{r['issue']}"
                  for r in result["review"]]
    return "\n".join(lines) + "\n"
