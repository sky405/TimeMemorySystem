"""访谈状态机（LangGraph StateGraph）。

判断交给 LLM，流程交给图：

    START → router → (ask → human → extract → validate → record) ↺
              │                                              │
              └──────────────→ closing → END                 │
                                   ↑                         │
                        fix → extract（验证失败修复环）────────┘

- router：三叉路口决策（LLM）。仅有的两处人工干预是护栏而非判断：
  告别安全词 → 强制收尾；轮次预算耗尽 → 强制收尾。
- human：interrupt() 等老人回答（Human-in-the-loop）。
- extract → validate → fix：ReAct 式抽取循环——抽取候选、验证器质检、
  不合格则带反馈修复，最多 max_fix_retries 次。
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .llm import InterviewLLM
from .models import ClosingContext, InterviewState, MemoryFragment, RouteContext, RouteDecision
from .topics import TOPIC_MAP, default_next_topic, get_topic, uncovered_topics
from .validators import is_farewell, validate_fragments


def _last_elder_text(state: InterviewState) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "elder":
            return m.get("text", "")
    return ""


def _recent_qa(state: InterviewState, window: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending = ""
    for m in state.get("messages", []):
        if m.get("role") == "ai":
            pending = m.get("text", "")
        elif m.get("role") == "elder" and pending:
            pairs.append((pending, m.get("text", "")))
            pending = ""
    return pairs[-window:]


def _best_highlight(state: InterviewState, topic_id: str) -> str:
    frags = [f for f in state.get("fragments", []) if f.get("topic_id") == topic_id]
    if not frags:
        return ""
    best = max(frags, key=lambda f: f.get("importance", 3))
    content = best.get("content", "")
    return content[:18] + ("…" if len(content) > 18 else "")


def build_interview_graph(llm: InterviewLLM):
    """构建访谈图（自带 MemorySaver checkpointer，支持 interrupt 暂停/恢复）。"""

    # -- router：三叉路口 ---------------------------------------------------------
    def router(state: InterviewState) -> dict:
        cfg = state["config"]
        turn = state.get("turn_count", 0)
        current = state.get("current_topic_id", cfg["first_topic_id"])
        covered = state.get("covered_topic_ids", [])
        topic = get_topic(current)

        if turn >= cfg["max_total_turns"]:
            d = RouteDecision(action="wrap", reasoning=f"已达单次访谈上限（{cfg['max_total_turns']} 轮）")
        else:
            answer = _last_elder_text(state)
            hit = is_farewell(answer) if answer else ""
            if hit:
                d = RouteDecision(action="wrap", reasoning=f"检测到告别「{hit}」，必须立刻收尾")
            else:
                ctx = RouteContext(
                    elder_name=state.get("elder", {}).get("name", "老人家"),
                    current_topic=topic,
                    topic_turns_current=state.get("topic_turns", {}).get(current, 0),
                    turn_count=turn,
                    max_turns=cfg["max_total_turns"],
                    covered_names=[TOPIC_MAP[i].name for i in covered if i in TOPIC_MAP],
                    uncovered=uncovered_topics(covered, current),
                    recent_qa=_recent_qa(state, cfg.get("recent_window", 6)),
                    last_answer=answer,
                )
                d = llm.route(ctx)
                if d.action == "switch":
                    if d.next_topic_id not in {t.id for t in ctx.uncovered}:
                        fb = default_next_topic(covered, current)  # 非法 id 兜底
                        d.next_topic_id = fb.id if fb else ""
                    if not d.next_topic_id:
                        d = RouteDecision(action="wrap", reasoning="所有话题已覆盖完，收尾")
                    else:
                        d.prev_topic_id = current

        out: dict = {
            "decision": d.model_dump(),
            "trail": [{"turn": turn, "topic": current, **d.model_dump()}],
        }
        if d.action == "switch":
            out["current_topic_id"] = d.next_topic_id
            out["covered_topic_ids"] = [current]
        return out

    def after_router(state: InterviewState) -> str:
        return "closing" if state["decision"]["action"] == "wrap" else "ask"

    # -- ask：生成下一句 -----------------------------------------------------------
    def ask(state: InterviewState) -> dict:
        d = RouteDecision(**state["decision"])
        current = state["current_topic_id"]
        topic = get_topic(current)
        elder = state.get("elder", {})
        extra: dict = {}

        # 补访提纲脚本优先：写作 Agent 指定的问题原样问出，并切到对应话题
        script_text = ""
        script = [dict(s) for s in state.get("script", [])]
        if script:
            item = script.pop(0)
            extra["script"] = script
            q_topic = item.get("topic_id") or current
            if q_topic not in TOPIC_MAP:
                q_topic = current
            if q_topic != current:
                extra["current_topic_id"] = q_topic
                extra["covered_topic_ids"] = [current]
                current, topic = q_topic, get_topic(q_topic)
            script_text = (item.get("question") or "").strip()
            if script_text and state.get("turn_count", 0) == 0 and not _last_elder_text(state):
                name = elder.get("name", "老人家")
                hometown = f"听说您老家是{elder.get('hometown')}，" if elder.get("hometown") else ""
                script_text = (f"{name}您好！我是您的回忆录访谈员小记。{hometown}"
                               f"上次聊完后，我们发现还有些故事没讲透，今天接着聊聊。"
                               f"先问您一个：{script_text}")

        if script_text:
            text = script_text
        elif state.get("turn_count", 0) == 0 and not _last_elder_text(state):
            name = elder.get("name", "老人家")
            hometown = f"听说您老家是{elder.get('hometown')}，" if elder.get("hometown") else ""
            text = (f"{name}您好！我是您的回忆录访谈员小记。{hometown}"
                    f"今天咱们就像拉家常一样，随便聊聊您这辈子的故事，"
                    f"想到哪儿说到哪儿，累了咱们就歇着。"
                    f"先问您一个：{llm.ask_opening(topic)}")
        elif d.action == "switch":
            prev = get_topic(d.prev_topic_id) if d.prev_topic_id in TOPIC_MAP else topic
            text = llm.ask_transition(prev, _best_highlight(state, d.prev_topic_id), topic)
        else:
            text = llm.ask_followup(topic, d.focus, _last_elder_text(state))

        msg = {"role": "ai", "text": text, "topic": current, "turn": state.get("turn_count", 0)}
        return {"messages": [msg], "pending_question": text, "last_reply": text, **extra}

    # -- human：等老人回答 ----------------------------------------------------------
    def human(state: InterviewState) -> dict:
        answer = ((interrupt(state.get("pending_question", "")) or "").strip())
        turn = state.get("turn_count", 0) + 1
        current = state["current_topic_id"]
        topic_turns = dict(state.get("topic_turns", {}))
        topic_turns[current] = topic_turns.get(current, 0) + 1
        msg = {"role": "elder", "text": answer, "topic": current, "turn": turn}
        return {"messages": [msg], "turn_count": turn, "topic_turns": topic_turns}

    # -- extract → validate → fix：ReAct 抽取循环 ------------------------------------
    def extract(state: InterviewState) -> dict:
        answer = _last_elder_text(state)
        if is_farewell(answer):
            return {"candidates": []}  # 告别语不是记忆，不抽取
        topic = get_topic(state["current_topic_id"])
        previous = [MemoryFragment(**c) for c in state.get("candidates", [])]
        committed = [MemoryFragment(**f) for f in state.get("fragments", [])]
        frags = llm.extract_fragments(
            answer, topic.name,
            previous, state.get("feedback", ""), committed,
        )
        return {"candidates": [f.model_dump() for f in frags]}

    def validate(state: InterviewState) -> dict:
        cands = [MemoryFragment(**c) for c in state.get("candidates", [])]
        committed = [MemoryFragment(**f) for f in state.get("fragments", [])]
        check = state["config"].get("llm_validate", True)
        report = validate_fragments(cands, committed, llm=llm if check else None, llm_check=check)
        return {"validation": report.to_dict()}

    def after_validate(state: InterviewState) -> str:
        v = state.get("validation", {})
        if not v.get("valid") and v.get("rejected") and state.get("retries", 0) < state["config"].get("max_fix_retries", 2):
            return "fix"
        return "record"

    def fix(state: InterviewState) -> dict:
        rej = state.get("validation", {}).get("rejected", [])
        details = "；".join(f"「{r['content'][:30]}」→ {'; '.join(r['reasons'])}" for r in rej)
        return {
            "feedback": f"上次抽取的 {len(rej)} 个候选全部未通过验证：{details}。请修正后重新输出。",
            "retries": state.get("retries", 0) + 1,
        }

    def record(state: InterviewState) -> dict:
        turn = state.get("turn_count", 0)
        current = state["current_topic_id"]
        stamped = [{**v, "topic_id": current, "source_turn": turn}
                   for v in state.get("validation", {}).get("valid", [])]
        return {"fragments": stamped, "new_fragments": stamped,
                "last_rejected": state.get("validation", {}).get("rejected", []),
                "candidates": [], "validation": {}, "feedback": "", "retries": 0}

    # -- closing：收尾 ---------------------------------------------------------------
    def closing(state: InterviewState) -> dict:
        frags = sorted(state.get("fragments", []), key=lambda f: -f.get("importance", 3))
        highlights = [(f["content"][:24] + ("…" if len(f["content"]) > 24 else "")) for f in frags[:2]]
        talked = {t for t, n in state.get("topic_turns", {}).items() if n > 0}
        ctx = ClosingContext(
            elder_name=state.get("elder", {}).get("name", "老人家"),
            highlights=highlights, n_topics=len(talked), n_fragments=len(frags),
        )
        text = llm.closing(ctx)
        msg = {"role": "ai", "text": text, "topic": state.get("current_topic_id", ""),
               "turn": state.get("turn_count", 0)}
        return {"messages": [msg], "last_reply": text, "status": "ended", "new_fragments": []}

    g = StateGraph(InterviewState)
    g.add_node("router", router)
    g.add_node("ask", ask)
    g.add_node("human", human)
    g.add_node("extract", extract)
    g.add_node("validate", validate)
    g.add_node("fix", fix)
    g.add_node("record", record)
    g.add_node("closing", closing)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router, {"ask": "ask", "closing": "closing"})
    g.add_edge("ask", "human")
    g.add_edge("human", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", after_validate, {"fix": "fix", "record": "record"})
    g.add_edge("fix", "extract")
    g.add_edge("record", "router")
    g.add_edge("closing", END)
    return g.compile(checkpointer=MemorySaver())
