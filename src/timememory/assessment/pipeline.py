"""Phase 3 评估流水线：gather → assess → plan → validate。

判断（打分、找缺口、出追问）由 LLM 做；validate 只做确定性格式修复。
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..interview.topics import TOPIC_MAP
from ..material.store import MaterialStore
from .llm import AssessmentLLM, get_assessment_llm
from .models import Assessment, PlanItem, SupplementPlan
from .stats import gather_stats


class AssessState(TypedDict, total=False):
    store: MaterialStore
    session_id: str | None
    llm: AssessmentLLM
    stats: dict[str, Any]
    assessment: Assessment
    plan: SupplementPlan


def gather_node(state: AssessState) -> dict:
    return {"stats": gather_stats(state["store"], state.get("session_id"))}


def assess_node(state: AssessState) -> dict:
    return {"assessment": state["llm"].assess(state["stats"])}


def plan_node(state: AssessState) -> dict:
    if not state["assessment"].gaps:
        return {"plan": SupplementPlan(items=[], note="素材充足，无需补充访谈。")}
    return {"plan": state["llm"].plan_questions(state["assessment"].gaps, state["stats"])}


def _merge_items(items: list[PlanItem]) -> list[PlanItem]:
    merged: dict[str, PlanItem] = {}
    pri = {"高": 0, "中": 1, "低": 2}
    for it in items:
        if it.topic_id in merged:
            old = merged[it.topic_id]
            qs = list(old.questions) + [q for q in it.questions if q not in old.questions]
            merged[it.topic_id] = PlanItem(
                topic_id=it.topic_id, topic_name=it.topic_name,
                priority=it.priority if pri.get(it.priority, 1) < pri.get(old.priority, 1)
                else old.priority,
                reason=old.reason or it.reason, questions=qs[:3])
        else:
            merged[it.topic_id] = it
    return list(merged.values())


def validate_node(state: AssessState) -> dict:
    """确定性修复：非法话题丢弃、问题去重截断、同话题合并。"""
    items: list[PlanItem] = []
    for it in state["plan"].items:
        t = TOPIC_MAP.get(it.topic_id)
        if not t:
            continue
        qs: list[str] = []
        for q in it.questions:
            q = (q or "").strip()
            if q and q not in qs:
                qs.append(q)
        if not qs:
            qs = list(t.opening_questions[:2])
        items.append(PlanItem(
            topic_id=t.id, topic_name=t.name,
            priority=it.priority if it.priority in ("高", "中", "低") else "中",
            reason=(it.reason or t.name).strip(), questions=qs[:3]))
    plan = SupplementPlan(items=_merge_items(items)[:8],
                          note=state["plan"].note or "按话题顺序逐个补访即可。")
    a = state["assessment"]
    ready = a.ready and not any(g.priority == "高" for g in a.gaps)
    return {"plan": plan, "assessment": a.model_copy(update={"ready": ready})}


def build_assessment_graph():
    g = StateGraph(AssessState)
    g.add_node("gather", gather_node)
    g.add_node("assess", assess_node)
    g.add_node("plan", plan_node)
    g.add_node("validate", validate_node)
    g.add_edge(START, "gather")
    g.add_edge("gather", "assess")
    g.add_edge("assess", "plan")
    g.add_edge("plan", "validate")
    g.add_edge("validate", END)
    return g.compile()


def run_assessment(store: MaterialStore, session_id: str | None = None,
                   llm: AssessmentLLM | None = None) -> dict[str, Any]:
    """一键评估：返回 {"stats", "assessment", "plan"}。"""
    out = build_assessment_graph().invoke({
        "store": store, "session_id": session_id, "llm": llm or get_assessment_llm()})
    return {"stats": out["stats"], "assessment": out["assessment"], "plan": out["plan"]}
