"""Phase 4 初稿流水线：gather → stages → outline → write ⇄ check → polish → render。

写作、回检、统稿由 LLM 做；normalize_outline 做确定性修补
（去重/丢未知 id/丢空章/不丢素材）；事实回检不通过不阻断，
挂批注进附录、留人工复核入口。
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..material.store import MaterialStore
from .llm import DraftingLLM, get_drafting_llm
from .manuscript import render_manuscript
from .models import ChapterDraft, ChapterSpec, Outline, ReviewItem, StageGroup
from .stages import group_stages


class DraftState(TypedDict, total=False):
    store: MaterialStore
    session_id: str | None
    llm: DraftingLLM
    elder: dict
    birth_year: int | None
    assessment_note: str
    fragments: list[dict]
    groups: list[StageGroup]
    undated: list[dict]
    outline: Outline
    chapter_index: int
    drafts: list[ChapterDraft]
    current_draft: ChapterDraft
    body: str
    manuscript: str
    review: list[ReviewItem]


def gather_node(state: DraftState) -> dict:
    return {"fragments": state["store"].all_fragments(state.get("session_id"))}


def stages_node(state: DraftState) -> dict:
    groups, undated = group_stages(state["fragments"], state.get("birth_year"))
    return {"groups": groups, "undated": undated}


def normalize_outline(outline: Outline, fragments: list[dict]) -> Outline:
    """修补 LLM 大纲：id 去重、丢未知 id、丢空章、遗漏素材补进末章。"""
    valid = {f["id"] for f in fragments}
    chapters: list[ChapterSpec] = []
    seen: set[str] = set()
    for ch in outline.chapters:
        ids = [i for i in dict.fromkeys(ch.fragment_ids) if i in valid and i not in seen]
        seen.update(ids)
        if ids:
            chapters.append(ch.model_copy(update={"fragment_ids": ids}))
    missing = [f["id"] for f in sorted(fragments, key=lambda x: x["id"]) if f["id"] not in seen]
    if missing and chapters:
        last = chapters[-1]
        chapters[-1] = last.model_copy(update={"fragment_ids": last.fragment_ids + missing})
    elif missing:
        chapters = [ChapterSpec(id="ch-0", title="岁月记忆", brief="老人的回忆。",
                                fragment_ids=missing)]
    return outline.model_copy(update={"chapters": chapters})


def outline_node(state: DraftState) -> dict:
    outline = state["llm"].make_outline(state["elder"], state["groups"],
                                        state["undated"],
                                        state.get("assessment_note", ""))
    return {"outline": normalize_outline(outline, state["fragments"]),
            "chapter_index": 0, "drafts": []}


def write_node(state: DraftState) -> dict:
    ch = state["outline"].chapters[state["chapter_index"]]
    frag_map = {f["id"]: f for f in state["fragments"]}
    mats = [frag_map[i] for i in ch.fragment_ids if i in frag_map]
    prev = state["drafts"][-1].summary if state["drafts"] else ""
    draft = state["llm"].write_chapter(state["elder"], ch, mats, prev)
    return {"current_draft": draft}


def check_node(state: DraftState) -> dict:
    draft = state["current_draft"]
    frag_map = {f["id"]: f for f in state["fragments"]}
    mats = [frag_map[i] for i in draft.fragment_ids if i in frag_map]
    fc = state["llm"].fact_check(draft.title, draft.text, mats)
    return {"drafts": state["drafts"] + [draft.model_copy(update={"factcheck": fc})],
            "chapter_index": state["chapter_index"] + 1}


def route_chapters(state: DraftState) -> str:
    if state["chapter_index"] < len(state["outline"].chapters):
        return "write"
    return "polish"


def polish_node(state: DraftState) -> dict:
    return {"body": state["llm"].polish(state["elder"], state["drafts"])}


def render_node(state: DraftState) -> dict:
    text, review = render_manuscript(state["outline"], state["drafts"], state["body"],
                                     state["elder"], state["fragments"])
    return {"manuscript": text, "review": review}


def build_drafting_graph():
    g = StateGraph(DraftState)
    g.add_node("gather", gather_node)
    g.add_node("stages", stages_node)
    g.add_node("outline", outline_node)
    g.add_node("write", write_node)
    g.add_node("check", check_node)
    g.add_node("polish", polish_node)
    g.add_node("render", render_node)
    g.add_edge(START, "gather")
    g.add_edge("gather", "stages")
    g.add_edge("stages", "outline")
    g.add_conditional_edges("outline", route_chapters, {"write": "write", "polish": "polish"})
    g.add_edge("write", "check")
    g.add_conditional_edges("check", route_chapters, {"write": "write", "polish": "polish"})
    g.add_edge("polish", "render")
    g.add_edge("render", END)
    return g.compile()


def _elder_dict(elder: Any) -> dict:
    if elder is None:
        return {"name": "老人家"}
    if is_dataclass(elder):
        return asdict(elder)
    if isinstance(elder, dict):
        return elder
    return {"name": str(elder)}


def _assessment_note(assessment: Any) -> str:
    if not assessment:
        return ""
    if isinstance(assessment, str):
        return assessment
    gaps = "; ".join(g.title for g in getattr(assessment, "gaps", [])[:6])
    overall = getattr(assessment, "overall", "")
    return f"{overall} 缺口：{gaps}" if gaps else overall


def run_drafting(store: MaterialStore, session_id: str | None = None, elder: Any = None,
                 birth_year: int | None = None, assessment: Any = None,
                 llm: DraftingLLM | None = None) -> dict[str, Any]:
    """一键成稿：返回 {"outline", "drafts", "manuscript", "review", "stats"}。"""
    elder_d = _elder_dict(elder)
    out = build_drafting_graph().invoke({
        "store": store, "session_id": session_id, "llm": llm or get_drafting_llm(),
        "elder": elder_d, "birth_year": birth_year or elder_d.get("birth_year"),
        "assessment_note": _assessment_note(assessment)})
    drafts: list[ChapterDraft] = out["drafts"]
    return {
        "outline": out["outline"], "drafts": drafts,
        "manuscript": out["manuscript"], "review": out["review"],
        "stats": {"fragments": len(out["fragments"]), "groups": len(out["groups"]),
                  "chapters": len(drafts),
                  "passed": sum(1 for d in drafts if d.factcheck.passed),
                  "flags": sum(len(d.factcheck.flags) for d in drafts)},
    }
