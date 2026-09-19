"""Phase 5 审核流水线：next → suggest → human ⇄ apply → … → finalize。

每条存疑走一轮"AI 建议 + 人拍板"；human 节点经 interrupt() 交出展示包、
收回裁决。run_review 接受 decide_fn 驱动（真人输入或脚本），可测可演示。

LLM 经闭包注入、不进 state，state 保持纯 JSON 可序列化。
"""
from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .book import render_final_book
from .llm import ReviewLLM, get_review_llm
from .models import VERDICTS, ChapterFinal, FixSuggestion, ItemReview

_SENT_SPLIT = re.compile(r"([。！？\n])")


def _replace_sentence(text: str, quote: str, new: str) -> tuple[str, bool]:
    """把包含 quote 的句子替换为 new（new 为空即删除）。返回 (新文本, 是否命中)。"""
    if not quote:
        return text, False
    parts = _SENT_SPLIT.split(text)
    for i in range(0, len(parts), 2):
        if quote in parts[i]:
            parts[i] = new
            if i + 1 < len(parts) and (not new or new[-1] in "。！？"):
                parts[i + 1] = ""  # 删除或新句自带句号时，吃掉原分隔符
            return "".join(parts), True
    return text, False


def apply_decision(text: str, quote: str, verdict: str, replacement: str = ""
                   ) -> tuple[str, bool, str]:
    """纯函数：把裁决应用到正文。返回 (新文本, 是否改动, 说明)。

    verdict 已修正时调用方须保证 replacement 非空（或走删除分支）。
    """
    if verdict in ("确认无误", "存疑保留"):
        return text, False, ""
    if verdict == "已修正" and not replacement:
        return text, False, "改写内容为空，未改动。"
    new = replacement if verdict == "已修正" else ""
    out, hit = _replace_sentence(text, quote, new)
    if not hit:
        return text, False, "未定位到原文句子，未改动。"
    return out, True, ""


def _excerpt(text: str, quote: str, radius: int = 120) -> str:
    if not quote or quote not in text:
        return text[:radius * 2]
    i = text.index(quote)
    lo, hi = max(0, i - radius), i + len(quote) + radius
    return ("…" if lo > 0 else "") + text[lo:hi] + ("…" if hi < len(text) else "")


class ReviewState(TypedDict, total=False):
    elder: dict
    reviewer: str
    title: str
    chapters: list[dict]  # [{chapter_id, title, text, fragment_ids}]
    fragments: list[dict]
    queue: list[dict]  # 待裁决 ReviewItem dicts
    index: int
    current: dict  # 当前展示包
    suggestion: dict
    decision: dict
    records: list[dict]  # ItemReview dicts
    finals: dict  # chapter_id → 当前正文
    status: str  # in_progress | done
    book: str
    stats: dict


def build_review_graph(llm: ReviewLLM):
    def next_node(state: ReviewState) -> dict:
        if state["index"] >= len(state["queue"]):
            return {"status": "done"}
        item = state["queue"][state["index"]]
        frag_map = {f["id"]: f for f in state["fragments"]}
        mats = [{"id": i, "content": frag_map[i].get("content", ""),
                 "topic_id": frag_map[i].get("topic_id", "")}
                for i in item.get("fragment_ids", []) if i in frag_map]
        ch = next((c for c in state["chapters"] if c["title"] == item["chapter"]),
                   {"chapter_id": "", "title": item["chapter"], "text": ""})
        text = state["finals"].get(ch["chapter_id"], ch["text"])
        return {"current": {"item": item, "chapter_id": ch["chapter_id"],
                            "chapter_title": ch["title"], "excerpt": _excerpt(text, item["quote"]),
                            "materials": mats}}

    def route_next(state: ReviewState) -> str:
        return "finalize" if state.get("status") == "done" else "suggest"

    def suggest_node(state: ReviewState) -> dict:
        cur, item = state["current"], state["current"]["item"]
        frag_map = {f["id"]: f for f in state["fragments"]}
        mats = [frag_map[i] for i in item.get("fragment_ids", []) if i in frag_map]
        text = state["finals"].get(cur["chapter_id"], "")
        sug = llm.suggest_fix(cur["chapter_title"], text, item["quote"], item["issue"], mats)
        return {"suggestion": sug.model_dump()}

    def human_node(state: ReviewState) -> dict:
        cur = state["current"]
        decision = interrupt({"position": state["index"] + 1, "total": len(state["queue"]),
                              "item": cur["item"], "suggestion": state["suggestion"],
                              "chapter_title": cur["chapter_title"],
                              "excerpt": cur["excerpt"], "materials": cur["materials"]})
        return {"decision": decision or {}}

    def apply_node(state: ReviewState) -> dict:
        cur, item, sug = state["current"], state["current"]["item"], state["suggestion"]
        d = state.get("decision", {}) or {}
        verdict = d.get("verdict", "存疑保留")
        note = (d.get("note") or "").strip()
        if verdict not in VERDICTS:
            note = f"未知裁决「{verdict}」，按存疑保留处理。" + (f" {note}" if note else "")
            verdict = "存疑保留"
        replacement = (d.get("replacement") or "").strip()
        if verdict == "已修正" and not replacement:
            if sug.get("action") == "删除该句":  # 采纳 AI 的删除建议
                verdict, replacement = "删除相关句", ""
            else:
                verdict = "存疑保留"
                note = ((note + " ") if note else "") + "（未提供改写内容）"
        text = state["finals"].get(cur["chapter_id"], "")
        new_text, applied, apply_note = apply_decision(text, item["quote"], verdict, replacement)
        if apply_note:
            note = ((note + " ") if note else "") + apply_note
        finals = dict(state["finals"])
        if cur["chapter_id"]:
            finals[cur["chapter_id"]] = new_text
        record = ItemReview(chapter=item["chapter"], quote=item["quote"], issue=item["issue"],
                            severity=item.get("severity", "存疑"),
                            fragment_ids=item.get("fragment_ids", []),
                            suggestion=FixSuggestion(**sug), verdict=verdict,
                            note=note, applied=applied)
        return {"finals": finals, "records": state["records"] + [record.model_dump()],
                "index": state["index"] + 1}

    def finalize_node(state: ReviewState) -> dict:
        finals = [ChapterFinal(chapter_id=c["chapter_id"], title=c["title"],
                               text=state["finals"].get(c["chapter_id"], c["text"]),
                               fragment_ids=c.get("fragment_ids", []), approved=True)
                  for c in state["chapters"]]
        records = [ItemReview(**r) for r in state["records"]]
        by_verdict: dict[str, int] = {}
        for r in records:
            by_verdict[r.verdict] = by_verdict.get(r.verdict, 0) + 1
        book = render_final_book(state.get("title", ""), state["elder"],
                                 state.get("reviewer", "家属"), finals, records,
                                 state["fragments"])
        return {"book": book, "status": "done",
                "stats": {"items": len(records), "by_verdict": by_verdict,
                          "applied": sum(1 for r in records if r.applied),
                          "chapters": len(finals)}}

    g = StateGraph(ReviewState)
    g.add_node("next", next_node)
    g.add_node("suggest", suggest_node)
    g.add_node("human", human_node)
    g.add_node("apply", apply_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "next")
    g.add_conditional_edges("next", route_next, {"suggest": "suggest", "finalize": "finalize"})
    g.add_edge("suggest", "human")
    g.add_edge("human", "apply")
    g.add_edge("apply", "next")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver())


def _as_dicts(objs: list) -> list[dict]:
    return [o.model_dump() if hasattr(o, "model_dump") else dict(o) for o in objs]


def _elder_dict(elder: Any) -> dict:
    if elder is None:
        return {"name": "老人家"}
    if is_dataclass(elder):
        return asdict(elder)
    if isinstance(elder, dict):
        return dict(elder)
    return {"name": str(elder)}


def run_review(drafts: list, review_items: list, fragments: list,
               elder: Any = None, title: str = "", reviewer: str = "家属",
               llm: ReviewLLM | None = None,
               decide_fn: Callable[[dict], dict] | None = None) -> dict[str, Any]:
    """跑完人工审核：返回 {"book", "records", "chapters", "stats", "reviewer"}。

    decide_fn(展示包) -> {"verdict": 裁决四选一, "replacement": 改写句(可选),
    "note": 备注(可选)}。展示包 keys：position/total/item/suggestion/
    chapter_title/excerpt/materials。
    """
    if decide_fn is None:
        raise ValueError("run_review 需要 decide_fn（真人输入或脚本）")
    chapters = [{"chapter_id": d.get("chapter_id", f"ch-{i}"), "title": d.get("title", ""),
                 "text": d.get("text", ""), "fragment_ids": d.get("fragment_ids", [])}
                for i, d in enumerate(_as_dicts(drafts))]
    graph = build_review_graph(llm or get_review_llm())
    thread = {"configurable": {"thread_id": f"review-{id(graph):x}"}}
    graph.invoke({"elder": _elder_dict(elder), "reviewer": reviewer, "title": title,
                  "chapters": chapters, "fragments": _as_dicts(fragments),
                  "queue": _as_dicts(review_items), "index": 0, "records": [],
                  "finals": {c["chapter_id"]: c["text"] for c in chapters},
                  "status": "in_progress"}, thread)
    while True:
        st = graph.get_state(thread)
        if st.values.get("status") == "done" and not any(
                getattr(t, "interrupts", None) for t in st.tasks):
            break
        task = next(t for t in st.tasks if getattr(t, "interrupts", None))
        graph.invoke(Command(resume=decide_fn(task.interrupts[0].value)), thread)
    vals = graph.get_state(thread).values
    return {"book": vals["book"], "records": vals["records"],
            "chapters": [{"chapter_id": c["chapter_id"], "title": c["title"],
                          "text": vals["finals"].get(c["chapter_id"], c["text"])}
                         for c in vals["chapters"]],
            "stats": vals["stats"], "reviewer": reviewer}
