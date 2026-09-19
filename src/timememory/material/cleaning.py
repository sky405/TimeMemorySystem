"""清洗：Phase 1 片段 → CleanFragment。

- 规范 id（f"{session}:{frag}"，跨会话唯一）、去空白、LLM 最小清洗。
- 过滤过短无料句；近重复（Jaccard ≥ 0.9）合并，保留更长的、引用求并集。
"""
from __future__ import annotations

import re

from .llm import MaterialLLM
from .models import CleanFragment

_WS_RE = re.compile(r"\s+")
_MIN_LEN = 6
_DUPE_THRESHOLD = 0.9


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _has_refs(raw: dict) -> bool:
    return bool(raw.get("time_refs") or raw.get("place_refs") or raw.get("person_refs"))


def clean_fragments(session_id: str, raw: list[dict], llm: MaterialLLM) -> list[CleanFragment]:
    cleaned: list[CleanFragment] = []
    for i, r in enumerate(raw):
        content = _WS_RE.sub(" ", (r.get("content") or "")).strip()
        if len(content) < _MIN_LEN and not _has_refs(r):
            continue
        content = llm.clean_text(content).strip() or content
        fid = r.get("id") or f"frag-{i:04d}"  # Phase 1 新版无 id 时按序补号
        cleaned.append(CleanFragment(
            id=f"{session_id}:{fid}",
            session_id=session_id,
            content=content,
            topic_id=r.get("topic_id", ""),
            source_turn=int(r.get("source_turn", 0)),
            time_refs=list(r.get("time_refs", [])),
            place_refs=list(r.get("place_refs", [])),
            person_refs=list(r.get("person_refs", [])),
            emotion=r.get("emotion", ""),
            importance=int(r.get("importance", 3)),
        ))
    merged = _merge_dupes(cleaned)
    merged.sort(key=lambda f: f.source_turn)
    return merged


def _merge_dupes(frags: list[CleanFragment]) -> list[CleanFragment]:
    """近重复合并：保留更长正文，引用求并集，重要度取高，轮次取早。"""
    out: list[CleanFragment] = []
    for f in frags:
        twin = next((o for o in out if _jaccard(f.content, o.content) >= _DUPE_THRESHOLD), None)
        if twin is None:
            out.append(f)
            continue
        if len(f.content) > len(twin.content):
            twin.content = f.content
        twin.time_refs = sorted(set(twin.time_refs) | set(f.time_refs))
        twin.place_refs = sorted(set(twin.place_refs) | set(f.place_refs))
        twin.person_refs = sorted(set(twin.person_refs) | set(f.person_refs))
        twin.importance = max(twin.importance, f.importance)
        twin.source_turn = min(twin.source_turn, f.source_turn)
    return out
