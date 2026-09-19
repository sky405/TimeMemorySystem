"""确定性素材摘要：把库里的片段和图谱压缩成 LLM 能一口吃下的 JSON。

纯规则统计，不做判断——判断是 assess 节点（LLM）的事。
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..interview.topics import TOPICS
from ..material.store import MaterialStore

_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
_MAX_SAMPLE_LEN = 80
_MAX_ENTITIES = 12


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def gather_stats(store: MaterialStore, session_id: str | None = None) -> dict[str, Any]:
    """统计某次访谈（或全库）的素材画像。"""
    frags = store.all_fragments(session_id)
    frags = sorted(frags, key=lambda f: f["id"])
    frag_map = {f["id"]: f for f in frags}
    fid_set = set(frag_map)

    by_topic = Counter(f.get("topic_id", "") for f in frags)
    empty_topics = [{"id": t.id, "name": t.name} for t in TOPICS if by_topic.get(t.id, 0) == 0]
    thin_topics = [{"id": t.id, "name": t.name, "count": by_topic[t.id]}
                   for t in TOPICS if by_topic.get(t.id, 0) == 1]

    years: list[int] = []
    for f in frags:
        hay = f.get("content", "") + " " + " ".join(f.get("time_refs", []))
        years.extend(int(y) for y in _YEAR_RE.findall(hay))
    years = sorted(set(years))
    decades = Counter(y // 10 * 10 for y in years)
    empty_decades: list[int] = []
    if years:
        for d in range(years[0] // 10 * 10, years[-1] // 10 * 10 + 1, 10):
            if d not in decades:
                empty_decades.append(d)

    entities: list[dict[str, Any]] = []
    for n in store.all_nodes():
        ev = [e["evidence_fragment_id"] for e, _ in store.neighbors(n["id"])
              if e.get("evidence_fragment_id") in fid_set]
        if session_id and not ev:
            continue  # 跨会话节点不计入本次画像
        topics = Counter(frag_map[e]["topic_id"] for e in ev if e in frag_map)
        entities.append({
            "name": n["name"], "type": n["type"], "mentions": len(ev),
            "topic": topics.most_common(1)[0][0] if topics else "",
            "evidence_ids": sorted(set(ev)),
        })
    entities.sort(key=lambda e: (-e["mentions"], e["name"]))
    thin_entities = [e for e in entities
                     if e["mentions"] <= 1 and e["type"] in ("person", "place")][:5]

    samples: dict[str, list[str]] = {}
    for t in TOPICS:
        got = [_truncate(f["content"], _MAX_SAMPLE_LEN) for f in frags
               if f.get("topic_id") == t.id][:2]
        if got:
            samples[t.id] = got

    n = len(frags) or 1
    return {
        "session_id": session_id or "*",
        "fragments": len(frags),
        "total_chars": sum(len(f.get("content", "")) for f in frags),
        "by_topic": dict(sorted(by_topic.items())),
        "empty_topics": empty_topics,
        "thin_topics": thin_topics,
        "years": years,
        "decades": {str(k): v for k, v in sorted(decades.items())},
        "empty_decades": empty_decades,
        "entities": entities[:_MAX_ENTITIES],
        "thin_entities": thin_entities,
        "emotion_coverage": round(sum(1 for f in frags if f.get("emotion")) / n, 2),
        "avg_importance": round(sum(f.get("importance", 0) for f in frags) / n, 2),
        "samples": samples,
    }
