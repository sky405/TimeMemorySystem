"""知识图谱归一：抽取草稿 → 去重/类型归一/挂证据 → 入库形态。

- 节点 id = sha1(type + name)，确定性，天然去重。
- 非法类型 → object；空名/空关系丢弃；边端点不在 nodes 里则自动补节点。
"""
from __future__ import annotations

import hashlib

from .models import KGEdge, KGExtraction, KGNode

NODE_TYPES = {"person", "place", "event", "time", "object", "org"}


def node_id(node_type: str, name: str) -> str:
    return hashlib.sha1(f"{node_type}\n{name}".encode("utf-8")).hexdigest()[:32]


def _norm_type(t: str) -> str:
    t = (t or "").strip().lower()
    return t if t in NODE_TYPES else "object"


def normalize_extraction(
    ext: KGExtraction, evidence_fragment_id: str
) -> tuple[list[KGNode], list[KGEdge]]:
    nodes: dict[str, KGNode] = {}

    def ensure(node_type: str, name: str, aliases: list[str], description: str) -> str:
        t, name = _norm_type(node_type), name.strip()
        nid = node_id(t, name)
        if nid in nodes:
            old = nodes[nid]
            old.aliases = sorted(set(old.aliases) | {a.strip() for a in aliases if a.strip()})
            if len(description.strip()) > len(old.description):
                old.description = description.strip()
        else:
            nodes[nid] = KGNode(id=nid, type=t, name=name,
                                aliases=sorted({a.strip() for a in aliases if a.strip()}),
                                description=description.strip())
        return nid

    for d in ext.nodes:
        if (d.name or "").strip():
            ensure(d.type, d.name, d.aliases, d.description)

    edges: list[KGEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for e in ext.edges:
        src_name, dst_name, rel = (e.src_name or "").strip(), (e.dst_name or "").strip(), (e.relation or "").strip()
        if not src_name or not dst_name or not rel:
            continue
        sid = ensure(e.src_type, src_name, [], "")
        did = ensure(e.dst_type, dst_name, [], "")
        key = (sid, did, rel[:32])
        if key in seen:
            continue
        seen.add(key)
        edges.append(KGEdge(src_id=sid, dst_id=did, relation=rel[:32],
                            evidence_fragment_id=evidence_fragment_id,
                            confidence=min(1.0, max(0.0, float(e.confidence or 0.5)))))
    return list(nodes.values()), edges
