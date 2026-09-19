"""混合检索（Phase 6 预演）：向量召回 top_k + 知识图谱一跳扩展补充。

证明向量库与图谱真实可用；Phase 6 的 RAG 对话将复用这里。
"""
from __future__ import annotations

from dataclasses import dataclass

from .embeddings import Embedder, cosine
from .store import MaterialStore


@dataclass
class ScoredFragment:
    fragment: dict
    score: float
    via: str  # "vector" 或 "kg:关系"


def retrieve(
    query_text: str,
    store: MaterialStore,
    embedder: Embedder,
    top_k: int = 5,
    expand_kg: bool = True,
    expand_extra: int = 2,
) -> list[ScoredFragment]:
    """返回 top_k 条向量结果 + 至多 expand_extra 条图谱扩展（via 标记来源）。"""
    qv = embedder.embed_texts([query_text])[0]
    ranked = sorted(
        ((cosine(qv, vec), fid) for fid, vec in store.all_embeddings()),
        key=lambda t: -t[0],
    )
    out: list[ScoredFragment] = []
    seen: set[str] = set()
    for score, fid in ranked[:top_k]:
        frag = store.get_fragment(fid)
        if frag:
            out.append(ScoredFragment(frag, round(score, 4), "vector"))
            seen.add(fid)

    extra: list[ScoredFragment] = []
    if expand_kg and expand_extra > 0:
        for seed in list(out):
            for e in store.edges_for_fragment(seed.fragment["id"]):
                for nid in (e["src_id"], e["dst_id"]):
                    for e2, _ in store.neighbors(nid):
                        fid2 = e2.get("evidence_fragment_id", "")
                        if fid2 and fid2 not in seen:
                            frag2 = store.get_fragment(fid2)
                            if frag2:
                                extra.append(ScoredFragment(
                                    frag2, round(seed.score * 0.5, 4), f"kg:{e2['relation']}"))
                                seen.add(fid2)
                                if len(extra) >= expand_extra:
                                    break
                    if len(extra) >= expand_extra:
                        break
            if len(extra) >= expand_extra:
                break
    return out + extra
