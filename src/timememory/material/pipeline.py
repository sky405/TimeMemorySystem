"""Phase 2 流水线（LangGraph）：clean → embed → extract_kg → persist。

线性 ETL，用图承载以便观察每步产出与将来扩展分支（如低质重采）。
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .cleaning import clean_fragments
from .embeddings import Embedder, get_embedder
from .kg import normalize_extraction
from .llm import MaterialLLM, get_material_llm
from .models import CleanFragment, KGEdge, KGNode
from .store import DB, MaterialStore


class MaterialState(TypedDict, total=False):
    session_id: str
    raw: list[dict]
    clean: list[dict]
    vectors: dict[str, list[float]]
    kg_nodes: list[dict]
    kg_edges: list[dict]
    stats: dict


def build_material_graph(llm: MaterialLLM, embedder: Embedder, store: MaterialStore):
    def clean_node(state: MaterialState) -> dict:
        frags = clean_fragments(state["session_id"], state.get("raw", []), llm)
        return {"clean": [f.model_dump() for f in frags]}

    def embed_node(state: MaterialState) -> dict:
        frags = [CleanFragment(**c) for c in state.get("clean", [])]
        vecs = embedder.embed_texts([f.content for f in frags])
        return {"vectors": {f.id: v for f, v in zip(frags, vecs)}}

    def kg_node(state: MaterialState) -> dict:
        nodes: dict[str, KGNode] = {}
        edges: list[KGEdge] = []
        for c in state.get("clean", []):
            frag = CleanFragment(**c)
            ns, es = normalize_extraction(llm.extract_kg(frag), frag.id)
            for n in ns:
                if n.id in nodes:
                    old = nodes[n.id]
                    old.aliases = sorted(set(old.aliases) | set(n.aliases))
                    if len(n.description) > len(old.description):
                        old.description = n.description
                else:
                    nodes[n.id] = n
            edges.extend(es)
        return {"kg_nodes": [n.model_dump() for n in nodes.values()],
                "kg_edges": [e.model_dump() for e in edges]}

    def persist_node(state: MaterialState) -> dict:
        frags = [CleanFragment(**c) for c in state.get("clean", [])]
        store.save_fragments(frags)
        for fid, vec in state.get("vectors", {}).items():
            store.save_embedding(fid, vec)
        for n in state.get("kg_nodes", []):
            store.upsert_node(KGNode(**n))
        new_edges = sum(store.add_edge(KGEdge(**e)) for e in state.get("kg_edges", []))
        return {"stats": {
            "raw": len(state.get("raw", [])), "clean": len(frags),
            "embedded": len(state.get("vectors", {})),
            "nodes": len(state.get("kg_nodes", "")), "edges": len(state.get("kg_edges", [])),
            "edges_new": new_edges, **store.stats(),
        }}

    g = StateGraph(MaterialState)
    g.add_node("clean", clean_node)
    g.add_node("embed", embed_node)
    g.add_node("extract_kg", kg_node)
    g.add_node("persist", persist_node)
    g.add_edge(START, "clean")
    g.add_edge("clean", "embed")
    g.add_edge("embed", "extract_kg")
    g.add_edge("extract_kg", "persist")
    g.add_edge("persist", END)
    return g.compile()


def run_material_pipeline(
    session_id: str,
    fragments: list[dict],
    llm: MaterialLLM | None = None,
    embedder: Embedder | None = None,
    store: MaterialStore | None = None,
    db: DB | None = None,
) -> dict:
    """跑完 Phase 2 全流程。返回 {"stats", "clean", "store"}。"""
    llm = llm or get_material_llm()
    embedder = embedder or get_embedder()
    store = store or MaterialStore(db)
    graph = build_material_graph(llm, embedder, store)
    result = graph.invoke({"session_id": session_id, "raw": fragments})
    return {"stats": result["stats"], "clean": result["clean"], "store": store}
