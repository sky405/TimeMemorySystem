"""Phase 2 素材处理：清洗 → 向量化 → 知识图谱（MySQL）。

    run_material_pipeline(session, phase1_fragments)
        → clean → embed → extract_kg → persist
"""

from .cleaning import clean_fragments
from .embeddings import DemoEmbedder, OpenAIEmbedder, cosine, get_embedder
from .kg import NODE_TYPES, node_id, normalize_extraction
from .llm import DemoMaterialLLM, LangChainMaterialLLM, get_material_llm
from .models import CleanFragment, KGEdge, KGEdgeDraft, KGExtraction, KGNode, KGNodeDraft
from .pipeline import build_material_graph, run_material_pipeline
from .query import ScoredFragment, retrieve
from .store import MaterialStore, connect_mysql, connect_sqlite, get_db

__all__ = [
    "clean_fragments",
    "DemoEmbedder",
    "OpenAIEmbedder",
    "cosine",
    "get_embedder",
    "NODE_TYPES",
    "node_id",
    "normalize_extraction",
    "DemoMaterialLLM",
    "LangChainMaterialLLM",
    "get_material_llm",
    "CleanFragment",
    "KGEdge",
    "KGEdgeDraft",
    "KGExtraction",
    "KGNode",
    "KGNodeDraft",
    "build_material_graph",
    "run_material_pipeline",
    "ScoredFragment",
    "retrieve",
    "MaterialStore",
    "connect_mysql",
    "connect_sqlite",
    "get_db",
]
