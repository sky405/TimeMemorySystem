"""Phase 6 家族记忆库：定稿入库 + RAG 问答。

    ingest_book(store, embedder, archive, chapters, fragments, records)
    ChatSession(store, ...).ask("王二哥是谁？")
        → rewrite → retrieve → gate → answer/noanswer
"""
from .ingest import ingest_book
from .llm import DemoLibraryLLM, LangChainLibraryLLM, LibraryLLM, get_library_llm
from .models import Answer, Caution, Citation, Passage
from .pipeline import ChatSession, build_library_graph

__all__ = [
    "ingest_book",
    "DemoLibraryLLM",
    "LangChainLibraryLLM",
    "LibraryLLM",
    "get_library_llm",
    "Answer",
    "Caution",
    "Citation",
    "Passage",
    "ChatSession",
    "build_library_graph",
]
