"""Phase 4 初稿生成：阶段分组 → 大纲 → 逐章写作 → 事实回检 → 统稿成稿。

    run_drafting(store, session_id, elder, birth_year)
        → gather → stages → outline → write ⇄ check → polish → render
"""
from .llm import (
    DemoDraftingLLM,
    DraftingLLM,
    LangChainDraftingLLM,
    get_drafting_llm,
)
from .manuscript import render_manuscript
from .models import (
    ChapterDraft,
    ChapterSpec,
    FactCheck,
    Flag,
    Outline,
    ReviewItem,
    StageGroup,
)
from .pipeline import build_drafting_graph, normalize_outline, run_drafting
from .stages import fragment_years, group_stages

__all__ = [
    "DemoDraftingLLM",
    "DraftingLLM",
    "LangChainDraftingLLM",
    "get_drafting_llm",
    "render_manuscript",
    "ChapterDraft",
    "ChapterSpec",
    "FactCheck",
    "Flag",
    "Outline",
    "ReviewItem",
    "StageGroup",
    "build_drafting_graph",
    "normalize_outline",
    "run_drafting",
    "fragment_years",
    "group_stages",
]
