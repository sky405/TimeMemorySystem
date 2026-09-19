"""ReviewLLM 协议 + LangChain 真模型 + Demo 实现。

有 `TMS_LLM_API_KEY` 用真模型（与前序阶段共用环境变量），
否则 Demo 建议删除无出处内容，离线可跑。
"""
from __future__ import annotations

import json
import os
from typing import Protocol

from .models import FixSuggestion
from .prompts import FIX_SYSTEM, FIX_USER


class ReviewLLM(Protocol):
    def suggest_fix(self, chapter: str, text: str, quote: str,
                    issue: str, materials: list[dict]) -> FixSuggestion: ...


class LangChainReviewLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float | None = None):
        from langchain_openai import ChatOpenAI

        from ..config import get_config
        cfg = get_config()
        if temperature is None:
            temperature = cfg.review.temperature
        if temperature is None:
            temperature = 0.2
        self.chat = ChatOpenAI(
            model=model or cfg.llm.model or "deepseek-chat",
            base_url=base_url or cfg.llm.base_url,
            api_key=api_key or cfg.llm.api_key,
            temperature=temperature,
        )

    def suggest_fix(self, chapter, text, quote, issue, materials) -> FixSuggestion:
        slim = [{k: m.get(k) for k in
                 ("id", "content", "topic_id", "time_refs", "place_refs", "person_refs")}
                for m in materials]
        return self.chat.with_structured_output(FixSuggestion).invoke([
            ("system", FIX_SYSTEM),
            ("user", FIX_USER.format(chapter=chapter, quote=quote, issue=issue,
                                     text=text,
                                     materials=json.dumps(slim, ensure_ascii=False))),
        ])


class DemoReviewLLM:
    def suggest_fix(self, chapter, text, quote, issue, materials) -> FixSuggestion:
        return FixSuggestion(action="删除该句", replacement="",
                             reason="Demo 模式：素材无出处，建议删除该句。")


def get_review_llm() -> ReviewLLM:
    from ..config import get_config
    cfg = get_config().llm
    if cfg.api_key:
        print("[TimeMemory] Phase 5 使用真模型提修订建议")
        return LangChainReviewLLM()
    print("[TimeMemory] Phase 5 使用 Demo 修订建议（离线）")
    return DemoReviewLLM()
