"""LibraryLLM 协议 + LangChain 真模型 + Demo 抽取式实现。

有 `TMS_LLM_API_KEY` 用真模型（与前序阶段共用环境变量），
否则 Demo 返回首段摘录 + 出处，离线可跑。
"""
from __future__ import annotations

import json
import os
from typing import Protocol

from .models import Answer, Citation
from .prompts import ANSWER_SYSTEM, ANSWER_USER, REWRITE_SYSTEM, REWRITE_USER


class LibraryLLM(Protocol):
    def rewrite(self, history: list[dict], question: str) -> str: ...
    def answer(self, question: str, passages: list[dict], triples: list[str]) -> Answer: ...


class LangChainLibraryLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.3):
        from langchain_openai import ChatOpenAI

        self.chat = ChatOpenAI(
            model=model or os.environ.get("TMS_LLM_MODEL", "deepseek-chat"),
            base_url=base_url or os.environ.get("TMS_LLM_BASE_URL"),
            api_key=api_key or os.environ.get("TMS_LLM_API_KEY"),
            temperature=temperature,
        )

    def rewrite(self, history, question) -> str:
        hist = "\n".join(f"[Q] {h.get('q', '')}\n[A] {h.get('a', '')[:100]}"
                         for h in history[-6:]) or "（无）"
        return (self.chat.invoke([
            ("system", REWRITE_SYSTEM),
            ("user", REWRITE_USER.format(history=hist, question=question)),
        ]).content or "").strip() or question.strip()

    def answer(self, question, passages, triples) -> Answer:
        return self.chat.with_structured_output(Answer).invoke([
            ("system", ANSWER_SYSTEM),
            ("user", ANSWER_USER.format(
                question=question,
                passages=json.dumps(passages, ensure_ascii=False),
                triples="\n".join(triples) or "（无）")),
        ])


class DemoLibraryLLM:
    """离线实现：改写原样返回；回答返回首段摘录 + 出处 + 图谱线索。"""

    def rewrite(self, history: list[dict], question: str) -> str:
        return question.strip()

    def answer(self, question: str, passages: list[dict], triples: list[str]) -> Answer:
        top = passages[0]
        kind_cn = "定稿章节" if top["kind"] == "chapter" else "访谈片段"
        text = f"根据记忆库记载：{top['content'][:150]}"
        if len(top["content"]) > 150:
            text += "…"
        text += f"\n\n（出自{kind_cn} {top['id']}）"
        if triples:
            text += f"\n相关线索：{'；'.join(triples[:3])}"
        if top.get("caution"):
            text += "\n注：该段内容存疑待考，转述时请谨慎。"
        cites = [Citation(passage_id=p["id"], quote=p["content"][:30])
                 for p in passages[:2]]
        return Answer(text=text, citations=cites, has_answer=True)


def get_library_llm() -> LibraryLLM:
    if os.environ.get("TMS_LLM_API_KEY"):
        print("[TimeMemory] Phase 6 使用真模型问答")
        return LangChainLibraryLLM()
    print("[TimeMemory] Phase 6 使用 Demo 抽取式问答（离线）")
    return DemoLibraryLLM()
