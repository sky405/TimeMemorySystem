"""Phase 5 数据模型：AI 修订建议 / 人工裁决 / 审核记录。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Verdict = Literal["确认无误", "已修正", "存疑保留", "删除相关句"]
FixAction = Literal["保留", "改写", "删除该句"]

VERDICTS: tuple[str, ...] = ("确认无误", "已修正", "存疑保留", "删除相关句")


class FixSuggestion(BaseModel):
    action: FixAction = "删除该句"
    replacement: str = ""  # 改写时的新句子
    reason: str = ""


class ItemReview(BaseModel):
    chapter: str
    quote: str
    issue: str
    severity: str = "存疑"
    fragment_ids: list[str] = []
    suggestion: FixSuggestion = FixSuggestion()
    verdict: str = ""  # 人工裁决
    note: str = ""
    applied: bool = False  # 是否实际改了正文


class ChapterFinal(BaseModel):
    chapter_id: str
    title: str
    text: str  # 修正后正文
    fragment_ids: list[str] = []
    approved: bool = False
