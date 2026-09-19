"""Phase 4 数据模型：大纲 / 章节草稿 / 事实核查 / 复核清单。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class StageGroup(BaseModel):
    """一年生阶段的素材分组（确定性分组产出）。"""

    id: str  # "stage-0"
    title: str  # "童年（1960年代）"或"1960年代"
    years: list[int] = []
    fragment_ids: list[str] = []
    topics: list[str] = []  # 覆盖的话题 id，供无年份素材归位


class ChapterSpec(BaseModel):
    id: str
    title: str
    stage: str = ""
    brief: str = ""  # 本章写什么
    fragment_ids: list[str] = []


class Outline(BaseModel):
    title: str
    chapters: list[ChapterSpec] = []
    notes: str = ""


class Flag(BaseModel):
    quote: str  # 存疑原文
    issue: str  # 问题说明
    severity: Literal["存疑", "错误"] = "存疑"


class FactCheck(BaseModel):
    passed: bool = True
    summary: str = ""
    flags: list[Flag] = []


class ChapterDraft(BaseModel):
    chapter_id: str
    title: str
    text: str
    summary: str = ""  # 本章摘要，供下一章保持连贯
    fragment_ids: list[str] = []
    factcheck: FactCheck = FactCheck()


class ReviewItem(BaseModel):
    """人工复核入口：一条存疑 + 查证范围。"""

    chapter: str
    quote: str
    issue: str
    severity: str = "存疑"
    fragment_ids: list[str] = []
