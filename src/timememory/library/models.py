"""Phase 6 数据模型：检索段落 / 引用 / 回答。"""
from __future__ import annotations

from pydantic import BaseModel


class Passage(BaseModel):
    id: str
    kind: str = "fragment"  # fragment 访谈片段 | chapter 定稿章节
    content: str = ""
    topic: str = ""
    score: float = 0.0
    via: str = "vector"  # vector | kg:关系
    caution: bool = False  # 是否含存疑待考内容


class Citation(BaseModel):
    passage_id: str
    quote: str = ""


class Answer(BaseModel):
    text: str
    citations: list[Citation] = []
    has_answer: bool = True


class Caution(BaseModel):
    quote: str
    chapter: str = ""
