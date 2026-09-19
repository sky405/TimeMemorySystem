"""Phase 3 数据模型：评估报告 + 补充访谈计划。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Priority = Literal["高", "中", "低"]

DIMENSIONS = ("话题覆盖", "时间线完整", "人物丰满度", "细节情感")


class DimensionScore(BaseModel):
    dimension: str
    score: int = Field(default=0, ge=0, le=10)
    summary: str = ""


class Gap(BaseModel):
    """一个素材缺口：缺什么、回哪个话题补。"""

    dimension: str = ""
    title: str
    detail: str = ""
    priority: Priority = "中"
    topic_id: str = ""  # Phase 1 话题 id，回访入口
    subject: str = ""  # 缺口主体（人名/年代等），用于生成追问
    evidence_ids: list[str] = []


class Assessment(BaseModel):
    ready: bool = False  # 素材是否足够动笔写传
    overall: str = ""
    dimensions: list[DimensionScore] = []
    strengths: list[str] = []
    gaps: list[Gap] = []


class PlanItem(BaseModel):
    """一个话题的补充访谈任务。"""

    topic_id: str
    topic_name: str = ""
    priority: Priority = "中"
    reason: str = ""
    questions: list[str] = []


class SupplementPlan(BaseModel):
    items: list[PlanItem] = []
    note: str = ""
