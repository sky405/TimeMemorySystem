"""编排层状态：访谈 Agent 与写作 Agent 的共享状态存储。

状态只放可序列化的数据（dict/list/字符串/数字）；store、answer_fn、
各 LLM 经 config.configurable 注入，不进 state。
"""
from __future__ import annotations

from typing import TypedDict

STATUS_COMPLETE = "complete"  # 素材就绪，成稿完毕
STATUS_COMPLETE_WITH_FLAGS = "complete_with_flags"  # 成稿完毕，但有存疑待人工审核
STATUS_MAX_ROUNDS = "max_rounds"  # 轮次用尽，带缺口成稿


class RoundReport(TypedDict):
    round: int
    session_id: str
    turns: int
    fragments: int
    focus_topics: list[str]  # 本轮补访话题（首轮为空）
    script_used: int  # 采用了写作 Agent 的几个问题
    questions_asked: list[str]


class MemoirState(TypedDict, total=False):
    archive_id: str
    elder: dict
    birth_year: int | None
    max_rounds: int
    round_index: int  # 已完成轮次数（下一轮序号）
    session_ids: list[str]
    round_reports: list[dict]
    round_fragments: list[dict]  # 本轮新片段
    total_fragments: int
    material_stats: dict
    assessment: dict
    plan_items: list[dict]
    ready: bool
    brief: str
    manuscript: str
    review: list[dict]
    drafts: list[dict]  # 章节草稿（Phase 5 审核的输入）
    draft_stats: dict
    outline_title: str
    status: str
