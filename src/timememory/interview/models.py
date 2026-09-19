"""数据模型：pydantic 结构化输出 schema + LangGraph 状态定义。

设计原则：判断交给 LLM（经由这些 schema 做结构化输出），
本文件只定义"形状"，不包含任何判断逻辑。
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

RouteAction = Literal["followup", "switch", "wrap"]


# ----------------------------------------------------------------------------
# 访谈领域模型
# ----------------------------------------------------------------------------
@dataclass
class Topic:
    id: str
    name: str
    description: str
    opening_questions: list[str] = field(default_factory=list)


@dataclass
class ElderProfile:
    name: str = "老人家"
    age: int | None = None
    hometown: str = ""
    known_facts: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    max_total_turns: int = 30  # 预算护栏：只管"聊多久"，不管"怎么聊"
    max_fix_retries: int = 2  # 抽取→验证失败后的修复重试次数
    llm_validate: bool = True  # 验证器是否启用 LLM 合理性检查
    first_topic_id: str = "childhood"
    recent_window: int = 6  # 路由时回看最近几轮对话


# ----------------------------------------------------------------------------
# LLM 结构化输出 schema（同时是验证器的输入）
# ----------------------------------------------------------------------------
class MemoryFragment(BaseModel):
    """一条有效记忆片段。content 尽量用老人原话，不改写。"""

    content: str = Field(description="记忆内容原文")
    time_refs: list[str] = Field(default_factory=list, description="时间表达，如 1962年/9岁/小时候")
    place_refs: list[str] = Field(default_factory=list, description="地点表达，如 嘉陵江/合川县")
    person_refs: list[str] = Field(default_factory=list, description="人物表达，如 邻居王二哥/我娘")
    emotion: str = Field(default="", description="情感：开心/怀念/难过/害怕/骄傲/愤怒/平静，无则空")
    importance: int = Field(default=3, ge=1, le=5, description="故事价值 1~5")
    needs_followup: bool = Field(default=False, description="是否值得追问")
    topic_id: str = Field(default="", description="所属话题，由 record 节点盖章")
    source_turn: int = Field(default=0, description="来源轮次，由 record 节点盖章")


class FragmentBatch(BaseModel):
    fragments: list[MemoryFragment] = Field(default_factory=list)


class RouteDecision(BaseModel):
    """三叉路口决策：全部由 LLM 做出。"""

    action: RouteAction = Field(description="followup 深挖 / switch 切换话题 / wrap 收尾")
    reasoning: str = Field(description="一句话中文理由")
    focus: str = Field(default="", description="followup 追问焦点；switch 可空")
    next_topic_id: str = Field(default="", description="switch 的新话题 id（须在候选内）")
    prev_topic_id: str = Field(default="", description="switch 时的老话题 id（过渡用，由程序回填）")


class PlausibilityVerdict(BaseModel):
    valid: bool = Field(description="该片段是否为一条真实有效的记忆")
    reason: str = Field(default="", description="理由")


# ----------------------------------------------------------------------------
# 传给 LLM 的上下文（纯数据，由 graph 组装）
# ----------------------------------------------------------------------------
@dataclass
class RouteContext:
    elder_name: str
    current_topic: Topic
    topic_turns_current: int
    turn_count: int
    max_turns: int
    covered_names: list[str]
    uncovered: list[Topic]  # 可切换的候选话题
    recent_qa: list[tuple[str, str]]  # [(AI提问, 老人回答)]
    last_answer: str


@dataclass
class ClosingContext:
    elder_name: str
    highlights: list[str]
    n_topics: int
    n_fragments: int


# ----------------------------------------------------------------------------
# LangGraph 状态
# ----------------------------------------------------------------------------
class InterviewState(TypedDict, total=False):
    elder: dict
    config: dict
    messages: Annotated[list[dict], operator.add]  # {role: ai/elder, text, topic, turn}
    trail: Annotated[list[dict], operator.add]  # 每轮路由决策轨迹
    fragments: Annotated[list[dict], operator.add]  # 已提交的有效片段
    covered_topic_ids: Annotated[list[str], operator.add]
    current_topic_id: str
    turn_count: int
    topic_turns: dict[str, int]
    candidates: list[dict]  # 本轮抽取候选
    validation: dict  # {"valid": [...], "rejected": [{"content":..., "reasons":[...]}]}
    last_rejected: list[dict]  # 上轮被驳回的候选（record 回填，便于观察验证器）
    feedback: str  # 修复反馈
    retries: int
    decision: dict
    pending_question: str
    last_reply: str  # 本轮对老人说的话（提问或收尾）
    new_fragments: list[dict]  # 本轮新提交的片段（供 reply 使用）
    script: list[dict]  # 补访提纲脚本 [{topic_id, question}]，问完即删（编排层注入）
    status: str  # in_progress | ended
