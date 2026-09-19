"""访谈 Agent 数据模型（纯标准库，无第三方依赖）。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class Speaker(str, Enum):
    AI = "ai"
    ELDER = "elder"
    SYSTEM = "system"


class SessionStatus(str, Enum):
    OPENING = "opening"
    IN_PROGRESS = "in_progress"
    CLOSING = "closing"
    ENDED = "ended"


class DecisionAction(str, Enum):
    """三叉路口的三个分支。"""

    DEEP_DIVE = "deep_dive"  # 分支 A：发现好故事，深度追问
    SWITCH_TOPIC = "switch_topic"  # 分支 B：话题聊干了，切换新话题
    WRAP_UP = "wrap_up"  # 分支 C：老人累了/要走，收尾结束


@dataclass
class Topic:
    """人生话题。"""

    id: str
    name: str
    description: str
    opening_questions: list[str]
    followup_angles: list[str]
    keywords: list[str]
    min_turns: int = 2
    max_turns: int = 6
    priority: int = 50
    sensitive: bool = False


@dataclass
class ElderProfile:
    """老人基本信息（注册时由家属填写）。"""

    name: str = "老人家"
    age: int | None = None
    gender: str = ""
    hometown: str = ""
    dialect: str = ""
    health_notes: str = ""
    known_facts: list[str] = field(default_factory=list)


@dataclass
class Message:
    speaker: Speaker
    text: str
    topic_id: str = ""
    turn_index: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryFragment:
    """从老人回答中提取的结构化记忆片段（交给 Phase 2/3 的核心产出）。"""

    id: str
    content: str
    topic_id: str
    source_turn: int
    time_refs: list[str] = field(default_factory=list)
    place_refs: list[str] = field(default_factory=list)
    person_refs: list[str] = field(default_factory=list)
    emotion: str = ""
    importance: int = 3  # 1~5
    entities: list[str] = field(default_factory=list)
    needs_followup: bool = False
    followup_hint: str = ""


@dataclass
class Decision:
    """三叉路口决策结果。"""

    action: DecisionAction
    confidence: float
    reasoning: str
    signals: dict = field(default_factory=dict)
    focus: str = ""  # DEEP_DIVE=追问焦点实体；SWITCH_TOPIC=新话题 id；WRAP_UP=收尾原因


@dataclass
class AgentConfig:
    """Agent 可调参数。"""

    # 轮次控制
    max_total_turns: int = 30  # 单次访谈老人回答轮次硬上限（红线）
    soft_total_turns: int = 24  # 软上限：接近时 fatigue 爬升
    # 决策阈值
    fatigue_wrap_threshold: float = 0.75
    exhaustion_switch_hard: float = 0.85  # 极高枯竭：即使没聊够 min_turns 也切换
    exhaustion_switch_soft: float = 0.60
    story_dive_threshold: float = 0.35
    # 提问
    max_questions_per_turn: int = 1  # 铁律：一次只问一个问题
    # LLM
    use_llm_judge: bool = True  # 是否启用 LLM 裁判（第 3 层）
    use_llm_polish: bool = True  # 是否用 LLM 润色提问
    llm_weight: float = 0.4  # 融合时 LLM 投票权重
    # 敏感话题
    sensitive_min_turns: int = 8  # 访谈满 N 轮后才允许切入敏感话题
    first_topic_id: str = "childhood"


@dataclass
class InterviewState:
    """整场访谈的可序列化状态。"""

    session_id: str = field(default_factory=lambda: f"iv-{uuid.uuid4().hex[:8]}")
    elder: ElderProfile = field(default_factory=ElderProfile)
    config: AgentConfig = field(default_factory=AgentConfig)
    status: SessionStatus = SessionStatus.OPENING
    current_topic_id: str = ""
    turn_count: int = 0  # 老人已回答的总轮数
    topic_turn_count: int = 0  # 当前话题已聊轮数
    covered_topic_ids: list[str] = field(default_factory=list)
    topic_turns: dict[str, int] = field(default_factory=dict)  # 各话题累计轮数
    messages: list[Message] = field(default_factory=list)
    fragments: list[MemoryFragment] = field(default_factory=list)
    expanded_entities: list[str] = field(default_factory=list)  # 已追问过的实体
    short_reply_streak: int = 0  # 连续极短回复计数
    decisions: list[Decision] = field(default_factory=list)

    def elder_messages(self) -> list[Message]:
        return [m for m in self.messages if m.speaker == Speaker.ELDER]

    def last_elder_text(self) -> str:
        msgs = self.elder_messages()
        return msgs[-1].text if msgs else ""

    def fragments_of(self, topic_id: str) -> list[MemoryFragment]:
        return [f for f in self.fragments if f.topic_id == topic_id]
