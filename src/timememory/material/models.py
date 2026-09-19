"""Phase 2 素材处理：数据模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CleanFragment(BaseModel):
    """清洗后的记忆片段（Phase 1 片段 + 规范 id + 清洗文本）。"""

    id: str  # f"{session_id}:{frag_id}"，跨会话唯一
    session_id: str = ""
    content: str
    topic_id: str = ""
    source_turn: int = 0
    time_refs: list[str] = Field(default_factory=list)
    place_refs: list[str] = Field(default_factory=list)
    person_refs: list[str] = Field(default_factory=list)
    emotion: str = ""
    importance: int = 3


# ----------------------------------------------------------------------------
# 知识图谱：抽取草稿（LLM 输出）→ 入库形态（id 归一后）
# ----------------------------------------------------------------------------
class KGNodeDraft(BaseModel):
    type: str = Field(description="person/place/event/time/object/org 之一")
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = Field(default="", description="一句话介绍，可空")


class KGEdgeDraft(BaseModel):
    src_name: str
    src_type: str = "object"
    dst_name: str
    dst_type: str = "object"
    relation: str = Field(description="关系，如 居住/救起/位于/发生于")
    confidence: float = 0.8


class KGExtraction(BaseModel):
    nodes: list[KGNodeDraft] = Field(default_factory=list)
    edges: list[KGEdgeDraft] = Field(default_factory=list)


class KGNode(BaseModel):
    id: str = ""  # sha1(type + name)，确定性 id，天然去重
    type: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class KGEdge(BaseModel):
    src_id: str
    dst_id: str
    relation: str
    evidence_fragment_id: str = ""  # 证据片段，可溯源
    confidence: float = 0.5
