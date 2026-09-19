"""Phase 1 初次访谈 Agent（LangGraph 版）。

    START → router → (ask → human → extract → validate → record) ↺
              │                                              │
              └──────────────→ closing → END                 │
                                   ↑                         │
                        fix → extract（验证失败修复环）────────┘

判断交给 LLM，流程交给图，质量交给验证器。
"""

from .agent import AgentReply, InterviewAgent
from .graph import build_interview_graph
from .llm import DemoInterviewLLM, LangChainInterviewLLM, get_llm
from .models import (
    AgentConfig,
    ElderProfile,
    FragmentBatch,
    InterviewState,
    MemoryFragment,
    RouteContext,
    RouteDecision,
    Topic,
)
from .validators import ValidationReport, is_farewell, validate_fragments

__all__ = [
    "AgentReply",
    "InterviewAgent",
    "build_interview_graph",
    "DemoInterviewLLM",
    "LangChainInterviewLLM",
    "get_llm",
    "AgentConfig",
    "ElderProfile",
    "FragmentBatch",
    "InterviewState",
    "MemoryFragment",
    "RouteContext",
    "RouteDecision",
    "Topic",
    "ValidationReport",
    "is_farewell",
    "validate_fragments",
]
