"""Phase 1 初次访谈 Agent。

核心循环（见 docs/phase1-interview-agent-design.md）：

    [1. 确定当前话题] → [2. AI 提问] → [3. 老人回答] → [4. 三叉路口决策]
                                                              ├── A: 深度追问 → 回到 [3]
                                                              ├── B: 切换话题 → 回到 [1]
                                                              └── C: 收尾总结 → 结束
"""

from .models import (
    AgentConfig,
    Decision,
    DecisionAction,
    ElderProfile,
    MemoryFragment,
    Message,
    SessionStatus,
    Speaker,
    Topic,
    InterviewState,
)
from .agent import AgentReply, InterviewAgent
from .llm import LLMClient, MockLLMClient, OpenAICompatibleClient, get_default_client

__all__ = [
    "AgentConfig",
    "Decision",
    "DecisionAction",
    "ElderProfile",
    "MemoryFragment",
    "Message",
    "SessionStatus",
    "Speaker",
    "Topic",
    "InterviewState",
    "AgentReply",
    "InterviewAgent",
    "LLMClient",
    "MockLLMClient",
    "OpenAICompatibleClient",
    "get_default_client",
]
