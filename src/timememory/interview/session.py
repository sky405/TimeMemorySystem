"""会话 JSON 持久化：老人中途离开，下次可恢复继续（补充访谈复用同一状态机）。"""
from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from .models import (
    AgentConfig,
    Decision,
    DecisionAction,
    ElderProfile,
    InterviewState,
    MemoryFragment,
    Message,
    SessionStatus,
    Speaker,
)


def _encode(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_encode(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    return obj


def state_to_dict(state: InterviewState) -> dict:
    return _encode(asdict(state))


def state_from_dict(d: dict) -> InterviewState:
    st = InterviewState()
    st.session_id = d.get("session_id", st.session_id)
    st.elder = ElderProfile(**d.get("elder", {}))
    st.config = AgentConfig(**d.get("config", {}))
    st.status = SessionStatus(d.get("status", "opening"))
    st.current_topic_id = d.get("current_topic_id", "")
    st.turn_count = d.get("turn_count", 0)
    st.topic_turn_count = d.get("topic_turn_count", 0)
    st.covered_topic_ids = d.get("covered_topic_ids", [])
    st.topic_turns = d.get("topic_turns", {})
    st.messages = [
        Message(speaker=Speaker(m["speaker"]), text=m["text"],
                topic_id=m.get("topic_id", ""), turn_index=m.get("turn_index", 0),
                timestamp=m.get("timestamp", 0.0))
        for m in d.get("messages", [])
    ]
    st.fragments = [MemoryFragment(**f) for f in d.get("fragments", [])]
    st.expanded_entities = d.get("expanded_entities", [])
    st.short_reply_streak = d.get("short_reply_streak", 0)
    st.decisions = [
        Decision(action=DecisionAction(x["action"]), confidence=x.get("confidence", 0.0),
                 reasoning=x.get("reasoning", ""), signals=x.get("signals", {}),
                 focus=x.get("focus", ""))
        for x in d.get("decisions", [])
    ]
    return st


def save_state(state: InterviewState, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state_to_dict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_state(path: str | Path) -> InterviewState:
    return state_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
