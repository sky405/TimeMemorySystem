"""InterviewAgent：图之上的薄封装。

对外 API（保持不变）：
    agent = InterviewAgent(elder, llm=...)   # llm 默认为 get_llm()
    agent.start()              # → 开场白 + 首个问题
    agent.step("老人回答...")   # → AgentReply（决策 + 新片段 + 下一句）
    agent.coverage_report()    # → Phase 3 完整度报告
    agent.export_transcript_markdown()
    agent.save_session(path) / resume via start(resume_state=...)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from langgraph.types import Command

from .graph import build_interview_graph
from .llm import InterviewLLM, get_llm
from .models import AgentConfig, ElderProfile, MemoryFragment, RouteDecision
from .topics import TOPIC_MAP, TOPICS


@dataclass
class AgentReply:
    text: str
    decision: RouteDecision | None
    new_fragments: list[MemoryFragment]
    session_ended: bool = False


class InterviewAgent:
    def __init__(
        self,
        elder: ElderProfile | None = None,
        config: AgentConfig | None = None,
        llm: InterviewLLM | None = None,
    ):
        self.elder = elder or ElderProfile()
        self.config = config or AgentConfig()
        self.llm = llm or get_llm()
        self.graph = build_interview_graph(self.llm)
        self.session_id = f"iv-{uuid.uuid4().hex[:8]}"
        self._thread = {"configurable": {"thread_id": self.session_id}}

    # -- 访谈驱动 ------------------------------------------------------------------
    def _initial_state(self, first_topic_id: str | None = None, resume: dict | None = None) -> dict:
        first = first_topic_id or self.config.first_topic_id
        if first not in TOPIC_MAP:
            first = self.config.first_topic_id
        state = {
            "elder": asdict(self.elder), "config": asdict(self.config),
            "messages": [], "trail": [], "fragments": [], "covered_topic_ids": [],
            "current_topic_id": first, "turn_count": 0, "topic_turns": {},
            "candidates": [], "validation": {}, "last_rejected": [],
            "feedback": "", "retries": 0,
            "status": "in_progress", "new_fragments": [],
            "pending_question": "", "last_reply": "",
        }
        if resume:
            state.update(resume)
        return state

    def _run(self, input_) -> dict:
        self.graph.invoke(input_, self._thread)
        return self.graph.get_state(self._thread).values

    @property
    def state(self) -> dict:
        return self.graph.get_state(self._thread).values

    def start(self, first_topic_id: str | None = None, resume_state: dict | None = None) -> AgentReply:
        """开始访谈（补充访谈：first_topic_id 直切缺口，或 resume_state 恢复会话）。"""
        values = self._run(self._initial_state(first_topic_id, resume_state))
        return self._reply(values)

    def step(self, elder_text: str) -> AgentReply:
        if self.state.get("status") == "ended":
            return AgentReply("咱们这次访谈已经结束了，谢谢您！好好休息。", None, [], True)
        if not (elder_text or "").strip():
            return AgentReply("我没太听清，您能再说一遍吗？", None, [], False)
        return self._reply(self._run(Command(resume=elder_text.strip())))

    def _reply(self, values: dict) -> AgentReply:
        d = values.get("decision")
        return AgentReply(
            text=values.get("last_reply", ""),
            decision=RouteDecision(**d) if d else None,
            new_fragments=[MemoryFragment(**f) for f in values.get("new_fragments", [])],
            session_ended=values.get("status") == "ended",
        )

    # -- 会话持久化 ------------------------------------------------------------------
    def save_session(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @staticmethod
    def load_session(path: str | Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # -- Phase 3 交接：完整度报告 ------------------------------------------------------
    def coverage_report(self) -> dict:
        values = self.state
        topic_turns = values.get("topic_turns", {})
        frags = [MemoryFragment(**f) for f in values.get("fragments", [])]
        by_topic: dict[str, list[MemoryFragment]] = {t.id: [] for t in TOPICS}
        for f in frags:
            by_topic.setdefault(f.topic_id, []).append(f)

        topics_report: dict[str, dict] = {}
        for t in TOPICS:
            turns = topic_turns.get(t.id, 0)
            fs = by_topic.get(t.id, [])
            coverage = round(0.5 * min(1.0, turns / 3) + 0.5 * min(1.0, len(fs) / 4), 2)
            gaps: list[str] = []
            if turns == 0:
                gaps = ["完全未覆盖"]
            else:
                if not any(f.time_refs for f in fs):
                    gaps.append("缺少具体时间（哪一年 / 多大岁数）")
                if not any(f.place_refs for f in fs):
                    gaps.append("缺少地点细节（哪里 / 什么样）")
                if not any(f.person_refs for f in fs):
                    gaps.append("缺少人物（和谁一起 / 印象最深的人）")
                if not any(f.importance >= 4 for f in fs):
                    gaps.append("缺少高价值故事（可再深挖具体经过）")
            asked = turns % len(t.opening_questions)
            topics_report[t.id] = {
                "name": t.name, "turns": turns, "fragments": len(fs),
                "coverage": coverage, "gaps": gaps,
                "suggested_questions": [t.opening_questions[asked]],
            }

        overall = round(sum(r["coverage"] for r in topics_report.values()) / len(topics_report), 2)
        recommendation = ("ready_for_writing" if overall >= 0.7
                          else "supplementary_interview" if overall >= 0.35 else "continue_interview")
        ranked = sorted(topics_report.items(), key=lambda kv: kv[1]["coverage"])
        resume, focus_q = [], []
        for tid, r in ranked:
            if r["coverage"] >= 1.0:
                continue
            resume.append(tid)
            focus_q.extend(r["suggested_questions"][:1])
            if len(resume) >= 3:
                break
        return {
            "session_id": self.session_id, "elder": self.elder.name,
            "total_turns": values.get("turn_count", 0), "total_fragments": len(frags),
            "topics": topics_report, "overall_coverage": overall,
            "recommendation": recommendation,
            "next_interview_plan": {"resume_topics": resume, "focus_questions": focus_q[:3]},
        }

    def fragments_json(self) -> list[dict]:
        return self.state.get("fragments", [])

    # -- Phase 5 交接：逐字稿 ------------------------------------------------------------
    def export_transcript_markdown(self) -> str:
        values = self.state
        trail_by_turn = {t.get("turn"): t for t in values.get("trail", [])}
        branch_cn = {"followup": "A·深挖", "switch": "B·切换", "wrap": "C·收尾"}
        lines = [
            f"# 访谈逐字稿（{self.session_id}）", "",
            f"- 受访老人：{self.elder.name}" + (f"（{self.elder.age} 岁）" if self.elder.age else ""),
            f"- 覆盖话题：{len(values.get('covered_topic_ids', []))} 个，"
            f"记忆片段：{len(values.get('fragments', []))} 条", "",
        ]
        for m in values.get("messages", []):
            who = "访谈员" if m.get("role") == "ai" else "老人"
            lines += [f"**{who}**：{m.get('text', '')}", ""]
            if m.get("role") == "elder":
                # 老人第 N 轮回答之后，展示路由对它的反应（trail.turn == N）
                t = trail_by_turn.get(m.get("turn"))
                if t:
                    lines += [f"> 🔀 决策：{branch_cn.get(t['action'], t['action'])}——{t.get('reasoning', '')}", ""]
        lines += ["---", "", "## 记忆片段", ""]
        for f in values.get("fragments", []):
            lines.append(
                f"- [{f.get('topic_id')}] {f.get('content')} "
                f"（⭐{f.get('importance', 3)}，时间{f.get('time_refs') or '-'}，"
                f"地点{f.get('place_refs') or '-'}，人物{f.get('person_refs') or '-'}，"
                f"情感{f.get('emotion') or '-'}）"
            )
        return "\n".join(lines) + "\n"
