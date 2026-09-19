"""InterviewAgent 编排器：把"话题 → 提问 → 回答 → 决策"闭环跑起来。

对外 API 只有三个动作：
    agent = InterviewAgent(elder_profile, llm=...)
    agent.start()              # → 开场白 + 首个问题
    agent.step("老人回答...")   # → AgentReply（含本轮决策、三叉路口分支）
    agent.coverage_report()    # → 交给 Phase 3 的完整度报告
"""
from __future__ import annotations

from dataclasses import dataclass

from .decision import DecisionEngine
from .extractor import FragmentExtractor
from .llm import LLMClient, MockLLMClient
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
from .questions import QuestionGenerator
from .report import build_coverage_report, export_fragments
from .topics import TOPIC_MAP, TopicPlanner


@dataclass
class AgentReply:
    text: str  # 对老人说的话
    decision: Decision | None  # 本轮三叉路口决策（start() 时为 None）
    new_fragments: list[MemoryFragment]  # 本轮新提取的记忆片段
    session_ended: bool = False


class InterviewAgent:
    def __init__(
        self,
        elder: ElderProfile | None = None,
        config: AgentConfig | None = None,
        llm: LLMClient | None = None,
        seed: int | None = None,
    ):
        self.state = InterviewState(elder=elder or ElderProfile(), config=config or AgentConfig())
        # 默认 Mock（确定性、离线可用）；想用真模型请显式传入 get_default_client()
        self.llm: LLMClient = llm or MockLLMClient()
        self.planner = TopicPlanner()
        self.decider = DecisionEngine(self.planner)
        self.extractor = FragmentExtractor()
        self.asker = QuestionGenerator(seed=seed)

    # -- 访谈开始 ---------------------------------------------------------------
    def start(
        self,
        first_topic_id: str | None = None,
        opening_override: str | None = None,
    ) -> AgentReply:
        """开始访谈（支持补充访谈：指定 first_topic_id / opening_override 直切缺口）。

        对应流程图 [1. 确定当前话题] → [2. AI 发起提问]。
        """
        st = self.state
        topic_id = first_topic_id or st.config.first_topic_id
        if topic_id not in TOPIC_MAP:
            topic_id = st.config.first_topic_id
        st.current_topic_id = topic_id
        st.topic_turn_count = 0
        st.status = SessionStatus.IN_PROGRESS

        topic = TOPIC_MAP[topic_id]
        first_q = opening_override or self.asker.opening(topic)
        text = self.asker.greeting(st, topic, first_q)
        st.messages.append(Message(speaker=Speaker.AI, text=text, topic_id=topic_id, turn_index=0))
        return AgentReply(text=text, decision=None, new_fragments=[], session_ended=False)

    # -- 核心循环：老人回答 → 提取 → 决策 → 下一句 ---------------------------------
    def step(self, elder_text: str) -> AgentReply:
        """处理一轮老人回答，走完 [3. 老人回答] → [提取片段] → [4. 三叉路口决策] → 下一句。"""
        st = self.state
        if st.status == SessionStatus.ENDED:
            return AgentReply(
                text="咱们这次访谈已经结束了，谢谢您！好好休息。",
                decision=None,
                new_fragments=[],
                session_ended=True,
            )
        text = (elder_text or "").strip()
        if not text:
            return AgentReply(
                text="我没太听清，您能再说一遍吗？",
                decision=None,
                new_fragments=[],
                session_ended=False,
            )

        # 计数与落盘消息
        st.turn_count += 1
        st.topic_turn_count += 1
        st.topic_turns[st.current_topic_id] = st.topic_turns.get(st.current_topic_id, 0) + 1
        if len(text) < 8:
            st.short_reply_streak += 1
        else:
            st.short_reply_streak = 0
        st.messages.append(
            Message(speaker=Speaker.ELDER, text=text, topic_id=st.current_topic_id,
                    turn_index=st.turn_count)
        )

        # 提取记忆片段
        new_frags = self.extractor.extract(text, st.current_topic_id, st.turn_count, llm=self.llm)
        st.fragments.extend(new_frags)

        # 三叉路口决策
        judge_llm = self.llm if st.config.use_llm_judge else None
        decision = self.decider.decide(st, text, new_frags, llm=judge_llm)
        st.decisions.append(decision)

        # R1 告别：告别语本身不是记忆，不留片段（逐字稿仍保留，可审计）
        if decision.signals.get("hard_rule") == "R1_exit_word" and new_frags:
            del st.fragments[-len(new_frags):]
            new_frags = []

        # 按分支执行
        if decision.action is DecisionAction.DEEP_DIVE:
            reply_text = self.asker.followup(st, text, new_frags, decision.focus, llm=self.llm)
            ended = False
        elif decision.action is DecisionAction.SWITCH_TOPIC:
            reply_text, ended = self._do_switch(decision)
        else:  # WRAP_UP
            reply_text = self._do_wrap_up(decision)
            ended = True

        st.messages.append(
            Message(speaker=Speaker.AI, text=reply_text, topic_id=st.current_topic_id,
                    turn_index=st.turn_count)
        )
        return AgentReply(text=reply_text, decision=decision,
                          new_fragments=new_frags, session_ended=ended)

    # -- 分支 B：切换话题 ----------------------------------------------------------
    def _do_switch(self, decision: Decision) -> tuple[str, bool]:
        st = self.state
        old_topic = TOPIC_MAP[st.current_topic_id]

        new_id = decision.focus if decision.focus in TOPIC_MAP else ""
        if not new_id:
            from .extractor import recent_entities

            nxt, _ = self.planner.next_topic(
                st, recent_entities(st) + [e for f in st.fragments[-3:] for e in f.entities]
            )
            new_id = nxt.id if nxt else ""
        if not new_id or new_id == st.current_topic_id:
            # 无话题可切 → 就地收尾
            return self._do_wrap_up(decision), True

        new_topic = TOPIC_MAP[new_id]
        # 搭桥词：新话题关键词 ∩ 老人最近提到的实体
        recent_blob = "".join(f.content for f in st.fragments[-4:])
        bridge = next((kw for kw in new_topic.keywords if kw and kw in recent_blob), "")
        # 本话题亮点（收尾过渡用）
        old_frags = sorted(st.fragments_of(old_topic.id), key=lambda f: -f.importance)
        highlight = ""
        if old_frags:
            h = old_frags[0].content
            highlight = (h[:18] + "…") if len(h) > 18 else h

        if old_topic.id not in st.covered_topic_ids:
            st.covered_topic_ids.append(old_topic.id)
        st.current_topic_id = new_id
        st.topic_turn_count = 0
        reply = self.asker.transition(st, old_topic, new_topic, bridge, highlight, llm=self.llm)
        return reply, False

    # -- 分支 C：收尾 ---------------------------------------------------------------
    def _do_wrap_up(self, decision: Decision) -> str:
        st = self.state
        if st.current_topic_id and st.current_topic_id not in st.covered_topic_ids:
            st.covered_topic_ids.append(st.current_topic_id)
        st.status = SessionStatus.ENDED
        return self.asker.closing(st)

    # -- 交接与导出 -----------------------------------------------------------------
    def coverage_report(self) -> dict:
        """素材完整度报告 → Phase 3 写作评估的输入。"""
        return build_coverage_report(self.state)

    def fragments_json(self) -> list[dict]:
        """记忆片段 → Phase 2 素材处理的输入。"""
        return export_fragments(self.state)

    def export_transcript_markdown(self) -> str:
        """逐字稿 Markdown → Phase 5 人工审核时家属可读。"""
        st = self.state
        lines = [
            f"# 访谈逐字稿（{st.session_id}）",
            "",
            f"- 受访老人：{st.elder.name}" + (f"（{st.elder.age} 岁）" if st.elder.age else ""),
            f"- 覆盖话题：{len(st.covered_topic_ids)} 个，记忆片段：{len(st.fragments)} 条",
            "",
        ]
        di = 0
        for m in st.messages:
            who = "访谈员" if m.speaker == Speaker.AI else "老人"
            lines.append(f"**{who}**：{m.text}")
            lines.append("")
            if m.speaker == Speaker.ELDER and di < len(st.decisions):
                d = st.decisions[di]
                di += 1
                branch = {"deep_dive": "A·深挖", "switch_topic": "B·切换", "wrap_up": "C·收尾"}[
                    d.action.value
                ]
                lines.append(f"> 🔀 决策：{branch}（置信度 {d.confidence}）——{d.reasoning}")
                lines.append("")
        lines += ["---", "", "## 记忆片段", ""]
        for f in st.fragments:
            lines.append(
                f"- [{f.topic_id}] {f.content} "
                f"（⭐{f.importance}，时间{f.time_refs or '-'}，地点{f.place_refs or '-'}，"
                f"人物{f.person_refs or '-'}，情感{f.emotion or '-'}）"
            )
        return "\n".join(lines) + "\n"
