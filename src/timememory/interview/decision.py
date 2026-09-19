"""三叉路口决策引擎 —— 访谈 Agent 最核心的一步。

三层融合架构：
    第 1 层 硬规则 (Rule Veto)：显式告别/疲惫、轮次上限 → 直接裁决，一票否决。
    第 2 层 特征打分 (Heuristic)：story_value / exhaustion / fatigue 三项 0~1 分。
    第 3 层 LLM 裁判 (LLM Judge)：语义理解，输出结构化裁决。

融合规则：硬规则永远优先；LLM 高置信 (≥0.8) 的 WRAP_UP 可直通（宁可早收尾，
不让老人家累着）；其余情况规则与 LLM 加权投票。
"""
from __future__ import annotations

import re

from .extractor import NARRATIVE_MARKERS, detect_emotion, recent_entities
from .llm import LLMClient, LLMError, extract_json_object
from .models import Decision, DecisionAction, InterviewState, MemoryFragment, Speaker, Topic
from .prompts import SYSTEM_JUDGE, build_judge_prompt
from .topics import TOPIC_MAP, TopicPlanner

# -- 关键词表 -----------------------------------------------------------------

# 第 1 层硬规则：告别/疲惫检测（必须高精度——故事里也常出现"吃饭/睡觉/走了"）。
# 四档设计（误判收尾会提前结束访谈，漏判则打扰疲惫老人——是安全关键路径）：
#   ALWAYS：出现即收尾。吃饭/睡觉/休息/做饭只认"动作告别"搭配
#           （"去吃饭了"算，"我妈每天做饭"/"天黑就睡觉"不算）。
#   SHORT：仅 ≤10 字短句视为告别（"下次再聊" vs "下次我再去的时候…"）。
#   SELF：仅第一人称告别（"我走了" vs "我爹走了"=去世；"再见" vs "再见他一面"=重逢愿望）。
#   DROWSY：犯困表达，"我/先"+近距离 或 极短句（"我困了" vs "孩子困了就睡"）。
_EXIT_ALWAYS = [
    "累了", "太累",
    "不聊了", "别聊了", "别问了", "结束吧", "就到这",
    "有事", "来人了", "拜拜",
    "去做饭", "得做饭", "要做饭", "做饭去", "做饭了", "先做饭",
    "去吃饭", "吃饭去", "吃饭了", "先吃饭", "吃完饭再",
    "去睡觉", "睡觉去", "睡觉了", "先睡", "想睡",
    "先休息", "休息一下", "休息一会", "休息会儿", "休息吧", "去休息", "休息了",
]
_EXIT_SHORT = ["下次", "改天", "改日", "下回", "歇会", "歇一会", "歇歇"]
_EXIT_SHORT_MAXLEN = 10
_EXIT_NEGATIVE = ["改天换地"]  # 含此短语时跳过 SHORT 判定（"当年改天换地修水库"是故事）
_EXIT_SELF = ["走了", "挂了", "再见"]
_EXIT_DROWSY_RE = re.compile(r"(我|咱们|先).{0,3}(睡了|困了|想睡)")
_EXIT_DROWSY_SHORT = ["睡了", "困了"]  # 极短句(≤5 字)才算
_EXIT_FIRST_SUCH_RE = re.compile(r"(咱们|那|今天|就)?先这样(吧|！|!|。|\s)*$")
_DEATH_WORDS = ["爹", "爸", "妈", "娘", "爷", "奶", "老伴", "去世", "过世", "死", "不在了"]
_REJOIN_RE = re.compile(r"再见[他她它你爸妈爷奶哥姐弟妹老]")

# 第 2 层：疲惫词（注意不用单字"困"，否则误伤"困难/贫困"；"累"保留，误伤词极罕见）
TIRED_WORDS = ["累", "困了", "发困", "犯困", "乏", "头晕", "不舒服", "没力气", "没劲"]

# 第 2 层：模糊/枯竭表达
VAGUE_PATTERNS = [
    "不知道", "不清楚", "不记得", "记不清", "记不得", "想不起来",
    "忘了", "忘记了", "没什么", "没啥", "就那样", "一般", "还行",
    "都忘了", "早忘了",
]

SHORT_REPLY_LEN = 8  # 少于此字数视为极短回复
SHORT_STREAK_LIMIT = 3


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_exit_signal(text: str) -> str:
    """检测告别信号。返回命中的告别词，无命中返回 ""。"""
    for w in _EXIT_ALWAYS:
        if w in text:
            return w
    if _EXIT_FIRST_SUCH_RE.search(text):
        return "先这样"  # 句尾"先这样(吧)"是告别；"先这样，再那样"是故事
    if not any(neg in text for neg in _EXIT_NEGATIVE):
        if len(text) <= _EXIT_SHORT_MAXLEN:
            for w in _EXIT_SHORT:
                if w in text:
                    return w
    for w in _EXIT_SELF:
        if w not in text:
            continue
        if w == "再见" and _REJOIN_RE.search(text):
            continue  # "再见他一面"是重逢愿望，不是告别
        if any(d in text for d in _DEATH_WORDS):
            continue  # "我爹走了"说的是去世，不是告别
        if "我" in text or "咱们" in text or "先" in text or len(text) <= 6:
            return w
    if _EXIT_DROWSY_RE.search(text):
        return "困了/睡了"
    if len(text) <= 5 and any(w in text for w in _EXIT_DROWSY_SHORT):
        return "困了/睡了"
    return ""


class DecisionEngine:
    """三叉路口决策引擎。"""

    def __init__(self, planner: TopicPlanner | None = None):
        self.planner = planner or TopicPlanner()

    # -- 对外主入口 ------------------------------------------------------------
    def decide(
        self,
        state: InterviewState,
        last_answer: str,
        last_fragments: list[MemoryFragment],
        llm: LLMClient | None = None,
    ) -> Decision:
        cfg = state.config
        topic = TOPIC_MAP[state.current_topic_id]
        text = (last_answer or "").strip()

        # ---- 第 1 层：硬规则（一票否决）----
        hard = self._hard_rules(state, topic, text, last_fragments)
        if hard is not None:
            return hard

        # ---- 第 2 层：特征打分 ----
        heuristic = self._heuristic_scores(state, topic, text, last_fragments)

        # ---- 第 3 层：LLM 裁判（可选）----
        llm_vote = self._llm_judge(state, topic, text, heuristic, llm) if llm else None

        # ---- 融合 ----
        rule_action, rule_conf, rule_why = self._rule_action(state, topic, heuristic)
        signals = dict(heuristic)
        signals["rule_action"] = rule_action.value
        signals["rule_confidence"] = round(rule_conf, 3)
        if llm_vote:
            signals["llm_action"] = llm_vote[0].value
            signals["llm_confidence"] = round(llm_vote[1], 3)
            signals["llm_reasoning"] = llm_vote[2]

        # LLM 高置信收尾直通：宁可早收尾，不让老人累着
        if llm_vote and llm_vote[0] is DecisionAction.WRAP_UP and llm_vote[1] >= 0.8:
            return Decision(
                action=DecisionAction.WRAP_UP,
                confidence=llm_vote[1],
                reasoning=f"LLM 高置信收尾直通：{llm_vote[2]}",
                signals=signals,
                focus="llm_wrap_up",
            )

        # 加权投票
        w_llm = cfg.llm_weight if llm_vote else 0.0
        w_rule = 1.0 - w_llm
        votes: dict[DecisionAction, float] = {rule_action: w_rule * rule_conf}
        if llm_vote:
            votes[llm_vote[0]] = votes.get(llm_vote[0], 0.0) + w_llm * llm_vote[1]
        winner = max(votes.items(), key=lambda kv: kv[1])
        final_conf = min(1.0, winner[1] + (0.1 if llm_vote and llm_vote[0] is winner[0] else 0.0))

        reasoning = rule_why
        if llm_vote:
            reasoning += f"；LLM 裁判：{llm_vote[0].value}（{llm_vote[1]:.2f}，{llm_vote[2]}）"

        focus = self._focus_for(state, winner[0], text, last_fragments)
        return Decision(
            action=winner[0],
            confidence=round(final_conf, 3),
            reasoning=reasoning,
            signals=signals,
            focus=focus,
        )

    # -- 第 1 层：硬规则 --------------------------------------------------------
    def _hard_rules(
        self,
        state: InterviewState,
        topic: Topic,
        text: str,
        last_fragments: list[MemoryFragment],
    ) -> Decision | None:
        cfg = state.config

        # R1: 显式告别/疲惫 → 立刻收尾
        hit = is_exit_signal(text)
        if hit:
            return Decision(
                action=DecisionAction.WRAP_UP,
                confidence=0.95,
                reasoning=f"硬规则 R1：检测到告别/疲惫词「{hit}」，必须立刻收尾",
                signals={"hard_rule": "R1_exit_word", "hit": hit},
                focus=f"老人说「{hit}」",
            )

        # R2: 总轮次硬上限 → 收尾
        if state.turn_count >= cfg.max_total_turns:
            return Decision(
                action=DecisionAction.WRAP_UP,
                confidence=1.0,
                reasoning=f"硬规则 R2：已达单次访谈上限（{cfg.max_total_turns} 轮）",
                signals={"hard_rule": "R2_max_total_turns"},
                focus="达到单次访谈轮次上限",
            )

        # R3: 单话题轮次超限 → 切换（无话题可切则收尾）
        if state.topic_turn_count >= topic.max_turns:
            entities = recent_entities(state) + [e for f in last_fragments for e in f.entities]
            nxt, bridge = self.planner.next_topic(state, entities)
            if nxt is None:
                return Decision(
                    action=DecisionAction.WRAP_UP,
                    confidence=0.9,
                    reasoning="硬规则 R3：话题轮次超限且无未覆盖话题，收尾",
                    signals={"hard_rule": "R3_no_more_topics"},
                    focus="所有话题已覆盖",
                )
            return Decision(
                action=DecisionAction.SWITCH_TOPIC,
                confidence=0.9,
                reasoning=f"硬规则 R3：「{topic.name}」已聊 {state.topic_turn_count} 轮达上限，切换到「{nxt.name}」",
                signals={"hard_rule": "R3_topic_max_turns", "bridge": bridge},
                focus=nxt.id,
            )
        return None

    # -- 第 2 层：特征打分 ------------------------------------------------------
    def _heuristic_scores(
        self,
        state: InterviewState,
        topic: Topic,
        text: str,
        last_fragments: list[MemoryFragment],
    ) -> dict[str, float]:
        cfg = state.config
        elder_texts = [m.text for m in state.messages if m.speaker == Speaker.ELDER]

        # ---- fatigue 疲惫分 ----
        fatigue = 0.0
        if any(w in text for w in TIRED_WORDS):
            fatigue = max(fatigue, 0.85)
        tail = elder_texts[-SHORT_STREAK_LIMIT:]
        if len(tail) >= SHORT_STREAK_LIMIT and all(len(t) < SHORT_REPLY_LEN for t in tail):
            fatigue = max(fatigue, 0.70)
        if state.turn_count >= cfg.soft_total_turns:
            span = max(1, cfg.max_total_turns - cfg.soft_total_turns)
            ramp = 0.45 + 0.45 * (state.turn_count - cfg.soft_total_turns + 1) / span
            fatigue = max(fatigue, min(0.9, ramp))

        # ---- story_value 故事价值分 ----
        story = 0.0
        ent_types = 0
        for f in last_fragments:
            if f.time_refs:
                ent_types += 1
            if f.place_refs:
                ent_types += 1
            if f.person_refs:
                ent_types += 1
        story += min(0.54, 0.18 * min(ent_types, 3))
        markers = sum(1 for m in NARRATIVE_MARKERS if m in text)
        story += min(0.24, 0.12 * markers)
        if detect_emotion(text):
            story += 0.10
        story += min(0.20, len(text) / 60 * 0.20)  # 长度加成（封顶）
        if len(text) < SHORT_REPLY_LEN and not last_fragments:
            story = min(story, 0.15)  # 极短且无片段：故事价值封顶
        story = min(1.0, story)

        # ---- exhaustion 话题枯竭分 ----
        exhaustion = 0.0
        # 含"忘"的表达只认句尾（"早忘了/都忘了"算枯竭，"忘了吃/带"不算）；其余模糊词直接命中
        vague_hit = any(p in text for p in VAGUE_PATTERNS if "忘" not in p) or bool(
            re.search(r"忘(?:记)?了($|[，。！？；、\s])", text)
        )
        if vague_hit:
            exhaustion += 0.55
        # 模糊与极短是相关信号，避免双重计分：命中模糊表达时不再叠加极短分
        elif len(text) < 6 and not any(f.entities for f in last_fragments):
            exhaustion += 0.35
        # 与历史回答重复？
        prev = elder_texts[:-1] if elder_texts else []
        if prev and text:
            sim = max((char_jaccard(text, p) for p in prev[-6:]), default=0.0)
            if sim >= 0.55:
                exhaustion += 0.40
        # 本轮无新实体（且话题已聊到最低轮次）？
        if state.topic_turn_count >= topic.min_turns:
            old_entities: set[str] = set()
            for f in state.fragments:
                if f.source_turn < state.turn_count:
                    old_entities.update(f.entities)
            new_entities = [e for f in last_fragments for e in f.entities if e not in old_entities]
            if not new_entities:
                exhaustion += 0.25
        # 单话题轮次爬升
        exhaustion += 0.40 * min(1.0, state.topic_turn_count / max(1, topic.max_turns))
        exhaustion = min(1.0, exhaustion)

        return {
            "story_value": round(story, 3),
            "exhaustion": round(exhaustion, 3),
            "fatigue": round(fatigue, 3),
        }

    # -- 第 2 层半：规则裁决 -----------------------------------------------------
    def _rule_action(
        self, state: InterviewState, topic: Topic, h: dict[str, float]
    ) -> tuple[DecisionAction, float, str]:
        cfg = state.config
        story, exh, fat = h["story_value"], h["exhaustion"], h["fatigue"]

        if fat >= cfg.fatigue_wrap_threshold:
            return (
                DecisionAction.WRAP_UP,
                fat,
                f"规则裁决：疲惫分 {fat:.2f}≥{cfg.fatigue_wrap_threshold}，收尾",
            )
        if exh >= cfg.exhaustion_switch_hard:
            return (
                DecisionAction.SWITCH_TOPIC,
                exh,
                f"规则裁决：枯竭分 {exh:.2f}≥{cfg.exhaustion_switch_hard}（老人没货了），切换话题",
            )
        if (
            exh >= cfg.exhaustion_switch_soft
            and story < 0.50
            and state.topic_turn_count >= topic.min_turns
        ):
            return (
                DecisionAction.SWITCH_TOPIC,
                (exh + (1 - story)) / 2,
                f"规则裁决：枯竭 {exh:.2f} 且故事价值低 {story:.2f}，切换话题",
            )
        if story >= cfg.story_dive_threshold:
            return (
                DecisionAction.DEEP_DIVE,
                story,
                f"规则裁决：故事价值 {story:.2f}≥{cfg.story_dive_threshold}，深挖",
            )
        if state.topic_turn_count < topic.min_turns:
            return (
                DecisionAction.DEEP_DIVE,
                0.55,
                f"规则裁决：话题仅聊 {state.topic_turn_count} 轮（<{topic.min_turns}），再给一次深挖机会",
            )
        return (
            DecisionAction.SWITCH_TOPIC,
            0.60,
            f"规则裁决：故事价值低（{story:.2f}）且已聊够 {topic.min_turns} 轮，切换话题",
        )

    # -- 第 3 层：LLM 裁判 -------------------------------------------------------
    def _llm_judge(
        self,
        state: InterviewState,
        topic: Topic,
        text: str,
        heuristic: dict[str, float],
        llm: LLMClient | None,
    ) -> tuple[DecisionAction, float, str] | None:
        if llm is None or not state.config.use_llm_judge:
            return None
        try:
            recent_qa = self._recent_qa(state, n=4)
            prompt = build_judge_prompt(state, topic, text, recent_qa, heuristic)
            raw = llm.chat(SYSTEM_JUDGE, prompt)
            obj = extract_json_object(raw)
            if not obj:
                return None
            action = DecisionAction(str(obj.get("action", "deep_dive")))
            conf = float(obj.get("confidence", 0.5))
            return action, max(0.0, min(1.0, conf)), str(obj.get("reasoning", ""))
        except (LLMError, ValueError, KeyError):
            return None

    @staticmethod
    def _recent_qa(state: InterviewState, n: int) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        pending_q = ""
        for m in state.messages:
            if m.speaker == Speaker.AI:
                pending_q = m.text
            elif m.speaker == Speaker.ELDER and pending_q:
                pairs.append((pending_q, m.text))
                pending_q = ""
        return pairs[-n:]

    # -- focus 计算 --------------------------------------------------------------
    def _focus_for(
        self,
        state: InterviewState,
        action: DecisionAction,
        text: str,
        last_fragments: list[MemoryFragment],
    ) -> str:
        if action is DecisionAction.DEEP_DIVE:
            # 优先选未展开过的地点/人物实体；越长越具体（"王二哥"优先于"娘"）
            cands: list[str] = []
            for f in sorted(last_fragments, key=lambda x: -x.importance):
                cands.extend(f.place_refs + f.person_refs + f.time_refs)
            for ent in sorted(set(cands), key=lambda e: (-len(e), e)):
                if ent not in state.expanded_entities:
                    return ent
            if last_fragments and last_fragments[0].followup_hint:
                return last_fragments[0].followup_hint
            return "事件经过"
        if action is DecisionAction.SWITCH_TOPIC:
            entities = recent_entities(state) + [e for f in last_fragments for e in f.entities]
            nxt, _ = self.planner.next_topic(state, entities)
            return nxt.id if nxt else ""
        return "收尾"
