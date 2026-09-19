"""提问生成器：开场 → 五板斧追问 → 过渡 → 收尾。

铁律：一次只问一个问题（max_questions_per_turn=1），短句口语，尊称"您"。
模板生成保证离线可用；LLM 润色（可选）让话术更像人。
"""
from __future__ import annotations

import random

from .llm import LLMClient, LLMError
from .models import InterviewState, MemoryFragment, Topic
from .prompts import SYSTEM_INTERVIEWER, SYSTEM_POLISH, build_polish_prompt


def _ack_plus_question(ack: str, question: str) -> str:
    """复述确认 + 追问（让老人感到被听见）。"""
    if ack:
        return f"{ack}，{question}"
    return question


class QuestionGenerator:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    # -- 开场 ------------------------------------------------------------------
    def greeting(self, state: InterviewState, topic: Topic, first_question: str) -> str:
        name = state.elder.name or "老人家"
        hometown = f"听说您老家是{state.elder.hometown}，" if state.elder.hometown else ""
        return (
            f"{name}您好！我是您的回忆录访谈员小记。{hometown}"
            f"今天咱们就像拉家常一样，随便聊聊您这辈子的故事，想到哪儿说到哪儿，"
            f"累了咱们就歇着。先问您一个：{first_question}"
        )

    def opening(self, topic: Topic, asked_count: int = 0) -> str:
        return topic.opening_questions[asked_count % len(topic.opening_questions)]

    # -- 分支 A：深度追问（五板斧）-----------------------------------------------
    def followup(
        self,
        state: InterviewState,
        last_answer: str,
        last_fragments: list[MemoryFragment],
        focus: str,
        llm: LLMClient | None = None,
    ) -> str:
        strategy, draft = self._pick_strategy(state, last_answer, last_fragments, focus)
        # 记录已展开实体，避免车轱辘话
        if focus and focus not in state.expanded_entities and len(focus) <= 12:
            state.expanded_entities.append(focus)
        return self._maybe_polish(draft, state, last_answer, llm)

    def _pick_strategy(
        self,
        state: InterviewState,
        last_answer: str,
        last_fragments: list[MemoryFragment],
        focus: str,
    ) -> tuple[str, str]:
        topic_id = state.current_topic_id
        frags = sorted(last_fragments, key=lambda f: -f.importance)
        top = frags[0] if frags else None

        # 1) entity_expand：有未展开过的地点/人物实体 → 优先深挖
        entity = focus if (focus and len(focus) <= 12 and "：" not in focus and focus != "事件经过") else ""
        if not entity and top:
            for cand in (top.place_refs + top.person_refs + top.time_refs):
                if cand not in state.expanded_entities:
                    entity = cand
                    break
        if entity:
            # 按实体类型选话术：人 / 地点 / 时间各有问法，避免"村长那个地方"类别扭话
            is_person = any(entity in f.person_refs for f in last_fragments)
            is_place = any(entity in f.place_refs for f in last_fragments)
            templates = [
                f"您刚才提到{entity}——这背后有没有哪段经历，是您最忘不了的？",
                f"说到{entity}，能再跟我讲讲当时具体是什么情形吗？",
            ]
            if is_person:
                templates += [
                    f"您刚才提到{entity}，他是个什么样的人？",
                    f"说到{entity}，你们之间有没有什么难忘的事？",
                ]
            elif is_place:
                templates += [
                    f"在{entity}，有没有发生过让您印象最深的一件事？",
                ]
            else:
                templates += [
                    f"那个时候的日子是怎么过的？还记得什么细节吗？",
                ]
            return "entity_expand", self._rng.choice(templates)

        # 2) emotion_deepen：有情感 → 追情绪
        if top and top.emotion:
            return (
                "emotion_deepen",
                f"听得出这事在您心里分量很重。那一刻您心里是什么滋味，还记得吗？",
            )

        # 3) event_chain：有情节标记 → 追前因后果
        if any(m in last_answer for m in ["后来", "然后", "结果", "有一次", "那次"]):
            templates = [
                "后来呢？这件事最后怎么样了？",
                "再往后呢？当时您是怎么做的？",
            ]
            return "event_chain", self._rng.choice(templates)

        # 4) sensory_detail：追画面细节
        if top and (top.place_refs or len(last_answer) >= 15):
            return (
                "sensory_detail",
                "您说的这个场面，我都想亲眼看看了。当时那儿是什么样子的？",
            )

        # 5) example_request：兜底，要个具体例子
        return "example_request", "能给我讲一个具体的例子吗？越细越好。"

    # -- 分支 B：切换话题（过渡语 + 新话题开场）------------------------------------
    def transition(
        self,
        state: InterviewState,
        old_topic: Topic,
        new_topic: Topic,
        bridge: str,
        last_highlight: str = "",
        llm: LLMClient | None = None,
    ) -> str:
        if last_highlight:
            ack = f"刚才您讲的{last_highlight}真有意思，我都记下来了"
        else:
            ack = f"刚才{old_topic.name}这段讲得真好，我都记下来了"
        if bridge:
            mid = f"说到{bridge}，我就想起"
        else:
            mid = "接下来我想问问"
        question = self.opening(new_topic)
        # 新话题的第一问：过渡 + 开场问题（仍只算一个问题）
        draft = f"{ack}。{mid}——{question}"
        return self._maybe_polish(draft, state, "", llm)

    # -- 分支 C：收尾 --------------------------------------------------------------
    def closing(self, state: InterviewState) -> str:
        name = state.elder.name or "老人家"
        highlights = [f.content[:24] + ("…" if len(f.content) > 24 else "")
                      for f in sorted(state.fragments, key=lambda x: -x.importance)[:2]]
        covered_names = [t for t in state.covered_topic_ids]
        n_frag = len(state.fragments)
        parts = [f"今天跟您聊得真开心，谢谢{name}！"]
        if highlights:
            parts.append(f"您讲的「{'」和「'.join(highlights)}」，我都好好记下来了。")
        if covered_names:
            parts.append(f"咱们今天一共聊了 {len(covered_names)} 段人生故事，记下了 {n_frag} 条回忆。")
        parts.append("您好好休息，咱们下次再接着聊！")
        return "".join(parts)

    # -- LLM 润色（可选，失败回退原文）---------------------------------------------
    def _maybe_polish(
        self, draft: str, state: InterviewState, last_answer: str, llm: LLMClient | None
    ) -> str:
        # Mock 不润色（保证演示确定性）；只给真模型润色
        if llm is None or not state.config.use_llm_polish or llm.name == "mock":
            return draft
        try:
            polished = llm.chat(
                SYSTEM_POLISH,
                build_polish_prompt(draft, state.elder.name, last_answer or "（新话题开场）"),
                temperature=0.5,
            ).strip()
            # 安全阀：润色结果必须仍是单问句，否则用原文
            if polished.count("？") + polished.count("?") != 1:
                return draft
            return polished
        except LLMError:
            return draft
