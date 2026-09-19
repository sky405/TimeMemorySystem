"""AssessmentLLM 协议 + LangChain 真模型 + Demo 规则实现。

有 `TMS_LLM_API_KEY` 用真模型（与 Phase 1/2 共用一套环境变量），
否则 Demo 按规则打分找缺口，离线可跑、输出确定。
"""
from __future__ import annotations

import json
import os
from typing import Protocol

from ..interview.topics import TOPIC_MAP, TOPICS
from .models import Assessment, DimensionScore, Gap, PlanItem, SupplementPlan
from .prompts import (
    ASSESS_SYSTEM,
    ASSESS_USER,
    PLAN_SYSTEM,
    PLAN_USER,
    topics_catalog,
)


class AssessmentLLM(Protocol):
    def assess(self, stats: dict) -> Assessment: ...
    def plan_questions(self, gaps: list[Gap], stats: dict) -> SupplementPlan: ...


class LangChainAssessmentLLM:
    """真模型：结构化输出评估报告与访谈提纲。"""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float | None = None):
        from langchain_openai import ChatOpenAI

        from ..config import get_config
        cfg = get_config()
        if temperature is None:
            temperature = cfg.assessment.temperature
        if temperature is None:
            temperature = 0.2
        self.chat = ChatOpenAI(
            model=model or cfg.llm.model or "deepseek-chat",
            base_url=base_url or cfg.llm.base_url,
            api_key=api_key or cfg.llm.api_key,
            temperature=temperature,
        )

    def assess(self, stats: dict) -> Assessment:
        return self.chat.with_structured_output(Assessment).invoke([
            ("system", ASSESS_SYSTEM),
            ("user", ASSESS_USER.format(stats=json.dumps(stats, ensure_ascii=False),
                                        catalog=topics_catalog())),
        ])

    def plan_questions(self, gaps: list[Gap], stats: dict) -> SupplementPlan:
        return self.chat.with_structured_output(SupplementPlan).invoke([
            ("system", PLAN_SYSTEM),
            ("user", PLAN_USER.format(
                gaps=json.dumps([g.model_dump() for g in gaps], ensure_ascii=False),
                catalog=topics_catalog(),
                samples=json.dumps(stats.get("samples", {}), ensure_ascii=False))),
        ])


def _thinnest_topic(stats: dict) -> str:
    counts = stats.get("by_topic", {})
    return min((t.id for t in TOPICS), key=lambda tid: counts.get(tid, 0))


def _valid_topic(tid: str, stats: dict) -> str:
    return tid if tid in TOPIC_MAP else _thinnest_topic(stats)


class DemoAssessmentLLM:
    """离线规则实现：覆盖率打分 + 模板追问。"""

    MAX_GAPS = 6

    def assess(self, stats: dict) -> Assessment:
        gaps: list[Gap] = []
        for t in stats["empty_topics"]:
            gaps.append(Gap(dimension="话题覆盖", title=f"缺少{t['name']}方面的回忆",
                            detail="该话题尚无任何片段。", priority="中", topic_id=t["id"]))
        for t in stats["thin_topics"]:
            gaps.append(Gap(dimension="话题覆盖", title=f"{t['name']}只有一条回忆，略显单薄",
                            detail="建议再补充一两段往事。", priority="低", topic_id=t["id"]))
        for e in stats["thin_entities"]:
            gaps.append(Gap(
                dimension="人物丰满度", title=f"{e['name']}缺少故事",
                detail=f"全库只出现 {e['mentions']} 次，没有展开讲过。",
                priority="中", topic_id=_valid_topic(e["topic"], stats),
                subject=e["name"], evidence_ids=e["evidence_ids"]))
        thin_tid = _thinnest_topic(stats)
        for d in stats["empty_decades"]:
            gaps.append(Gap(dimension="时间线完整", title=f"{d}年代是空白",
                            detail="前后年代都有回忆，唯独这一段缺失。", priority="低",
                            topic_id=thin_tid, subject=f"{d}年代"))
        if stats["emotion_coverage"] < 0.3 and stats["fragments"] >= 3:
            gaps.append(Gap(dimension="细节情感", title="缺少情感和细节描写",
                            detail="多数片段只记了事，没记当时的心情和细节。",
                            priority="低", topic_id=thin_tid))
        order = {"高": 0, "中": 1, "低": 2}
        gaps.sort(key=lambda g: order[g.priority])
        gaps = gaps[:self.MAX_GAPS]

        n_empty = len(stats["empty_topics"])
        n_thin_ent = len(stats["thin_entities"])
        years = stats["years"]
        dims = [
            DimensionScore(dimension="话题覆盖", score=round((9 - n_empty) / 9 * 10),
                           summary=f"9 个话题覆盖了 {9 - n_empty} 个。"),
            DimensionScore(
                dimension="时间线完整",
                score=max(0, min(10, 10 - 2 * len(stats["empty_decades"]) - (0 if years else 4))),
                summary=(f"时间跨度 {years[0]}–{years[-1]} 年。" if years else "没有明确年份。")),
            DimensionScore(dimension="人物丰满度", score=max(0, 10 - 2 * n_thin_ent),
                           summary=f"{n_thin_ent} 个人物/地点只提到过一次。"
                           if n_thin_ent else "主要人物都有故事展开。"),
            DimensionScore(dimension="细节情感", score=round(stats["emotion_coverage"] * 10),
                           summary=f"{round(stats['emotion_coverage'] * 100)}% 的片段记录了情感。"),
        ]
        strengths: list[str] = []
        if stats["by_topic"]:
            top, cnt = max(stats["by_topic"].items(), key=lambda kv: (kv[1], kv[0]))
            strengths.append(f"「{TOPIC_MAP[top].name}」素材最丰富（{cnt} 条）。"
                             if top in TOPIC_MAP else f"「{top}」有 {cnt} 条素材。")
        if years:
            strengths.append(f"时间跨度 {years[0]}–{years[-1]} 年，有年代感。")
        if stats["emotion_coverage"] >= 0.5:
            strengths.append("情感记录细腻。")
        ready = not gaps
        overall = ("素材充足，可以动笔写传。" if ready
                   else f"共 {stats['fragments']} 条片段，覆盖 {9 - n_empty}/9 个话题，"
                        f"还有 {len(gaps)} 个缺口值得补充访谈。")
        return Assessment(ready=ready, overall=overall, dimensions=dims,
                          strengths=strengths, gaps=gaps)

    def plan_questions(self, gaps: list[Gap], stats: dict) -> SupplementPlan:
        items: list[PlanItem] = []
        for g in gaps:
            t = TOPIC_MAP[_valid_topic(g.topic_id, stats)]
            if g.dimension == "人物丰满度" and g.subject:
                qs = [f"{g.subject}是个什么样的人？",
                      f"能讲一件和{g.subject}有关的、印象最深的事吗？",
                      t.opening_questions[0] if t.opening_questions else ""]
            elif g.dimension == "时间线完整" and g.subject:
                qs = [f"{g.subject}您多大？那几年日子是怎么过的？",
                      f"{g.subject}印象最深的一件事是什么？",
                      t.opening_questions[0] if t.opening_questions else ""]
            elif g.dimension == "细节情感":
                qs = ["当时您心里是什么滋味？还记得什么细节？",
                      t.opening_questions[0] if t.opening_questions else ""]
            else:
                qs = list(t.opening_questions[:3])
            items.append(PlanItem(topic_id=t.id, topic_name=t.name, priority=g.priority,
                                  reason=g.title, questions=[q for q in qs if q][:3]))
        merged: dict[str, PlanItem] = {}
        for it in items:  # 同话题合并，取最高优先级
            if it.topic_id in merged:
                old = merged[it.topic_id]
                qs = list(old.questions) + [q for q in it.questions if q not in old.questions]
                pri = {"高": 0, "中": 1, "低": 2}
                merged[it.topic_id] = PlanItem(
                    topic_id=it.topic_id, topic_name=it.topic_name,
                    priority=it.priority if pri[it.priority] < pri[old.priority] else old.priority,
                    reason=old.reason, questions=qs[:3])
            else:
                merged[it.topic_id] = it
        return SupplementPlan(
            items=list(merged.values()),
            note="缺口按重要性排序，建议从第一个话题开始补访；老人累了就停，下次接着聊。")


def get_assessment_llm() -> AssessmentLLM:
    from ..config import get_config
    cfg = get_config().llm
    if cfg.api_key:
        print("[TimeMemory] Phase 3 使用真模型评估")
        return LangChainAssessmentLLM()
    print("[TimeMemory] Phase 3 使用 Demo 规则评估（离线）")
    return DemoAssessmentLLM()
