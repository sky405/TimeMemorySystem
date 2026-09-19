"""Phase 3 交接：素材完整度报告 + 缺口分析 + 下次访谈计划。

Phase 1 结束后（或每轮结束后），调用 build_coverage_report(state) 得到：
- 每个话题的覆盖度、缺口、建议追问
- 整体覆盖度与建议动作（直接写作 / 补充访谈 / 继续访谈）
- 下次访谈计划（resume_topics + focus_questions），供补充访谈的 start() 使用
"""
from __future__ import annotations

from dataclasses import asdict

from .models import InterviewState
from .topics import TOPICS

FRAGMENTS_FOR_FULL = 4  # 单话题满覆盖参考片段数


def _topic_gaps(state: InterviewState, topic_id: str) -> list[str]:
    frags = state.fragments_of(topic_id)
    turns = state.topic_turns.get(topic_id, 0)
    if turns == 0:
        return ["完全未覆盖"]
    gaps: list[str] = []
    if not any(f.time_refs for f in frags):
        gaps.append("缺少具体时间（哪一年 / 多大岁数）")
    if not any(f.place_refs for f in frags):
        gaps.append("缺少地点细节（哪里 / 什么样）")
    if not any(f.person_refs for f in frags):
        gaps.append("缺少人物（和谁一起 / 印象最深的人）")
    if not any(f.importance >= 4 for f in frags):
        gaps.append("缺少高价值故事（可再深挖具体经过）")
    return gaps


def _suggested_questions(state: InterviewState, topic_id: str) -> list[str]:
    from .topics import TOPIC_MAP

    topic = TOPIC_MAP[topic_id]
    out: list[str] = []
    if state.topic_turns.get(topic_id, 0) == 0:
        out.append(topic.opening_questions[0])
    for angle in topic.followup_angles[:2]:
        out.append(f"关于{topic.name}的「{angle}」，还有什么印象深的事吗？")
    return out[:3]


def build_coverage_report(state: InterviewState) -> dict:
    topics_report: dict[str, dict] = {}
    for t in TOPICS:
        turns = state.topic_turns.get(t.id, 0)
        frags = state.fragments_of(t.id)
        turn_cov = min(1.0, turns / max(1, t.min_turns))
        frag_cov = min(1.0, len(frags) / FRAGMENTS_FOR_FULL)
        coverage = round(0.5 * turn_cov + 0.5 * frag_cov, 2)
        topics_report[t.id] = {
            "name": t.name,
            "turns": turns,
            "fragments": len(frags),
            "coverage": coverage,
            "gaps": _topic_gaps(state, t.id),
            "suggested_questions": _suggested_questions(state, t.id),
        }

    overall = round(sum(r["coverage"] for r in topics_report.values()) / len(topics_report), 2)
    if overall >= 0.7:
        recommendation = "ready_for_writing"  # 素材充足 → Phase 4
    elif overall >= 0.35:
        recommendation = "supplementary_interview"  # 部分缺口 → 补充访谈
    else:
        recommendation = "continue_interview"  # 覆盖太少 → 继续访谈

    # 下次访谈计划：挑覆盖度最低的 3 个非敏感/已解锁话题
    ranked = sorted(topics_report.items(), key=lambda kv: kv[1]["coverage"])
    resume: list[str] = []
    focus_q: list[str] = []
    for tid, r in ranked:
        if r["coverage"] >= 1.0:
            continue
        resume.append(tid)
        focus_q.extend(r["suggested_questions"][:1])
        if len(resume) >= 3:
            break

    return {
        "session_id": state.session_id,
        "elder": state.elder.name,
        "total_turns": state.turn_count,
        "total_fragments": len(state.fragments),
        "topics": topics_report,
        "overall_coverage": overall,
        "recommendation": recommendation,
        "next_interview_plan": {"resume_topics": resume, "focus_questions": focus_q[:3]},
    }


def export_fragments(state: InterviewState) -> list[dict]:
    """记忆片段列表（JSON 可序列化）→ Phase 2 清洗/向量化/知识图谱的输入。"""
    return [asdict(f) for f in state.fragments]
