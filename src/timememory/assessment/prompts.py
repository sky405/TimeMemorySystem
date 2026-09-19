"""Phase 3 提示词：评估打分 / 补充访谈计划。只返回 JSON。"""
from __future__ import annotations

from ..interview.topics import TOPICS


def topics_catalog() -> str:
    return "\n".join(f"- {t.id}（{t.name}）：{t.description}" for t in TOPICS)


ASSESS_SYSTEM = """你是传记写作顾问。判断当前素材够不够为老人写一本人生传记，
从四个维度打分（0-10），并找出最值得补充访谈的缺口。
只返回 JSON，不要解释。"""

ASSESS_USER = """素材摘要：
{stats}

话题清单（缺口的 topic_id 必须取自这里）：
{catalog}

返回 JSON：
{{
  "ready": false,
  "overall": "一两句话总体评价",
  "dimensions": [
    {{"dimension": "话题覆盖", "score": 0, "summary": "…"}},
    {{"dimension": "时间线完整", "score": 0, "summary": "…"}},
    {{"dimension": "人物丰满度", "score": 0, "summary": "…"}},
    {{"dimension": "细节情感", "score": 0, "summary": "…"}}
  ],
  "strengths": ["素材的优点…"],
  "gaps": [
    {{"dimension": "话题覆盖", "title": "缺少求学经历",
      "detail": "…", "priority": "高/中/低",
      "topic_id": "education", "subject": "", "evidence_ids": []}}
  ]
}}
要求：缺口按重要性排序，最多 6 个；priority 只能是 高/中/低；
人物类缺口把人名填进 subject；只有素材确实够动笔才 ready=true。"""

PLAN_SYSTEM = """你是资深访谈记者。根据素材缺口，为下一次补充访谈准备追问提纲。
问题要口语化、具体、一次只问一件事，适合跟老人聊天时自然问出。
只返回 JSON，不要解释。"""

PLAN_USER = """素材缺口：
{gaps}

话题清单：
{catalog}

各话题已有素材（追问不要重复问已知内容）：
{samples}

返回 JSON：
{{
  "items": [
    {{"topic_id": "education", "priority": "高",
      "reason": "为什么补这个话题",
      "questions": ["问题1", "问题2", "问题3"]}}
  ],
  "note": "给访谈员的一句话提醒"
}}
要求：每个缺口对应一个 item（同话题合并）；每项 1-3 个问题；
topic_id 必须取自话题清单；priority 只能是 高/中/低。"""
