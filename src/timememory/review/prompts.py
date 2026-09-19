"""Phase 5 提示词：AI 修订建议。只返回 JSON。"""
from __future__ import annotations

FIX_SYSTEM = """你是校对编辑。初稿中有一处事实存疑，给出最小化修订建议。
只返回 JSON，不要解释。"""

FIX_USER = """章节：{chapter}
存疑原文：{quote}
问题：{issue}

本章正文：
{text}

原始素材：
{materials}

返回 JSON：
{{"action": "保留/改写/删除该句",
  "replacement": "改写时填写新句子，否则空",
  "reason": "一句话理由"}}
要求：优先"改写"（用素材内容纠正）；素材完全没提的选"删除该句"；
只有核查有误（素材其实有出处）才选"保留"并说明出处。"""
