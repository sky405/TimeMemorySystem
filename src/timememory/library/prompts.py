"""Phase 6 提示词：多轮改写 / 带出处回答。只返回指定格式。"""
from __future__ import annotations

REWRITE_SYSTEM = """你是检索助手。把子孙的追问改写成无须上下文也能独立检索的一句话。
只返回改写后的一句话，不要解释。"""

REWRITE_USER = """历史对话（[Q] 子孙提问，[A] 已给回答）：
{history}

当前追问：{question}

改写（把"他/她/它/那/这"还原成具体人名地名；无需改写就原样返回）："""

ANSWER_SYSTEM = """你是家族记忆库的问答助手。只依据下面给定的材料回答，
不要编造；引用材料时在 citations 里给出段落 id 与原文。
只返回 JSON，不要解释。"""

ANSWER_USER = """问题：{question}

检索段落（kind=chapter 是定稿章节，fragment 是访谈原话）：
{passages}

知识图谱线索：
{triples}

返回 JSON：
{{"text": "回答正文（口语化，有温度；引用处用【段落id】标注）",
  "citations": [{{"passage_id": "…", "quote": "支撑原文（30字内）"}}],
  "has_answer": true}}
要求：材料答不上来就 has_answer=false、text="记忆库里没有相关记载。"；
caution=true 的段落要在正文里提醒"存疑待考"。"""
