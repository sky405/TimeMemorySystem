"""验证器：记忆片段"有效"的守门员 + 安全护栏。

- validate_fragments：候选片段 → 有效 / 驳回（附理由）。
  检查项：schema 合法 → 内容非空 → 非纯模糊 → 非重复 →（可选）LLM 合理性。
- is_farewell：安全护栏。只收录无歧义的告别表达，其余交给 LLM 路由判断。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .models import MemoryFragment

if TYPE_CHECKING:
    from .llm import InterviewLLM

# ----------------------------------------------------------------------------
# 安全护栏：无歧义告别表达（命中则强制收尾；"困了/走了/再见/下次"等有歧义的
# 一律不收录，交给 LLM 按语境判断）
# ----------------------------------------------------------------------------
FAREWELL_PHRASES = [
    "累了", "太累", "不聊了", "别聊了", "别问了", "结束吧", "就到这",
    "拜拜", "有事", "来人了",
    "先睡", "想睡", "睡觉了", "吃饭了", "休息了",
    "去做饭", "得做饭", "要做饭", "做饭去", "做饭了", "先做饭",
    "去吃饭", "吃饭去", "先吃饭", "去睡觉", "睡觉去",
    "先休息", "去休息", "休息一下", "休息一会", "休息会儿", "休息吧",
    "我走了", "先走了", "下次再聊", "改天再聊", "下回再聊",
    "咱们下次", "咱们改天", "先这样吧", "先这样。", "先这样！",
]


def is_farewell(text: str) -> str:
    """命中返回告别词，否则返回 ""。"""
    return next((w for w in FAREWELL_PHRASES if w in text), "")


# ----------------------------------------------------------------------------
# 片段验证
# ----------------------------------------------------------------------------
_VAGUE_WORDS = ["不知道", "不清楚", "不记得", "记不清", "记不得", "想不起来",
                "没什么", "没啥", "就那样", "一般", "还行"]
# "忘了/忘记了"只认句尾："早忘了"算模糊，"忘了吃"不算。
_FORGOT_ALONE_RE = re.compile(r"忘(?:记)?了($|[，。！？；、\s])")


def is_pure_vague(text: str, has_substance: bool) -> bool:
    """纯模糊回答判定：有实质内容（地点/人物/情节/情感）就不算纯模糊。"""
    if has_substance:
        return False
    if any(w in text for w in _VAGUE_WORDS):
        return True
    return bool(_FORGOT_ALONE_RE.search(text))


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class RejectedFragment:
    content: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    valid: list[MemoryFragment] = field(default_factory=list)
    rejected: list[RejectedFragment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": [f.model_dump() for f in self.valid],
            "rejected": [{"content": r.content, "reasons": r.reasons} for r in self.rejected],
        }


def validate_fragments(
    candidates: list[MemoryFragment | dict],
    committed: list[MemoryFragment],
    *,
    llm: "InterviewLLM | None" = None,
    llm_check: bool = False,
    dedupe_threshold: float = 0.8,
    min_content_len: int = 4,
) -> ValidationReport:
    """验证候选片段。规则检查全过后，可选再过 LLM 合理性检查。"""
    report = ValidationReport()
    for c in candidates:
        frag, reasons = _parse_and_check(c, committed, dedupe_threshold, min_content_len)
        if reasons or frag is None:
            content = c.get("content", "") if isinstance(c, dict) else (frag.content if frag else "")
            report.rejected.append(RejectedFragment(content=str(content), reasons=reasons))
        else:
            report.valid.append(frag)

    if llm_check and llm is not None:
        kept: list[MemoryFragment] = []
        for f in report.valid:
            try:
                ok = llm.plausibility(f)
            except Exception:
                ok = True  # LLM 挂了不挡路：规则已通过就放行
            if ok:
                kept.append(f)
            else:
                report.rejected.append(RejectedFragment(f.content, ["LLM 质检未通过"]))
        report.valid = kept
    return report


def _parse_and_check(
    c: MemoryFragment | dict,
    committed: list[MemoryFragment],
    dedupe_threshold: float,
    min_len: int,
) -> tuple[MemoryFragment | None, list[str]]:
    reasons: list[str] = []
    if isinstance(c, dict):
        try:
            frag = MemoryFragment(**c)
        except Exception as e:  # pydantic schema 错误
            return None, [f"schema 非法：{e}"]
    else:
        frag = c

    content = (frag.content or "").strip()
    if len(content) < min_len:
        reasons.append("内容过短或为空")
        return frag, reasons

    has_substance = bool(
        frag.time_refs or frag.place_refs or frag.person_refs or frag.emotion
    )
    # 注意：纯时间（"很久以前"）撑不起一条记忆
    has_substance = bool(frag.place_refs or frag.person_refs or frag.emotion)
    if is_pure_vague(content, has_substance) and len(content) < 20:
        reasons.append("纯模糊回答，无实质内容")

    if committed and content:
        sim = max((char_jaccard(content, f.content) for f in committed), default=0.0)
        if sim >= dedupe_threshold:
            reasons.append("与已有片段重复")
    return frag, reasons
