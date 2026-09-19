"""DraftingLLM 协议 + LangChain 真模型 + Demo 拼接实现。

有 `TMS_LLM_API_KEY` 用真模型（与前序阶段共用环境变量），
否则 Demo 拼接原文成章，离线可跑、输出确定。
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol

from .models import (
    ChapterDraft,
    ChapterSpec,
    FactCheck,
    Flag,
    Outline,
    StageGroup,
)
from .prompts import (
    FACTCHECK_SYSTEM,
    FACTCHECK_USER,
    OUTLINE_SYSTEM,
    OUTLINE_USER,
    POLISH_SYSTEM,
    POLISH_USER,
    WRITE_SYSTEM,
    WRITE_USER,
)

_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")


class DraftingLLM(Protocol):
    def make_outline(self, elder: dict, groups: list[StageGroup],
                     undated: list[dict], assessment_note: str) -> Outline: ...
    def write_chapter(self, elder: dict, chapter: ChapterSpec,
                      materials: list[dict], prev_summary: str) -> ChapterDraft: ...
    def fact_check(self, title: str, text: str, materials: list[dict]) -> FactCheck: ...
    def polish(self, elder: dict, chapters: list[ChapterDraft]) -> str: ...


def _mat_json(materials: list[dict]) -> str:
    slim = [{k: m.get(k) for k in
             ("id", "content", "topic_id", "time_refs", "place_refs", "person_refs", "emotion")}
            for m in materials]
    return json.dumps(slim, ensure_ascii=False)


class LangChainDraftingLLM:
    """真模型：大纲 / 写作 / 回检结构化输出，统稿返回 Markdown。"""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.4):
        from langchain_openai import ChatOpenAI

        self.chat = ChatOpenAI(
            model=model or os.environ.get("TMS_LLM_MODEL", "deepseek-chat"),
            base_url=base_url or os.environ.get("TMS_LLM_BASE_URL"),
            api_key=api_key or os.environ.get("TMS_LLM_API_KEY"),
            temperature=temperature,
        )

    def make_outline(self, elder, groups, undated, assessment_note="") -> Outline:
        return self.chat.with_structured_output(Outline).invoke([
            ("system", OUTLINE_SYSTEM),
            ("user", OUTLINE_USER.format(
                elder=json.dumps(elder, ensure_ascii=False),
                assessment_note=assessment_note or "（无）",
                groups=json.dumps([g.model_dump() for g in groups], ensure_ascii=False),
                undated=_mat_json(undated))),
        ])

    def write_chapter(self, elder, chapter, materials, prev_summary="") -> ChapterDraft:
        return self.chat.with_structured_output(ChapterDraft).invoke([
            ("system", WRITE_SYSTEM),
            ("user", WRITE_USER.format(
                elder=json.dumps(elder, ensure_ascii=False),
                chapter=json.dumps(chapter.model_dump(), ensure_ascii=False),
                prev_summary=prev_summary or "（第一章）",
                materials=_mat_json(materials))),
        ])

    def fact_check(self, title, text, materials) -> FactCheck:
        return self.chat.with_structured_output(FactCheck).invoke([
            ("system", FACTCHECK_SYSTEM),
            ("user", FACTCHECK_USER.format(title=title, text=text,
                                           materials=_mat_json(materials))),
        ])

    def polish(self, elder, chapters) -> str:
        body = "\n\n".join(f"## {c.title}\n\n{c.text}" for c in chapters)
        return self.chat.invoke([
            ("system", POLISH_SYSTEM),
            ("user", POLISH_USER.format(elder=json.dumps(elder, ensure_ascii=False),
                                        chapters=body or "（暂无章节）")),
        ]).content


class DemoDraftingLLM:
    """离线实现：大纲按阶段分章、无年份按话题归位；写作拼接原文；回检查年份出处。"""

    def make_outline(self, elder, groups, undated, assessment_note="") -> Outline:
        name = elder.get("name", "老人家")
        if not groups:
            chapters = [ChapterSpec(id="ch-0", title="岁月记忆", stage="",
                                    brief="老人的回忆。",
                                    fragment_ids=[f["id"] for f in undated])]
        else:
            buckets = {g.id: list(g.fragment_ids) for g in groups}
            for f in undated:
                t = f.get("topic_id", "")
                best = max(groups, key=lambda g: (t in g.topics, len(g.fragment_ids)))
                buckets[best.id].append(f["id"])
            chapters = [ChapterSpec(id=f"ch-{i}", title=g.title, stage=g.title,
                                    brief=f"讲述{g.title}的往事。",
                                    fragment_ids=buckets[g.id])
                        for i, g in enumerate(groups)]
        return Outline(title=f"{name}的回忆录（初稿）", chapters=chapters,
                       notes=assessment_note or "")

    def write_chapter(self, elder, chapter, materials, prev_summary="") -> ChapterDraft:
        mats = sorted(materials, key=lambda f: (f.get("source_turn", 0), f.get("id", "")))
        name = elder.get("name", "老人家")
        if mats:
            body = "\n\n".join(m.get("content", "") for m in mats)
            text = f"关于{chapter.title}，{name}这样回忆：\n\n{body}"
        else:
            text = f"关于{chapter.title}，暂无素材。（待补充访谈）"
        return ChapterDraft(chapter_id=chapter.id, title=chapter.title, text=text,
                            summary=text[:60], fragment_ids=[m["id"] for m in mats])

    def fact_check(self, title, text, materials) -> FactCheck:
        mat_years: set[str] = set()
        for m in materials:
            hay = m.get("content", "") + " " + " ".join(m.get("time_refs", []))
            mat_years.update(_YEAR_RE.findall(hay))
        # 素材年份所在的年代也算有出处（如 1962 → "1960年代"可用）
        mat_years.update(str(int(y) // 10 * 10) for y in list(mat_years))
        flags: list[Flag] = []
        for y in sorted(set(_YEAR_RE.findall(text))):
            if y not in mat_years:
                quote = next((s.strip() for s in re.split(r"[。！？\n]", text) if y in s), y)
                flags.append(Flag(quote=quote[:40], issue=f"年份 {y} 在素材中没有出处"))
        return FactCheck(passed=not flags,
                         summary="未发现编造内容。" if not flags else f"发现 {len(flags)} 处存疑。",
                         flags=flags)

    def polish(self, elder, chapters) -> str:
        if not chapters:
            return "（暂无素材，待补充访谈后生成。）"
        return "\n\n* * *\n\n".join(f"## {c.title}\n\n{c.text}" for c in chapters)


def get_drafting_llm() -> DraftingLLM:
    if os.environ.get("TMS_LLM_API_KEY"):
        print("[TimeMemory] Phase 4 使用真模型写作")
        return LangChainDraftingLLM()
    print("[TimeMemory] Phase 4 使用 Demo 拼接写作（离线）")
    return DemoDraftingLLM()
