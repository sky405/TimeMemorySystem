"""定稿渲染：封面 + 目录 + 修正后正文 + 审核记录 + 素材溯源。"""
from __future__ import annotations

from datetime import date

from .models import ChapterFinal, ItemReview


def render_final_book(title: str, elder: dict, reviewer: str,
                      chapters: list[ChapterFinal], records: list[ItemReview],
                      fragments: list[dict]) -> str:
    name = elder.get("name", "老人家") if isinstance(elder, dict) else str(elder)
    title = (title or f"{name}的回忆录（定稿）").replace("初稿", "定稿")
    frag_map = {f["id"]: f for f in fragments}
    lines = [f"# {title}", "",
             f"口述：{name}　审核：{reviewer}　定稿日期：{date.today().isoformat()}　版本：V1.0",
             "", "## 目录"]
    if chapters:
        for i, c in enumerate(chapters, 1):
            lines.append(f"{i}. {c.title}")
    else:
        lines.append("（暂无章节）")
    lines.append("")
    for c in chapters:
        lines += [f"## {c.title}", "", c.text, ""]
    lines.append("## 附录一：审核记录")
    if not records:
        lines.append("初稿零存疑，审核直接通过。")
    for r in records:
        sug = f"{r.suggestion.action}" + (f"→“{r.suggestion.replacement}”"
                                           if r.suggestion.replacement else "")
        lines.append(f"- 【{r.chapter}】“{r.quote}”——{r.issue}（{r.severity}）")
        lines.append(f"  AI 建议：{sug}——{r.suggestion.reason}")
        lines.append(f"  裁决：{r.verdict}" + (f"｜备注：{r.note}" if r.note else "")
                     + ("｜已改动正文" if r.applied else "｜正文未动"))
    lines += ["", "## 附录二：素材溯源"]
    if not chapters:
        lines.append("（空）")
    for c in chapters:
        lines.append(f"### {c.title}")
        for fid in c.fragment_ids:
            f = frag_map.get(fid, {})
            lines.append(f"- `{fid}`（{f.get('topic_id', '')}）{f.get('content', '')[:40]}")
    return "\n".join(lines) + "\n"
