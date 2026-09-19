"""成稿渲染：封面 + 目录 + 正文 + 存疑批注 + 人工复核清单 + 素材溯源。"""
from __future__ import annotations

from datetime import date

from .models import ChapterDraft, Outline, ReviewItem


def render_manuscript(outline: Outline, drafts: list[ChapterDraft], body: str,
                      elder: dict, fragments: list[dict]
                      ) -> tuple[str, list[ReviewItem]]:
    name = elder.get("name", "老人家") if isinstance(elder, dict) else str(elder)
    frag_map = {f["id"]: f for f in fragments}
    lines = [f"# {outline.title}", "",
             f"口述：{name}　整理：时光记忆系统　{date.today().isoformat()}",
             "", "## 目录"]
    if drafts:
        for i, d in enumerate(drafts, 1):
            lines.append(f"{i}. {d.title}")
    else:
        lines.append("（暂无章节）")
    lines += ["", body, ""]
    review = [ReviewItem(chapter=d.title, quote=fl.quote, issue=fl.issue,
                         severity=fl.severity, fragment_ids=d.fragment_ids)
              for d in drafts for fl in d.factcheck.flags]
    lines.append("## 附录一：存疑批注与人工复核清单")
    if not drafts:
        lines.append("暂无素材，未生成章节。")
    elif not review:
        lines.append("全部章节通过事实回检，无存疑。")
    else:
        for r in review:
            lines.append(f"- 【{r.chapter}】“{r.quote}”——{r.issue}（{r.severity}）"
                         f"｜查证片段：{', '.join(r.fragment_ids)}")
    lines += ["", "## 附录二：素材溯源"]
    if not drafts:
        lines.append("（空）")
    for d in drafts:
        lines.append(f"### {d.title}")
        for fid in d.fragment_ids:
            f = frag_map.get(fid, {})
            lines.append(f"- `{fid}`（{f.get('topic_id', '')}）{f.get('content', '')[:40]}")
    return "\n".join(lines) + "\n", review
