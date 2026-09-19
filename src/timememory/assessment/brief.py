"""补充访谈提纲渲染：评估报告 → 给访谈员看的 Markdown。"""
from __future__ import annotations

from .models import Assessment, SupplementPlan


def render_brief(assessment: Assessment, plan: SupplementPlan) -> str:
    a, p = assessment, plan
    lines = ["# 写作评估报告", ""]
    lines.append(f"**总体**：{'✅ 素材充足，可以动笔' if a.ready else '⚠️ 建议补充访谈'}——{a.overall}")
    lines.append("")
    lines.append("## 维度打分")
    for d in a.dimensions:
        lines.append(f"- {d.dimension}：{d.score}/10——{d.summary}")
    if a.strengths:
        lines.append("")
        lines.append("## 素材亮点")
        lines.extend(f"- {s}" for s in a.strengths)
    if a.gaps:
        lines.append("")
        lines.append(f"## 素材缺口（{len(a.gaps)}）")
        for i, g in enumerate(a.gaps, 1):
            lines.append(f"{i}. [{g.priority}] {g.title}——{g.detail}")
    lines.append("")
    lines.append("## 补充访谈提纲")
    if not p.items:
        lines.append("无需补访。")
    for i, it in enumerate(p.items, 1):
        lines.append(f"### {i}. {it.topic_name} [{it.priority}]")
        lines.append(f"回访原因：{it.reason}")
        lines.extend(f"- {q}" for q in it.questions)
    if p.note:
        lines.append("")
        lines.append(f"> {p.note}")
    return "\n".join(lines) + "\n"
