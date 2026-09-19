"""人生阶段分组：按年份把素材分到人生阶段。

有出生年 → 按年龄映射到 童年/少年/青年/中年/晚年；
无出生年 → 按年代分组（"1960年代"）。
无年份的片段单独列出，由大纲环节归位到各章。纯规则，不做判断。
"""
from __future__ import annotations

import re

from .models import StageGroup

_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")

# (年龄下限, 阶段名)
STAGES = [(0, "童年"), (13, "少年"), (19, "青年"), (36, "中年"), (60, "晚年")]


def fragment_years(frag: dict) -> list[int]:
    hay = frag.get("content", "") + " " + " ".join(frag.get("time_refs", []))
    return sorted({int(y) for y in _YEAR_RE.findall(hay)})


def _stage_for_age(age: int) -> str:
    name = STAGES[0][1]
    for low, label in STAGES:
        if age >= low:
            name = label
    return name


def group_stages(fragments: list[dict], birth_year: int | None = None
                 ) -> tuple[list[StageGroup], list[dict]]:
    """返回 (阶段分组, 无年份片段)，分组按年份升序。"""
    buckets: dict[str, dict] = {}
    undated: list[dict] = []
    for f in sorted(fragments, key=lambda x: x.get("id", "")):
        years = fragment_years(f)
        if not years:
            undated.append(f)
            continue
        first = years[0]
        if birth_year:
            key = _stage_for_age(first - birth_year)
        else:
            key = f"{first // 10 * 10}年代"
        b = buckets.setdefault(key, {"years": set(), "ids": [], "topics": set()})
        b["years"].update(years)
        b["ids"].append(f["id"])
        if f.get("topic_id"):
            b["topics"].add(f["topic_id"])

    groups: list[StageGroup] = []
    for i, (key, b) in enumerate(sorted(buckets.items(), key=lambda kv: min(kv[1]["years"]))):
        years = sorted(b["years"])
        decade = f"{years[0] // 10 * 10}年代"
        title = f"{key}（{decade}）" if birth_year else key
        groups.append(StageGroup(id=f"stage-{i}", title=title, years=years,
                                 fragment_ids=b["ids"], topics=sorted(b["topics"])))
    return groups, undated
