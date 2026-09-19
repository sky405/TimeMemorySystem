"""定稿入库：章节 → book 片段 + 向量（复用 Phase 2 四表，无新表）；
审核记录 → 待考清单（问答时标注）。
"""
from __future__ import annotations

from ..drafting.stages import fragment_years
from ..material.embeddings import Embedder
from ..material.models import CleanFragment
from ..material.store import MaterialStore
from .models import Caution


def ingest_book(store: MaterialStore, embedder: Embedder, archive_id: str,
                chapters: list[dict], fragments: list[dict],
                records: list[dict] | None = None) -> dict:
    """chapters: run_review 返回的章节（含 fragment_ids）；
    fragments: 整档片段；records: 审核记录 dicts。返回 {"chapters", "cautions"}。
    章节 id 形如 "{archive}:book:{chapter_id}"，topic 记为 "book"。"""
    frag_map = {f["id"]: f for f in fragments}
    n = 0
    for ch in chapters:
        cid = ch.get("chapter_id", f"ch-{n}")
        years: set[int] = set()
        persons: set[str] = set()
        places: set[str] = set()
        for fid in ch.get("fragment_ids", []):
            f = frag_map.get(fid)
            if not f:
                continue
            years.update(fragment_years(f))
            persons.update(f.get("person_refs", []))
            places.update(f.get("place_refs", []))
        frag = CleanFragment(
            id=f"{archive_id}:book:{cid}", session_id=archive_id,
            content=ch.get("text", ""), topic_id="book", source_turn=0,
            time_refs=[f"{y}年" for y in sorted(years)],
            place_refs=sorted(places)[:10], person_refs=sorted(persons)[:10],
            emotion="", importance=5)
        store.save_fragments([frag])
        store.save_embedding(frag.id, embedder.embed_texts([frag.content])[0])
        n += 1
    cautions = [Caution(quote=r.get("quote", ""), chapter=r.get("chapter", ""))
                for r in (records or [])
                if r.get("verdict") == "存疑保留" and r.get("quote")]
    return {"chapters": n, "cautions": [c.model_dump() for c in cautions]}
