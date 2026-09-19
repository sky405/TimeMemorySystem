"""Phase 6 家族记忆库测试：定稿入库 + RAG 问答（Demo，离线）。"""
import unittest

from test_drafting_pipeline import run_demo

from timememory.drafting import DemoDraftingLLM, run_drafting
from timememory.library import (
    ChatSession,
    DemoLibraryLLM,
    ingest_book,
)
from timememory.material import DemoEmbedder


def _ingested(records=None):
    _, store = run_demo()
    emb = DemoEmbedder()
    out = run_drafting(store, "iv-test", {"name": "张爷爷"}, birth_year=1953,
                       llm=DemoDraftingLLM())
    chapters = [d.model_dump() for d in out["drafts"]]
    frags = store.all_fragments("iv-test")
    report = ingest_book(store, emb, "iv-test", chapters, frags, records or [])
    return store, emb, report


class TestLibrary(unittest.TestCase):
    def test_ingest(self):
        store, _, report = _ingested()
        self.assertEqual(report, {"chapters": 1, "cautions": []})
        book = store.get_fragment("iv-test:book:ch-0")
        self.assertIsNotNone(book)
        self.assertEqual(book["topic_id"], "book")
        self.assertEqual(book["importance"], 5)
        self.assertIn("1962年", book["time_refs"])  # 继承成员片段年份
        self.assertIn("王二哥", book["person_refs"])
        self.assertIn("iv-test:book:ch-0", dict(store.all_embeddings()))

    def test_ask_answers_with_citation(self):
        store, emb, _ = _ingested()
        session = ChatSession(store, emb, DemoLibraryLLM())
        res = session.ask("谁把我捞起来？")
        self.assertTrue(res["answer"].has_answer)
        self.assertIn("捞", res["answer"].text)
        self.assertTrue(res["answer"].citations)
        scores = [p["score"] for p in res["passages"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(res["standalone"], "谁把我捞起来？")  # Demo 改写原样返回
        self.assertTrue(any("出现于" in t for t in res["triples"]))  # 图谱线索在场

    def test_gate_rejects_low_score(self):
        store, emb, _ = _ingested()
        session = ChatSession(store, emb, DemoLibraryLLM(), threshold=0.99)
        res = session.ask("谁把我捞起来？")
        self.assertFalse(res["answer"].has_answer)
        self.assertIn("没有找到相关记载", res["answer"].text)
        self.assertEqual(res["answer"].citations, [])

    def test_gibberish_no_answer(self):
        store, emb, _ = _ingested()
        session = ChatSession(store, emb, DemoLibraryLLM())
        res = session.ask("火星移民计划zxq")
        self.assertFalse(res["answer"].has_answer)

    def test_caution_marked(self):
        records = [{"chapter": "童年", "quote": "把我拖上了嘉陵江的岸",
                    "issue": "x", "severity": "存疑", "verdict": "存疑保留"}]
        store, emb, report = _ingested(records)
        self.assertEqual(len(report["cautions"]), 1)
        session = ChatSession(store, emb, DemoLibraryLLM(),
                              cautions=report["cautions"])
        # 用原文自查：与自身余弦恒为 1.0，必为 top1
        res = session.ask("后来王二哥跳下水，把我拖上了嘉陵江的岸。")
        self.assertTrue(res["passages"][0]["caution"])
        self.assertIn("待考", res["answer"].text)

    def test_history_kept(self):
        store, emb, _ = _ingested()
        session = ChatSession(store, emb, DemoLibraryLLM())
        session.ask("谁把我捞起来？")
        res = session.ask("后来呢？")
        self.assertEqual(len(session.history), 2)
        self.assertEqual(res["standalone"], "后来呢？")


if __name__ == "__main__":
    unittest.main()
