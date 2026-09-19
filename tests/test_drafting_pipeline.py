"""Phase 4 全流程测试：分组 → 大纲 → 写作 ⇄ 回检 → 统稿（Demo LLM）。"""
import unittest

from test_material_pipeline import run_demo

from timememory.drafting import (
    DemoDraftingLLM,
    normalize_outline,
    run_drafting,
)
from timememory.drafting.models import ChapterSpec, Outline
from timememory.material.models import CleanFragment
from timememory.material.store import MaterialStore, connect_sqlite


class InventingLLM(DemoDraftingLLM):
    """编造年份的写手：回检必须抓住 1978。"""

    def write_chapter(self, elder, chapter, materials, prev_summary=""):
        d = super().write_chapter(elder, chapter, materials, prev_summary)
        return d.model_copy(update={"text": d.text + "\n\n1978年，我还去过北京。"})


def _elder():
    return {"name": "张爷爷", "hometown": "四川合川"}


class TestDraftingPipeline(unittest.TestCase):
    def test_demo_full_run(self):
        _, store = run_demo()
        out = run_drafting(store, "iv-test", _elder(), birth_year=1953,
                           llm=DemoDraftingLLM())
        self.assertEqual(out["stats"],
                         {"fragments": 4, "groups": 1, "chapters": 1,
                          "passed": 1, "flags": 0})
        self.assertIn("张爷爷", out["outline"].title)
        self.assertIn("童年", out["drafts"][0].title)
        # 素材不重不漏
        covered = [i for d in out["drafts"] for i in d.fragment_ids]
        self.assertEqual(sorted(covered), sorted(
            f["id"] for f in store.all_fragments("iv-test")))
        self.assertEqual(out["review"], [])
        for kw in ("目录", "附录一", "附录二", "无存疑", "经常去游泳"):
            self.assertIn(kw, out["manuscript"])

    def test_factcheck_catches_invention(self):
        _, store = run_demo()
        out = run_drafting(store, "iv-test", _elder(), birth_year=1953,
                           llm=InventingLLM())
        self.assertEqual(out["stats"]["flags"], 1)
        self.assertEqual(out["stats"]["passed"], 0)
        self.assertFalse(out["drafts"][0].factcheck.passed)
        self.assertEqual(len(out["review"]), 1)
        self.assertIn("1978", out["review"][0].quote + out["review"][0].issue)
        # 批注与复核入口进附录
        self.assertIn("存疑批注与人工复核清单", out["manuscript"])
        self.assertIn("1978", out["manuscript"])
        self.assertIn("查证片段", out["manuscript"])

    def test_normalize_outline(self):
        _, store = run_demo()
        frags = store.all_fragments("iv-test")
        bad = Outline(title="t", chapters=[
            ChapterSpec(id="ch-0", title="A", fragment_ids=[
                "iv-test:frag-0001", "iv-test:frag-0001", "nope"]),
            ChapterSpec(id="ch-1", title="B", fragment_ids=[]),  # 空章丢弃
        ])
        fixed = normalize_outline(bad, frags)
        self.assertEqual(len(fixed.chapters), 1)
        self.assertEqual(fixed.chapters[0].fragment_ids,
                         ["iv-test:frag-0001", "iv-test:frag-0002",
                          "iv-test:frag-0005", "iv-test:frag-0006"])

    def test_empty_store(self):
        store = MaterialStore(connect_sqlite(":memory:"))
        out = run_drafting(store, "iv-empty", _elder(), llm=DemoDraftingLLM())
        self.assertEqual(out["stats"]["chapters"], 0)
        self.assertIn("暂无素材", out["manuscript"])
        self.assertEqual(out["review"], [])

    def test_undated_only_single_chapter(self):
        store = MaterialStore(connect_sqlite(":memory:"))
        store.save_fragments([
            CleanFragment(id="s:f1", session_id="s", content="小时候爱掏鸟窝。",
                          topic_id="childhood", source_turn=1),
            CleanFragment(id="s:f2", session_id="s", content="村口有棵大槐树。",
                          topic_id="hometown", source_turn=2),
        ])
        out = run_drafting(store, "s", _elder(), birth_year=1953, llm=DemoDraftingLLM())
        self.assertEqual(len(out["drafts"]), 1)
        self.assertEqual(out["drafts"][0].title, "岁月记忆")
        self.assertEqual(sorted(out["drafts"][0].fragment_ids), ["s:f1", "s:f2"])

    def test_deterministic(self):
        texts = []
        for _ in range(2):
            _, store = run_demo()
            texts.append(run_drafting(store, "iv-test", _elder(), birth_year=1953,
                                      llm=DemoDraftingLLM())["manuscript"])
        self.assertEqual(texts[0], texts[1])


if __name__ == "__main__":
    unittest.main()
