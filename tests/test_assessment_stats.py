"""Phase 3 素材摘要测试（确定性统计，不依赖 LLM）。"""
import unittest

from test_material_pipeline import run_demo

from timememory.assessment.stats import gather_stats


class TestGatherStats(unittest.TestCase):
    def test_topic_coverage(self):
        _, store = run_demo()
        s = gather_stats(store, "iv-test")
        self.assertEqual(s["fragments"], 4)
        self.assertEqual(s["by_topic"], {"childhood": 4})
        self.assertEqual(len(s["empty_topics"]), 8)
        self.assertEqual(s["empty_topics"][0]["id"], "hometown")  # 保持人生线顺序
        self.assertNotIn("childhood", [t["id"] for t in s["empty_topics"]])
        self.assertEqual(s["thin_topics"], [])

    def test_years(self):
        _, store = run_demo()
        s = gather_stats(store, "iv-test")
        self.assertEqual(s["years"], [1962])
        self.assertEqual(s["empty_decades"], [])

    def test_entities(self):
        _, store = run_demo()
        s = gather_stats(store, "iv-test")
        thin = [e["name"] for e in s["thin_entities"]]
        self.assertIn("邻居", thin)  # 只出现一次
        self.assertNotIn("王二哥", thin)  # 出现两次，有故事
        wang = next(e for e in s["entities"] if e["name"] == "王二哥")
        self.assertEqual(wang["mentions"], 2)
        self.assertEqual(wang["topic"], "childhood")

    def test_quality_numbers(self):
        _, store = run_demo()
        s = gather_stats(store, "iv-test")
        self.assertEqual(s["emotion_coverage"], 0.25)  # 只有一条记了"害怕"
        self.assertEqual(s["avg_importance"], 4.5)
        self.assertEqual(len(s["samples"]["childhood"]), 2)


if __name__ == "__main__":
    unittest.main()
