"""Phase 4 阶段分组测试（确定性规则）。"""
import unittest

from test_material_pipeline import run_demo

from timememory.drafting.stages import fragment_years, group_stages


class TestGroupStages(unittest.TestCase):
    def test_with_birth_year(self):
        _, store = run_demo()
        frags = store.all_fragments("iv-test")
        groups, undated = group_stages(frags, birth_year=1953)
        self.assertEqual(len(groups), 1)
        self.assertIn("童年", groups[0].title)  # 1962 年时 9 岁
        self.assertIn("1960年代", groups[0].title)
        self.assertEqual(groups[0].fragment_ids, ["iv-test:frag-0005"])
        self.assertEqual(sorted(f["id"] for f in undated),
                         ["iv-test:frag-0001", "iv-test:frag-0002", "iv-test:frag-0006"])

    def test_without_birth_year(self):
        _, store = run_demo()
        groups, _ = group_stages(store.all_fragments("iv-test"))
        self.assertEqual(groups[0].title, "1960年代")

    def test_all_undated(self):
        frags = [{"id": "s:f1", "content": "没有年份的回忆。", "time_refs": []},
                 {"id": "s:f2", "content": "也没有年份。", "time_refs": []}]
        groups, undated = group_stages(frags, birth_year=1953)
        self.assertEqual(groups, [])
        self.assertEqual(len(undated), 2)

    def test_fragment_years(self):
        self.assertEqual(fragment_years({"content": "1962年到了重庆，1978年回乡。",
                                         "time_refs": []}),
                         [1962, 1978])


if __name__ == "__main__":
    unittest.main()
