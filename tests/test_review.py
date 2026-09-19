"""Phase 5 人工审核测试（脚本化 decide_fn，离线）。"""
import unittest

from test_drafting_pipeline import InventingLLM, run_demo

from timememory.drafting import DemoDraftingLLM, run_drafting
from timememory.review import DemoReviewLLM, apply_decision, run_review


def _flagged_out():
    _, store = run_demo()
    out = run_drafting(store, "iv-test", {"name": "张爷爷"}, birth_year=1953,
                       llm=InventingLLM())
    return out, store


def _review(out, store, decide_fn, reviewer="儿子"):
    return run_review(out["drafts"], out["review"], store.all_fragments("iv-test"),
                      elder={"name": "张爷爷"}, title=out["outline"].title,
                      reviewer=reviewer, llm=DemoReviewLLM(), decide_fn=decide_fn)


class TestReview(unittest.TestCase):
    def test_suggest_demo(self):
        sug = DemoReviewLLM().suggest_fix("童年", "1978年去过北京。",
                                          "1978年去过北京", "年份无出处", [])
        self.assertEqual(sug.action, "删除该句")
        self.assertTrue(sug.reason)

    def test_apply_decision_units(self):
        text = "1962年到了重庆。1978年去过北京。后来回乡。"
        out, changed, _ = apply_decision(text, "1978年去过北京", "删除相关句")
        self.assertTrue(changed)
        self.assertNotIn("北京", out)
        self.assertIn("重庆", out)
        out, changed, _ = apply_decision(text, "1978年去过北京", "已修正", "那几年常出远门。")
        self.assertTrue(changed)
        self.assertIn("常出远门", out)
        self.assertNotIn("北京", out)
        for v in ("确认无误", "存疑保留"):
            out, changed, _ = apply_decision(text, "1978年去过北京", v)
            self.assertFalse(changed)
            self.assertEqual(out, text)
        out, changed, note = apply_decision(text, "不存在的话", "删除相关句")
        self.assertFalse(changed)
        self.assertIn("未定位", note)

    def test_confirm_keeps_text(self):
        out, store = _flagged_out()
        calls = []

        def decide(view):
            calls.append(view)
            self.assertEqual(view["total"], 1)
            self.assertIn("1978", view["item"]["quote"] + view["item"]["issue"])
            self.assertTrue(view["materials"])  # 查证素材在场
            self.assertIn("删除该句", view["suggestion"]["action"])  # AI 建议在场
            return {"verdict": "确认无误", "note": "问过老人，确有其事。"}

        res = _review(out, store, decide)
        self.assertEqual(len(calls), 1)
        self.assertIn("去过北京", res["book"])  # 正文保留
        self.assertIn("确认无误", res["book"])
        self.assertIn("问过老人", res["book"])
        self.assertIn("审核：儿子", res["book"])
        self.assertEqual(res["stats"]["by_verdict"], {"确认无误": 1})
        self.assertEqual(res["stats"]["applied"], 0)

    def test_fix_applies_replacement(self):
        out, store = _flagged_out()
        res = _review(out, store, lambda v: {"verdict": "已修正",
                                             "replacement": "那几年还去过很多地方。"})
        final = res["chapters"][0]["text"]
        self.assertIn("去过很多地方", final)
        self.assertNotIn("去过北京", final)  # 附录审计记录仍保留原文，此处查定稿正文
        self.assertEqual(res["stats"]["applied"], 1)
        self.assertIn("已改动正文", res["book"])

    def test_delete_removes_sentence(self):
        out, store = _flagged_out()
        res = _review(out, store, lambda v: {"verdict": "删除相关句"})
        final = res["chapters"][0]["text"]
        self.assertNotIn("北京", final)
        self.assertIn("1962年", final)  # 其它句子不受影响
        self.assertEqual(res["stats"]["by_verdict"], {"删除相关句": 1})

    def test_fix_without_replacement_adopts_delete(self):
        out, store = _flagged_out()
        res = _review(out, store, lambda v: {"verdict": "已修正"})  # 没给改写句
        self.assertEqual(res["records"][0]["verdict"], "删除相关句")  # 采纳 AI 删除建议
        self.assertNotIn("北京", res["chapters"][0]["text"])

    def test_unknown_verdict_downgrades(self):
        out, store = _flagged_out()
        res = _review(out, store, lambda v: {"verdict": "随便"})
        self.assertEqual(res["records"][0]["verdict"], "存疑保留")
        self.assertIn("去过北京", res["book"])  # 正文不动

    def test_no_flags_passthrough(self):
        _, store = run_demo()
        out = run_drafting(store, "iv-test", {"name": "张爷爷"}, birth_year=1953,
                           llm=DemoDraftingLLM())
        calls = []
        res = _review(out, store, lambda v: calls.append(v) or {"verdict": "确认无误"})
        self.assertEqual(calls, [])  # 零存疑，不打扰人
        self.assertIn("零存疑", res["book"])
        self.assertIn("定稿日期", res["book"])
        self.assertIn("# 张爷爷的回忆录（定稿）", res["book"])
        self.assertEqual(res["stats"]["items"], 0)


if __name__ == "__main__":
    unittest.main()
