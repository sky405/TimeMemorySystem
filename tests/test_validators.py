"""验证器测试：有效片段放行，问题片段驳回并给理由。"""
import unittest

from timememory.interview.models import MemoryFragment
from timememory.interview.validators import is_farewell, validate_fragments


def frag(content, **kw):
    return MemoryFragment(content=content, **kw)


class StubLLM:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls: list[str] = []

    def plausibility(self, fragment):
        self.calls.append(fragment.content)
        return self.verdicts.pop(0) if self.verdicts else True


class TestValidateFragments(unittest.TestCase):
    def test_valid_passes(self):
        r = validate_fragments(
            [frag("我在四川嘉陵江边长大的。", place_refs=["四川嘉陵江"])], [])
        self.assertEqual(len(r.valid), 1)
        self.assertEqual(r.rejected, [])

    def test_schema_error_rejected(self):
        r = validate_fragments([{"content": "内容足够长的一句话", "importance": 99}], [])
        self.assertEqual(len(r.valid), 0)
        self.assertEqual(len(r.rejected), 1)
        self.assertIn("schema", r.rejected[0].reasons[0])

    def test_short_rejected(self):
        r = validate_fragments([frag("ab")], [])
        self.assertEqual(len(r.valid), 0)
        self.assertIn("过短", r.rejected[0].reasons[0])

    def test_pure_vague_rejected(self):
        r = validate_fragments([frag("记不清了。")], [])
        self.assertEqual(len(r.valid), 0)
        self.assertIn("模糊", r.rejected[0].reasons[0])

    def test_vague_with_substance_passes(self):
        r = validate_fragments(
            [frag("小时候家里很穷，别的记不清了。", time_refs=["小时候"], emotion="难过")], [])
        self.assertEqual(len(r.valid), 1)

    def test_forgot_with_object_passes(self):
        # "忘了吃"是故事细节，不是"忘记了"
        r = validate_fragments([frag("那天玩得太开心，连晚饭都忘了吃。")], [])
        self.assertEqual(len(r.valid), 1)

    def test_duplicate_rejected(self):
        committed = [frag("我在嘉陵江边长大，经常游泳。")]
        r = validate_fragments([frag("我在嘉陵江边长大，经常游泳。")], committed)
        self.assertEqual(len(r.valid), 0)
        self.assertIn("重复", r.rejected[0].reasons[0])

    def test_llm_check_rejects(self):
        llm = StubLLM([False])
        r = validate_fragments([frag("今天天气不错。")], [], llm=llm, llm_check=True)
        self.assertEqual(len(r.valid), 0)
        self.assertIn("质检", r.rejected[0].reasons[0])
        self.assertEqual(llm.calls, ["今天天气不错。"])

    def test_llm_error_fail_open(self):
        class Boom:
            def plausibility(self, f):
                raise RuntimeError("down")

        r = validate_fragments([frag("今天天气不错。")], [], llm=Boom(), llm_check=True)
        self.assertEqual(len(r.valid), 1)  # LLM 挂了不挡路


class TestFarewellGuardrail(unittest.TestCase):
    def test_hits(self):
        for text in ["我有点累了，下次再聊吧。", "我先走了。", "我要去吃饭了。", "先睡了。"]:
            self.assertTrue(is_farewell(text), text)

    def test_misses_ambiguous(self):
        # 有歧义的一律不收录，交给 LLM 按语境判断
        for text in ["那年头日子太困难了。", "那年我爹走了。", "当年改天换地修水库。",
                     "下次我再去的时候，他已经搬走了。", "想再见他一面。"]:
            self.assertEqual(is_farewell(text), "", text)


if __name__ == "__main__":
    unittest.main()
