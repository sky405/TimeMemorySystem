"""记忆片段提取器测试。"""
import unittest

from timememory.interview.extractor import FragmentExtractor


class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.ex = FragmentExtractor()

    def test_place_and_followup(self):
        frags = self.ex.extract("我在四川嘉陵江边长大的，经常去游泳。", "childhood", 1)
        self.assertEqual(len(frags), 1)
        f = frags[0]
        self.assertIn("四川嘉陵江", f.place_refs)
        self.assertTrue(f.needs_followup)
        self.assertGreaterEqual(f.importance, 3)

    def test_time_place_person(self):
        frags = self.ex.extract("1962年，我9岁，跟着父亲从合川县到了重庆市。", "childhood", 2)
        self.assertEqual(len(frags), 1)
        f = frags[0]
        self.assertIn("1962年", f.time_refs)
        self.assertIn("9岁", f.time_refs)
        self.assertIn("合川县", f.place_refs)
        self.assertIn("重庆市", f.place_refs)
        self.assertIn("父亲", f.person_refs)
        self.assertGreaterEqual(f.importance, 4)

    def test_noise_filtered(self):
        for noise in ["嗯", "哦", "好的", ""]:
            self.assertEqual(self.ex.extract(noise, "childhood", 1), [], noise)

    def test_long_answer_splits_sentences(self):
        text = "小时候家里很穷，兄弟姐妹五个。我每天放学就去放牛，牛跑丢过一次，害我被父亲狠狠骂了一顿。"
        frags = self.ex.extract(text, "childhood", 3)
        self.assertGreaterEqual(len(frags), 2)
        self.assertTrue(all(f.content for f in frags))

    def test_forgot_with_object_not_empty(self):
        # "忘了吃"是故事细节，不是"忘记了"——不能过滤
        frags = self.ex.extract("1962年夏天，我在嘉陵江边玩得太开心，连晚饭都忘了吃。", "childhood", 1)
        self.assertEqual(len(frags), 1)
        self.assertIn("嘉陵江", frags[0].place_refs)

    def test_vague_only_filtered(self):
        for vague in ["记不清了，都是很久以前的事了。", "别的也记不清了。", "不知道，早都忘了。"]:
            self.assertEqual(self.ex.extract(vague, "childhood", 1), [], vague)

    def test_mixed_vague_kept(self):
        # 半句有料半句模糊 → 保留（有料子句撑住）
        frags = self.ex.extract("小时候家里很穷，别的记不清了。", "childhood", 1)
        self.assertEqual(len(frags), 1)

    def test_relation_name_full(self):
        frags = self.ex.extract("多亏邻居王二哥把我捞起来。", "childhood", 1)
        self.assertTrue(frags)
        self.assertIn("王二哥", frags[0].person_refs)
        self.assertNotIn("二哥", frags[0].person_refs)
        self.assertNotIn("哥", frags[0].person_refs)

    def test_collective_not_place(self):
        frags = self.ex.extract("全村人夏天都在树下乘凉。", "hometown", 1)
        self.assertTrue(frags)
        self.assertNotIn("全村", frags[0].place_refs)

    def test_chinese_age_time_clean(self):
        frags = self.ex.extract("那是七八岁的时候，水流特别急。", "childhood", 1)
        self.assertTrue(frags)
        self.assertIn("七八岁", frags[0].time_refs)
        self.assertNotIn("那是七八岁的时候", frags[0].time_refs)

    def test_idiom_not_place(self):
        frags = self.ex.extract("集上人山人海，热闹极了。", "childhood", 1)
        self.assertTrue(frags)
        self.assertEqual(frags[0].place_refs, [])

    def test_emotion_detected(self):
        frags = self.ex.extract("想起去世的老伴，我心里就难过得不行。", "love_marriage", 4)
        self.assertTrue(frags)
        self.assertEqual(frags[0].emotion, "难过")


if __name__ == "__main__":
    unittest.main()
