"""路由测试（Demo LLM）：好故事深挖 / 模糊切换 / 告别收尾。"""
import unittest

from timememory.interview.llm import DemoInterviewLLM
from timememory.interview.models import RouteContext
from timememory.interview.topics import TOPIC_MAP, TOPICS


def make_ctx(answer: str, current: str = "childhood", turn: int = 1) -> RouteContext:
    uncovered = [t for t in TOPICS if t.id != current]
    return RouteContext(
        elder_name="张爷爷", current_topic=TOPIC_MAP[current],
        topic_turns_current=1, turn_count=turn, max_turns=30,
        covered_names=[], uncovered=uncovered, recent_qa=[], last_answer=answer,
    )


class TestDemoRouter(unittest.TestCase):
    def setUp(self):
        self.llm = DemoInterviewLLM()

    def test_rich_answer_followup(self):
        d = self.llm.route(make_ctx("我在四川嘉陵江边长大的。有一次差点被水冲走，多亏邻居王二哥把我捞起来。"))
        self.assertEqual(d.action, "followup")
        self.assertEqual(d.focus, "四川嘉陵江")

    def test_vague_answer_switch(self):
        ctx = make_ctx("记不清了。")
        d = self.llm.route(ctx)
        self.assertEqual(d.action, "switch")
        self.assertEqual(d.next_topic_id, ctx.uncovered[0].id)

    def test_vague_no_topic_left_wrap(self):
        ctx = make_ctx("记不清了。")
        ctx.uncovered = []
        d = self.llm.route(ctx)
        self.assertEqual(d.action, "wrap")

    def test_farewell_wrap(self):
        d = self.llm.route(make_ctx("我累了，咱们下次再聊吧。"))
        self.assertEqual(d.action, "wrap")

    def test_opening_no_answer(self):
        d = self.llm.route(make_ctx(""))
        self.assertEqual(d.action, "followup")

    def test_mixed_substance_followup(self):
        # 半句有料（"穷"有情感分量）→ 深挖而非切换
        d = self.llm.route(make_ctx("小时候家里很穷，别的记不清了。"))
        self.assertEqual(d.action, "followup")


if __name__ == "__main__":
    unittest.main()
