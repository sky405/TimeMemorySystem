"""三叉路口决策引擎测试：分支 A（深挖）/ B（切换）/ C（收尾）。"""
import unittest

from timememory.interview.decision import DecisionEngine
from timememory.interview.extractor import FragmentExtractor
from timememory.interview.llm import MockLLMClient
from timememory.interview.models import (
    AgentConfig,
    DecisionAction,
    ElderProfile,
    InterviewState,
    Message,
    Speaker,
)


def make_state(
    topic_id: str = "childhood",
    turn_count: int = 1,
    topic_turns: int = 1,
    history: list[str] | None = None,
) -> InterviewState:
    st = InterviewState(
        elder=ElderProfile(name="张爷爷", age=82, hometown="四川合川"),
        config=AgentConfig(),
    )
    st.current_topic_id = topic_id
    st.turn_count = turn_count
    st.topic_turn_count = topic_turns
    st.topic_turns[topic_id] = topic_turns
    for i, h in enumerate(history or []):
        st.messages.append(Message(Speaker.AI, f"问题{i}", topic_id, i))
        st.messages.append(Message(Speaker.ELDER, h, topic_id, i))
    return st


class TestDecisionBranches(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine()
        self.extractor = FragmentExtractor()
        self.mock = MockLLMClient()

    def decide(self, state, answer, use_llm=False):
        frags = self.extractor.extract(answer, state.current_topic_id, state.turn_count)
        state.fragments.extend(frags)
        state.messages.append(Message(Speaker.ELDER, answer, state.current_topic_id, state.turn_count))
        llm = self.mock if use_llm else None
        return self.engine.decide(state, answer, frags, llm=llm), frags

    # -- 分支 A：发现好故事 → 深挖 ----------------------------------------------
    def test_rich_story_deep_dive(self):
        st = make_state()
        d, _ = self.decide(st, "我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。")
        self.assertEqual(d.action, DecisionAction.DEEP_DIVE, d.reasoning)
        self.assertGreaterEqual(d.confidence, 0.4)

    def test_rich_story_deep_dive_with_mock_llm(self):
        st = make_state()
        d, _ = self.decide(
            st, "1962年我9岁，跟着父亲从合川县到了重庆市，在朝天门码头扛过包。",
            use_llm=True,
        )
        self.assertEqual(d.action, DecisionAction.DEEP_DIVE, d.reasoning)

    # -- 分支 B：话题聊干了 → 切换 -----------------------------------------------
    def test_vague_after_min_turns_switch(self):
        st = make_state(topic_turns=2, turn_count=2,
                        history=["我在嘉陵江边长大，经常游泳。"])
        d, _ = self.decide(st, "记不清了，都是很久以前的事了。")
        self.assertEqual(d.action, DecisionAction.SWITCH_TOPIC, d.reasoning)
        self.assertTrue(d.focus)  # focus 应为新话题 id

    def test_vague_with_mock_llm_switch(self):
        st = make_state(topic_turns=3, turn_count=3,
                        history=["我在嘉陵江边长大。", "小时候的事。"])
        d, _ = self.decide(st, "不知道，早都忘了。", use_llm=True)
        self.assertEqual(d.action, DecisionAction.SWITCH_TOPIC, d.reasoning)

    def test_topic_max_turns_hard_rule_switch(self):
        st = make_state(topic_turns=6, turn_count=6)  # childhood max_turns=6
        d, _ = self.decide(st, "还有好多故事可以讲呢。")
        self.assertEqual(d.action, DecisionAction.SWITCH_TOPIC, d.reasoning)
        self.assertIn("R3", d.reasoning)

    # -- 分支 C：老人累了 → 收尾 --------------------------------------------------
    def test_tired_wrap_up(self):
        st = make_state()
        d, _ = self.decide(st, "我有点累了，咱们下次再聊吧。")
        self.assertEqual(d.action, DecisionAction.WRAP_UP, d.reasoning)
        self.assertIn("R1", d.reasoning)

    def test_goodbye_wrap_up(self):
        st = make_state(topic_turns=3, turn_count=5)
        d, _ = self.decide(st, "今天就到这吧，我要去吃饭了。")
        self.assertEqual(d.action, DecisionAction.WRAP_UP, d.reasoning)

    def test_max_total_turns_wrap_up(self):
        st = make_state(topic_turns=2, turn_count=30)  # 达硬上限
        d, _ = self.decide(st, "我精神好着呢，接着聊！")
        self.assertEqual(d.action, DecisionAction.WRAP_UP, d.reasoning)
        self.assertIn("R2", d.reasoning)

    # -- 边界：话题没聊够时再给一次机会 -------------------------------------------
    def test_min_turns_grace_one_more_chance(self):
        st = make_state(topic_turns=1, turn_count=1)
        d, _ = self.decide(st, "记不清了。")
        self.assertEqual(d.action, DecisionAction.DEEP_DIVE, d.reasoning)

    # -- 防误伤：故事里的"告别词"不能触发收尾 --------------------------------------
    def test_story_hardship_not_wrap(self):
        st = make_state()
        d, _ = self.decide(st, "那年头日子太困难了，一家人经常吃了上顿没下顿。")
        self.assertNotEqual(d.action, DecisionAction.WRAP_UP, d.reasoning)

    def test_story_death_euphemism_not_wrap(self):
        st = make_state(topic_turns=2, turn_count=2)
        d, _ = self.decide(st, "那年我爹走了，全家人哭成一团。")
        self.assertNotEqual(d.action, DecisionAction.WRAP_UP, d.reasoning)

    def test_story_meal_not_wrap(self):
        st = make_state()
        d, _ = self.decide(st, "每天早起吃完饭就下地干活，中午在田埂上歇一会儿。")
        self.assertNotEqual(d.action, DecisionAction.WRAP_UP, d.reasoning)

    def test_story_land_reform_not_wrap(self):
        st = make_state()
        d, _ = self.decide(st, "当年改天换地修水库，全村老少齐上阵。")
        self.assertNotEqual(d.action, DecisionAction.WRAP_UP, d.reasoning)

    def test_story_cooking_not_wrap(self):
        st = make_state()
        d, _ = self.decide(st, "我妈每天回家做饭，日子过得紧巴巴的。")
        self.assertNotEqual(d.action, DecisionAction.WRAP_UP, d.reasoning)

    def test_forgot_with_object_still_dive(self):
        # "忘了吃"是故事细节，不能触发枯竭切换
        st = make_state(topic_turns=2, turn_count=2)
        d, _ = self.decide(st, "1962年夏天，我在嘉陵江边玩得太开心，连晚饭都忘了吃。", use_llm=True)
        self.assertEqual(d.action, DecisionAction.DEEP_DIVE, d.reasoning)

    def test_first_person_goodbye_wrap(self):
        st = make_state(topic_turns=2, turn_count=3)
        d, _ = self.decide(st, "我先走了，孙子放学了我得去接。")
        self.assertEqual(d.action, DecisionAction.WRAP_UP, d.reasoning)

    def test_repetition_pushes_exhaustion(self):
        st = make_state(topic_turns=2, turn_count=2, history=["我在嘉陵江边长大，经常游泳。"])
        d, _ = self.decide(st, "我在嘉陵江边长大，经常去游泳。")  # 高度重复
        self.assertEqual(d.action, DecisionAction.SWITCH_TOPIC, d.reasoning)


if __name__ == "__main__":
    unittest.main()
