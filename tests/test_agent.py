"""InterviewAgent 全流程测试：开场 → 深挖 → 切换 → 收尾 → 报告 → 持久化。"""
import tempfile
import unittest
from pathlib import Path

from timememory.interview.agent import InterviewAgent
from timememory.interview.llm import MockLLMClient
from timememory.interview.models import DecisionAction, ElderProfile
from timememory.interview.session import load_state, save_state


class TestAgentFlow(unittest.TestCase):
    def setUp(self):
        self.agent = InterviewAgent(
            elder=ElderProfile(name="张爷爷", age=82, hometown="四川合川"),
            llm=MockLLMClient(),
            seed=42,
        )

    def test_start_greeting(self):
        reply = self.agent.start()
        self.assertIn("张爷爷", reply.text)
        self.assertIn("?", reply.text.replace("？", "?"))
        self.assertEqual(self.agent.state.current_topic_id, "childhood")

    def test_full_flow_abc_branches(self):
        self.agent.start()
        # 第 1 轮：好故事 → A 深挖
        r1 = self.agent.step("我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。")
        self.assertEqual(r1.decision.action, DecisionAction.DEEP_DIVE)
        self.assertTrue(r1.new_fragments)
        self.assertFalse(r1.session_ended)
        # 追问里应带上焦点实体（嘉陵江/王二哥/邻居其一）
        self.assertTrue(any(e in r1.text for e in ["嘉陵江", "王二哥", "邻居", "游泳"]))

        # 第 2 轮：继续深挖
        r2 = self.agent.step("那是七八岁的时候，水流特别急，我越扑腾离岸越远，吓得直哭。")
        self.assertEqual(r2.decision.action, DecisionAction.DEEP_DIVE)
        self.assertFalse(r2.session_ended)

        # 第 3 轮：聊干了 → B 切换
        r3 = self.agent.step("记不清了，都是很久以前的事了。")
        self.assertEqual(r3.decision.action, DecisionAction.SWITCH_TOPIC)
        self.assertNotEqual(self.agent.state.current_topic_id, "childhood")
        self.assertIn("childhood", self.agent.state.covered_topic_ids)
        self.assertFalse(r3.session_ended)

        # 第 4 轮：新话题好故事 → A 深挖
        r4 = self.agent.step("我们村口有棵大槐树，全村人夏天都在树下乘凉，村长还常在那儿开会。")
        self.assertEqual(r4.decision.action, DecisionAction.DEEP_DIVE)

        # 第 5 轮：累了 → C 收尾
        r5 = self.agent.step("我有点累了，咱们下次再聊吧。")
        self.assertEqual(r5.decision.action, DecisionAction.WRAP_UP)
        self.assertTrue(r5.session_ended)

        # 结束后不再接受输入
        r6 = self.agent.step("我还想聊。")
        self.assertTrue(r6.session_ended)

    def test_farewell_leaves_no_fragment(self):
        # R1 告别语不是记忆，不留片段
        self.agent.start()
        self.agent.step("我在四川嘉陵江边长大的，经常去游泳。")
        n_before = len(self.agent.state.fragments)
        r = self.agent.step("我有点累了，咱们下次再聊吧。")
        self.assertTrue(r.session_ended)
        self.assertEqual(r.new_fragments, [])
        self.assertEqual(len(self.agent.state.fragments), n_before)

    def test_vague_answer_leaves_no_fragment(self):
        self.agent.start()
        self.agent.step("我在四川嘉陵江边长大的。")
        n_before = len(self.agent.state.fragments)
        self.agent.step("记不清了。")
        self.assertEqual(len(self.agent.state.fragments), n_before)

    def test_single_question_rule(self):
        self.agent.start()
        r = self.agent.step("我在嘉陵江边长大，六岁那年跟爷爷去赶集，集上人山人海。")
        marks = r.text.count("？") + r.text.count("?")
        self.assertLessEqual(marks, 1, r.text)

    def test_coverage_report(self):
        self.agent.start()
        self.agent.step("我在四川嘉陵江边长大的，经常去游泳。")
        self.agent.step("我有点累了，下次再聊吧。")
        report = self.agent.coverage_report()
        self.assertIn("topics", report)
        self.assertIn("childhood", report["topics"])
        self.assertGreater(report["topics"]["childhood"]["coverage"], 0)
        self.assertIn(report["recommendation"],
                      ["ready_for_writing", "supplementary_interview", "continue_interview"])
        self.assertIn("resume_topics", report["next_interview_plan"])
        self.assertTrue(self.agent.fragments_json())

    def test_transcript_export(self):
        self.agent.start()
        self.agent.step("我在四川嘉陵江边长大的，经常去游泳。")
        md = self.agent.export_transcript_markdown()
        self.assertIn("访谈逐字稿", md)
        self.assertIn("决策", md)
        self.assertIn("记忆片段", md)

    def test_session_persistence_roundtrip(self):
        self.agent.start()
        self.agent.step("我在四川嘉陵江边长大的，经常去游泳。")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            save_state(self.agent.state, p)
            st2 = load_state(p)
        self.assertEqual(st2.turn_count, 1)
        self.assertEqual(len(st2.fragments), len(self.agent.state.fragments))
        self.assertEqual(st2.current_topic_id, "childhood")

    def test_supplementary_interview_resume(self):
        # 补充访谈：直切缺口话题
        reply = self.agent.start(first_topic_id="education",
                                 opening_override="您小时候读过书吗？还记得学校吗？")
        self.assertEqual(self.agent.state.current_topic_id, "education")
        self.assertIn("读过书", reply.text)


if __name__ == "__main__":
    unittest.main()
