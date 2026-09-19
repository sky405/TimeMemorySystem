"""全流程测试：LangGraph 开场 → 深挖 → 切换 → 收尾 → 修复环 → 报告 → 恢复。"""
import tempfile
import unittest
from pathlib import Path

from timememory.interview.agent import InterviewAgent
from timememory.interview.llm import DemoInterviewLLM
from timememory.interview.models import AgentConfig, ElderProfile, MemoryFragment


def make_agent(**cfg_kw) -> InterviewAgent:
    return InterviewAgent(
        elder=ElderProfile(name="张爷爷", age=82, hometown="四川合川"),
        config=AgentConfig(**cfg_kw),
        llm=DemoInterviewLLM(),
    )


class FailOnceLLM(DemoInterviewLLM):
    """抽取第一次返回废片段（触发 fix 修复环），之后正常。"""

    def __init__(self):
        self.calls: list[str] = []
        self.feedbacks: list[str] = []

    def extract_fragments(self, answer, topic_name, previous, feedback, committed):
        self.calls.append(answer)
        self.feedbacks.append(feedback)
        if len(self.calls) == 1:
            return [MemoryFragment(content="ab")]
        return self._rule_extract(answer)


class TestGraphFlow(unittest.TestCase):
    def test_start_greeting(self):
        agent = make_agent()
        reply = agent.start()
        self.assertIn("张爷爷", reply.text)
        self.assertIn("?", reply.text.replace("？", "?"))
        self.assertEqual(agent.state["current_topic_id"], "childhood")

    def test_full_flow_abc(self):
        agent = make_agent()
        agent.start()
        r1 = agent.step("我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。")
        self.assertEqual(r1.decision.action, "followup")
        self.assertTrue(r1.new_fragments)
        self.assertTrue(any(e in r1.text for e in ["嘉陵江", "王二哥", "邻居"]))
        self.assertFalse(r1.session_ended)

        r2 = agent.step("那是七八岁的时候，水流特别急，我越扑腾离岸越远，吓得直哭。")
        self.assertEqual(r2.decision.action, "followup")

        r3 = agent.step("记不清了，都是很久以前的事了。")
        self.assertEqual(r3.decision.action, "switch")
        self.assertNotEqual(agent.state["current_topic_id"], "childhood")
        self.assertIn("childhood", agent.state["covered_topic_ids"])

        r4 = agent.step("我们村口有棵大槐树，全村人夏天都在树下乘凉，村长还常在那儿给大家开会。")
        self.assertEqual(r4.decision.action, "followup")

        r5 = agent.step("我有点累了，咱们下次再聊吧。")
        self.assertEqual(r5.decision.action, "wrap")
        self.assertTrue(r5.session_ended)
        self.assertEqual(r5.new_fragments, [])  # 告别语不抽取

        r6 = agent.step("我还想聊。")
        self.assertTrue(r6.session_ended)

    def test_single_question_rule(self):
        agent = make_agent()
        agent.start()
        r = agent.step("我在嘉陵江边长大，六岁那年跟爷爷去赶集，集上人山人海。")
        self.assertLessEqual(r.text.count("？") + r.text.count("?"), 1, r.text)

    def test_max_turns_wrap(self):
        agent = make_agent(max_total_turns=2)
        agent.start()
        agent.step("我在嘉陵江边长大。")
        r = agent.step("村里有条小河。")
        self.assertTrue(r.session_ended)
        self.assertEqual(r.decision.action, "wrap")

    def test_fix_loop_repairs(self):
        agent = InterviewAgent(
            elder=ElderProfile(name="张爷爷"), llm=FailOnceLLM())
        agent.start()
        r = agent.step("我在四川嘉陵江边长大的，经常去游泳。")
        llm = agent.llm
        self.assertEqual(len(llm.calls), 2)  # 抽取被调用两次
        self.assertEqual(llm.feedbacks[0], "")
        self.assertIn("过短", llm.feedbacks[1])  # 第二次带上了验证反馈
        self.assertTrue(r.new_fragments)  # 修复后提交成功
        self.assertIn("四川嘉陵江", r.new_fragments[0].place_refs)
        self.assertEqual(agent.state["retries"], 0)
        self.assertEqual(agent.state["last_rejected"], [])

    def test_coverage_report(self):
        agent = make_agent()
        agent.start()
        agent.step("我在四川嘉陵江边长大的，经常去游泳。")
        agent.step("我累了，下次再聊吧。")
        report = agent.coverage_report()
        self.assertGreater(report["topics"]["childhood"]["coverage"], 0)
        self.assertIn(report["recommendation"],
                      ["ready_for_writing", "supplementary_interview", "continue_interview"])
        self.assertIn("resume_topics", report["next_interview_plan"])

    def test_transcript_export(self):
        agent = make_agent()
        agent.start()
        agent.step("我在四川嘉陵江边长大的，经常去游泳。")
        md = agent.export_transcript_markdown()
        self.assertIn("访谈逐字稿", md)
        self.assertIn("决策", md)
        self.assertIn("记忆片段", md)

    def test_save_load_resume(self):
        agent = make_agent()
        agent.start()
        agent.step("我在四川嘉陵江边长大的，经常去游泳。")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            agent.save_session(p)
            snapshot = InterviewAgent.load_session(p)
        agent2 = make_agent()
        r = agent2.start(resume_state=snapshot)
        self.assertEqual(agent2.state["turn_count"], 1)
        self.assertTrue(r.text)  # 恢复后直接拿到下一问
        r2 = agent2.step("江水很清，夏天好多人在里面游泳。")
        self.assertFalse(r2.session_ended)

    def test_supplementary_interview(self):
        agent = make_agent()
        r = agent.start(first_topic_id="education")
        self.assertEqual(agent.state["current_topic_id"], "education")
        self.assertIn("读过书", r.text)


if __name__ == "__main__":
    unittest.main()
