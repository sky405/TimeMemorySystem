"""编排层测试：访谈 ↔ 写作协作循环（全 Demo/Stub，离线）。"""
import unittest

from test_assessment_pipeline import StubLLM

from timememory.assessment import DemoAssessmentLLM
from timememory.assessment.models import Assessment, Gap, PlanItem, SupplementPlan
from timememory.drafting import DemoDraftingLLM
from timememory.interview.agent import InterviewAgent
from timememory.interview.llm import DemoInterviewLLM
from timememory.interview.models import ElderProfile
from timememory.material import (
    DemoEmbedder,
    DemoMaterialLLM,
    MaterialStore,
    connect_sqlite,
)
from timememory.orchestration import (
    STATUS_COMPLETE,
    STATUS_MAX_ROUNDS,
    render_memoir_report,
    run_memoir,
)

FAREWELL = "今天有点累了，咱们下次再聊吧。"


def _script(routes: dict) -> callable:
    def fn(question: str, round_index: int, turn: int) -> str:
        script = routes.get(round_index, [])
        return script[turn] if turn < len(script) else FAREWELL
    return fn


def _llms(**overrides) -> dict:
    base = dict(interview_llm=DemoInterviewLLM(), material_llm=DemoMaterialLLM(),
                embedder=DemoEmbedder(), assess_llm=DemoAssessmentLLM(),
                draft_llm=DemoDraftingLLM())
    base.update(overrides)
    return base


def _store() -> MaterialStore:
    return MaterialStore(connect_sqlite(":memory:"))


class TestOrchestration(unittest.TestCase):
    def test_ready_first_round(self):
        stub = StubLLM(Assessment(ready=True, overall="很好", gaps=[]),
                        SupplementPlan(items=[], note=""))
        out = run_memoir(ElderProfile(name="张爷爷"),
                         _script({0: ["我在嘉陵江边长大。", "王二哥救过我。"]}),
                         archive_id="t-ready", store=_store(), **_llms(assess_llm=stub))
        self.assertEqual(out["status"], STATUS_COMPLETE)
        self.assertEqual(len(out["rounds"]), 1)
        self.assertIn("回忆录", out["manuscript"])
        self.assertEqual(out["review"], [])
        report = render_memoir_report(out)
        self.assertIn("各轮访谈", report)

    def test_max_rounds_with_supplement(self):
        stub = StubLLM(
            Assessment(ready=False, overall="缺",
                       gaps=[Gap(title="缺故乡", topic_id="hometown")]),
            SupplementPlan(items=[PlanItem(topic_id="hometown",
                                           questions=["老家是什么样的？", "还记得老屋吗？"])],
                           note=""))
        out = run_memoir(
            ElderProfile(name="张爷爷"),
            _script({0: ["我在嘉陵江边长大。"], 1: ["老家在四川合川，村口有棵大槐树。"]}),
            archive_id="t-max", store=_store(), max_rounds=2, **_llms(assess_llm=stub))
        self.assertEqual(out["status"], STATUS_MAX_ROUNDS)
        self.assertEqual(len(out["rounds"]), 2)
        self.assertEqual([r["session_id"] for r in out["rounds"]],
                         ["t-max-r0", "t-max-r1"])
        r1 = out["rounds"][1]
        self.assertEqual(r1["focus_topics"], ["hometown"])
        self.assertEqual(r1["script_used"], 2)
        # 写作 Agent 的问题真的被访谈 Agent 问出来了
        self.assertTrue(any("老家是什么样的" in q for q in r1["questions_asked"]))
        self.assertIn("附录二", out["manuscript"])

    def test_no_progress_stops(self):
        out = run_memoir(ElderProfile(name="张爷爷"), _script({}),
                         archive_id="t-empty", store=_store(),
                         max_rounds=3, **_llms())
        self.assertEqual(len(out["rounds"]), 1)  # 首轮零产出，不再空转
        self.assertEqual(out["status"], STATUS_MAX_ROUNDS)
        self.assertIn("暂无素材", out["manuscript"])

    def test_demo_full_collaboration(self):
        out = run_memoir(
            ElderProfile(name="张爷爷", hometown="四川合川"),
            _script({0: ["我在嘉陵江边长大。", "王二哥救过我。", "我娘给我煮姜汤。",
                          "1962年到了重庆。"],
                     1: ["老家在四川合川。", "私塾先生姓陈。", "和老伴经人介绍认识。"]}),
            birth_year=1953, archive_id="t-demo", store=_store(),
            max_rounds=2, **_llms())
        self.assertEqual(len(out["rounds"]), 2)
        self.assertGreater(out["rounds"][1]["script_used"], 0)
        self.assertIn("1962", out["manuscript"])  # 素材进成稿
        self.assertIn("附录二", out["manuscript"])

    def test_interview_script_mechanism(self):
        # Phase 1 脚本机制：指定问题原样问出 + 自动切话题
        agent = InterviewAgent(elder=ElderProfile(name="爷"), llm=DemoInterviewLLM())
        reply = agent.start(resume_state={"script": [
            {"topic_id": "hometown", "question": "老家门前有河吗？"}]})
        self.assertIn("老家门前有河吗？", reply.text)
        self.assertEqual(agent.state["current_topic_id"], "hometown")
        self.assertEqual(agent.state["script"], [])


if __name__ == "__main__":
    unittest.main()
