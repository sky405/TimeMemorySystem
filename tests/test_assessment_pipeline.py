"""Phase 3 全流程测试：gather → assess → plan → validate（Demo LLM + Stub LLM）。"""
import unittest

from test_material_pipeline import run_demo

from timememory.assessment import DemoAssessmentLLM, render_brief, run_assessment
from timememory.assessment.models import Assessment, Gap, PlanItem, SupplementPlan
from timememory.interview.topics import TOPIC_MAP


class StubLLM:
    """假 LLM：返回预设的评估与计划，用于测试 validate 修复逻辑。"""

    def __init__(self, assessment: Assessment, plan: SupplementPlan):
        self._a, self._p = assessment, plan

    def assess(self, stats: dict) -> Assessment:
        return self._a

    def plan_questions(self, gaps: list, stats: dict) -> SupplementPlan:
        return self._p


class TestAssessmentPipeline(unittest.TestCase):
    def test_demo_assessment_gaps(self):
        _, store = run_demo()
        out = run_assessment(store, "iv-test", DemoAssessmentLLM())
        a = out["assessment"]
        self.assertFalse(a.ready)
        self.assertEqual(len(a.gaps), 6)  # 缺口截断
        self.assertTrue(all(g.topic_id in TOPIC_MAP for g in a.gaps))
        dims = {d.dimension: d.score for d in a.dimensions}
        self.assertEqual(set(dims), {"话题覆盖", "时间线完整", "人物丰满度", "细节情感"})
        self.assertEqual(dims["话题覆盖"], 1)  # 只覆盖 1/9
        self.assertTrue(all(0 <= v <= 10 for v in dims.values()))

    def test_plan_questions(self):
        _, store = run_demo()
        out = run_assessment(store, "iv-test", DemoAssessmentLLM())
        items = out["plan"].items
        self.assertTrue(len(items) > 0)
        self.assertEqual(items[0].topic_id, "hometown")  # 按人生线顺序补访
        for it in items:
            self.assertIn(it.topic_id, TOPIC_MAP)
            self.assertEqual(it.topic_name, TOPIC_MAP[it.topic_id].name)
            self.assertTrue(1 <= len(it.questions) <= 3)
            self.assertTrue(all(q.strip() for q in it.questions))

    def test_validate_repairs_bad_llm(self):
        _, store = run_demo()
        stub = StubLLM(
            Assessment(ready=False, overall="x",
                       gaps=[Gap(title="缺", topic_id="education")]),
            SupplementPlan(items=[
                PlanItem(topic_id="not-a-topic", questions=[" q1 ", "q1", "  "]),
                PlanItem(topic_id="education", questions=[]),
            ]))
        out = run_assessment(store, "iv-test", stub)
        items = out["plan"].items
        self.assertEqual(len(items), 1)  # 非法话题被丢弃
        self.assertEqual(items[0].topic_id, "education")
        self.assertEqual(len(items[0].questions), 2)  # 空问题用话题示例补齐

    def test_ready_when_complete(self):
        _, store = run_demo()
        stub = StubLLM(Assessment(ready=True, overall="很好", gaps=[]),
                        SupplementPlan(items=[], note=""))
        out = run_assessment(store, "iv-test", stub)
        self.assertTrue(out["assessment"].ready)
        self.assertEqual(out["plan"].items, [])
        self.assertIn("无需", out["plan"].note)

    def test_high_gap_forces_not_ready(self):
        _, store = run_demo()
        stub = StubLLM(
            Assessment(ready=True, overall="x",
                       gaps=[Gap(title="严重缺", priority="高", topic_id="childhood")]),
            SupplementPlan(items=[PlanItem(topic_id="childhood", questions=["还记得吗？"])]))
        out = run_assessment(store, "iv-test", stub)
        self.assertFalse(out["assessment"].ready)  # 有高优先级缺口就不能 ready

    def test_deterministic(self):
        outs = [render_brief(*self._assess()) for _ in range(2)]
        self.assertEqual(outs[0], outs[1])

    def _assess(self):
        _, store = run_demo()
        out = run_assessment(store, "iv-test", DemoAssessmentLLM())
        return out["assessment"], out["plan"]

    def test_render_brief(self):
        _, store = run_demo()
        out = run_assessment(store, "iv-test", DemoAssessmentLLM())
        text = render_brief(out["assessment"], out["plan"])
        for kw in ("写作评估报告", "维度打分", "素材缺口", "补充访谈提纲",
                   "故乡与故土", "老家是什么样的"):
            self.assertIn(kw, text)


if __name__ == "__main__":
    unittest.main()
