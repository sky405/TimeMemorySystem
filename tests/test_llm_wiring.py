"""LangChain 真模型链路测试：用不可达地址验证接线与失败兜底。"""
import unittest

from langchain_openai import ChatOpenAI

from timememory.interview.llm import LangChainInterviewLLM
from timememory.interview.models import RouteContext
from timememory.interview.topics import TOPIC_MAP, TOPICS


def make_dead_llm() -> LangChainInterviewLLM:
    model = ChatOpenAI(
        model="test-model", api_key="sk-test",
        base_url="http://127.0.0.1:1/v1",  # 不可达：连接被拒绝，快速失败
        timeout=5, max_retries=0,
    )
    return LangChainInterviewLLM(model=model)


def make_ctx(answer: str) -> RouteContext:
    return RouteContext(
        elder_name="张爷爷", current_topic=TOPIC_MAP["childhood"],
        topic_turns_current=1, turn_count=1, max_turns=30,
        covered_names=[], uncovered=[t for t in TOPICS if t.id != "childhood"],
        recent_qa=[], last_answer=answer,
    )


class TestLangChainWiring(unittest.TestCase):
    """真模型挂掉时，各判断入口必须返回安全默认值，不能抛异常中断访谈。"""

    def test_route_fallback(self):
        d = make_dead_llm().route(make_ctx("我在嘉陵江边长大。"))
        self.assertEqual(d.action, "followup")

    def test_extract_fallback(self):
        frags = make_dead_llm().extract_fragments("我在嘉陵江边长大。", "童年经历", [], "", [])
        self.assertEqual(frags, [])

    def test_ask_fallback(self):
        q = make_dead_llm().ask_followup(TOPIC_MAP["childhood"], "嘉陵江", "我在嘉陵江边长大。")
        self.assertIn("嘉陵江", q)

    def test_plausibility_fail_open(self):
        from timememory.interview.models import MemoryFragment
        ok = make_dead_llm().plausibility(MemoryFragment(content="今天天气不错。"))
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
