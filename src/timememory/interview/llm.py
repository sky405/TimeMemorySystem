"""LLM 抽象层：判断一律走这里。

- InterviewLLM：协议。路由 / 提问 / 抽取 / 收尾 / 质检五个判断入口。
- LangChainInterviewLLM：真模型（ChatOpenAI 兼容接口 + 结构化输出）。
- DemoInterviewLLM：离线确定性实现，用于测试与无 Key 演示。
- get_llm()：有 TMS_LLM_API_KEY 用真模型，否则降级 Demo。

环境变量：TMS_LLM_BASE_URL / TMS_LLM_API_KEY / TMS_LLM_MODEL
"""
from __future__ import annotations

import os
import re
import sys
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import prompts
from .models import (
    ClosingContext,
    FragmentBatch,
    MemoryFragment,
    PlausibilityVerdict,
    RouteContext,
    RouteDecision,
    Topic,
)


class InterviewLLM(Protocol):
    name: str

    def route(self, ctx: RouteContext) -> RouteDecision: ...
    def ask_opening(self, topic: Topic) -> str: ...
    def ask_followup(self, topic: Topic, focus: str, last_answer: str) -> str: ...
    def ask_transition(self, prev: Topic, highlight: str, new: Topic) -> str: ...
    def extract_fragments(
        self,
        answer: str,
        topic_name: str,
        previous: list[MemoryFragment],
        feedback: str,
        committed: list[MemoryFragment],
    ) -> list[MemoryFragment]: ...
    def closing(self, ctx: ClosingContext) -> str: ...
    def plausibility(self, fragment: MemoryFragment) -> bool: ...


# ----------------------------------------------------------------------------
# 真模型：LangChain ChatOpenAI（兼容 OpenAI / DeepSeek / 千问 / Ollama）
# ----------------------------------------------------------------------------
class LangChainInterviewLLM:
    name = "langchain"

    def __init__(self, model=None, temperature: float = 0.3):
        self.model = model or ChatOpenAI(
            model=os.getenv("TMS_LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("TMS_LLM_BASE_URL") or None,
            api_key=os.getenv("TMS_LLM_API_KEY", ""),
            temperature=temperature,
        )

    def _ask_text(self, system: str, user: str, temperature: float = 0.5) -> str:
        model = self.model.bind(temperature=temperature) if hasattr(self.model, "bind") else self.model
        return model.invoke([SystemMessage(system), HumanMessage(user)]).content.strip()

    def route(self, ctx: RouteContext) -> RouteDecision:
        try:
            return (
                self.model.with_structured_output(RouteDecision)
                .invoke([
                    SystemMessage(prompts.ROUTER_SYSTEM),
                    HumanMessage(prompts.build_router_user(ctx)),
                ])
            )
        except Exception as e:
            print(f"[TimeMemory] 路由 LLM 失败，默认继续深挖：{e}", file=sys.stderr)
            return RouteDecision(action="followup", reasoning="LLM 调用失败，默认继续深挖")

    def ask_opening(self, topic: Topic) -> str:
        try:
            return self._ask_text(
                prompts.ASK_SYSTEM,
                prompts.build_opening_user(topic.name, topic.description, topic.opening_questions[0]),
                temperature=0.6,
            )
        except Exception as e:
            print(f"[TimeMemory] 提问 LLM 失败，用默认开场：{e}", file=sys.stderr)
            return topic.opening_questions[0]

    def ask_followup(self, topic: Topic, focus: str, last_answer: str) -> str:
        try:
            return self._ask_text(
                prompts.ASK_SYSTEM,
                prompts.build_followup_user(topic.name, focus, last_answer),
                temperature=0.6,
            )
        except Exception as e:
            print(f"[TimeMemory] 提问 LLM 失败，用兜底追问：{e}", file=sys.stderr)
            return f"您刚才提到的{focus}，能再跟我讲讲吗？" if focus else "能再跟我讲讲吗？"

    def ask_transition(self, prev: Topic, highlight: str, new: Topic) -> str:
        try:
            return self._ask_text(
                prompts.ASK_SYSTEM,
                prompts.build_transition_user(prev.name, highlight, new.name, new.description,
                                              new.opening_questions[0]),
                temperature=0.6,
            )
        except Exception as e:
            print(f"[TimeMemory] 提问 LLM 失败，用默认过渡：{e}", file=sys.stderr)
            return f"刚才{prev.name}这段讲得真好。接下来我想问问——{new.opening_questions[0]}"

    def extract_fragments(self, answer, topic_name, previous, feedback, committed) -> list[MemoryFragment]:
        try:
            batch = (
                self.model.with_structured_output(FragmentBatch)
                .invoke([
                    SystemMessage(prompts.EXTRACT_SYSTEM),
                    HumanMessage(prompts.build_extract_user(answer, topic_name, previous, feedback)),
                ])
            )
            return batch.fragments
        except Exception as e:
            print(f"[TimeMemory] 抽取 LLM 失败，本轮无片段：{e}", file=sys.stderr)
            return []

    def closing(self, ctx: ClosingContext) -> str:
        try:
            return self._ask_text(prompts.ASK_SYSTEM, prompts.build_closing_user(ctx), temperature=0.6)
        except Exception as e:
            print(f"[TimeMemory] 收尾 LLM 失败，用默认收尾：{e}", file=sys.stderr)
            return f"今天跟您聊得真开心，谢谢{ctx.elder_name}！您好好休息，咱们下次再接着聊！"

    def plausibility(self, fragment: MemoryFragment) -> bool:
        try:
            verdict = (
                self.model.with_structured_output(PlausibilityVerdict)
                .invoke([
                    SystemMessage(prompts.PLAUSIBILITY_SYSTEM),
                    HumanMessage(prompts.build_plausibility_user(fragment)),
                ])
            )
            return verdict.valid
        except Exception as e:
            print(f"[TimeMemory] 质检 LLM 失败，默认放行：{e}", file=sys.stderr)
            return True


# ----------------------------------------------------------------------------
# 离线 Demo 实现：确定性、小体量，仅用于测试与无 Key 演示
# ----------------------------------------------------------------------------
_TIME_RE = re.compile(r"(\d{4}年|\d{1,2}岁|[一二三四五六七八九十\d两几]{1,4}岁|"
                      r"小时候|那时候|当年|后来|以前|夏天|冬天|春节|正月|过年时)")
_PLACE_RE = re.compile(r"[\u4e00-\u9fa5]{1,6}(?:省|市|县|镇|乡|村|江|河|湖|海|山|街|路|桥|庙|小学|中学|学校)")
_PLACE_CUT = "我在到去从回跟和与把被叫让给比着了过是有你他她它们这那就便住往"
_PLACE_STOP = {"现在", "在那", "在这", "全村", "全国", "全市", "全省", "全县"}
_PERSON_WORDS = ["爸爸", "父亲", "妈妈", "母亲", "爷爷", "奶奶", "外公", "外婆",
                 "哥哥", "弟弟", "姐姐", "妹妹", "老伴", "儿子", "女儿",
                 "老师", "先生", "同学", "邻居", "乡亲", "师傅", "同事",
                 "村长", "书记", "哥", "姐", "娘", "爹"]
_RELATION_RE = re.compile(r"[王李张刘陈杨黄赵吴周徐孙马朱胡郭何高罗郑梁谢宋唐许韩冯董萧程曹袁邓傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文欧][一二三四五大小老]?[哥姐弟妹叔伯姨姑舅]")
_EMO_WORDS = ["开心", "高兴", "怀念", "想念", "忘不了", "难过", "伤心", "哭",
              "害怕", "怕", "吓", "骄傲", "自豪", "生气", "苦", "穷"]
_VAGUE_WORDS = ["不知道", "不清楚", "不记得", "记不清", "记不得", "想不起来",
                "没什么", "没啥", "就那样", "一般", "还行"]
_FORGOT_RE = re.compile(r"忘(?:记)?了($|[，。！？；、\s])")
_MARKERS = ["后来", "然后", "结果", "没想到", "有一次", "最难忘", "那次", "多亏"]
_NOISE = {"嗯", "哦", "啊", "好的", "好", "是", "对", "行", "知道了"}


def _demo_clean_place(m: str) -> str:
    cut = max((i for i, ch in enumerate(m) if ch in _PLACE_CUT), default=-1)
    c = m[cut + 1:]
    return "" if len(c) < 2 or c in _PLACE_STOP else c


def _demo_scan(text: str) -> tuple[list[str], list[str], list[str]]:
    time_refs = sorted(set(_TIME_RE.findall(text)))
    scan = text
    for idiom in ("人山人海", "山珍海味", "天涯海角", "五湖四海"):
        scan = scan.replace(idiom, "")
    place_refs = sorted({_demo_clean_place(m) for m in _PLACE_RE.findall(scan)} - {""})
    persons = {w for w in _PERSON_WORDS if w in text} | set(_RELATION_RE.findall(text))
    person_refs = sorted(p for p in persons if not any(p != o and p in o for o in persons))
    return time_refs, place_refs, person_refs


def _demo_vague(text: str) -> bool:
    if any(w in text for w in _VAGUE_WORDS):
        return True
    return bool(_FORGOT_RE.search(text))


class DemoInterviewLLM:
    """离线确定性实现：让测试/演示在无 Key 时跑通全链路。

    注意：它是"真模型的替身演员"，不是产品逻辑——生产环境请用 LangChainInterviewLLM。
    """

    name = "demo"

    # -- 路由 ------------------------------------------------------------------
    def route(self, ctx: RouteContext) -> RouteDecision:
        from .validators import is_farewell  # 延迟导入，避免循环

        answer = ctx.last_answer.strip()
        if not answer:  # 开场（无回答）：从首话题开始深挖
            return RouteDecision(action="followup", reasoning="访谈开场", focus="")
        if is_farewell(answer):
            return RouteDecision(action="wrap", reasoning="检测到告别")
        frags = self._rule_extract(answer)
        if frags:
            focus = self._pick_focus(frags)
            return RouteDecision(action="followup", reasoning="回答中有实质内容，继续深挖", focus=focus)
        if _demo_vague(answer):
            if ctx.uncovered:
                return RouteDecision(action="switch", reasoning="回答模糊，切换话题",
                                     next_topic_id=ctx.uncovered[0].id)
            return RouteDecision(action="wrap", reasoning="话题已覆盖完，收尾")
        return RouteDecision(action="followup", reasoning="再给一次深挖机会", focus="")

    @staticmethod
    def _pick_focus(frags: list[MemoryFragment]) -> str:
        cands: list[str] = []
        for f in frags:
            cands += f.place_refs + f.person_refs + f.time_refs
        cands = sorted(set(cands), key=lambda e: (-len(e), e))
        return cands[0] if cands else ""

    # -- 提问 ------------------------------------------------------------------
    def ask_opening(self, topic: Topic) -> str:
        return topic.opening_questions[0]

    def ask_followup(self, topic: Topic, focus: str, last_answer: str) -> str:
        if not focus:
            return "能给我讲一个具体的例子吗？越细越好。"
        return f"您刚才提到{focus}——这背后有没有哪段经历，是您最忘不了的？"

    def ask_transition(self, prev: Topic, highlight: str, new: Topic) -> str:
        h = f"刚才您讲的{highlight}真有意思，我都记下来了。" if highlight else f"刚才{prev.name}这段讲得真好。"
        return f"{h}接下来我想问问——{new.opening_questions[0]}"

    # -- 抽取 ------------------------------------------------------------------
    def extract_fragments(self, answer, topic_name, previous, feedback, committed) -> list[MemoryFragment]:
        if feedback and previous:
            # 修复轮：Demo 模型不会"改写"，做诚实的自我过滤（复用验证器规则）
            from .validators import validate_fragments
            return validate_fragments(previous, committed or []).valid
        return self._rule_extract(answer)

    def _rule_extract(self, answer: str) -> list[MemoryFragment]:
        text = (answer or "").strip()
        if not text or text in _NOISE:
            return []
        sentences = [s for s in re.split(r"[。！？；…]+", text) if s.strip()]
        if len("".join(sentences)) <= 40 or len(sentences) <= 1:
            sentences = [text]
        out: list[MemoryFragment] = []
        for sent in sentences:
            if len(sent) < 2 or sent in _NOISE:
                continue
            time_refs, place_refs, person_refs = _demo_scan(sent)
            emo = next((w for w in _EMO_WORDS if w in sent), "")
            has_plot = any(m in sent for m in _MARKERS)
            # 空回答过滤：模糊 + 无地点/人物/情节/情感 + 短句
            if _demo_vague(sent) and not (place_refs or person_refs or has_plot or emo) and len(sent) < 20:
                continue
            importance = 2 + bool(place_refs or person_refs) + bool(time_refs) + bool(has_plot or len(sent) >= 25)
            out.append(MemoryFragment(
                content=sent, time_refs=time_refs, place_refs=place_refs,
                person_refs=person_refs, emotion=emo, importance=min(5, importance),
                needs_followup=bool((place_refs or person_refs or has_plot) and len(sent) >= 8),
            ))
        return out

    # -- 收尾与质检 --------------------------------------------------------------
    def closing(self, ctx: ClosingContext) -> str:
        parts = [f"今天跟您聊得真开心，谢谢{ctx.elder_name}！"]
        if ctx.highlights:
            parts.append(f"您讲的「{'」和「'.join(ctx.highlights[:2])}」，我都好好记下来了。")
        if ctx.n_topics:
            parts.append(f"咱们今天一共聊了 {ctx.n_topics} 段人生故事，记下了 {ctx.n_fragments} 条回忆。")
        parts.append("您好好休息，咱们下次再接着聊！")
        return "".join(parts)

    def plausibility(self, fragment: MemoryFragment) -> bool:
        return True  # Demo 信任规则验证器


def get_llm() -> InterviewLLM:
    """有 Key 用真模型，无 Key 降级 Demo。"""
    if os.getenv("TMS_LLM_API_KEY"):
        model = os.getenv("TMS_LLM_MODEL", "gpt-4o-mini")
        base = os.getenv("TMS_LLM_BASE_URL", "https://api.openai.com/v1")
        print(f"[TimeMemory] 使用真模型：{model}（{base}）")
        return LangChainInterviewLLM()
    print("[TimeMemory] 未检测到 TMS_LLM_API_KEY，使用 DemoInterviewLLM（离线确定性）")
    return DemoInterviewLLM()
