"""Phase 2 LLM 抽象：清洗 / 图谱抽取两个判断入口。

- MaterialLLM 协议；LangChainMaterialLLM 真模型；DemoMaterialLLM 离线实现。
- get_material_llm()：有 TMS_LLM_API_KEY 用真模型，否则降级 Demo。
"""
from __future__ import annotations

import os
import sys
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import prompts
from .models import CleanFragment, KGEdgeDraft, KGExtraction, KGNodeDraft


class MaterialLLM(Protocol):
    name: str

    def clean_text(self, content: str) -> str: ...
    def extract_kg(self, fragment: CleanFragment) -> KGExtraction: ...


def build_chat_model(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("TMS_LLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("TMS_LLM_BASE_URL") or None,
        api_key=os.getenv("TMS_LLM_API_KEY", ""),
        temperature=temperature,
    )


class LangChainMaterialLLM:
    name = "langchain"

    def __init__(self, model=None):
        self.model = model or build_chat_model()

    def clean_text(self, content: str) -> str:
        try:
            out = self.model.invoke([SystemMessage(prompts.CLEAN_SYSTEM),
                                     HumanMessage(prompts.build_clean_user(content))]).content.strip()
            return out or content
        except Exception as e:
            print(f"[TimeMemory] 清洗 LLM 失败，保留原文：{e}", file=sys.stderr)
            return content

    def extract_kg(self, fragment: CleanFragment) -> KGExtraction:
        try:
            return self.model.with_structured_output(KGExtraction).invoke([
                SystemMessage(prompts.KG_SYSTEM),
                HumanMessage(prompts.build_kg_user(fragment.content, fragment.time_refs,
                                                   fragment.place_refs, fragment.person_refs)),
            ])
        except Exception as e:
            print(f"[TimeMemory] 图谱抽取 LLM 失败，本条无三元组：{e}", file=sys.stderr)
            return KGExtraction()


class DemoMaterialLLM:
    """离线确定性实现：清洗只做去空白；图谱从 Phase 1 已标实体直转。

    注意：替身演员——人物→地点关系只能标"出现于"，生产请用真模型。
    """

    name = "demo"

    def clean_text(self, content: str) -> str:
        return " ".join(content.split())

    def extract_kg(self, fragment: CleanFragment) -> KGExtraction:
        times = [t for t in fragment.time_refs if t not in ("后来", "然后", "结果")]  # 副词不成节点
        nodes = ([KGNodeDraft(type="person", name=p) for p in fragment.person_refs]
                 + [KGNodeDraft(type="place", name=p) for p in fragment.place_refs]
                 + [KGNodeDraft(type="time", name=t) for t in times])
        edges = [KGEdgeDraft(src_name=p, src_type="person", dst_name=q, dst_type="place",
                             relation="出现于", confidence=0.6)
                 for p in fragment.person_refs for q in fragment.place_refs]
        return KGExtraction(nodes=nodes, edges=edges)


def get_material_llm() -> MaterialLLM:
    if os.getenv("TMS_LLM_API_KEY"):
        print(f"[TimeMemory] Phase 2 LLM：真模型 {os.getenv('TMS_LLM_MODEL', 'gpt-4o-mini')}")
        return LangChainMaterialLLM()
    print("[TimeMemory] Phase 2 LLM：Demo 实现（离线确定性）")
    return DemoMaterialLLM()
