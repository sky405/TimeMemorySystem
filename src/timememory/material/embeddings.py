"""向量化：文本 → 向量（纯 Python cosine，无 numpy 依赖）。

- OpenAIEmbedder：OpenAI 兼容 Embedding 接口。
    TMS_EMB_API_KEY（必填才启用）/ TMS_EMB_BASE_URL（默认复用 TMS_LLM_BASE_URL）
    / TMS_EMB_MODEL（默认 text-embedding-3-small）
- DemoEmbedder：离线确定性（字符 bigram 哈希，64 维），用于测试与演示。
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model=None):
        if model is None:
            from langchain_openai import OpenAIEmbeddings
            model = OpenAIEmbeddings(
                model=os.getenv("TMS_EMB_MODEL", "text-embedding-3-small"),
                base_url=os.getenv("TMS_EMB_BASE_URL") or os.getenv("TMS_LLM_BASE_URL") or None,
                api_key=os.getenv("TMS_EMB_API_KEY", ""),
            )
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(map(float, v)) for v in self.model.embed_documents(texts)]


class DemoEmbedder:
    """离线确定性向量：字符 bigram 哈希到固定维度并归一化。

    同文本必同向量；语义能力弱，仅用于测试与无 Key 演示。
    """

    name = "demo"

    def __init__(self, dim: int = 256):  # 64 维碰撞噪声太大（无关注入可达 0.37），256 维才有区分度
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        grams = [text[i:i + 2] for i in range(max(0, len(text) - 1))] or [text]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def get_embedder() -> Embedder:
    if os.getenv("TMS_EMB_API_KEY"):
        print(f"[TimeMemory] 向量化：真模型 {os.getenv('TMS_EMB_MODEL', 'text-embedding-3-small')}")
        return OpenAIEmbedder()
    print("[TimeMemory] 向量化：Demo 实现（离线确定性；设 TMS_EMB_API_KEY 可切真模型）")
    return DemoEmbedder()
