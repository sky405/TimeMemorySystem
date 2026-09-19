"""Phase 6 问答图：rewrite → retrieve → gate → answer/noanswer。

依赖（store/embedder/llm）经闭包注入；state 纯 JSON。
ChatSession 持有多轮历史、待考清单与检索配置。
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..material.embeddings import Embedder, get_embedder
from ..material.query import retrieve
from ..material.store import MaterialStore
from .llm import LibraryLLM, get_library_llm
from .models import Answer

CHAPTER_SNIPPET = 800  # 章节进提示词前截断
MAX_TRIPLES = 8


class ChatState(TypedDict, total=False):
    question: str
    history: list[dict]
    standalone: str
    passages: list[dict]
    triples: list[str]
    max_score: float
    answer: dict
    threshold: float
    top_k: int
    cautions: list[dict]


def _caution_hit(content: str, cautions: list[dict]) -> bool:
    return any(c.get("quote") and c["quote"] in content for c in cautions)


def build_library_graph(store: MaterialStore, embedder: Embedder, llm: LibraryLLM):
    def rewrite_node(state: ChatState) -> dict:
        return {"standalone": llm.rewrite(state.get("history", []), state["question"])}

    def retrieve_node(state: ChatState) -> dict:
        hits = retrieve(state["standalone"], store, embedder,
                        top_k=state.get("top_k", 4), expand_kg=True)
        cautions = state.get("cautions", [])
        passages: list[dict] = []
        for h in hits:
            f = h.fragment
            kind = "chapter" if ":book:" in f["id"] else "fragment"
            content = f["content"]
            if kind == "chapter" and len(content) > CHAPTER_SNIPPET:
                content = content[:CHAPTER_SNIPPET] + "…"
            passages.append({"id": f["id"], "kind": kind, "content": content,
                             "topic": f.get("topic_id", ""), "score": h.score,
                             "via": h.via,
                             "caution": _caution_hit(f["content"], cautions)})
        triples: list[str] = []
        for h in hits:
            for e in store.edges_for_fragment(h.fragment["id"]):
                s = (store.get_node(e["src_id"]) or {}).get("name", "？")
                d = (store.get_node(e["dst_id"]) or {}).get("name", "？")
                t = f"{s} -{e['relation']}→ {d}"
                if t not in triples:
                    triples.append(t)
                if len(triples) >= MAX_TRIPLES:
                    break
            if len(triples) >= MAX_TRIPLES:
                break
        return {"passages": passages, "triples": triples,
                "max_score": passages[0]["score"] if passages else 0.0}

    def route_gate(state: ChatState) -> str:
        return ("answer" if state.get("max_score", 0.0) >= state.get("threshold", 0.15)
                else "noanswer")

    def answer_node(state: ChatState) -> dict:
        ans = llm.answer(state["standalone"], state["passages"], state["triples"])
        return {"answer": ans.model_dump()}

    def noanswer_node(state: ChatState) -> dict:
        return {"answer": Answer(text="抱歉，记忆库里没有找到相关记载。",
                                 citations=[], has_answer=False).model_dump()}

    g = StateGraph(ChatState)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.add_node("noanswer", noanswer_node)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("retrieve", route_gate,
                            {"answer": "answer", "noanswer": "noanswer"})
    g.add_edge("answer", END)
    g.add_edge("noanswer", END)
    return g.compile()


class ChatSession:
    """多轮问答会话。"""

    def __init__(self, store: MaterialStore, embedder: Embedder | None = None,
                 llm: LibraryLLM | None = None, cautions: list[dict] | None = None,
                 threshold: float | None = None, top_k: int | None = None):
        from ..config import get_config
        tuning = get_config().library
        self.store = store
        self.embedder = embedder or get_embedder()
        self.llm = llm or get_library_llm()
        self.cautions = [dict(c) for c in (cautions or [])]
        self.threshold = tuning.threshold if threshold is None else threshold
        self.top_k = tuning.top_k if top_k is None else top_k
        self.history: list[dict] = []
        self._graph = build_library_graph(store, self.embedder, self.llm)

    def ask(self, question: str) -> dict[str, Any]:
        """问一轮：返回 {"answer", "passages", "triples", "standalone"}。"""
        out = self._graph.invoke({"question": question, "history": list(self.history),
                                  "threshold": self.threshold, "top_k": self.top_k,
                                  "cautions": self.cautions})
        ans = Answer(**out["answer"])
        self.history.append({"q": question, "a": ans.text})
        return {"answer": ans, "passages": out["passages"],
                "triples": out["triples"], "standalone": out["standalone"]}
