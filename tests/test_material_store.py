"""Phase 2 存储测试（SQLite :memory:；同一套 schema 也用于 MySQL）。"""
import unittest

from timememory.material.models import CleanFragment, KGEdge, KGNode
from timememory.material.store import MaterialStore, connect_sqlite


def frag(i: int, content: str, **kw) -> CleanFragment:
    d = {"id": f"iv-t:frag-{i:04d}", "session_id": "iv-t", "content": content,
         "topic_id": "childhood", "source_turn": i}
    d.update(kw)
    return CleanFragment(**d)


class TestMaterialStore(unittest.TestCase):
    def setUp(self):
        self.store = MaterialStore(connect_sqlite(":memory:"))

    def test_fragment_save_get(self):
        self.store.save_fragments([
            frag(1, "我在四川嘉陵江边长大的。", place_refs=["四川嘉陵江"], importance=4),
            frag(2, "王二哥救了我。", person_refs=["王二哥"]),
        ])
        got = self.store.get_fragment("iv-t:frag-0001")
        self.assertEqual(got["content"], "我在四川嘉陵江边长大的。")
        self.assertEqual(got["place_refs"], ["四川嘉陵江"])
        self.assertEqual(len(self.store.all_fragments()), 2)
        self.assertEqual(len(self.store.all_fragments("iv-t")), 2)

    def test_fragment_upsert_idempotent(self):
        self.store.save_fragments([frag(1, "第一版。")])
        self.store.save_fragments([frag(1, "第二版。")])
        self.assertEqual(self.store.get_fragment("iv-t:frag-0001")["content"], "第二版。")
        self.assertEqual(self.store.stats()["fragments"], 1)

    def test_embedding_roundtrip(self):
        vec = [0.1 * i for i in range(8)]
        self.store.save_embedding("iv-t:frag-0001", vec)
        got = dict(self.store.all_embeddings())
        self.assertIn("iv-t:frag-0001", got)
        for a, b in zip(got["iv-t:frag-0001"], vec):
            self.assertAlmostEqual(a, b, places=5)

    def test_node_upsert_merge(self):
        self.store.upsert_node(KGNode(id="n1", type="person", name="王二哥",
                                      aliases=["二哥"], description="邻居"))
        self.store.upsert_node(KGNode(id="n1", type="person", name="王二哥",
                                      aliases=["王二"], description="邻居王二哥，救过爷爷"))
        node = self.store.find_node("person", "王二哥")
        self.assertEqual(sorted(node["aliases"]), ["二哥", "王二"])
        self.assertIn("救过爷爷", node["description"])
        self.assertEqual(self.store.stats()["kg_nodes"], 1)

    def test_edge_add_ignore_dup(self):
        e = KGEdge(src_id="a", dst_id="b", relation="救起", evidence_fragment_id="f1")
        self.assertTrue(self.store.add_edge(e))
        self.assertFalse(self.store.add_edge(e))
        self.assertEqual(len(self.store.all_edges()), 1)

    def test_neighbors_and_evidence(self):
        for n in [KGNode(id="a", type="person", name="王二哥"),
                  KGNode(id="b", type="place", name="嘉陵江"),
                  KGNode(id="c", type="person", name="我娘")]:
            self.store.upsert_node(n)
        self.store.add_edge(KGEdge(src_id="a", dst_id="b", relation="出现于", evidence_fragment_id="f1"))
        self.store.add_edge(KGEdge(src_id="c", dst_id="a", relation="感谢", evidence_fragment_id="f2"))
        nbs = self.store.neighbors("a")
        self.assertEqual(len(nbs), 2)
        self.assertEqual({n["name"] for _, n in nbs}, {"嘉陵江", "我娘"})
        self.assertEqual(len(self.store.edges_for_fragment("f1")), 1)


if __name__ == "__main__":
    unittest.main()
