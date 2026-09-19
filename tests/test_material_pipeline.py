"""Phase 2 全流程测试：清洗 → 向量化 → 图谱 → 混合检索（Demo 后端）。"""
import unittest

from timememory.material.embeddings import DemoEmbedder, cosine
from timememory.material.llm import DemoMaterialLLM
from timememory.material.pipeline import run_material_pipeline
from timememory.material.query import retrieve
from timememory.material.store import MaterialStore, connect_sqlite

RAW = [
    {"id": "frag-0001", "content": "我在四川嘉陵江边长大的，经常去游泳。",
     "topic_id": "childhood", "source_turn": 1, "time_refs": [], "place_refs": ["四川嘉陵江"],
     "person_refs": [], "emotion": "", "importance": 4},
    {"id": "frag-0002", "content": "有一次在嘉陵江里差点被水冲走，多亏邻居王二哥把我捞起来。",
     "topic_id": "childhood", "source_turn": 1, "time_refs": [], "place_refs": ["嘉陵江"],
     "person_refs": ["邻居", "王二哥"], "emotion": "害怕", "importance": 5},
    {"id": "frag-0003", "content": "我在四川嘉陵江边长大，经常去游泳。",
     "topic_id": "childhood", "source_turn": 2, "time_refs": [], "place_refs": ["四川嘉陵江"],
     "person_refs": [], "emotion": "", "importance": 3},
    {"id": "frag-0004", "content": "嗯。",
     "topic_id": "childhood", "source_turn": 2, "time_refs": [], "place_refs": [],
     "person_refs": [], "emotion": "", "importance": 1},
    {"id": "frag-0005", "content": "1962年，我9岁，跟着父亲从合川县到了重庆市。",
     "topic_id": "childhood", "source_turn": 3, "time_refs": ["1962年", "9岁"],
     "place_refs": ["合川县", "重庆市"], "person_refs": ["父亲"], "emotion": "", "importance": 5},
    {"id": "frag-0006", "content": "后来王二哥跳下水，把我拖上了嘉陵江的岸。",
     "topic_id": "childhood", "source_turn": 4, "time_refs": ["后来"],
     "place_refs": ["嘉陵江"], "person_refs": ["王二哥"], "emotion": "", "importance": 4},
]


def run_demo():
    store = MaterialStore(connect_sqlite(":memory:"))
    result = run_material_pipeline("iv-test", RAW, DemoMaterialLLM(), DemoEmbedder(), store)
    return result, store


class TestMaterialPipeline(unittest.TestCase):
    def test_end_to_end_stats(self):
        result, store = run_demo()
        s = result["stats"]
        self.assertEqual(s["raw"], 6)
        self.assertEqual(s["clean"], 4)  # 近重复合并一对 + 丢弃"嗯"
        self.assertEqual(s["embedded"], 4)
        # 节点：王二哥/邻居/父亲 + 四川嘉陵江/嘉陵江/合川县/重庆市 + 1962年/9岁（"后来"不成节点）
        self.assertEqual(s["kg_nodes"], 9)
        # 边：frag2 两条 + frag6 一条 + frag5 两条
        self.assertEqual(s["edges"], 5)
        self.assertTrue(all(f["id"].startswith("iv-test:") for f in result["clean"]))

    def test_missing_ids_autonumbered(self):
        # Phase 1 新版片段无 id：清洗时按序补号，入库不冲突
        raw = [
            {"content": "我在四川嘉陵江边长大的，经常去游泳。",
             "topic_id": "childhood", "source_turn": 1, "place_refs": ["四川嘉陵江"]},
            {"content": "村长王老三给大家讲古。",
             "topic_id": "hometown", "source_turn": 2, "person_refs": ["村长"]},
        ]
        result = run_material_pipeline("iv-noid", raw, DemoMaterialLLM(), DemoEmbedder(),
                                       MaterialStore(connect_sqlite(":memory:")))
        ids = [f["id"] for f in result["clean"]]
        self.assertEqual(ids, ["iv-noid:frag-0000", "iv-noid:frag-0001"])
        self.assertEqual(result["stats"]["fragments"], 2)

    def test_merge_keeps_longer(self):
        result, _ = run_demo()
        merged = [f for f in result["clean"] if "经常去游泳" in f["content"] or "经常去游泳" in f["content"]]
        self.assertEqual(len(merged), 1)
        self.assertIn("长大的", merged[0]["content"])  # 更长的原文被保留

    def test_kg_nodes_edges(self):
        _, store = run_demo()
        node = store.find_node("person", "王二哥")
        self.assertIsNotNone(node)
        names = {(e["relation"], e["evidence_fragment_id"]) for e, _ in store.neighbors(node["id"])}
        self.assertTrue(any(fid.endswith("frag-0006") for _, fid in names))

    def test_retrieve_vector(self):
        _, store = run_demo()
        res = retrieve("嘉陵江游泳", store, DemoEmbedder(), top_k=2, expand_kg=False)
        self.assertIn("嘉陵江", res[0].fragment["content"])
        self.assertEqual(res[0].via, "vector")

    def test_retrieve_kg_expansion(self):
        _, store = run_demo()
        res = retrieve("邻居捞起", store, DemoEmbedder(), top_k=1, expand_kg=True)
        self.assertIn("捞", res[0].fragment["content"])
        self.assertEqual(res[0].via, "vector")
        # 王二哥节点把 frag-0006 经图谱带回来（向量 top1 之外）
        self.assertTrue(any(r.via.startswith("kg:") and "拖" in r.fragment["content"] for r in res),
                        [f"{r.via}:{r.fragment['content'][:12]}" for r in res])


class TestDemoEmbedder(unittest.TestCase):
    def test_deterministic(self):
        emb = DemoEmbedder()
        a = emb.embed_texts(["嘉陵江游泳"])[0]
        b = emb.embed_texts(["嘉陵江游泳"])[0]
        c = emb.embed_texts(["完全不同的内容"])[
            0]
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=5)
        self.assertLess(cosine(a, c), 0.9)


if __name__ == "__main__":
    unittest.main()
