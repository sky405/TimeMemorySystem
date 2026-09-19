"""MySQL 后端冒烟测试：仅在配置 TMS_MYSQL_URL/HOST 时运行。

本地起 MySQL：docker run -d --name tm-mysql -e MYSQL_ROOT_PASSWORD=pass -p 3306:3306 mysql:8
    TMS_MYSQL_URL=mysql://root:pass@127.0.0.1:3306/timememory_test python3 -m unittest tests.test_material_mysql
"""
import os
import unittest

NEED_MYSQL = bool(os.getenv("TMS_MYSQL_URL") or os.getenv("TMS_MYSQL_HOST"))


@unittest.skipUnless(NEED_MYSQL, "需 MySQL（设 TMS_MYSQL_URL 或 TMS_MYSQL_HOST）")
class TestMySQLBackend(unittest.TestCase):
    def test_smoke(self):
        from timememory.material.models import CleanFragment, KGEdge, KGNode
        from timememory.material.store import MaterialStore, get_db

        store = MaterialStore(get_db())
        self.assertEqual(store.db.backend, "mysql")
        store.save_fragments([CleanFragment(id="smoke:1", session_id="smoke", content="冒烟测试片段。")])
        self.assertEqual(store.get_fragment("smoke:1")["content"], "冒烟测试片段。")
        store.save_embedding("smoke:1", [0.5, 0.5])
        self.assertIn("smoke:1", dict(store.all_embeddings()))
        store.upsert_node(KGNode(id="n-smoke", type="person", name="冒烟老人"))
        self.assertIsNotNone(store.find_node("person", "冒烟老人"))
        store.add_edge(KGEdge(src_id="n-smoke", dst_id="n-smoke", relation="自检"))
        self.assertGreaterEqual(len(store.all_edges()), 1)


if __name__ == "__main__":
    unittest.main()
