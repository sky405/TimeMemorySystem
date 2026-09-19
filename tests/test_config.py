"""统一配置测试：默认值 / YAML / 环境变量优先级（显式 > 环境 > YAML > 默认）。"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from timememory.config import configure, get_config, load_config

ENV_KEYS = ["TMS_CONFIG", "TMS_LLM_BASE_URL", "TMS_LLM_API_KEY", "TMS_LLM_MODEL",
            "TMS_EMB_BASE_URL", "TMS_EMB_API_KEY", "TMS_EMB_MODEL",
            "TMS_MYSQL_URL", "TMS_MYSQL_HOST", "TMS_MYSQL_PORT", "TMS_MYSQL_USER",
            "TMS_MYSQL_PASSWORD", "TMS_MYSQL_DB", "TMS_SQLITE_PATH"]


class TestConfig(unittest.TestCase):
    def setUp(self):
        self._paths: list[str] = []
        self._saved = {}
        for k in ENV_KEYS:
            if k in os.environ:
                self._saved[k] = os.environ.pop(k)

    def tearDown(self):
        os.environ.update(self._saved)
        configure(None)
        for p in self._paths:
            Path(p).unlink(missing_ok=True)

    def _yaml(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.write(fd, content.encode())
        os.close(fd)
        self._paths.append(path)
        return path

    def test_defaults(self):
        cfg = load_config(self._yaml("{}\n"))
        self.assertIsNone(cfg.llm.api_key)
        self.assertEqual(cfg.embedding.model, "text-embedding-3-small")
        self.assertEqual(cfg.sqlite.path, "data/material.db")
        self.assertEqual(cfg.mysql.port, 3306)
        self.assertEqual(cfg.library.threshold, 0.15)
        self.assertEqual(cfg.library.top_k, 4)
        self.assertEqual(cfg.orchestration.max_rounds, 3)

    def test_yaml_values(self):
        cfg = load_config(self._yaml(
            "llm:\n  model: qwen-max\n  base_url: https://x\nlibrary:\n  threshold: 0.5\n"))
        self.assertEqual(cfg.llm.model, "qwen-max")
        self.assertEqual(cfg.library.threshold, 0.5)
        self.assertEqual(cfg.library.top_k, 4)  # 未写项保持默认

    def test_env_overrides_yaml(self):
        p = self._yaml("llm:\n  model: qwen-max\n")
        with patch.dict(os.environ, {"TMS_LLM_MODEL": "gpt-x"}):
            cfg = load_config(p)
        self.assertEqual(cfg.llm.model, "gpt-x")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/tm.yaml")

    def test_bad_yaml_raises(self):
        with self.assertRaises(ValueError):
            load_config(self._yaml(":\n- bad: [unclosed\n"))

    def test_configure_global(self):
        configure(self._yaml("sqlite:\n  path: /tmp/x.db\n"))
        self.assertEqual(get_config().sqlite.path, "/tmp/x.db")

    def test_llm_selection(self):
        from timememory.library.llm import (
            DemoLibraryLLM,
            LangChainLibraryLLM,
            get_library_llm,
        )
        self.assertIsInstance(get_library_llm(), DemoLibraryLLM)
        with patch.dict(os.environ, {"TMS_LLM_API_KEY": "sk-test"}):
            self.assertIsInstance(get_library_llm(), LangChainLibraryLLM)
        configure(self._yaml("llm:\n  api_key: sk-yaml\n"))
        self.assertIsInstance(get_library_llm(), LangChainLibraryLLM)

    def test_embedder_selection(self):
        from timememory.material.embeddings import (
            DemoEmbedder,
            OpenAIEmbedder,
            get_embedder,
        )
        self.assertIsInstance(get_embedder(), DemoEmbedder)
        with patch.dict(os.environ, {"TMS_EMB_API_KEY": "sk-test"}):
            self.assertIsInstance(get_embedder(), OpenAIEmbedder)

    def test_get_db_sqlite_from_config(self):
        from timememory.material.store import get_db
        configure(self._yaml("sqlite:\n  path: ':memory:'\n"))
        db = get_db()
        self.assertEqual(db.backend, "sqlite")
        db.close()

    def test_chat_defaults_from_config(self):
        from timememory.library import ChatSession
        from timememory.material import DemoEmbedder, MaterialStore, connect_sqlite
        configure(self._yaml("library:\n  threshold: 0.5\n  top_k: 9\n"))
        store = MaterialStore(connect_sqlite(":memory:"))
        s = ChatSession(store, DemoEmbedder())
        self.assertEqual((s.threshold, s.top_k), (0.5, 9))
        s2 = ChatSession(store, DemoEmbedder(), threshold=0.1)  # 显式优先
        self.assertEqual((s2.threshold, s2.top_k), (0.1, 9))

    def test_memoir_rounds_from_config(self):
        from timememory.assessment import DemoAssessmentLLM
        from timememory.drafting import DemoDraftingLLM
        from timememory.interview.llm import DemoInterviewLLM
        from timememory.interview.models import ElderProfile
        from timememory.material import DemoEmbedder, DemoMaterialLLM
        from timememory.material import MaterialStore, connect_sqlite
        from timememory.orchestration import run_memoir
        configure(self._yaml("orchestration:\n  max_rounds: 1\n"))
        out = run_memoir(
            ElderProfile(name="爷"), lambda q, r, t: "今天有点累了，咱们下次再聊吧。",
            store=MaterialStore(connect_sqlite(":memory:")),
            interview_llm=DemoInterviewLLM(), material_llm=DemoMaterialLLM(),
            embedder=DemoEmbedder(), assess_llm=DemoAssessmentLLM(),
            draft_llm=DemoDraftingLLM())
        self.assertEqual(len(out["rounds"]), 1)


if __name__ == "__main__":
    unittest.main()
