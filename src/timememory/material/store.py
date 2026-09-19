"""Phase 2 存储：MySQL（生产）/ SQLite（测试与本地），同一套 schema。

四张表：fragments（清洗片段）+ embeddings（向量 BLOB）+ kg_nodes / kg_edges（知识图谱）。

MySQL 配置（二选一）：
    TMS_MYSQL_URL=mysql://user:pass@host:port/db
    或 TMS_MYSQL_HOST / TMS_MYSQL_PORT / TMS_MYSQL_USER / TMS_MYSQL_PASSWORD / TMS_MYSQL_DB
都不设 → SQLite（TMS_SQLITE_PATH，默认 data/material.db；":memory:" 用于测试）。

说明：向量以 float32 BLOB 存 MySQL，家族数据量（数百片段）下 Python 内
暴力 cosine 足够用；数据量大了再迁专用向量库，检索接口不变。
"""
from __future__ import annotations

import json
import os
import sqlite3
import struct
from pathlib import Path
from urllib.parse import unquote, urlparse

from .models import CleanFragment, KGEdge, KGNode

# 注意：MySQL 不允许 TEXT 列设 DEFAULT，因此 TEXT 列一律 NOT NULL 无默认值，
# 写入时代码保证传值（与 SQLite 保持一致）。

MYSQL_DDL = """
CREATE TABLE IF NOT EXISTS fragments (
  id VARCHAR(64) PRIMARY KEY,
  session_id VARCHAR(64) NOT NULL,
  content TEXT NOT NULL,
  topic_id VARCHAR(64) NOT NULL DEFAULT '',
  source_turn INT NOT NULL DEFAULT 0,
  time_refs TEXT NOT NULL,
  place_refs TEXT NOT NULL,
  person_refs TEXT NOT NULL,
  emotion VARCHAR(32) NOT NULL DEFAULT '',
  importance INT NOT NULL DEFAULT 3,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_fragments_session (session_id),
  INDEX idx_fragments_topic (topic_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS embeddings (
  fragment_id VARCHAR(64) PRIMARY KEY,
  dim INT NOT NULL,
  vector BLOB NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_nodes (
  id VARCHAR(64) PRIMARY KEY,
  type VARCHAR(32) NOT NULL,
  name VARCHAR(255) NOT NULL,
  aliases TEXT NOT NULL,
  description TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_node (type, name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS kg_edges (
  id INTEGER PRIMARY KEY AUTO_INCREMENT,
  src_id VARCHAR(64) NOT NULL,
  dst_id VARCHAR(64) NOT NULL,
  relation VARCHAR(64) NOT NULL,
  evidence_fragment_id VARCHAR(64) NOT NULL DEFAULT '',
  confidence FLOAT NOT NULL DEFAULT 0.5,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_edge (src_id, dst_id, relation, evidence_fragment_id),
  INDEX idx_edge_src (src_id),
  INDEX idx_edge_dst (dst_id),
  INDEX idx_edge_evidence (evidence_fragment_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS fragments (
  id VARCHAR(64) PRIMARY KEY,
  session_id VARCHAR(64) NOT NULL,
  content TEXT NOT NULL,
  topic_id VARCHAR(64) NOT NULL DEFAULT '',
  source_turn INT NOT NULL DEFAULT 0,
  time_refs TEXT NOT NULL,
  place_refs TEXT NOT NULL,
  person_refs TEXT NOT NULL,
  emotion VARCHAR(32) NOT NULL DEFAULT '',
  importance INT NOT NULL DEFAULT 3,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fragments_session ON fragments(session_id);
CREATE INDEX IF NOT EXISTS idx_fragments_topic ON fragments(topic_id);
CREATE TABLE IF NOT EXISTS embeddings (
  fragment_id VARCHAR(64) PRIMARY KEY,
  dim INT NOT NULL,
  vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS kg_nodes (
  id VARCHAR(64) PRIMARY KEY,
  type VARCHAR(32) NOT NULL,
  name VARCHAR(255) NOT NULL,
  aliases TEXT NOT NULL,
  description TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(type, name)
);
CREATE TABLE IF NOT EXISTS kg_edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_id VARCHAR(64) NOT NULL,
  dst_id VARCHAR(64) NOT NULL,
  relation VARCHAR(64) NOT NULL,
  evidence_fragment_id VARCHAR(64) NOT NULL DEFAULT '',
  confidence FLOAT NOT NULL DEFAULT 0.5,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(src_id, dst_id, relation, evidence_fragment_id)
);
CREATE INDEX IF NOT EXISTS idx_edge_src ON kg_edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON kg_edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edge_evidence ON kg_edges(evidence_fragment_id);
"""


def pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    if len(blob) // 4 != dim:
        raise ValueError(f"向量维度不符：blob={len(blob)//4}，dim={dim}")
    return list(struct.unpack(f"<{dim}f", blob))


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _load(text: str, default):
    try:
        return json.loads(text) if text else default
    except (json.JSONDecodeError, TypeError):
        return default


class DB:
    """最小 DB 适配器：SQL 统一用 ? 占位，MySQL 自动转 %s。"""

    def __init__(self, conn, backend: str):
        self.conn = conn
        self.backend = backend  # "mysql" | "sqlite"

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.backend == "mysql" else sql

    def execute(self, sql: str, params: tuple = ()):
        cur = self.conn.cursor()
        cur.execute(self._sql(sql), params)
        return cur

    def executemany(self, sql: str, rows: list[tuple]):
        cur = self.conn.cursor()
        cur.executemany(self._sql(sql), rows)
        return cur

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(self._sql(sql), params)
        rows = cur.fetchall()
        if self.backend == "mysql":
            return [dict(r) for r in rows]
        return [dict(r) for r in rows]

    def script(self, ddl: str) -> None:
        for stmt in ddl.split(";"):
            if stmt.strip():
                self.execute(stmt)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DB":
        return self

    def __exit__(self, *exc) -> None:
        self.commit()
        self.close()


def connect_sqlite(path: str | Path) -> DB:
    p = str(path)
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    db = DB(conn, "sqlite")
    db.script(SQLITE_DDL)
    db.commit()
    return db


def connect_mysql(host: str, port: int, user: str, password: str, database: str) -> DB:
    import pymysql
    import pymysql.cursors

    try:
        conn = pymysql.connect(host=host, port=port, user=user, password=password,
                               database=database, charset="utf8mb4",
                               cursorclass=pymysql.cursors.DictCursor, autocommit=False)
    except pymysql.err.OperationalError as e:
        if e.args and e.args[0] != 1049:  # 1049 = Unknown database
            raise
        admin = pymysql.connect(host=host, port=port, user=user, password=password,
                                charset="utf8mb4", autocommit=True)
        with admin.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4")
        admin.close()
        conn = pymysql.connect(host=host, port=port, user=user, password=password,
                               database=database, charset="utf8mb4",
                               cursorclass=pymysql.cursors.DictCursor, autocommit=False)
    db = DB(conn, "mysql")
    db.script(MYSQL_DDL)
    db.commit()
    return db


def get_db() -> DB:
    """按统一配置连接：优先 MySQL，否则 SQLite。"""
    from ..config import get_config
    cfg = get_config()
    url = cfg.mysql.url or ""
    host = cfg.mysql.host or ""
    if url or host:
        if url:
            u = urlparse(url)
            host = u.hostname or "localhost"
            port = u.port or 3306
            user = unquote(u.username or "root")
            password = unquote(u.password or "")
            database = (u.path or "/timememory").lstrip("/")
        else:
            port = cfg.mysql.port
            user = cfg.mysql.user
            password = cfg.mysql.password
            database = cfg.mysql.database
        print(f"[TimeMemory] Phase 2 存储：MySQL {user}@{host}:{port}/{database}")
        return connect_mysql(host, port, user, password, database)
    path = cfg.sqlite.path
    print(f"[TimeMemory] Phase 2 存储：SQLite {path}（设 TMS_MYSQL_URL 可切 MySQL）")
    return connect_sqlite(path)


class MaterialStore:
    """Phase 2 仓储：清洗片段 + 向量 + 知识图谱的统一读写。"""

    def __init__(self, db: DB | None = None):
        self.db = db or get_db()

    # -- 片段 ------------------------------------------------------------------
    def save_fragments(self, frags: list[CleanFragment]) -> int:
        if not frags:
            return 0
        cols = ("id", "session_id", "content", "topic_id", "source_turn",
                "time_refs", "place_refs", "person_refs", "emotion", "importance")
        rows = [(f.id, f.session_id, f.content, f.topic_id, f.source_turn,
                 _dump(f.time_refs), _dump(f.place_refs), _dump(f.person_refs),
                 f.emotion, f.importance) for f in frags]
        if self.db.backend == "mysql":
            self.db.executemany(
                "INSERT INTO fragments (id, session_id, content, topic_id, source_turn,"
                " time_refs, place_refs, person_refs, emotion, importance)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON DUPLICATE KEY UPDATE content=VALUES(content), topic_id=VALUES(topic_id),"
                " source_turn=VALUES(source_turn), time_refs=VALUES(time_refs),"
                " place_refs=VALUES(place_refs), person_refs=VALUES(person_refs),"
                " emotion=VALUES(emotion), importance=VALUES(importance)", rows)
        else:
            self.db.executemany(
                "INSERT INTO fragments (id, session_id, content, topic_id, source_turn,"
                " time_refs, place_refs, person_refs, emotion, importance)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET content=excluded.content,"
                " topic_id=excluded.topic_id, source_turn=excluded.source_turn,"
                " time_refs=excluded.time_refs, place_refs=excluded.place_refs,"
                " person_refs=excluded.person_refs, emotion=excluded.emotion,"
                " importance=excluded.importance", rows)
        self.db.commit()
        return len(rows)

    def get_fragment(self, fragment_id: str) -> dict | None:
        rows = self.db.query("SELECT * FROM fragments WHERE id=?", (fragment_id,))
        return self._decode_fragment(rows[0]) if rows else None

    def all_fragments(self, session_id: str | None = None) -> list[dict]:
        if session_id:
            rows = self.db.query("SELECT * FROM fragments WHERE session_id=? ORDER BY source_turn",
                                 (session_id,))
        else:
            rows = self.db.query("SELECT * FROM fragments ORDER BY source_turn")
        return [self._decode_fragment(r) for r in rows]

    @staticmethod
    def _decode_fragment(row: dict) -> dict:
        row = dict(row)
        for k in ("time_refs", "place_refs", "person_refs"):
            row[k] = _load(row.get(k, "[]"), [])
        return row

    # -- 向量 ------------------------------------------------------------------
    def save_embedding(self, fragment_id: str, vector: list[float]) -> None:
        self.db.execute("DELETE FROM embeddings WHERE fragment_id=?", (fragment_id,))
        self.db.execute("INSERT INTO embeddings (fragment_id, dim, vector) VALUES (?,?,?)",
                        (fragment_id, len(vector), pack_vector(vector)))
        self.db.commit()

    def all_embeddings(self) -> list[tuple[str, list[float]]]:
        return [(r["fragment_id"], unpack_vector(bytes(r["vector"]), r["dim"]))
                for r in self.db.query("SELECT fragment_id, dim, vector FROM embeddings")]

    # -- 知识图谱：节点 -----------------------------------------------------------
    def upsert_node(self, node: KGNode) -> str:
        """按 (type, name) 去重：别名合并，介绍取更长的。"""
        old = self.find_node(node.type, node.name)
        if old:
            aliases = sorted(set(old["aliases"]) | set(node.aliases))
            desc = node.description if len(node.description) > len(old["description"] or "") else (old["description"] or "")
            self.db.execute("UPDATE kg_nodes SET aliases=?, description=? WHERE id=?",
                            (_dump(aliases), desc, old["id"]))
            self.db.commit()
            return old["id"]
        self.db.execute("INSERT INTO kg_nodes (id, type, name, aliases, description) VALUES (?,?,?,?,?)",
                        (node.id, node.type, node.name, _dump(node.aliases), node.description))
        self.db.commit()
        return node.id

    def find_node(self, node_type: str, name: str) -> dict | None:
        rows = self.db.query("SELECT * FROM kg_nodes WHERE type=? AND name=?", (node_type, name))
        return self._decode_node(rows[0]) if rows else None

    def get_node(self, node_id: str) -> dict | None:
        rows = self.db.query("SELECT * FROM kg_nodes WHERE id=?", (node_id,))
        return self._decode_node(rows[0]) if rows else None

    def all_nodes(self, node_type: str | None = None) -> list[dict]:
        rows = (self.db.query("SELECT * FROM kg_nodes WHERE type=?", (node_type,))
                if node_type else self.db.query("SELECT * FROM kg_nodes"))
        return [self._decode_node(r) for r in rows]

    @staticmethod
    def _decode_node(row: dict) -> dict:
        row = dict(row)
        row["aliases"] = _load(row.get("aliases", "[]"), [])
        return row

    # -- 知识图谱：边 -------------------------------------------------------------
    def add_edge(self, edge: KGEdge) -> bool:
        """(src, dst, relation, evidence) 冲突则忽略。同一三元组来自不同证据
        会保留多行（provenance），重复跑流水线则天然幂等。返回是否新插入。"""
        if self.db.backend == "mysql":
            cur = self.db.execute(
                "INSERT IGNORE INTO kg_edges (src_id, dst_id, relation, evidence_fragment_id, confidence)"
                " VALUES (?,?,?,?,?)",
                (edge.src_id, edge.dst_id, edge.relation, edge.evidence_fragment_id, edge.confidence))
        else:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO kg_edges (src_id, dst_id, relation, evidence_fragment_id, confidence)"
                " VALUES (?,?,?,?,?)",
                (edge.src_id, edge.dst_id, edge.relation, edge.evidence_fragment_id, edge.confidence))
        self.db.commit()
        return cur.rowcount > 0

    def neighbors(self, node_id: str) -> list[tuple[dict, dict]]:
        """返回 [(edge, neighbor_node)]，含出边与入边。"""
        out: list[tuple[dict, dict]] = []
        for row in self.db.query("SELECT * FROM kg_edges WHERE src_id=?", (node_id,)):
            node = self.get_node(row["dst_id"])
            if node:
                out.append((dict(row), node))
        for row in self.db.query("SELECT * FROM kg_edges WHERE dst_id=?", (node_id,)):
            node = self.get_node(row["src_id"])
            if node:
                out.append((dict(row), node))
        return out

    def edges_for_fragment(self, fragment_id: str) -> list[dict]:
        return self.db.query("SELECT * FROM kg_edges WHERE evidence_fragment_id=?", (fragment_id,))

    def all_edges(self) -> list[dict]:
        return self.db.query("SELECT * FROM kg_edges")

    # -- 统计 --------------------------------------------------------------------
    def stats(self) -> dict:
        out = {}
        for table in ("fragments", "embeddings", "kg_nodes", "kg_edges"):
            out[table] = self.db.query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
        return out
