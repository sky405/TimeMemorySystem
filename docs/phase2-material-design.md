# Phase 2 素材处理设计文档

> 状态：已实现（`src/timememory/material/`）
> 输入：Phase 1 的有效记忆片段 → 输出：MySQL 里的清洗片段 + 向量 + 知识图谱

---

## 1. 流水线（LangGraph）

```mermaid
flowchart LR
    P1[Phase 1 片段] --> C[clean<br/>清洗]
    C --> E[embed<br/>向量化]
    E --> K[extract_kg<br/>图谱抽取]
    K --> S[persist<br/>MySQL 入库]
    S --> Q[(MySQL)]
```

`run_material_pipeline(session_id, fragments)` 一键跑完。线性 ETL，用图承载以便观察每步产出。

## 2. 清洗（`cleaning.py`）

- 规范 id：`f"{session}:{frag}"`，跨会话唯一（缺 id 时按序补号）。
- 去空白 + LLM 最小清洗（改错别字标点，不改写原意；Demo 实现只做去空白）。
- 过滤过短无料句；近重复（Jaccard ≥ 0.9）合并：留更长的、引用求并集、重要度取高。

## 3. 向量化（`embeddings.py`）

- `OpenAIEmbedder`：OpenAI 兼容 Embedding 接口。
  `TMS_EMB_API_KEY`（必填才启用）/ `TMS_EMB_BASE_URL`（默认复用 LLM 的）/ `TMS_EMB_MODEL`。
- `DemoEmbedder`：离线确定性（字符 bigram 哈希，64 维），仅测试演示。
- 向量以 float32 BLOB 存 MySQL `embeddings` 表；家族数据量下 Python 暴力 cosine 足够，
  数据量大了再迁专用向量库（检索接口不变）。

## 4. 知识图谱（`kg.py` + MySQL）

LLM 从每条记忆抽 `KGExtraction{nodes, edges}`（结构化输出），归一后入库：

- 节点 id = `sha1(type + name)`，确定性，天然去重；同名节点合并别名、介绍取长。
- 类型：`person / place / event / time / object / org`，非法类型归入 `object`。
- 边唯一键 `(src, dst, relation, evidence)`：同一三元组来自多条记忆会保留多行（provenance），
  重跑流水线天然幂等；每条边挂 `evidence_fragment_id`，可溯源到原句。

### MySQL Schema（`store.py`，InnoDB + utf8mb4）

```sql
fragments (id PK, session_id, content, topic_id, source_turn,
           time_refs, place_refs, person_refs, emotion, importance, created_at)
embeddings (fragment_id PK, dim, vector BLOB)
kg_nodes  (id PK, type, name, aliases, description, created_at, UNIQUE(type, name))
kg_edges  (id AUTO_INCREMENT PK, src_id, dst_id, relation,
           evidence_fragment_id, confidence, created_at,
           UNIQUE(src_id, dst_id, relation, evidence_fragment_id))
```

### 存储后端

`MaterialStore` 统一读写，SQL 用 `?` 占位、MySQL 自动转 `%s`：

- 生产：MySQL。`TMS_MYSQL_URL=mysql://user:pass@host:port/db`
  （或 `TMS_MYSQL_HOST/PORT/USER/PASSWORD/DB`），库不存在自动建库。
- 测试/本地：SQLite（标准库）。`TMS_SQLITE_PATH`（默认 `data/material.db`，`:memory:` 用于测试）。

本地起 MySQL：`docker run -d --name tm-mysql -e MYSQL_ROOT_PASSWORD=pass -p 3306:3306 mysql:8`

## 5. 混合检索（`query.py`，Phase 6 预演）

`retrieve(query, store, embedder)`：向量召回 top_k + 知识图谱一跳扩展（`via` 标记来源）。
证明向量库与图谱真实可用，Phase 6 RAG 对话将复用。

## 6. 交给下游的接口

- Phase 3 写作评估：读 `fragments`（清洗后全文）+ `kg_nodes/edges`（人物关系、时间线）。
- Phase 6 RAG：`retrieve()` 混合检索。

## 7. 文件结构

```text
src/timememory/material/
├── models.py      CleanFragment / KG 草稿与入库模型
├── store.py       MySQL/SQLite 适配 + 四表 schema + MaterialStore
├── llm.py         MaterialLLM 协议 + LangChain 真模型 + Demo 实现
├── prompts.py     清洗 / 图谱抽取提示词
├── cleaning.py    清洗与去重合并
├── embeddings.py  OpenAI 向量 + Demo 向量 + cosine
├── kg.py          抽取归一（id/去重/挂证据）
├── pipeline.py    StateGraph（clean → embed → extract_kg → persist）
└── query.py       混合检索（向量 + 图谱扩展）
```
