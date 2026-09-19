# TimeMemorySystem 时光记忆系统

为老人书写人生回忆录的多智能体系统：从访谈采集，到素材处理，到传记写作，再到后代可检索的家族记忆库。

```text
用户（老人家属）注册 → 填写老人基本信息
        │
        ▼
┌───────────────────────────┐
│ Phase 1: 初次访谈 ✅ 已实现 │  LangGraph 访谈 Agent，产出有效记忆片段
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│ Phase 2: 素材处理 ✅ 已实现 │  清洗 → 向量化 → 知识图谱（MySQL）
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│ Phase 3: 写作评估 ⬜ 规划中 │  分析素材完整度，发现缺口（缺口 → 回到 Phase 1 补充访谈）
└────────────┬──────────────┘
             ▼
     Phase 4 初稿生成 → Phase 5 人工审核 → Phase 6 家族记忆库（RAG 对话）
```

## Phase 1：初次访谈 Agent（LangGraph）

设计原则：**判断交给 LLM，流程交给图，质量交给验证器。**

```text
START → router → (ask → human → extract → validate → record) ↺
          │                                              │
          └──────────────→ closing → END                 │
                               ↑                         │
                    fix → extract（验证失败修复环）────────┘
```

详见 [Phase 1 设计文档](docs/phase1-interview-agent-design.md)。

## Phase 2：素材处理

```text
Phase 1 片段 → clean 清洗 → embed 向量化 → extract_kg 图谱抽取 → persist MySQL 入库
```

- 四张表：`fragments`（清洗片段）+ `embeddings`（向量）+ `kg_nodes / kg_edges`（知识图谱）。
- 生产用 MySQL（`TMS_MYSQL_URL`），测试/本地用 SQLite，同一套 schema。
- 自带混合检索（向量 + 图谱一跳扩展），为 Phase 6 RAG 预演。

详见 [Phase 2 设计文档](docs/phase2-material-design.md)。

## 快速开始

```bash
pip install -r requirements.txt

# 跑测试（标准库 unittest；MySQL 测试需设 TMS_MYSQL_URL，否则跳过）
PYTHONPATH=src python3 -m unittest discover -s tests

# Phase 1 模拟访谈 / 交互访谈
PYTHONPATH=src python3 examples/simulated_interview.py
PYTHONPATH=src python3 examples/demo_interview.py

# Phase 2 演示：访谈 → 素材处理 → 打印图谱 + 混合检索
PYTHONPATH=src python3 examples/phase2_demo.py

# 接真模型（可选，不设则用离线 Demo 实现）
export TMS_LLM_BASE_URL="https://api.deepseek.com/v1"
export TMS_LLM_API_KEY="sk-..." TMS_LLM_MODEL="deepseek-chat"
export TMS_EMB_API_KEY="..." TMS_EMB_MODEL="text-embedding-3-small"  # 真向量
export TMS_MYSQL_URL="mysql://root:pass@127.0.0.1:3306/timememory"  # 生产库
```

## 最小代码示例

```python
from timememory.interview import InterviewAgent, ElderProfile
from timememory.material import run_material_pipeline, retrieve, get_embedder

# Phase 1：访谈
agent = InterviewAgent(elder=ElderProfile(name="张爷爷"))
print(agent.start().text)
while True:
    reply = agent.step(input("老人："))
    print(f"[{reply.decision.action}] {reply.text}")
    if reply.session_ended:
        break

# Phase 2：素材处理
result = run_material_pipeline(agent.session_id, agent.fragments_json())
print(result["stats"])  # 清洗/向量/节点/边统计

# 混合检索（Phase 6 预演）
for r in retrieve("王二哥是谁？", result["store"], get_embedder()):
    print(f"[{r.via} {r.score}] {r.fragment['content']}")
```

## 目录结构

```text
src/timememory/
├── interview/     Phase 1：访谈 Agent（router/ask/human/extract/validate/fix/record/closing）
└── material/      Phase 2：素材处理（clean/embed/extract_kg/persist + 混合检索）
docs/                       设计文档
tests/                      单元测试（unittest）
examples/                   各阶段演示脚本
```
