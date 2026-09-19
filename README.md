# TimeMemorySystem 时光记忆系统

为老人书写人生回忆录的多智能体系统：从访谈采集，到素材处理，到传记写作，再到后代可检索的家族记忆库。

```text
用户（老人家属）注册 → 填写老人基本信息
        │
        ▼
┌─────────────────────────────────────────┐
│ 编排层 ✅ 已实现                          │
│  访谈 Agent ◄──提纲脚本── 写作 Agent      │
│   采集素材      缺口补访     评估+成稿     │
└────────────┬────────────────────────────┘
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
│ Phase 3: 写作评估 ✅ 已实现 │  素材够不够写？缺口 → 补充访谈提纲
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│ Phase 4: 初稿生成 ✅ 已实现 │  阶段分组 → 大纲 → 逐章写作 → 事实回检 → 统稿
└────────────┬──────────────┘
             ▼
┌───────────────────────────┐
│ Phase 5: 人工审核 ✅ 已实现 │  AI 修订建议 + 人逐条裁决 → 定稿
└────────────┬──────────────┘
             ▼
     Phase 6 家族记忆库（RAG 对话）
```

## 编排层：访谈 ↔ 写作协作

```text
interview → material → assess → draft → END
                ↑_________│（有缺口且预算未尽 → 带提纲再访一轮）
```

- 写作 Agent 的追问经**提纲脚本**注入访谈 Agent，原样问出、自动切话题——不是只打印在报告里。
- 同档多轮（`{archive}-r0/r1…`）素材追加到同一 store，评估与成稿跑在整档上。
- 停止条件：评估就绪 / 轮次用尽 / 某轮零产出；复核清单随成稿交付人工审核。

详见 [编排层设计文档](docs/orchestration-design.md)。

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

## Phase 3：写作评估

```text
MySQL 素材 → gather 摘要 → assess 打分找缺口 → plan 追问 → validate 修复 → 补充访谈提纲
```

- 四个维度打分：话题覆盖 / 时间线完整 / 人物丰满度 / 细节情感。
- 每个缺口绑定 Phase 1 话题 id，直接成为下一轮补访提纲。

详见 [Phase 3 设计文档](docs/phase3-assessment-design.md)。

## Phase 4：初稿生成

```text
素材 → 阶段分组 → 大纲 → write ⇄ check（逐章写作+事实回检循环）→ 统稿 → 回忆录初稿
```

- 有出生年按年龄分 童年/少年/青年/中年/晚年，无则按年代分组。
- 事实回检不通过不阻断：挂批注、留人工复核入口（存疑原文 + 查证片段范围）。
- 统稿只做衔接（序/过渡/尾声），不改写正文、不增加事实。

详见 [Phase 4 设计文档](docs/phase4-drafting-design.md)。

## Phase 5：人工审核

```text
初稿+复核清单 → suggest（AI 修订建议）→ human（逐条裁决）→ apply → 定稿
```

- 每条存疑先看 AI 建议（保留/改写/删除）+ 查证素材，再四选一：确认无误 / 已修正 / 存疑保留 / 删除相关句。
- 零存疑直接通过；所有裁决写入审核记录附录，原文永久可查。
- 交互入口：`examples/review_cli.py`（访谈成书 → 逐条裁决 → 定稿落盘）。

详见 [Phase 5 设计文档](docs/review-design.md)。

## 快速开始

```bash
pip install -r requirements.txt

# 跑测试（标准库 unittest；MySQL 测试需设 TMS_MYSQL_URL，否则跳过）
PYTHONPATH=src python3 -m unittest discover -s tests

# 一键跑完整本回忆录工程（访谈 ↔ 写作协作到成稿）
PYTHONPATH=src python3 examples/memoir_demo.py

# 交互式审核（访谈成书 → 逐条裁决 → 定稿写入 data/final_book.md）
PYTHONPATH=src python3 examples/review_cli.py

# 各阶段演示
PYTHONPATH=src python3 examples/simulated_interview.py  # Phase 1 模拟访谈
PYTHONPATH=src python3 examples/phase2_demo.py          # Phase 2 素材处理
PYTHONPATH=src python3 examples/phase3_demo.py          # Phase 3 写作评估
PYTHONPATH=src python3 examples/phase4_demo.py          # Phase 4 初稿生成
PYTHONPATH=src python3 examples/review_demo.py          # Phase 5 人工审核

# 接真模型（可选，不设则用离线 Demo 实现）
export TMS_LLM_BASE_URL="https://api.deepseek.com/v1"
export TMS_LLM_API_KEY="sk-..." TMS_LLM_MODEL="deepseek-chat"
export TMS_EMB_API_KEY="..." TMS_EMB_MODEL="text-embedding-3-small"  # 真向量
export TMS_MYSQL_URL="mysql://root:pass@127.0.0.1:3306/timememory"  # 生产库
```

## 最小代码示例

```python
from timememory.interview import ElderProfile
from timememory.orchestration import run_memoir, render_memoir_report
from timememory.review import run_review

# 编排层一键成书：访谈 ↔ 写作协作到就绪，然后成稿
result = run_memoir(
    elder=ElderProfile(name="张爷爷", age=82, hometown="四川合川"),
    answer_fn=lambda q, r, t: input(f"[第{r+1}轮] {q}\n老人："),
    birth_year=1953, max_rounds=3)
print(render_memoir_report(result))

# Phase 5：人工审核 → 定稿
final = run_review(result["drafts"], result["review"],
                   [f for f in result["store"].all_fragments()
                    if f["id"].startswith(result["archive_id"])],
                   elder={"name": "张爷爷"}, title=result["outline_title"],
                   reviewer="儿子", decide_fn=lambda v: input(f"{v['item']} 裁决："))
print(final["book"])
```

## 目录结构

```text
src/timememory/
├── interview/     Phase 1：访谈 Agent（router/ask/human/extract/validate/fix/record/closing）
├── material/      Phase 2：素材处理（clean/embed/extract_kg/persist + 混合检索）
├── assessment/    Phase 3：写作评估（gather/assess/plan/validate + 访谈提纲）
├── drafting/      Phase 4：初稿生成（stages/outline/write ⇄ check/polish/render）
├── review/        Phase 5：人工审核（suggest/human/apply + 定稿渲染）
└── orchestration/ 编排层（interview → material → assess ⇄ interview … → draft）
docs/                       设计文档
tests/                      单元测试（unittest）
examples/                   演示脚本（一键成书 + 各阶段 + 交互审核）
```
