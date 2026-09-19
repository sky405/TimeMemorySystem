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
│ Phase 2: 素材处理 ⬜ 规划中 │  清洗 → 向量化 → 知识图谱
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

- **router**：三叉路口决策（A 深挖 / B 切换 / C 收尾），LLM 结构化输出。
- **extract → validate → fix**：ReAct 式抽取循环，验证器（schema/模糊/重复/LLM 质检）
  不合格就带反馈修复，直到产出有效记忆片段。
- **人工规则只剩两处**：告别安全护栏 + 轮次预算护栏。

详见 [设计文档](docs/phase1-interview-agent-design.md)。

## 快速开始

```bash
pip install -r requirements.txt

# 跑测试（标准库 unittest）
PYTHONPATH=src python3 -m unittest discover -s tests

# 模拟访谈演示（含 A/B/C 三分支 + 验证器驳回展示 + 完整度报告 + 逐字稿）
PYTHONPATH=src python3 examples/simulated_interview.py

# 亲自扮演老人，和 Agent 聊天
PYTHONPATH=src python3 examples/demo_interview.py

# 接真模型（可选，不设则用离线 Demo 实现）
export TMS_LLM_BASE_URL="https://api.deepseek.com/v1"
export TMS_LLM_API_KEY="sk-..."
export TMS_LLM_MODEL="deepseek-chat"
```

## 最小代码示例

```python
from timememory.interview import InterviewAgent, ElderProfile

agent = InterviewAgent(elder=ElderProfile(name="张爷爷", age=82, hometown="四川合川"))
print(agent.start().text)                       # 开场白 + 首个问题

while True:
    reply = agent.step(input("老人："))          # 老人回答 → 图运转 → 下一句
    print(f"[{reply.decision.action}] {reply.text}")
    if reply.session_ended:
        break

print(agent.coverage_report())                  # → Phase 3：素材完整度报告
print(agent.fragments_json())                   # → Phase 2：有效记忆片段
print(agent.export_transcript_markdown())       # → Phase 5：逐字稿
```

## 目录结构

```text
src/timememory/interview/   Phase 1 核心包
├── models.py      pydantic schema（片段/决策/质检）+ 图状态
├── topics.py      人生九话题（纯数据）
├── prompts.py     全部中文提示词
├── llm.py         InterviewLLM 协议 + LangChain 真模型 + Demo 离线实现
├── validators.py  片段验证器 + 告别安全护栏
├── graph.py       StateGraph（router/ask/human/extract/validate/fix/record/closing）
└── agent.py       薄封装（start/step/报告/逐字稿/会话快照）
docs/                       设计文档
tests/                      单元测试（unittest）
examples/                   模拟演示 + 交互式 CLI + 逐字稿示例
```
