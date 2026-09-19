# TimeMemorySystem 时光记忆系统

为老人书写人生回忆录的多智能体系统：从访谈采集，到素材处理，到传记写作，再到后代可检索的家族记忆库。

```text
用户（老人家属）注册 → 填写老人基本信息
        │
        ▼
┌───────────────────────────┐
│ Phase 1: 初次访谈 ✅ 已实现 │  访谈 Agent 与老人对话，提取初始记忆片段
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

## Phase 1：初次访谈 Agent

核心循环（详见 [设计文档](docs/phase1-interview-agent-design.md)）：

```text
[1. 确定当前话题] → [2. AI 提问] → [3. 老人回答] → [4. 三叉路口决策]
                                                        ├── A: 发现好故事 → 深度追问 → 回 [3]
                                                        ├── B: 话题聊干了 → 切换话题 → 回 [1]
                                                        └── C: 老人累了 → 收尾总结 → 结束
```

**特点**：零第三方依赖（纯标准库）；离线可跑（Mock LLM + 规则兜底）；有 Key 自动切真模型
（OpenAI / DeepSeek / 通义千问 / 本地 Ollama，OpenAI 协议）。

## 快速开始

```bash
# 1. 跑测试（38 个，标准库 unittest）
PYTHONPATH=src python3 -m unittest discover -s tests

# 2. 看模拟访谈演示（含 A/B/C 三分支 + 完整度报告 + 逐字稿）
PYTHONPATH=src python3 examples/simulated_interview.py

# 3. 亲自扮演老人，和 Agent 聊天
PYTHONPATH=src python3 examples/demo_interview.py

# 4. 接真模型（可选，不设则用 Mock）
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
    reply = agent.step(input("老人："))          # 老人回答 → 决策 → 下一句
    print(f"[{reply.decision.action.value}] {reply.text}")
    if reply.session_ended:
        break

print(agent.coverage_report())                  # → Phase 3：素材完整度报告
print(agent.fragments_json())                   # → Phase 2：记忆片段
print(agent.export_transcript_markdown())       # → Phase 5：逐字稿
```

## 目录结构

```text
src/timememory/interview/   Phase 1 核心包
├── models.py      数据模型（话题/消息/记忆片段/决策/会话状态）
├── topics.py      人生九话题库 + 搭桥式话题规划器
├── decision.py    三叉路口决策引擎（硬规则 → 特征分 → LLM 裁判）
├── extractor.py   记忆片段提取（规则打底 + LLM 精修）
├── questions.py   提问生成（开场/五板斧追问/过渡/收尾，一次只问一个问题）
├── llm.py         LLM 抽象（Mock + OpenAI 协议，标准库 urllib）
├── prompts.py     全部中文提示词
├── agent.py       编排器（状态机 + 对外 API）
├── session.py     会话 JSON 持久化（中途离开可恢复）
└── report.py      Phase 3 交接（完整度报告 + 缺口 + 下次访谈计划）
docs/                       设计文档
tests/                      单元测试（unittest，零依赖）
examples/                   模拟演示 + 交互式 CLI + 逐字稿示例
```
