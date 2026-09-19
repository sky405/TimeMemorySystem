"""模拟访谈演示（LangGraph 版）：脚本化"老人"走完 A→B→C 全流程。

运行：pip install -r requirements.txt && PYTHONPATH=src python3 examples/simulated_interview.py
如需真模型：export TMS_LLM_API_KEY=... TMS_LLM_BASE_URL=... TMS_LLM_MODEL=...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from timememory.interview.agent import InterviewAgent
from timememory.interview.llm import get_llm
from timememory.interview.models import ElderProfile

SCRIPT = [
    "我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。",  # A
    "那是七八岁的时候，水流特别急，我越扑腾离岸越远，吓得直哭。",  # A
    "后来王二哥跳下水，一把抓住我的胳膊，把我拖上了岸。我娘知道后，一边哭一边给我煮姜汤。",  # A
    "记不清了，都是很久以前的事了。",  # B → 故乡
    "我们村口有棵大槐树，全村人夏天都在树下乘凉，村长还常在那儿给大家开会。",  # A
    "有一年夏天特别热，树下挤了几十口人，村长王老三给大家讲古，我听得入了迷，连晚饭都忘了吃。",  # A
    "别的也记不清了。",  # B → 求学
    "我只读了三年书，私塾先生姓陈，特别严厉。有一次我逃学去掏鸟窝，被他拿戒尺打了手心。",  # A
    "今天有点累了，咱们下次再聊吧。",  # C
]

BRANCH_CN = {"followup": "A·深挖", "switch": "B·切换", "wrap": "C·收尾"}


def main() -> None:
    agent = InterviewAgent(
        elder=ElderProfile(name="张爷爷", age=82, hometown="四川合川"),
        llm=get_llm(),
    )
    print("=" * 72)
    print("Phase 1 初次访谈 Agent（LangGraph）· 模拟演示")
    print("=" * 72)

    print(f"\n【访谈员】{agent.start().text}\n")
    for i, answer in enumerate(SCRIPT, 1):
        print(f"【老人·第{i}轮】{answer}")
        reply = agent.step(answer)
        d = reply.decision
        if d:
            print(f"  🔀 决策：{BRANCH_CN[d.action]}——{d.reasoning}")
            for f in reply.new_fragments:
                print(f"     📝 片段[{f.topic_id} ⭐{f.importance}]：{f.content}")
            for r in agent.state.get("last_rejected", []):
                print(f"     🚫 驳回「{r['content'][:20]}」：{'; '.join(r['reasons'])}")
        print(f"【访谈员】{reply.text}\n")
        if reply.session_ended:
            break

    report = agent.coverage_report()
    print("=" * 72)
    print("素材完整度报告（→ Phase 3）")
    print("=" * 72)
    print(f"整体覆盖度：{report['overall_coverage']}，建议动作：{report['recommendation']}")
    for r in report["topics"].values():
        if r["turns"]:
            print(f"- {r['name']}：{r['turns']} 轮 / {r['fragments']} 片段 / 覆盖度 {r['coverage']}")
    print(f"下次访谈计划：{report['next_interview_plan']}")

    out = Path(__file__).resolve().parent / "sample_transcript.md"
    out.write_text(agent.export_transcript_markdown(), encoding="utf-8")
    print(f"\n逐字稿已保存：{out}")


if __name__ == "__main__":
    main()
