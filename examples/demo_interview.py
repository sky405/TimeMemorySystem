"""交互式演示：你扮演老人，在终端里和访谈 Agent 聊天。

运行：pip install -r requirements.txt && PYTHONPATH=src python3 examples/demo_interview.py
退出：输入 quit / q，或说累了让 Agent 自动收尾。
如需真模型：export TMS_LLM_API_KEY=... TMS_LLM_BASE_URL=... TMS_LLM_MODEL=...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from timememory.interview.agent import InterviewAgent
from timememory.interview.llm import get_llm
from timememory.interview.models import ElderProfile

BRANCH_CN = {"followup": "A·深挖", "switch": "B·切换", "wrap": "C·收尾"}


def main() -> None:
    name = input("老人怎么称呼？（默认：张爷爷）").strip() or "张爷爷"
    hometown = input("老人老家是哪？（可空）").strip()
    agent = InterviewAgent(elder=ElderProfile(name=name, hometown=hometown), llm=get_llm())
    print("\n—— 访谈开始（你扮演老人，直接输入说的话；quit 退出）——\n")
    print(f"【访谈员】{agent.start().text}\n")

    while True:
        try:
            answer = input(f"【{name}】").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if answer.lower() in {"quit", "q", "退出"}:
            break
        if not answer:
            continue
        reply = agent.step(answer)
        if reply.decision:
            print(f"  🔀 {BRANCH_CN[reply.decision.action]}｜{reply.decision.reasoning}｜"
                  f"新片段 {len(reply.new_fragments)} 条")
        print(f"【访谈员】{reply.text}\n")
        if reply.session_ended:
            break

    report = agent.coverage_report()
    print(f"\n—— 访谈结束：{report['total_turns']} 轮 / {report['total_fragments']} 片段 / "
          f"整体覆盖度 {report['overall_coverage']} ——")
    out = Path(__file__).resolve().parent / "interactive_transcript.md"
    out.write_text(agent.export_transcript_markdown(), encoding="utf-8")
    agent.save_session(Path(__file__).resolve().parent / "interactive_session.json")
    print(f"逐字稿：{out}")


if __name__ == "__main__":
    main()
