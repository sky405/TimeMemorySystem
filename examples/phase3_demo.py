"""Phase 3 演示：访谈 → 素材处理 → 写作评估 → 补充访谈提纲。

离线可跑：访谈/素材/评估全用 Demo 实现；设 TMS_LLM_API_KEY 切真模型评估。
用法：PYTHONPATH=src python3 examples/phase3_demo.py
"""
import sys

sys.path.insert(0, "src")

from timememory.assessment import render_brief, run_assessment
from timememory.interview.agent import InterviewAgent
from timememory.interview.models import ElderProfile
from timememory.material.pipeline import run_material_pipeline

SCRIPT = [
    "我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。",
    "后来王二哥跳下水，一把抓住我的胳膊，把我拖上了岸。我娘知道后，一边哭一边给我煮姜汤。",
    "那是七八岁的时候，水流特别急，我越扑腾离岸越远，吓得直哭。",
    "我只读了三年书，私塾先生姓陈，特别严厉。有一次我逃学去掏鸟窝，被他拿戒尺打了手心。",
    "夏天天热，村长带着大家修沟，我也跟着去抬土，肩膀都磨破了。",
]


def main() -> None:
    print("=" * 72)
    print("Phase 1：访谈采集")
    print("=" * 72)
    agent = InterviewAgent(elder=ElderProfile(name="张爷爷", age=82, hometown="四川合川"))
    agent.start()
    for answer in SCRIPT:
        reply = agent.step(answer)
        if reply.session_ended:
            break
    fragments = agent.fragments_json()
    print(f"访谈结束：{len(fragments)} 条原始片段\n")

    print("=" * 72)
    print("Phase 2：素材处理")
    print("=" * 72)
    result = run_material_pipeline(agent.session_id, fragments)
    print(f"统计：{result['stats']}\n")

    print("=" * 72)
    print("Phase 3：写作评估")
    print("=" * 72)
    out = run_assessment(result["store"], agent.session_id)
    print(render_brief(out["assessment"], out["plan"]))


if __name__ == "__main__":
    main()
