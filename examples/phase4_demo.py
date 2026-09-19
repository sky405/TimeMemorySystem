"""Phase 4 演示：访谈 → 素材处理 → 写作评估 → 初稿生成。

离线可跑：全用 Demo 实现；设 TMS_LLM_API_KEY 切真模型写作。
用法：PYTHONPATH=src python3 examples/phase4_demo.py
"""
import sys

sys.path.insert(0, "src")

from timememory.assessment import run_assessment
from timememory.drafting import run_drafting
from timememory.interview.agent import InterviewAgent
from timememory.interview.models import ElderProfile
from timememory.material.pipeline import run_material_pipeline

SCRIPT = [
    "我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。",
    "后来王二哥跳下水，一把抓住我的胳膊，把我拖上了岸。我娘知道后，一边哭一边给我煮姜汤。",
    "1962年，我9岁，跟着父亲从合川县到了重庆市。",
    "我只读了三年书，私塾先生姓陈，特别严厉。有一次我逃学去掏鸟窝，被他拿戒尺打了手心。",
    "1978年，我去公社当了会计，第一个月领到工资，给娘扯了一块新布。",
    "夏天天热，村长带着大家修沟，我也跟着去抬土，肩膀都磨破了。",
]

ELDER = {"name": "张爷爷", "hometown": "四川合川"}
BIRTH_YEAR = 1953


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
    assessed = run_assessment(result["store"], agent.session_id)
    a = assessed["assessment"]
    print(f"评估：{a.overall} 缺口 {len(a.gaps)} 个\n")

    print("=" * 72)
    print("Phase 4：初稿生成")
    print("=" * 72)
    out = run_drafting(result["store"], agent.session_id, ELDER,
                       birth_year=BIRTH_YEAR, assessment=a)
    print(f"统计：{out['stats']}")
    print(f"大纲：{' ｜ '.join(c.title for c in out['outline'].chapters)}\n")
    print(out["manuscript"])


if __name__ == "__main__":
    main()
