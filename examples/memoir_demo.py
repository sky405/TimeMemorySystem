"""编排层演示：一键跑完整本回忆录工程。

访谈 Agent ↔ 写作 Agent 协作：首轮访谈 → 素材处理 → 评估找缺口 →
带着提纲补访 → 再评估 → 成稿。
离线可跑；设 TMS_LLM_API_KEY 切真模型。
用法：PYTHONPATH=src python3 examples/memoir_demo.py
"""
import sys

sys.path.insert(0, "src")

from timememory.interview.models import ElderProfile
from timememory.material import MaterialStore, connect_sqlite
from timememory.orchestration import render_memoir_report, run_memoir

ANSWERS = {
    0: [
        "我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。",
        "后来王二哥跳下水，一把抓住我的胳膊，把我拖上了岸。我娘知道后，一边哭一边给我煮姜汤。",
        "1962年，我9岁，跟着父亲从合川县到了重庆市。",
        "我只读了三年书，私塾先生姓陈，特别严厉。有一次我逃学去掏鸟窝，被他拿戒尺打了手心。",
        "1978年，我去公社当了会计，第一个月领到工资，给娘扯了一块新布。",
        "夏天天热，村长带着大家修沟，我也跟着去抬土，肩膀都磨破了。",
    ],
    1: [
        "老家在四川合川，村口有棵大槐树，全村人夏天都在树下乘凉。",
        "我娘裹小脚，走路一摇一晃，但干活一点不含糊，纳的鞋底最结实。",
        "私塾在祠堂里，先生姓陈，教《三字经》，背不出来要打手心。",
        "和老伴是经人介绍认识的，见了两面就定了亲，彩礼是一辆自行车。",
    ],
}

FAREWELL = "今天有点累了，咱们下次再聊吧。"


def answer_fn(question: str, round_index: int, turn: int) -> str:
    script = ANSWERS.get(round_index, [])
    return script[turn] if turn < len(script) else FAREWELL


def main() -> None:
    store = MaterialStore(connect_sqlite(":memory:"))  # 整档落内存，不受历史残留干扰
    result = run_memoir(elder=ElderProfile(name="张爷爷", age=82, hometown="四川合川"),
                        answer_fn=answer_fn, birth_year=1953, max_rounds=2, store=store)
    print(render_memoir_report(result))
    print("=" * 72)
    print("初稿全文")
    print("=" * 72)
    print(result["manuscript"])


if __name__ == "__main__":
    main()
