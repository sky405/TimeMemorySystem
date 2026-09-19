"""Phase 5 演示：访谈成书 → 人工审核（脚本模拟审核人） → 定稿。

离线可跑；设 TMS_LLM_API_KEY 切真模型提修订建议。
用法：PYTHONPATH=src python3 examples/review_demo.py
"""
import sys

sys.path.insert(0, "src")

from timememory.interview.models import ElderProfile
from timememory.material import MaterialStore, connect_sqlite
from timememory.orchestration import run_memoir
from timememory.review import run_review

ANSWERS = [
    "我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。",
    "后来王二哥跳下水，一把抓住我的胳膊，把我拖上了岸。我娘知道后，一边哭一边给我煮姜汤。",
    "1962年，我9岁，跟着父亲从合川县到了重庆市。",
    "1978年，我去公社当了会计，第一个月领到工资，给娘扯了一块新布。",
]

FAREWELL = "今天有点累了，咱们下次再聊吧。"


def answer_fn(question: str, round_index: int, turn: int) -> str:
    return ANSWERS[turn] if turn < len(ANSWERS) else FAREWELL


def decide_fn(view: dict) -> dict:
    item = view["item"]
    print(f"\n[审核 {view['position']}/{view['total']}] {view['chapter_title']}")
    print(f"  原文：{item['quote']}\n  问题：{item['issue']}")
    print(f"  AI 建议：{view['suggestion']['action']}——{view['suggestion']['reason']}")
    print("  裁决：确认无误（模拟审核人已向老人核实）")
    return {"verdict": "确认无误", "note": "已向老人核实。"}


def main() -> None:
    store = MaterialStore(connect_sqlite(":memory:"))
    print("=" * 72)
    print("编排层：访谈 → 成书")
    print("=" * 72)
    mem = run_memoir(ElderProfile(name="张爷爷", hometown="四川合川"),
                     answer_fn, birth_year=1953, max_rounds=1, store=store,
                     archive_id="demo-book")
    print(f"状态：{mem['status']}｜章节：{mem['draft_stats']['chapters']}｜"
          f"存疑：{mem['draft_stats']['flags']}")

    print("\n" + "=" * 72)
    print("Phase 5：人工审核")
    print("=" * 72)
    frags = [f for f in store.all_fragments() if f["id"].startswith("demo-book")]
    res = run_review(mem["drafts"], mem["review"], frags,
                     elder={"name": "张爷爷"}, title=mem["outline_title"],
                     reviewer="儿子", decide_fn=decide_fn)
    print(f"\n审核完毕：{res['stats']}")

    print("\n" + "=" * 72)
    print("合成示例：一条存疑的改写流程（展示裁决如何改动正文）")
    print("=" * 72)
    before = "1962年到了重庆。1978年去过北京。后来回乡。"
    res2 = run_review(
        [{"chapter_id": "ch-0", "title": "童年", "text": before,
          "fragment_ids": ["demo:f1"]}],
        [{"chapter": "童年", "quote": "1978年去过北京",
          "issue": "年份 1978 在素材中没有出处", "severity": "存疑",
          "fragment_ids": ["demo:f1"]}],
        [{"id": "demo:f1", "content": "1962年到了重庆。", "topic_id": "childhood"}],
        elder={"name": "张爷爷"}, title="合成示例", reviewer="儿子",
        decide_fn=lambda v: {"verdict": "已修正",
                             "replacement": "那几年还去过很多地方。",
                             "note": "老人记不清哪年，改成虚写。"})
    print(f"改前：{before}")
    print(f"改后：{res2['chapters'][0]['text']}")
    print(f"记录：{res2['records'][0]['verdict']}｜已改动正文：{res2['records'][0]['applied']}")


if __name__ == "__main__":
    main()
