"""Phase 6 演示：访谈成书 → 定稿入库 → 子孙问答。

离线可跑；设 TMS_LLM_API_KEY / TMS_EMB_API_KEY 切真模型。
用法：PYTHONPATH=src python3 examples/library_demo.py
"""
import sys

sys.path.insert(0, "src")

from timememory.interview.models import ElderProfile
from timememory.library import ChatSession, ingest_book
from timememory.material import MaterialStore, connect_sqlite, get_embedder
from timememory.orchestration import run_memoir

ANSWERS = [
    "我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。",
    "后来王二哥跳下水，一把抓住我的胳膊，把我拖上了岸。我娘知道后，一边哭一边给我煮姜汤。",
    "1962年，我9岁，跟着父亲从合川县到了重庆市。",
    "我只读了三年书，私塾先生姓陈，特别严厉。有一次我逃学去掏鸟窝，被他拿戒尺打了手心。",
    "1978年，我去公社当了会计，第一个月领到工资，给娘扯了一块新布。",
]

QUESTIONS = ["王二哥是谁？", "1962年发生了什么？", "私塾先生严厉吗？", "量子波动速读术"]

FAREWELL = "今天有点累了，咱们下次再聊吧。"


def main() -> None:
    store = MaterialStore(connect_sqlite(":memory:"))
    print("=" * 72)
    print("编排层：访谈 → 成书（审核略过：Demo 成稿零存疑）")
    print("=" * 72)
    mem = run_memoir(ElderProfile(name="张爷爷", hometown="四川合川"),
                     lambda q, r, t: ANSWERS[t] if t < len(ANSWERS) else FAREWELL,
                     birth_year=1953, max_rounds=1, store=store, archive_id="demo-lib")
    print(f"状态：{mem['status']}｜章节：{mem['draft_stats']['chapters']}")

    print("\n" + "=" * 72)
    print("Phase 6：定稿入库 + 子孙问答")
    print("=" * 72)
    frags = [f for f in store.all_fragments() if f["id"].startswith("demo-lib")]
    report = ingest_book(store, get_embedder(), "demo-lib", mem["drafts"],
                         frags, mem["review"])
    print(f"入库：{report['chapters']} 章，待考 {len(report['cautions'])} 条\n")
    session = ChatSession(store, cautions=report["cautions"])
    for q in QUESTIONS:
        res = session.ask(q)
        a = res["answer"]
        print(f"❓ {q}")
        print(f"   {a.text.splitlines()[0]}")
        for c in a.citations:
            print(f"   📖 {c.passage_id}｜{c.quote}")
        if not a.has_answer:
            print("   （门控：低分拒答）")
        print()


if __name__ == "__main__":
    main()
