"""Phase 5 交互式审核：访谈成书 → 逐条人工裁决 → 定稿落盘。

用法：PYTHONPATH=src python3 examples/review_cli.py
定稿写入 data/final_book.md（本机文件，不入库）。
"""
import sys

sys.path.insert(0, "src")

from timememory.interview.models import ElderProfile
from timememory.material import MaterialStore, connect_sqlite
from timememory.orchestration import run_memoir
from timememory.review import VERDICTS, run_review


def answer_fn(question: str, round_index: int, turn: int) -> str:
    print(f"\n[第{round_index + 1}轮·访谈员] {question}")
    return input("老人：")


def decide_fn(view: dict) -> dict:
    item = view["item"]
    print("\n" + "=" * 72)
    print(f"存疑 {view['position']}/{view['total']}｜{view['chapter_title']}")
    print(f"原文：{item['quote']}\n问题：{item['issue']}（{item['severity']}）")
    print(f"\n上下文：{view['excerpt']}")
    print("\n查证素材：")
    for m in view["materials"]:
        print(f"  - [{m.get('topic_id', '')}] {m.get('content', '')[:60]}")
    s = view["suggestion"]
    print(f"\nAI 建议：{s['action']}"
          + (f"→“{s['replacement']}”" if s.get("replacement") else "")
          + f"——{s['reason']}")
    print("\n裁决：1确认无误 2已修正 3存疑保留 4删除相关句")
    while True:
        c = input("请选择 [1/2/3/4]：").strip()
        if c in "1234":
            break
        print("请输入 1-4。")
    verdict = VERDICTS[int(c) - 1]
    replacement = ""
    if verdict == "已修正":
        replacement = input("改写后的句子（直接回车则采纳 AI 建议）：").strip()
    note = input("备注（可空）：").strip()
    return {"verdict": verdict, "replacement": replacement, "note": note}


def main() -> None:
    name = input("老人姓名：").strip() or "老人家"
    birth = input("出生年份（可空）：").strip()
    mem = run_memoir(ElderProfile(name=name), answer_fn,
                     birth_year=int(birth) if birth.isdigit() else None,
                     max_rounds=2, store=MaterialStore(connect_sqlite(":memory:")))
    print(f"\n初稿已生成：{mem['draft_stats']['chapters']} 章，"
          f"{mem['draft_stats']['flags']} 条存疑，开始逐条审核……")
    frags = [f for f in mem["store"].all_fragments()
             if f.get("session_id", "").startswith(mem["archive_id"])]
    reviewer = input("审核人姓名（默认：家属）：").strip() or "家属"
    res = run_review(mem["drafts"], mem["review"], frags,
                     elder={"name": name}, title=mem["outline_title"],
                     reviewer=reviewer, decide_fn=decide_fn)
    with open("data/final_book.md", "w", encoding="utf-8") as fh:
        fh.write(res["book"])
    print(f"\n审核完毕：{res['stats']}，定稿已写入 data/final_book.md")


if __name__ == "__main__":
    main()
