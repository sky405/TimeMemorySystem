"""Phase 2 演示：Phase 1 访谈 → 素材处理 → 打印知识图谱 + 混合检索。

运行：pip install -r requirements.txt && PYTHONPATH=src python3 examples/phase2_demo.py
演示默认用 :memory: SQLite；设 TMS_MYSQL_URL 可切 MySQL，设 TMS_LLM_API_KEY /
TMS_EMB_API_KEY 可分别切真模型与真向量。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TMS_SQLITE_PATH", ":memory:")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from timememory.interview.agent import InterviewAgent
from timememory.interview.models import ElderProfile
from timememory.material.pipeline import run_material_pipeline
from timememory.material.query import retrieve
from timememory.material.embeddings import get_embedder

SCRIPT = [
    "我在四川嘉陵江边长大的，经常去游泳。有一次差点被水冲走，多亏邻居王二哥把我捞起来。",
    "那是七八岁的时候，水流特别急，我越扑腾离岸越远，吓得直哭。",
    "后来王二哥跳下水，一把抓住我的胳膊，把我拖上了岸。我娘知道后，一边哭一边给我煮姜汤。",
    "记不清了，都是很久以前的事了。",
    "我们村口有棵大槐树，全村人夏天都在树下乘凉，村长还常在那儿给大家开会。",
    "我只读了三年书，私塾先生姓陈，特别严厉。有一次我逃学去掏鸟窝，被他拿戒尺打了手心。",
    "今天有点累了，咱们下次再聊吧。",
]

QUERIES = ["在嘉陵江游泳", "王二哥是谁？", "私塾先生严厉吗？"]


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
    print("Phase 2：素材处理（清洗 → 向量化 → 知识图谱）")
    print("=" * 72)
    result = run_material_pipeline(agent.session_id, fragments)
    store = result["store"]
    print(f"统计：{result['stats']}\n")

    print("— 知识图谱节点 —")
    for n in sorted(store.all_nodes(), key=lambda x: (x["type"], x["name"])):
        print(f"  [{n['type']}] {n['name']}")
    print("\n— 知识图谱三元组 —")
    for e in store.all_edges():
        src = store.get_node(e["src_id"]) or {}
        dst = store.get_node(e["dst_id"]) or {}
        print(f"  {src.get('name')} -[{e['relation']}]-> {dst.get('name')}"
              f"  （证据：{e['evidence_fragment_id']}）")

    print("\n" + "=" * 72)
    print("混合检索预演（向量 + 图谱扩展）")
    print("=" * 72)
    embedder = get_embedder()
    for q in QUERIES:
        print(f"\n❓ {q}")
        for r in retrieve(q, store, embedder, top_k=2):
            print(f"  [{r.via} {r.score}] {r.fragment['content'][:40]}")


if __name__ == "__main__":
    main()
