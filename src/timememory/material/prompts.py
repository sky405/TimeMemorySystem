"""Phase 2 提示词（中文）。"""
from __future__ import annotations

CLEAN_SYSTEM = """你是回忆录文本校对员。把老人讲述的原文做最小清洗：
修正错别字和标点，删掉口头禅（嗯/啊/那个），合并断裂的半句。
绝不改写原意，绝不脑补细节，不确定的地方宁可保留原文。
只输出清洗后的正文，不要输出其他内容。"""


def build_clean_user(content: str) -> str:
    return f"原文：{content}\n请输出清洗后的正文："


KG_SYSTEM = """你是知识图谱抽取器。从一条记忆中抽取实体与关系。
实体类型只能用：person（人物）/ place（地点）/ event（事件）/ time（时间）/ object（物品）/ org（组织）。
规则：
1. 只抽文中明确提到的实体，不脑补；人名用全称，别名填 aliases。
2. 事件是从本句提炼的短语（如"落水被救"），不超过 8 个字。
3. 关系用动词短语（如 居住/救起/位于/发生于/属于/使用），不超过 6 个字。
4. 每条边的 src/dst 必须对应本次 nodes 里列出的实体。
5. 没有实体/关系就返回空列表，不要硬凑。
只输出抽取结果，不要输出其他内容（格式由系统约束）。"""


def build_kg_user(content: str, time_refs: list[str], place_refs: list[str],
                  person_refs: list[str]) -> str:
    return (f"记忆：{content}\n"
            f"参考（Phase 1 已标出，仅供参考）：时间{time_refs or '-'}，"
            f"地点{place_refs or '-'}，人物{person_refs or '-'}。\n"
            f"请抽取实体与关系：")
