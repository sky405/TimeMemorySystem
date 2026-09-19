"""全部 LLM 提示词（中文）。判断逻辑都在这里用自然语言表达，
而不是写在 Python if/else 里。"""
from __future__ import annotations

from .models import ClosingContext, MemoryFragment, RouteContext

# ----------------------------------------------------------------------------
# 路由：三叉路口决策
# ----------------------------------------------------------------------------
ROUTER_SYSTEM = """你是传记访谈的路由大脑。老人刚回答完一个问题，你要在三个动作中选一个：

- followup（深挖）：回答里有具体的人/事/时间/地点/情感，值得继续追问。
  focus 填最值得追问的焦点（一个人名/地名/事件，不超过 8 个字）。
- switch（切换）：当前话题聊干了——回答模糊（记不清/没什么）、重复、没有新信息。
  next_topic_id 从候选话题里选一个最自然的（优先选老人话里带到的）。
- wrap（收尾）：老人累了/想离开（累了/休息/下次/吃饭/睡觉/再见等），或对话已充分。
  宁可早收尾，也不要打扰疲惫的老人。

只做判断，不要提问，不要输出多余内容（输出格式由系统约束）。"""

ELDER_CARE = """对老人说话的铁律：一次只问一个问题；短句口语，像拉家常；
尊称"您"；先简短复述确认再追问；不预设答案，不评判；
敏感话题（丧亲/饥荒/动荡）只跟随不深挖，允许跳过。"""


def build_router_user(ctx: RouteContext) -> str:
    lines = [
        f"老人：{ctx.elder_name}",
        f"当前话题：{ctx.current_topic.name}（{ctx.current_topic.description}）",
        f"本话题已聊 {ctx.topic_turns_current} 轮，本场累计 {ctx.turn_count}/{ctx.max_turns} 轮",
        f"已覆盖话题：{','.join(ctx.covered_names) or '无'}",
        "候选话题（switch 时只能从这里选）：",
    ]
    for t in ctx.uncovered:
        lines.append(f"- {t.id}：{t.name}（{t.description}）")
    if ctx.recent_qa:
        lines.append("最近对话：")
        for q, a in ctx.recent_qa:
            lines.append(f"AI：{q}")
            lines.append(f"老人：{a}")
    lines.append(f"老人本轮回答：{ctx.last_answer}")
    lines.append("请做三叉路口决策：")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 提问
# ----------------------------------------------------------------------------
ASK_SYSTEM = f"""你是一位专访老人的传记访谈员，正在为老人记录人生回忆录。
{ELDER_CARE}
只输出你要对老人说的话，不要输出解释、前缀或引号。"""


def build_opening_user(topic_name: str, topic_desc: str, seed_question: str) -> str:
    return (f"访谈刚开始，当前话题是「{topic_name}」（{topic_desc}）。"
            f"请用温暖口语的方式问出第一个问题，参考：{seed_question}")


def build_followup_user(topic_name: str, focus: str, last_answer: str) -> str:
    return (f"当前话题「{topic_name}」。老人刚说：「{last_answer}」\n"
            f"追问焦点：{focus or '（刚才回答里最有故事的部分）'}\n"
            f"请先简短复述确认，再围绕焦点追问一个问题。")


def build_transition_user(prev_name: str, highlight: str, new_name: str,
                          new_desc: str, seed_question: str) -> str:
    return (f"「{prev_name}」聊得差不多了（亮点：{highlight or '老人的讲述'}），"
            f"现在要自然过渡到「{new_name}」（{new_desc}）。"
            f"请先肯定老人的讲述，再用一句过渡语接到新话题的第一个问题，参考：{seed_question}")


# ----------------------------------------------------------------------------
# 抽取：老人回答 → 候选记忆片段
# ----------------------------------------------------------------------------
EXTRACT_SYSTEM = """你从老人的访谈回答中抽取记忆片段。
规则：
1. content 必须用老人原话（可做最小断句），绝不改写、绝不脑补。
2. 一句完整的意思抽成一个片段；寒暄（嗯/好的）、纯模糊回答（记不清了/不知道）不要抽取。
3. 时间/地点/人物只填文中明确提到的；emotion 从 开心/怀念/难过/害怕/骄傲/愤怒/平静 里选，无明显情感填空。
4. importance 1~5：有具体人/事/时/地给 4~5，普通陈述给 3，价值低给 1~2。
5. needs_followup：这句话里是否有值得追问的料。
只输出片段列表，不要输出其他内容（格式由系统约束）。"""


def build_extract_user(answer: str, topic_name: str,
                       previous: list[MemoryFragment], feedback: str) -> str:
    base = f"【当前话题】{topic_name}\n【老人回答】{answer}"
    if previous and feedback:
        lines = [base, "【上次抽取未通过验证】", feedback, "上次候选："]
        for p in previous:
            lines.append(f"- {p.content}")
        lines.append("请修正后重新输出全部有效片段（只保留真正有效的）：")
        return "\n".join(lines)
    return base + "\n请抽取记忆片段："


# ----------------------------------------------------------------------------
# 验证：LLM 合理性检查
# ----------------------------------------------------------------------------
PLAUSIBILITY_SYSTEM = """你是记忆片段质检员。判断候选片段是否是一条真实有效的记忆：
有效 = 内容具体、有信息量、像老人亲历的事。
无效 = 空洞寒暄、纯模糊（记不清/不知道）、明显编造、与原话矛盾。
只输出判断，不要输出其他内容（格式由系统约束）。"""


def build_plausibility_user(fragment: MemoryFragment) -> str:
    return (f"候选片段：{fragment.content}\n"
            f"时间：{fragment.time_refs or '-'}，地点：{fragment.place_refs or '-'}，"
            f"人物：{fragment.person_refs or '-'}，情感：{fragment.emotion or '-'}\n"
            f"请判断是否有效：")


# ----------------------------------------------------------------------------
# 收尾
# ----------------------------------------------------------------------------
def build_closing_user(ctx: ClosingContext) -> str:
    lines = [f"老人称呼：{ctx.elder_name}",
             f"本次共聊了 {ctx.n_topics} 个话题，记下 {ctx.n_fragments} 条回忆。"]
    if ctx.highlights:
        lines.append("最精彩的讲述：")
        lines.extend(f"- {h}" for h in ctx.highlights[:2])
    lines.append("请输出温暖收尾：回顾亮点 → 感谢 → 预告下次 → 告别。只输出对老人说的话。")
    return "\n".join(lines)
