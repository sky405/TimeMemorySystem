"""人生话题库（纯数据）。切到哪个话题由 LLM 路由决定，
这里只提供候选列表和顺序兜底。"""
from __future__ import annotations

from .models import Topic

TOPICS: list[Topic] = [
    Topic("childhood", "童年经历", "出生、幼年生活、父母与兄弟姐妹、童年趣事。",
          ["您小时候在哪里长大？家里是什么样的？", "您还记得小时候最常玩的地方是哪里吗？", "小时候家里都有些什么人？您跟谁最亲？"]),
    Topic("hometown", "故乡与故土", "老屋、村庄街巷、山水风物、乡亲邻里。",
          ["您的老家是什么样的？还记得老屋吗？", "老家那条街、那个村，现在还能想起样子吗？", "老家有没有哪处风景，是您一辈子都忘不了的？"]),
    Topic("education", "求学经历", "上学、老师、同学、读书的难与乐。",
          ["您小时候读过书吗？还记得学校和老师吗？", "上学那会儿，印象最深的老师是哪一位？", "读书的时候有没有特别难忘的事？"]),
    Topic("youth_work", "青年与工作", "第一份工作、职业生涯、师傅同事、奋斗岁月。",
          ["您年轻时做过什么工作？第一份工还记得吗？", "刚参加工作那会儿，日子是怎么过的？", "工作这些年，有没有哪件事让您特别骄傲？"]),
    Topic("love_marriage", "爱情与婚姻", "相识、恋爱、结婚、婚后生活。",
          ["您和老伴是怎么认识的？", "结婚那天您还记得吗？当时是什么场面？", "婚后两个人是怎么过日子的？"]),
    Topic("family_children", "家庭与子女", "孩子出生、养育、家风、儿孙趣事。",
          ["孩子出生时您还记得吗？当时高兴成啥样？", "带孩子最操心的是什么事？", "孩子们小时候，谁最调皮？"]),
    Topic("craft_hobby", "手艺与爱好", "拿手本事、业余爱好、绝活与消遣。",
          ["您有什么拿手的本事，或者喜欢的消遣？", "您这门手艺是跟谁学的？学了多久？", "闲下来的时候，您最喜欢干点啥？"]),
    Topic("hard_times", "艰难岁月", "饥荒、动荡、病痛、离别，以及如何熬过来。敏感话题，只跟随不主动深挖。",
          ["那些年最难的时候，您是怎么熬过来的？", "走过那么难的日子，回头看，是什么撑着您走过来的？"]),
    Topic("wisdom", "人生感悟", "最想告诉子孙的话、一生的总结。",
          ["如果跟孙辈说一句最想说的话，您会说什么？", "活了这一辈子，您觉得人最要紧的是啥？"]),
]

TOPIC_MAP: dict[str, Topic] = {t.id: t for t in TOPICS}


def get_topic(topic_id: str) -> Topic:
    return TOPIC_MAP[topic_id]


def uncovered_topics(covered_ids: list[str], current_id: str) -> list[Topic]:
    """未覆盖的话题（排除当前话题），保持人生时间线顺序。"""
    done = set(covered_ids) | {current_id}
    return [t for t in TOPICS if t.id not in done]


def default_next_topic(covered_ids: list[str], current_id: str) -> Topic | None:
    """顺序兜底：LLM 没给出合法新话题时，按顺序取下一个。"""
    rest = uncovered_topics(covered_ids, current_id)
    return rest[0] if rest else None
