"""人生话题库 + 话题规划器（TopicPlanner）。

九个话题按中国老人一生时间线编排。切换话题时优先"搭桥"——
如果老人刚才提到的实体命中某未覆盖话题的关键词，就自然地切过去。
"""
from __future__ import annotations

from .models import InterviewState, Topic


TOPICS: list[Topic] = [
    Topic(
        id="childhood",
        name="童年经历",
        description="出生、幼年生活、父母与兄弟姐妹、童年趣事。",
        opening_questions=[
            "您小时候在哪里长大？家里是什么样的？",
            "您还记得小时候最常玩的地方是哪里吗？",
            "小时候家里都有些什么人？您跟谁最亲？",
        ],
        followup_angles=["玩伴与游戏", "父母印象", "童年趣事", "吃的穿的", "过年过节"],
        keywords=["小时候", "童年", "出生", "父母", "爸爸", "妈妈", "兄弟", "姐妹",
                  "玩伴", "放牛", "捉鱼", "过家家", "弹珠", "毽子"],
        min_turns=2,
        max_turns=6,
        priority=100,
    ),
    Topic(
        id="hometown",
        name="故乡与故土",
        description="老屋、村庄/街巷、山水风物、乡亲邻里。",
        opening_questions=[
            "您的老家是什么样的？还记得老屋吗？",
            "老家那条街、那个村，现在还能想起样子吗？",
            "老家有没有哪处风景，是您一辈子都忘不了的？",
        ],
        followup_angles=["老屋格局", "村庄街巷", "山水风物", "邻里乡亲", "家乡吃食"],
        keywords=["老家", "老屋", "村", "街", "巷", "邻居", "乡亲", "山", "河", "江",
                  "湖", "田", "地", "祠堂", "庙", "集市"],
        min_turns=2,
        max_turns=5,
        priority=90,
    ),
    Topic(
        id="education",
        name="求学经历",
        description="上学、老师、同学、读书的难与乐。",
        opening_questions=[
            "您小时候读过书吗？还记得学校和老师吗？",
            "上学那会儿，印象最深的老师是哪一位？",
            "读书的时候有没有特别难忘的事？",
        ],
        followup_angles=["学校模样", "印象最深的老师", "同窗好友", "辍学/升学", "最喜欢的课"],
        keywords=["上学", "读书", "学校", "小学", "中学", "老师", "先生", "同学",
                  "同桌", "课堂", "考试", "毕业", "私塾"],
        min_turns=2,
        max_turns=5,
        priority=80,
    ),
    Topic(
        id="youth_work",
        name="青年与工作",
        description="第一份工作、职业生涯、师傅同事、奋斗岁月。",
        opening_questions=[
            "您年轻时做过什么工作？第一份工还记得吗？",
            "刚参加工作那会儿，日子是怎么过的？",
            "工作这些年，有没有哪件事让您特别骄傲？",
        ],
        followup_angles=["第一份工作", "师傅与同事", "职业变迁", "最骄傲的事", "最苦的活"],
        keywords=["工作", "上班", "工厂", "单位", "师傅", "同事", "学徒", "当兵",
                  "下乡", "知青", "种地", "手艺", "工资", "退休"],
        min_turns=2,
        max_turns=6,
        priority=70,
    ),
    Topic(
        id="love_marriage",
        name="爱情与婚姻",
        description="相识、恋爱、结婚、婚后生活。",
        opening_questions=[
            "您和老伴是怎么认识的？",
            "结婚那天您还记得吗？当时是什么场面？",
            "婚后两个人是怎么过日子的？",
        ],
        followup_angles=["初识经过", "定情信物", "婚礼场面", "婚后磨合", "最恩爱的时刻"],
        keywords=["老伴", "媳妇", "丈夫", "妻子", "对象", "相亲", "介绍", "结婚",
                  "婚礼", "嫁", "娶", "彩礼", "洞房", "恋爱"],
        min_turns=2,
        max_turns=6,
        priority=60,
    ),
    Topic(
        id="family_children",
        name="家庭与子女",
        description="孩子出生、养育、家风、儿孙趣事。",
        opening_questions=[
            "孩子出生时您还记得吗？当时高兴成啥样？",
            "带孩子最操心的是什么事？",
            "孩子们小时候，谁最调皮？",
        ],
        followup_angles=["孩子出生", "养育艰辛", "儿孙趣事", "家风家训", "团圆时刻"],
        keywords=["孩子", "儿子", "女儿", "孙子", "孙女", "出生", "满月", "带孩子",
                  "上学", "成家", "团圆", "家风"],
        min_turns=2,
        max_turns=6,
        priority=50,
    ),
    Topic(
        id="hard_times",
        name="艰难岁月",
        description="饥荒、动荡、病痛、离别，以及如何熬过来。敏感话题，只跟随不主动深挖。",
        opening_questions=[
            "那些年最难的时候，您是怎么熬过来的？",
            "如果愿意讲讲，那段日子里最让您挂念的是什么？要是不想提，咱们就跳过，没关系。",
            "走过那么难的日子，回头看，是什么撑着您走过来的？",
        ],
        followup_angles=["如何熬过来", "互相扶持的人", "走出低谷的转折", "对苦难的看法"],
        keywords=["饥荒", "挨饿", "灾", "动乱", "打仗", "生病", "住院", "去世",
                  "离别", "下岗", "受苦", "熬", "难"],
        min_turns=1,
        max_turns=4,
        priority=25,
        sensitive=True,
    ),
    Topic(
        id="craft_hobby",
        name="手艺与爱好",
        description="拿手本事、业余爱好、绝活与消遣。",
        opening_questions=[
            "您有什么拿手的本事，或者喜欢的消遣？",
            "您这门手艺是跟谁学的？学了多久？",
            "闲下来的时候，您最喜欢干点啥？",
        ],
        followup_angles=["手艺来历", "学艺过程", "得意之作", "爱好趣事", "是否传给后人"],
        keywords=["手艺", "木匠", "裁缝", "厨艺", "做饭", "种菜", "钓鱼", "下棋",
                  "唱戏", "拉二胡", "书法", "绣花", "爱好", "绝活"],
        min_turns=1,
        max_turns=5,
        priority=40,
    ),
    Topic(
        id="wisdom",
        name="人生感悟",
        description="最想告诉子孙的话、一生的总结。",
        opening_questions=[
            "如果跟孙辈说一句最想说的话，您会说什么？",
            "活了这一辈子，您觉得人最要紧的是啥？",
            "有没有哪句老话，是您一直记在心里的？",
        ],
        followup_angles=["家训", "人生信条", "对子孙的期望", "最后悔/最欣慰的事"],
        keywords=["道理", "老话", "俗话", "做人", "善良", "勤快", "知足", "感恩",
                  "希望", "嘱咐", "交代"],
        min_turns=1,
        max_turns=4,
        priority=10,
    ),
]

TOPIC_MAP: dict[str, Topic] = {t.id: t for t in TOPICS}


def get_topic(topic_id: str) -> Topic:
    return TOPIC_MAP[topic_id]


class TopicPlanner:
    """话题规划器：决定"下一个话题去哪"，支持搭桥切换。"""

    def __init__(self, topics: list[Topic] | None = None):
        self.topics = topics or TOPICS
        self.by_id = {t.id: t for t in self.topics}

    def uncovered(self, state: InterviewState) -> list[Topic]:
        covered = set(state.covered_topic_ids) | {state.current_topic_id}
        return [t for t in self.topics if t.id not in covered]

    def _sensitive_allowed(self, topic: Topic, state: InterviewState) -> bool:
        if not topic.sensitive:
            return True
        # 敏感话题：访谈轮次达标（建立信任）后才允许
        return state.turn_count >= state.config.sensitive_min_turns

    def bridge_scores(self, state: InterviewState, recent_entities: list[str]) -> dict[str, float]:
        """计算每个未覆盖话题的"搭桥分"：老人刚提到的实体命中话题关键词则加分。"""
        scores: dict[str, float] = {}
        blob = "".join(recent_entities)
        for t in self.uncovered(state):
            if not self._sensitive_allowed(t, state):
                continue
            hits = sum(1 for kw in t.keywords if kw and kw in blob)
            scores[t.id] = hits
        return scores

    def next_topic(
        self, state: InterviewState, recent_entities: list[str] | None = None
    ) -> tuple[Topic | None, str]:
        """返回 (新话题, 搭桥词)。无话题可切时返回 (None, "")。

        优先级：搭桥命中 > priority 顺序。敏感话题默认沉底。
        """
        candidates = [t for t in self.uncovered(state) if self._sensitive_allowed(t, state)]
        if not candidates:
            return None, ""
        bridge = ""
        if recent_entities:
            scores = self.bridge_scores(state, recent_entities)
            best = max(scores.items(), key=lambda kv: kv[1], default=(None, 0))
            if best[0] and best[1] > 0:
                topic = self.by_id[best[0]]
                # 找出命中的关键词作为搭桥词
                blob = "".join(recent_entities)
                for kw in topic.keywords:
                    if kw and kw in blob:
                        bridge = kw
                        break
                return topic, bridge
        # 无搭桥：按 priority 从高到低取第一个
        candidates.sort(key=lambda t: t.priority, reverse=True)
        return candidates[0], bridge

    def opening_question(self, topic: Topic, asked_count: int = 0) -> str:
        """按轮换取开场问题，避免重复。"""
        if not topic.opening_questions:
            return f"咱们聊聊{topic.name}吧？"
        return topic.opening_questions[asked_count % len(topic.opening_questions)]
