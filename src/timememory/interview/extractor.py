"""记忆片段提取器：规则打底 + LLM 精修（可选）。

每轮老人回答 → 切分为 1~N 个 MemoryFragment。
规则层零依赖、确定性、可测试；LLM 精修只做"修正与补全"，失败即回退规则结果。
"""
from __future__ import annotations

import itertools
import re

from .llm import LLMClient, LLMError, extract_json_object
from .models import InterviewState, MemoryFragment
from .prompts import SYSTEM_EXTRACTOR, build_extract_prompt

# -- 正则与词表 ---------------------------------------------------------------

TIME_RE = re.compile(
    r"(\d{4}年|\d{1,2}岁|[一二三四五六七八九十\d两几]{1,4}岁|"
    r"[一二三四五六七八九十\d两几岁小多年那当春夏秋冬正]{1,6}的时候|"
    r"上世纪\d{2}年代|\d{2}年代|几年前|去年|今年|以前|以后|"
    r"小时候|童年|少年|青年|年轻时|那年|那时候|当年|后来|解放前|解放后|"
    r"三年困难时期|七十年代|八十年代|六十年代|五十年代|正月|除夕|过年时|"
    r"春天|夏天|秋天|冬天|春节|端午|中秋|清明)"
)
PLACE_RE = re.compile(r"[\u4e00-\u9fa5]{1,8}(?:省|市|县|区|镇|乡|村|屯|寨|江|河|湖|海|山|岭|街|巷|路|桥|庙|祠堂|小学|中学|学校|工厂)")
PERSON_WORDS = [
    "爸爸", "父亲", "爹", "妈妈", "母亲", "娘", "爷爷", "奶奶", "外公", "外婆",
    "哥哥", "弟弟", "姐姐", "妹妹", "二哥", "大哥", "大姐",
    "伯伯", "叔叔", "婶婶", "姑姑", "舅舅",
    "老伴", "媳妇", "丈夫", "妻子", "对象", "儿子", "女儿", "孙子", "孙女",
    "老师", "先生", "同学", "同桌", "邻居", "乡亲", "师傅", "同事", "战友", "朋友",
    "村长", "书记", "队长", "干部",
    "哥", "姐",
]
# 姓氏 + 排行/称谓："王二哥""李大姐"。姓氏含"江"等字，但必须后接称谓才算人，不会误伤江河。
_SURNAMES = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高罗郑梁谢宋唐许韩冯董萧程曹袁邓傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文欧"
_RELATION_NAME_RE = re.compile(f"[{_SURNAMES}][一二三四五六七八九大小老]?[哥姐弟妹叔伯姨姑舅]")
EMOTION_MAP = {
    "开心": ["开心", "高兴", "快乐", "欢喜", "乐呵", "美滋滋", "幸福"],
    "怀念": ["怀念", "想念", "记得", "忘不了", "一辈子", "老惦记"],
    "难过": ["难过", "伤心", "哭", "流泪", "心酸", "可怜", "苦", "穷"],
    "害怕": ["害怕", "怕", "吓", "提心吊胆"],
    "骄傲": ["骄傲", "自豪", "光荣", "争气", "了不起"],
    "愤怒": ["生气", "愤怒", "气", "冤", "不公"],
}
NARRATIVE_MARKERS = ["后来", "然后", "结果", "没想到", "有一次", "最难忘", "记得有",
                     "那次", "竟然", "幸亏", "多亏", "要不是"]
NOISE_UTTERANCES = {"嗯", "哦", "啊", "好的", "好", "是", "对", "行", "知道了", "呵呵"}

# 地点清洗：介词/动词/代词边界。贪婪匹配常把前面的成分带进来
# （如"跟着父亲从合川县"），按"最后一个边界字"切断，只保留真正的地名。
# 注意：上/下/里/中 不在其中，否则会切坏"中山路"等地名。
_PLACE_BOUNDARIES = "我在到去从回跟和与把被叫让给比着了过是有你他她它们这那就便住往朝向将"

# "地点后缀"误匹配保护：命中后缀但不是地点的词。
# （"全州县"这类真地名罕见，用停用词而不用边界字"全"，避免切坏它。）
_PLACE_STOPWORDS = {"现在", "正在", "在那", "在这", "时候", "时间", "学校里",
                    "全村", "全国", "全球", "全市", "全省", "全县", "全队", "全家"}

# 空回答过滤（与 decision.VAGUE_PATTERNS 同步）：无实质内容的纯模糊句不成片段。
_FRAGMENT_VAGUE = ["不知道", "不清楚", "不记得", "记不清", "记不得", "想不起来",
                   "没什么", "没啥", "就那样", "一般", "还行"]
# 含"忘"的表达只认句尾："早忘了/都忘了"算空（_FORGOT_ALONE_RE 覆盖），"忘了吃/带"不算。
_FORGOT_ALONE_RE = re.compile(r"忘(?:记)?了($|[，。！？；、\s])")


def _clean_place(match: str) -> str:
    """清洗地点匹配，只保留最后一个边界字之后的地名。

    "跟着父亲从合川县" → "合川县"；"我在四川嘉陵江" → "四川嘉陵江"。
    """
    cut = -1
    for i, ch in enumerate(match):
        if ch in _PLACE_BOUNDARIES:
            cut = i
    cleaned = match[cut + 1:]
    if len(cleaned) < 2 or cleaned in _PLACE_STOPWORDS:
        return ""
    return cleaned


def _drop_subsumed(refs: list[str]) -> list[str]:
    """去掉被更长结果包含的短结果（"二哥" vs "王二哥" → 留"王二哥"）。"""
    return sorted({r for r in refs if r and not any(r != o and r in o for o in refs)})


def _scan_refs(sent: str) -> tuple[list[str], list[str], list[str]]:
    """一次扫描出时间/地点/人物三类表达。"""
    time_refs = _drop_subsumed(TIME_RE.findall(sent))
    # 地点扫描前先剔除含"后缀字"的成语，避免"人山人海"被当成地名
    scan = sent
    for idiom in ("人山人海", "山珍海味", "天涯海角", "五湖四海", "海阔天空", "山清水秀"):
        scan = scan.replace(idiom, "")
    place_refs = _drop_subsumed([_clean_place(m) for m in PLACE_RE.findall(scan)])
    persons = [w for w in PERSON_WORDS if w in sent]
    persons += _RELATION_NAME_RE.findall(sent)
    person_refs = _drop_subsumed(persons)
    return time_refs, place_refs, person_refs


def _clause_has_vague(clause: str) -> bool:
    if any(p in clause for p in _FRAGMENT_VAGUE):
        return True
    return bool(_FORGOT_ALONE_RE.search(clause))


def detect_emotion(text: str) -> str:
    for emo, words in EMOTION_MAP.items():
        if any(w in text for w in words):
            return emo
    return ""


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？；…]+", text)
    return [p.strip() for p in parts if p.strip()]


_counter = itertools.count(1)


def _new_id() -> str:
    return f"frag-{next(_counter):04d}"


class FragmentExtractor:
    """记忆片段提取器。"""

    def extract(
        self,
        text: str,
        topic_id: str,
        turn_index: int,
        llm: LLMClient | None = None,
    ) -> list[MemoryFragment]:
        text = (text or "").strip()
        if not text or text in NOISE_UTTERANCES:
            return []
        fragments = self._rule_extract(text, topic_id, turn_index)
        if llm is not None and fragments:
            try:
                refined = self._llm_refine(text, topic_id, turn_index, fragments, llm)
                if refined:
                    fragments = refined
            except LLMError:
                pass  # LLM 失败 → 用规则结果
        return fragments

    # -- 规则层 ---------------------------------------------------------------
    def _rule_extract(self, text: str, topic_id: str, turn: int) -> list[MemoryFragment]:
        sentences = split_sentences(text)
        # 短回答整体作为一个片段，避免碎片化
        if len("".join(sentences)) <= 40 or len(sentences) <= 1:
            sentences = [text] if text else []
        fragments: list[MemoryFragment] = []
        for sent in sentences:
            if len(sent) < 2 or sent in NOISE_UTTERANCES:
                continue
            # 空回答过滤：所有子句都无实质内容（模糊且无地点/人物/情节/情感）→ 不成片段。
            # 例："记不清了"被过滤；"小时候家里很穷，别的记不清了"被保留（"穷"有情感分量）。
            # 注意只看地点/人物/情节/情感：纯时间（"很久以前"）撑不起一条记忆。
            clauses = split_sentences(sent)
            if all(self._clause_is_empty(c) for c in clauses):
                continue
            time_refs, place_refs, person_refs = _scan_refs(sent)
            emotion = detect_emotion(sent)
            entities = time_refs + place_refs + person_refs
            has_plot = any(m in sent for m in NARRATIVE_MARKERS)

            importance = 2
            if place_refs or person_refs:
                importance += 1
            if time_refs:
                importance += 1
            if has_plot or len(sent) >= 25:
                importance += 1
            importance = max(1, min(5, importance))

            needs_followup = bool((place_refs or person_refs or has_plot) and len(sent) >= 8)
            hint = ""
            if needs_followup:
                focus = (place_refs + person_refs + time_refs)[:2]
                hint = f"{'、'.join(focus)}：可追问具体经过/难忘细节" if focus else "可追问具体经过"

            fragments.append(
                MemoryFragment(
                    id=_new_id(),
                    content=sent,
                    topic_id=topic_id,
                    source_turn=turn,
                    time_refs=time_refs,
                    place_refs=place_refs,
                    person_refs=person_refs,
                    emotion=emotion,
                    importance=importance,
                    entities=entities,
                    needs_followup=needs_followup,
                    followup_hint=hint,
                )
            )
        return fragments

    @staticmethod
    def _clause_is_empty(clause: str) -> bool:
        """子句是否无实质内容：模糊表达 + 无地点/人物/情节/情感 + 短句。"""
        if not _clause_has_vague(clause):
            return False
        _, place_refs, person_refs = _scan_refs(clause)
        if place_refs or person_refs:
            return False
        if any(m in clause for m in NARRATIVE_MARKERS):
            return False
        if detect_emotion(clause):
            return False
        return len(clause) < 20

    # -- LLM 精修层 ------------------------------------------------------------
    def _llm_refine(
        self,
        text: str,
        topic_id: str,
        turn: int,
        base: list[MemoryFragment],
        llm: LLMClient,
    ) -> list[MemoryFragment]:
        from .topics import TOPIC_MAP  # 延迟导入，避免循环

        topic_name = TOPIC_MAP.get(topic_id).name if topic_id in TOPIC_MAP else topic_id
        raw = llm.chat(SYSTEM_EXTRACTOR, build_extract_prompt(topic_name, text))
        obj = extract_json_object(raw)
        if not obj or not isinstance(obj.get("fragments"), list):
            return []
        refined: list[MemoryFragment] = []
        for i, item in enumerate(obj["fragments"]):
            if not isinstance(item, dict) or not item.get("content"):
                continue
            old = base[min(i, len(base) - 1)]
            refined.append(
                MemoryFragment(
                    id=old.id,
                    content=str(item.get("content", old.content)),
                    topic_id=topic_id,
                    source_turn=turn,
                    time_refs=list(item.get("time_refs") or old.time_refs),
                    place_refs=list(item.get("place_refs") or old.place_refs),
                    person_refs=list(item.get("person_refs") or old.person_refs),
                    emotion=str(item.get("emotion") or old.emotion),
                    importance=int(item.get("importance") or old.importance),
                    entities=list(
                        (item.get("time_refs") or [])
                        + (item.get("place_refs") or [])
                        + (item.get("person_refs") or [])
                    )
                    or old.entities,
                    needs_followup=bool(item.get("needs_followup", old.needs_followup)),
                    followup_hint=str(item.get("followup_hint") or old.followup_hint),
                )
            )
        return refined or base


def recent_entities(state: InterviewState, last_n_turns: int = 2) -> list[str]:
    """取最近 N 轮片段中的实体（供搭桥/追问使用）。"""
    out: list[str] = []
    cutoff = state.turn_count - last_n_turns
    for f in state.fragments:
        if f.source_turn > cutoff:
            out.extend(f.entities)
    # 去重保序
    seen, deduped = set(), []
    for e in out:
        if e and e not in seen:
            seen.add(e)
            deduped.append(e)
    return deduped
