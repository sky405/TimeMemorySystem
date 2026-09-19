"""LLM 抽象层（纯标准库，零第三方依赖）。

- LLMClient: 抽象接口，只需实现 chat(system, user) -> str。
- MockLLMClient: 离线默认，确定性模板回复，用于测试与演示。
- OpenAICompatibleClient: 用 urllib 实现 OpenAI Chat Completions 协议，
  兼容 OpenAI / DeepSeek / 通义千问 / 月之暗面 / 本地 Ollama 等。
- get_default_client(): 有 TMS_LLM_API_KEY 就用真模型，否则降级 Mock。

环境变量：
    TMS_LLM_BASE_URL  默认 https://api.openai.com/v1
    TMS_LLM_API_KEY   为空则使用 Mock
    TMS_LLM_MODEL     默认 gpt-4o-mini
    TMS_LLM_TIMEOUT   默认 30（秒）
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class LLMClient:
    """LLM 客户端抽象接口。"""

    name: str = "base"

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """离线 Mock：按意图返回确定性回复，保证无 Key 也能跑通全链路。"""

    name: str = "mock"

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        low = (system + "\n" + user).lower()
        if "judge" in low or "三叉路口" in user or "decision" in low:
            return self._mock_judge(user)
        if "extract" in low or "记忆片段" in user and "json" in low:
            return '{"fragments": []}'
        # 提问润色：原样返回最后一行问句
        for line in reversed(user.splitlines()):
            line = line.strip()
            if line.endswith(("?", "？")):
                return line
        return user.strip().splitlines()[-1] if user.strip() else "您能再讲讲吗？"

    def _mock_judge(self, user: str) -> str:
        """Mock 裁判：用关键词做一个粗糙但可用的裁决，保证 JSON 链路可测。"""
        # 只扫描"老人本轮回答"片段，避免误命中提示词模板里的字（如"累计"含"累"）
        text = user
        if "【老人本轮回答】" in user:
            text = user.split("【老人本轮回答】", 1)[1].split("【规则引擎", 1)[0]
        text = text.strip()
        # 与 decision.is_exit_signal 同策略：ALWAYS 词直接命中，SHORT 词仅短句命中
        # （不用单字"困"，否则误伤"困难"；"走了/再见"在 Mock 层不判定，交给规则层；
        #  Mock 置信度封顶 0.75，避免单独触发"LLM 高置信收尾直通"）
        tired_always = ["累", "困了", "发困", "不聊", "别问", "先睡", "想睡",
                        "吃饭了", "睡觉了", "休息了", "先这样"]
        tired_short = ["下次", "改天", "改日", "下回", "歇", "休息", "做饭", "吃饭", "睡觉"]
        vague_words = ["不知道", "不记得", "记不清", "没什么", "就那样"]
        # "忘了/忘记了"只认句尾或标点前（"早忘了"算，"忘了吃"不算）
        forgot_alone = text.rstrip("，。！？；、 \t").endswith(("忘了", "忘记了")) or any(
            p in text for p in ["忘了，", "忘了。", "忘了！", "忘了？", "忘了；", "忘记了，", "忘记了。"]
        )
        negative = "改天换地" in text
        is_tired = any(w in text for w in tired_always) or (
            not negative and len(text) <= 10 and any(w in text for w in tired_short)
        )
        vague_hit = any(w in text for w in vague_words) or forgot_alone
        if is_tired:
            action, conf, reason = "wrap_up", 0.75, "Mock: 检测到疲惫/告别信号"
        elif vague_hit:
            action, conf, reason = "switch_topic", 0.7, "Mock: 回答模糊，话题可能已枯竭"
        else:
            action, conf, reason = "deep_dive", 0.6, "Mock: 默认继续深挖"
        return json.dumps(
            {"action": action, "confidence": conf, "reasoning": reason, "focus": ""},
            ensure_ascii=False,
        )


class OpenAICompatibleClient(LLMClient):
    """OpenAI Chat Completions 协议客户端（标准库 urllib 实现）。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self.api_key = api_key or os.getenv("TMS_LLM_API_KEY", "")
        self.base_url = (base_url or os.getenv("TMS_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("TMS_LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout or int(os.getenv("TMS_LLM_TIMEOUT", "30"))
        self.name = f"openai-compatible:{self.model}"
        if not self.api_key:
            raise LLMError("缺少 TMS_LLM_API_KEY，无法创建 OpenAICompatibleClient")

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:500]
            raise LLMError(f"LLM HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise LLMError(f"LLM 调用失败: {e}") from e
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise LLMError(f"LLM 返回格式异常: {str(data)[:300]}") from e


def get_default_client() -> LLMClient:
    """有 Key 用真模型，无 Key 降级 Mock（并打印一行提示）。"""
    if os.getenv("TMS_LLM_API_KEY"):
        try:
            client = OpenAICompatibleClient()
            print(f"[TimeMemory] 使用 LLM: {client.name} ({client.base_url})")
            return client
        except LLMError as e:
            print(f"[TimeMemory] 真模型不可用，回退 Mock: {e}")
    else:
        print("[TimeMemory] 未检测到 TMS_LLM_API_KEY，使用 MockLLM（规则兜底全链路可用）")
    return MockLLMClient()


def extract_json_object(text: str) -> dict | None:
    """从 LLM 回复中提取第一个 JSON 对象，失败返回 None。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
