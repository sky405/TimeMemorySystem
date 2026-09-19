"""统一配置中心：timememory.yaml + 环境变量 + 默认值。

优先级：显式传参 > 环境变量 > YAML > 默认值。
YAML 只放非敏感的结构配置；API Key 等密钥建议走环境变量
（环境变量永远覆盖 YAML 同名项）。

配置文件查找顺序：显式 path > configure() > TMS_CONFIG > ./timememory.yaml。
找不到文件不报错，直接用默认值 + 环境变量。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None  # 空则各阶段用自己的默认模型


class EmbeddingConfig(BaseModel):
    base_url: str | None = None  # 空则复用 llm.base_url
    api_key: str | None = None
    model: str = "text-embedding-3-small"


class MySQLConfig(BaseModel):
    url: str | None = None
    host: str | None = None
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "timememory"


class SQLiteConfig(BaseModel):
    path: str = "data/material.db"


class PhaseTuning(BaseModel):
    temperature: float | None = None  # 空则各阶段用原来的默认值


class LibraryTuning(BaseModel):
    temperature: float | None = None
    threshold: float = 0.15  # 检索门控：低于此分拒答
    top_k: int = 4


class OrchestrationTuning(BaseModel):
    max_rounds: int = 3


class TimememoryConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    mysql: MySQLConfig = Field(default_factory=MySQLConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    interview: PhaseTuning = Field(default_factory=PhaseTuning)
    material: PhaseTuning = Field(default_factory=PhaseTuning)
    assessment: PhaseTuning = Field(default_factory=PhaseTuning)
    drafting: PhaseTuning = Field(default_factory=PhaseTuning)
    review: PhaseTuning = Field(default_factory=PhaseTuning)
    library: LibraryTuning = Field(default_factory=LibraryTuning)
    orchestration: OrchestrationTuning = Field(default_factory=OrchestrationTuning)


_CONFIG_PATH: str | None = None


def configure(path: str | Path | None) -> None:
    """指定配置文件（测试或多环境切换用）；传 None 清除。"""
    global _CONFIG_PATH
    _CONFIG_PATH = str(path) if path else None


def _resolve_path(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在：{explicit}")
        return p
    for cand in (_CONFIG_PATH, os.environ.get("TMS_CONFIG"), "timememory.yaml"):
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _apply_env(cfg: TimememoryConfig) -> TimememoryConfig:
    e = os.environ.get
    if v := e("TMS_LLM_BASE_URL"):
        cfg.llm.base_url = v
    if v := e("TMS_LLM_API_KEY"):
        cfg.llm.api_key = v
    if v := e("TMS_LLM_MODEL"):
        cfg.llm.model = v
    if v := e("TMS_EMB_BASE_URL"):
        cfg.embedding.base_url = v
    if v := e("TMS_EMB_API_KEY"):
        cfg.embedding.api_key = v
    if v := e("TMS_EMB_MODEL"):
        cfg.embedding.model = v
    if v := e("TMS_MYSQL_URL"):
        cfg.mysql.url = v
    if v := e("TMS_MYSQL_HOST"):
        cfg.mysql.host = v
    if v := e("TMS_MYSQL_PORT"):
        try:
            cfg.mysql.port = int(v)
        except ValueError:
            pass
    if v := e("TMS_MYSQL_USER"):
        cfg.mysql.user = v
    if (v := e("TMS_MYSQL_PASSWORD")) is not None:
        cfg.mysql.password = v
    if v := e("TMS_MYSQL_DB"):
        cfg.mysql.database = v
    if v := e("TMS_SQLITE_PATH"):
        cfg.sqlite.path = v
    return cfg


def load_config(path: str | Path | None = None) -> TimememoryConfig:
    """加载配置（YAML + 环境变量覆盖）。每次调用都重新读，保证环境变量实时生效。"""
    data: dict[str, Any] = {}
    p = _resolve_path(path)
    if p:
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"配置文件 {p} 解析失败：{exc}")
        if not isinstance(data, dict):
            raise ValueError(f"配置文件 {p} 顶层必须是 mapping。")
    return _apply_env(TimememoryConfig(**data))


def get_config() -> TimememoryConfig:
    """取当前配置（load_config 的快捷方式）。"""
    return load_config()
