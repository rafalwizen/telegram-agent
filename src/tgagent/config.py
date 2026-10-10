"""Environment-based configuration with fail-fast validation."""

from __future__ import annotations

import os
from pathlib import Path


class ConfigError(Exception):
    """Raised when required environment variables are missing or invalid."""


class Config:
    """Typed, immutable runtime configuration loaded from the environment."""

    def __init__(
        self,
        *,
        telegram_token: str,
        allowed_ids: frozenset[int],
        glm_api_key: str,
        glm_base_url: str,
        glm_model: str,
        glm_vision_model: str,
        glm_asr_model: str,
        workspace_dir: Path,
        data_dir: Path,
        max_tool_iters: int,
        tool_result_limit: int,
        context_token_budget: int,
        py_timeout_s: int,
        py_mem_mb: int,
        max_voice_min: int,
        log_level: str,
    ) -> None:
        self.telegram_token = telegram_token
        self.allowed_ids = allowed_ids
        self.glm_api_key = glm_api_key
        self.glm_base_url = glm_base_url
        self.glm_model = glm_model
        self.glm_vision_model = glm_vision_model
        self.glm_asr_model = glm_asr_model
        self.workspace_dir = workspace_dir
        self.data_dir = data_dir
        self.max_tool_iters = max_tool_iters
        self.tool_result_limit = tool_result_limit
        self.context_token_budget = context_token_budget
        self.py_timeout_s = py_timeout_s
        self.py_mem_mb = py_mem_mb
        self.max_voice_min = max_voice_min
        self.log_level = log_level


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got: {value}")
    return value


def _parse_ids(raw: str) -> frozenset[int]:
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.lstrip("-").isdigit():
            raise ConfigError(f"ALLOWED_TELEGRAM_IDS must be integers, got: {part!r}")
        ids.add(int(part))
    if not ids:
        raise ConfigError("ALLOWED_TELEGRAM_IDS must contain at least one Telegram user id")
    return frozenset(ids)


def _path_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def load_config() -> Config:
    """Read the environment and return a validated Config (raises ConfigError)."""
    return Config(
        telegram_token=_require("TELEGRAM_BOT_TOKEN"),
        allowed_ids=_parse_ids(_require("ALLOWED_TELEGRAM_IDS")),
        glm_api_key=_require("GLM_API_KEY"),
        glm_base_url=os.environ.get("GLM_BASE_URL", "").strip() or "https://api.z.ai/api/paas/v4",
        glm_model=os.environ.get("GLM_MODEL", "").strip() or "glm-4.6",
        glm_vision_model=os.environ.get("GLM_VISION_MODEL", "").strip() or "glm-5.3-flash",
        glm_asr_model=os.environ.get("GLM_ASR_MODEL", "").strip() or "glm-asr-2512",
        workspace_dir=_path_env("WORKSPACE_DIR", "data/workspace"),
        data_dir=_path_env("DATA_DIR", "data/bot"),
        max_tool_iters=_int_env("MAX_TOOL_ITERS", 8),
        tool_result_limit=_int_env("TOOL_RESULT_LIMIT", 8000),
        context_token_budget=_int_env("CONTEXT_TOKEN_BUDGET", 32000),
        py_timeout_s=_int_env("PY_TIMEOUT", 60),
        py_mem_mb=_int_env("PY_MEM_MB", 1536),
        max_voice_min=_int_env("MAX_VOICE_MINUTES", 5),
        log_level=os.environ.get("LOG_LEVEL", "INFO").strip().upper(),
    )
