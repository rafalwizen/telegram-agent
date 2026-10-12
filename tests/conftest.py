"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tgagent.config import Config


def make_config(tmp_path: Path, **overrides: object) -> Config:
    """Build a Config with dummy values suitable for unit tests."""
    defaults: dict[str, object] = {
        "telegram_token": "123:test",
        "allowed_ids": frozenset({1}),
        "glm_api_key": "key",
        "glm_base_url": "https://example.invalid/v4",
        "glm_model": "test-model",
        "glm_vision_model": "test-vision-model",
        "glm_asr_model": "test-asr-model",
        "workspace_dir": tmp_path / "ws",
        "data_dir": tmp_path / "data",
        "max_tool_iters": 2,
        "tool_result_limit": 8000,
        "context_token_budget": 1000,
        "py_timeout_s": 5,
        "py_mem_mb": 256,
        "max_voice_min": 1,
        "log_level": "INFO",
    }
    defaults.update(overrides)
    return Config(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def make_cfg(tmp_path: Path) -> Callable[..., Config]:
    """Factory for test Config objects rooted at tmp_path (workspace pre-created)."""

    def _factory(**overrides: object) -> Config:
        cfg = make_config(tmp_path, **overrides)
        cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    return _factory


@pytest.fixture
def config(make_cfg: Callable[..., Config]) -> Config:
    return make_cfg()
