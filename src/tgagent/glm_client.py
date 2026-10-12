"""Thin async wrapper over the OpenAI-compatible GLM APIs (chat + audio transcription)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI

from .config import Config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class GLMError(Exception):
    """Raised when a GLM API call fails (after retries, if retryable)."""


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in RETRYABLE_STATUS


class GLMClient:
    """All GLM traffic goes through this class."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._client = AsyncOpenAI(
            api_key=cfg.glm_api_key,
            base_url=cfg.glm_base_url,
            timeout=180.0,
            max_retries=0,  # retries handled here with readable logging
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4000,
    ) -> Any:
        """Call chat/completions and return the assistant message object."""
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=model or self._cfg.glm_model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    # GLM-specific: disable thinking where supported (ignored otherwise).
                    extra_body={"thinking": {"type": "disabled"}},
                )
                return response.choices[0].message
            except Exception as exc:
                last_exc = exc
                if not _retryable(exc) or attempt == MAX_RETRIES:
                    raise GLMError(str(exc)) from exc
                await self._sleep_retry(attempt, exc)
        raise GLMError(str(last_exc))  # pragma: no cover - for type checkers

    async def transcribe(self, audio_path: Path) -> str:
        """Transcribe an MP3/WAV file via the audio/transcriptions endpoint."""
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with audio_path.open("rb") as fh:
                    response = await self._client.audio.transcriptions.create(
                        model=self._cfg.glm_asr_model,
                        file=fh,
                    )
                return (response.text or "").strip()
            except Exception as exc:
                last_exc = exc
                if not _retryable(exc) or attempt == MAX_RETRIES:
                    raise GLMError(str(exc)) from exc
                await self._sleep_retry(attempt, exc)
        raise GLMError(str(last_exc))  # pragma: no cover - for type checkers

    async def _sleep_retry(self, attempt: int, exc: Exception) -> None:
        delay = 2.0**attempt
        logger.warning(
            "GLM attempt %d/%d failed (%s); retrying in %.0fs", attempt, MAX_RETRIES, exc, delay
        )
        await asyncio.sleep(delay)
