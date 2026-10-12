"""Thin async wrapper over the GLM APIs used by the bot.

Both endpoints are covered by the GLM Coding Plan:
- OpenAI-compatible chat/completions — text conversation and tool calling
- Anthropic-compatible messages — image-capable turns (no tool loop)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from .config import Config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
ANTHROPIC_VERSION = "2023-06-01"

# Error fragments meaning "plan window exhausted / no balance": retrying is pointless
# (the 5-hour window resets on its own), so we fail fast with a clear message instead.
_QUOTA_PATTERNS = ("1113", "insufficient balance", "recharge", "quota", "usage limit", "window")


class GLMError(Exception):
    """Raised when a GLM API call fails (after retries, if retryable)."""


def is_quota_error(text: str) -> bool:
    """True when an error text indicates a billing/quota problem, not a transient one."""
    lowered = text.lower()
    return any(pattern in lowered for pattern in _QUOTA_PATTERNS)


def _retryable_http(exc: Exception) -> bool:
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
                if not _retryable_http(exc) or is_quota_error(str(exc)) or attempt == MAX_RETRIES:
                    raise GLMError(str(exc)) from exc
                await self._sleep_retry(attempt, exc)
        raise GLMError(str(last_exc))  # pragma: no cover - for type checkers

    async def vision_chat(
        self, system: str, messages: list[dict[str, Any]], image_data_urls: list[str]
    ) -> str:
        """Single image-capable completion via the Anthropic-compatible endpoint.

        No tool loop: image turns get a plain answer; tools can be used on the
        follow-up text turn. History stays text-only — images are attached to
        the current (last user) turn only.
        """
        payload_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        if not payload_messages or payload_messages[-1]["role"] != "user":
            payload_messages.append({"role": "user", "content": ""})
        last = payload_messages[-1]
        text = last["content"] if isinstance(last["content"], str) else ""
        blocks = [_image_block(url) for url in image_data_urls]
        blocks.append({"type": "text", "text": text or "Describe this image."})
        last["content"] = blocks

        body = {
            "model": self._cfg.glm_model,
            "max_tokens": 4000,
            "system": system,
            "messages": payload_messages,
        }
        url = f"{self._cfg.glm_anthropic_base_url.rstrip('/')}/v1/messages"
        headers = {
            "Authorization": f"Bearer {self._cfg.glm_api_key}",
            "anthropic-version": ANTHROPIC_VERSION,
        }

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(url, json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt == MAX_RETRIES:
                    raise GLMError(str(exc)) from exc
                await self._sleep_retry(attempt, exc)
                continue
            if response.status_code == 200:
                data = response.json()
                texts = [
                    block.get("text", "")
                    for block in data.get("content", [])
                    if block.get("type") == "text"
                ]
                return "\n".join(t for t in texts if t).strip()
            error_text = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code in RETRYABLE_STATUS and not is_quota_error(response.text):
                if attempt == MAX_RETRIES:
                    raise GLMError(error_text)
                await self._sleep_retry(attempt, GLMError(error_text))
                continue
            raise GLMError(error_text)
        raise GLMError(str(last_exc))  # pragma: no cover - for type checkers

    async def _sleep_retry(self, attempt: int, exc: Exception) -> None:
        delay = 2.0**attempt
        logger.warning(
            "GLM attempt %d/%d failed (%s); retrying in %.0fs", attempt, MAX_RETRIES, exc, delay
        )
        await asyncio.sleep(delay)


def _image_block(data_url: str) -> dict[str, Any]:
    """Convert 'data:image/jpeg;base64,xxx' into an Anthropic image content block."""
    header, _, data = data_url.partition(",")
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }
