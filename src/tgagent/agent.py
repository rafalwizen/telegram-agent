"""Agent turn handling: GLM completion with a tool-calling loop and image routing."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from .config import Config
from .glm_client import GLMClient, GLMError, is_quota_error
from .history import History
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SPECS, ToolContext
from .tools import dispatch as dispatch_tool
from .vision import prepare_image

logger = logging.getLogger(__name__)

# Reports (iteration, tool_name, args_summary) while a turn is running.
ProgressFn = Callable[[int, str, str], Awaitable[None]]


class AgentError(Exception):
    """User-facing error, shown to the user as the result of the turn."""


def friendly_glm_error(exc: GLMError) -> str | None:
    """Map known GLM billing/quota errors to a clear, human explanation."""
    text = str(exc).lower()
    if "1113" in text or "insufficient balance" in text or "recharge" in text:
        return (
            "GLM account has no pay-as-you-go balance for this feature. "
            "Chat and tools run on the Coding Plan; this feature would need standard API balance."
        )
    if is_quota_error(text):
        return (
            "GLM Coding Plan usage limit reached for the current 5-hour window — "
            "please try again after the window resets."
        )
    return None


class Agent:
    def __init__(self, cfg: Config, glm: GLMClient, history: History) -> None:
        self._cfg = cfg
        self._glm = glm
        self._history = history

    async def handle_turn(
        self,
        chat_id: int,
        user_text: str,
        image_paths: list[Path] | None = None,
        on_progress: ProgressFn | None = None,
    ) -> str:
        """Process one user turn; returns the final answer text for Telegram."""
        self._history.append(chat_id, "user", user_text)
        messages = self._history.build_context(
            chat_id, SYSTEM_PROMPT, self._cfg.context_token_budget
        )

        data_urls = [url for url in (prepare_image(p) for p in (image_paths or [])) if url]
        if data_urls:
            # Image turn: single completion on the Anthropic-compatible endpoint,
            # no tool loop (tools are used on follow-up text turns).
            try:
                final = await self._glm.vision_chat(SYSTEM_PROMPT, messages, data_urls)
            except GLMError as exc:
                logger.error("GLM vision_chat failed: %s", exc)
                raise AgentError(
                    friendly_glm_error(exc) or f"GLM API error — please try again. ({exc})"
                ) from exc
            final = final.strip() or "*(empty response)*"
            self._history.append(chat_id, "assistant", final)
            return final

        ctx = ToolContext(chat_id=chat_id, workspace=self._cfg.workspace_dir, cfg=self._cfg)
        final: str | None = None
        for iteration in range(1, self._cfg.max_tool_iters + 1):
            try:
                answer = await self._glm.chat(messages, model=self._cfg.glm_model, tools=TOOL_SPECS)
            except GLMError as exc:
                logger.error("GLM chat failed: %s", exc)
                raise AgentError(
                    friendly_glm_error(exc) or f"GLM API error — please try again. ({exc})"
                ) from exc

            tool_calls = getattr(answer, "tool_calls", None)
            if not tool_calls:
                final = (answer.content or "").strip() or "*(empty response)*"
                break

            # Re-serialize the assistant message as plain dicts (content may be None
            # when the model jumps straight to tool calls — a known GLM quirk).
            messages.append(
                {
                    "role": "assistant",
                    "content": answer.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    # Known GLM quirk: arguments arrive as a sometimes-broken JSON string.
                    args = {}
                    result = "ERROR: tool arguments were not valid JSON"
                else:
                    summary = json.dumps(args, ensure_ascii=False)[:80]
                    if on_progress is not None:
                        await on_progress(iteration, name, summary)
                    result = await dispatch_tool(name, args, ctx)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                logger.info(
                    "chat=%s iter=%d tool=%s result_len=%d", chat_id, iteration, name, len(result)
                )

        if final is None:
            final = "I reached my tool-use limit for this message — please rephrase or continue."
        self._history.append(chat_id, "assistant", final)
        return final
