"""Agent turn handling: one user message in, final answer out.

Phase 2: single GLM completion. Tool calling and a scripting workspace
are layered on top of this class in later phases.
"""

from __future__ import annotations

import logging

from .config import Config
from .glm_client import GLMClient, GLMError
from .history import History
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """User-facing error, shown to the user as the result of the turn."""


class Agent:
    def __init__(self, cfg: Config, glm: GLMClient, history: History) -> None:
        self._cfg = cfg
        self._glm = glm
        self._history = history

    async def handle_turn(self, chat_id: int, user_text: str) -> str:
        """Process one user turn: append, build context, complete, persist, return."""
        self._history.append(chat_id, "user", user_text)
        messages = self._history.build_context(
            chat_id, SYSTEM_PROMPT, self._cfg.context_token_budget
        )
        try:
            answer = await self._glm.chat(messages, model=self._cfg.glm_model)
        except GLMError as exc:
            logger.error("GLM chat failed: %s", exc)
            raise AgentError(f"GLM API error — please try again. ({exc})") from exc
        final = (answer.content or "").strip() or "*(empty response)*"
        self._history.append(chat_id, "assistant", final)
        return final
