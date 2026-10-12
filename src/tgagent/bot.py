"""Application factory: handler registration, heartbeat and wiring of shared objects."""

from __future__ import annotations

import asyncio
import logging

from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)

from . import handlers
from .agent import Agent
from .config import Config
from .glm_client import GLMClient
from .history import History

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 60


async def _heartbeat_loop(application: Application) -> None:
    """Touch the heartbeat file periodically; the Docker healthcheck watches its age."""
    beat = application.bot_data["config"].data_dir / "heartbeat"
    while True:
        try:
            beat.touch()
        except OSError:
            logger.warning("Heartbeat touch failed", exc_info=True)
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)


async def _post_init(application: Application) -> None:
    application.create_task(_heartbeat_loop(application))


def build_application(cfg: Config, agent: Agent, history: History, glm: GLMClient) -> Application:
    """Build the python-telegram-bot Application with all handlers registered."""
    application: Application = (
        ApplicationBuilder()
        .token(cfg.telegram_token)
        .get_updates_read_timeout(30)
        .connect_timeout(30)
        .build()
    )
    application.bot_data["config"] = cfg
    application.bot_data["agent"] = agent
    application.bot_data["history"] = history
    application.bot_data["glm"] = glm
    application.post_init = _post_init

    application.add_handler(CommandHandler(["start", "help"], handlers.cmd_start))
    application.add_handler(CommandHandler("new", handlers.cmd_new))
    application.add_handler(CommandHandler("status", handlers.cmd_status))
    application.add_handler(MessageHandler(~filters.COMMAND, handlers.on_message))
    application.add_error_handler(handlers.on_error)
    return application
