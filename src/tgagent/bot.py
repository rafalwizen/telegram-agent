"""Application factory: handler registration, heartbeat and wiring of shared objects."""

from __future__ import annotations

import asyncio
import logging

from telegram import BotCommand
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

# Shown by Telegram in the "/" command menu of the chat.
BOT_COMMANDS = [
    BotCommand("start", "what I can do"),
    BotCommand("new", "start a fresh conversation (clear context)"),
    BotCommand("status", "model and context stats"),
    BotCommand("files", "list files in my workspace"),
    BotCommand("clear_files", "delete all workspace files (confirm with 'yes')"),
    BotCommand("import", "load a conversation from a Claude Code .jsonl or JSON file"),
    BotCommand("export", "download this conversation as a JSON file"),
]


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
    # Register the command menu so Telegram's "/" button lists our commands.
    try:
        await application.bot.set_my_commands(BOT_COMMANDS)
    except Exception:
        logger.warning("set_my_commands failed (menu may be stale)", exc_info=True)
    # Plain asyncio task: PTB warns about Application.create_task before running state.
    application.bot_data["_heartbeat_task"] = asyncio.create_task(_heartbeat_loop(application))


async def _post_shutdown(application: Application) -> None:
    task = application.bot_data.pop("_heartbeat_task", None)
    if task is not None:
        task.cancel()


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
    application.post_shutdown = _post_shutdown

    application.add_handler(CommandHandler(["start", "help"], handlers.cmd_start))
    application.add_handler(CommandHandler("new", handlers.cmd_new))
    application.add_handler(CommandHandler("status", handlers.cmd_status))
    application.add_handler(CommandHandler("files", handlers.cmd_files))
    application.add_handler(CommandHandler("clear_files", handlers.cmd_clear_files))
    application.add_handler(CommandHandler("import", handlers.cmd_import))
    application.add_handler(CommandHandler("export", handlers.cmd_export))
    application.add_handler(MessageHandler(~filters.COMMAND, handlers.on_message))
    application.add_error_handler(handlers.on_error)
    return application
