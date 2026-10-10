"""Application factory: handler registration and wiring of shared objects."""

from __future__ import annotations

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
from .history import History


def build_application(cfg: Config, agent: Agent, history: History) -> Application:
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

    application.add_handler(CommandHandler(["start", "help"], handlers.cmd_start))
    application.add_handler(CommandHandler("new", handlers.cmd_new))
    application.add_handler(CommandHandler("status", handlers.cmd_status))
    application.add_handler(MessageHandler(~filters.COMMAND, handlers.on_message))
    application.add_error_handler(handlers.on_error)
    return application
