"""Telegram update routing: allowlist enforcement and message handlers."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import ContextTypes

from .config import Config

logger = logging.getLogger(__name__)


def _config(context: ContextTypes.DEFAULT_TYPE) -> Config:
    return context.bot_data["config"]


def is_allowed(update: Update, allowed_ids: frozenset[int]) -> bool:
    """Return True only for updates coming from a whitelisted user."""
    user = update.effective_user
    return user is not None and user.id in allowed_ids


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start and /help."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        return
    message = update.effective_message
    if message is None:
        return
    await message.reply_text(
        "Hi! I'm your personal assistant bot.\n"
        "Phase 1 (echo mode): send me any text and I'll echo it back. "
        "Full AI answers, file analysis and a scripting workspace are coming next."
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular (non-command) messages; echo for now."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        user_id = getattr(update.effective_user, "id", None)
        logger.warning("Ignoring unauthorized user id=%s", user_id)
        return
    message = update.message
    if message is None or not message.text:
        return
    await message.reply_text(f"echo: {message.text}")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log unhandled errors; a single bad update must never crash the bot."""
    logger.error("Unhandled error while processing update: %s", update, exc_info=context.error)
