"""Telegram update routing: allowlist enforcement, chat flow and commands."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import telegram.error
from telegram import Message, Update
from telegram.ext import ContextTypes

from .agent import Agent, AgentError
from .attachments import (
    AttachmentError,
    attachment_note,
    save_document,
    save_photo,
)
from .config import Config
from .glm_client import GLMClient, GLMError
from .history import History
from .splitter import split_for_telegram
from .voice import VoiceError, voice_to_text

logger = logging.getLogger(__name__)

# Serialize turns per chat so rapid messages queue instead of racing.
LOCKS: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
SEND_ATTEMPTS = 3
PROGRESS_EDIT_INTERVAL = 2.0  # seconds between placeholder progress edits
DEFAULT_IMAGE_PROMPT = "Describe what is in this image, briefly."


def _config(context: ContextTypes.DEFAULT_TYPE) -> Config:
    return context.bot_data["config"]


def _agent(context: ContextTypes.DEFAULT_TYPE) -> Agent:
    return context.bot_data["agent"]


def _history(context: ContextTypes.DEFAULT_TYPE) -> History:
    return context.bot_data["history"]


def _glm(context: ContextTypes.DEFAULT_TYPE) -> GLMClient:
    return context.bot_data["glm"]


def is_allowed(update: Update, allowed_ids: frozenset[int]) -> bool:
    """Return True only for updates coming from a whitelisted user."""
    user = update.effective_user
    return user is not None and user.id in allowed_ids


async def _edit_text(message: Message, text: str) -> None:
    """Edit a message, tolerating 'message is not modified' and rate limits."""
    try:
        await message.edit_text(text)
    except telegram.error.BadRequest as exc:
        if "not modified" not in str(exc).lower():
            logger.warning("edit_text failed: %s", exc)
    except telegram.error.RetryAfter as exc:
        await asyncio.sleep(exc.retry_after + 0.5)
        try:
            await message.edit_text(text)
        except telegram.error.TelegramError as retry_exc:
            logger.warning("edit_text retry failed: %s", retry_exc)


async def _reply_safe(message: Message, text: str) -> None:
    """Reply with basic retry handling (rate limits, transient network errors)."""
    for attempt in range(1, SEND_ATTEMPTS + 1):
        try:
            await message.reply_text(text)
            return
        except telegram.error.RetryAfter as exc:
            await asyncio.sleep(exc.retry_after + 0.5)
        except (telegram.error.TimedOut, telegram.error.NetworkError) as exc:
            if attempt == SEND_ATTEMPTS:
                logger.error("Failed to send message after %d attempts: %s", attempt, exc)
                return
            await asyncio.sleep(2.0)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start and /help."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        return
    message = update.effective_message
    if message is None:
        return
    await _reply_safe(
        message,
        "Hi! I'm your personal AI assistant.\n\n"
        "Send me text, a photo, a voice note or a file (as a document). Commands:\n"
        "/new — start a fresh conversation (clear context)\n"
        "/status — current model and context stats",
    )


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new: clear the conversation history for this chat."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        return
    message = update.effective_message
    if message is None:
        return
    _history(context).reset(message.chat_id)
    await _reply_safe(message, "🧹 Context cleared — we start fresh.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status: show model ids and context statistics."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        return
    message = update.effective_message
    if message is None:
        return
    stats = _history(context).stats(message.chat_id)
    await _reply_safe(
        message,
        f"Model: {cfg.glm_model} (vision: {cfg.glm_vision_model}, ASR: {cfg.glm_asr_model})\n"
        f"Messages in context: {stats['messages']}\n"
        f"Context ≈ {stats['est_tokens']} / {cfg.context_token_budget} tokens\n"
        f"Workspace: {cfg.workspace_dir}",
    )


async def _transcribe_voice(message: Message, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Download and transcribe a voice note; returns the user-facing text."""
    cfg = _config(context)
    voice = message.voice
    if voice is None:
        return ""
    if voice.file_size and voice.file_size > 20 * 1024 * 1024:
        raise AttachmentError("Voice message too large (over 20 MB).")
    tg_file = await context.bot.get_file(voice.file_id)
    with tempfile.TemporaryDirectory(prefix="tgvoice-") as tmp:
        ogg = Path(tmp) / "voice.ogg"
        await tg_file.download_to_drive(ogg)
        return await voice_to_text(
            ogg, voice.duration or 0, _glm(context), Path(tmp), cfg.max_voice_min
        )


async def _collect_user_text(
    message: Message, context: ContextTypes.DEFAULT_TYPE
) -> tuple[str, list[Path]] | None:
    """Turn an incoming message into (user_text, image_paths), or None if not usable."""
    cfg = _config(context)
    workspace = cfg.workspace_dir
    images: list[Path] = []

    if message.voice is not None:
        text = await _transcribe_voice(message, context)
        return f"[voice message transcription]\n{text}", []

    if message.video_note is not None or message.audio is not None:
        await _reply_safe(
            message, "I can't process audio/video files yet — send a voice note instead."
        )
        return None

    if message.photo:
        photo = message.photo[-1]  # highest resolution variant
        rel = await save_photo(context.bot, photo, workspace)
        note = attachment_note(rel, f"photo, {(photo.file_size or 0) / 1024:.0f} KB")
        text = f"{message.caption or DEFAULT_IMAGE_PROMPT}\n\n{note}"
        images.append(workspace / rel)
        return text, images

    if message.document is not None:
        doc = message.document
        rel = await save_document(context.bot, doc, workspace)
        meta = f"{doc.mime_type or 'unknown type'}, {(doc.file_size or 0) / 1024:.0f} KB"
        text = f"{message.caption or ''}\n\n{attachment_note(rel, meta)}".strip()
        if (doc.mime_type or "").startswith("image/"):
            images.append(workspace / rel)
        return text, images

    if message.text:
        return message.text, []

    return None


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a regular message: run one agent turn and deliver the answer in chunks."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        user_id = getattr(update.effective_user, "id", None)
        logger.warning("Ignoring unauthorized user id=%s", user_id)
        return
    message = update.message
    if message is None:
        return

    try:
        collected = await _collect_user_text(message, context)
    except (AttachmentError, VoiceError, GLMError) as exc:
        await _reply_safe(message, f"⚠️ {exc}")
        return
    if collected is None:
        return
    user_text, image_paths = collected

    chat_id = message.chat_id
    async with LOCKS[chat_id]:
        placeholder = await message.reply_text("…")
        last_edit = 0.0

        async def on_progress(iteration: int, tool: str, summary: str) -> None:
            nonlocal last_edit
            now = time.monotonic()
            if now - last_edit < PROGRESS_EDIT_INTERVAL:
                return
            last_edit = now
            await _edit_text(placeholder, f"⚙️ {tool} ({iteration}/{cfg.max_tool_iters})…")

        try:
            final = await _agent(context).handle_turn(
                chat_id, user_text, image_paths=image_paths or None, on_progress=on_progress
            )
        except AgentError as exc:
            await _edit_text(placeholder, f"⚠️ {exc}")
            return
        chunks = split_for_telegram(final)
        await _edit_text(placeholder, chunks[0])
        for chunk in chunks[1:]:
            # ~1 message/sec per chat keeps us within Telegram's rate limits.
            await asyncio.sleep(1.0)
            await _reply_safe(message, chunk)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log unhandled errors; a single bad update must never crash the bot."""
    logger.error("Unhandled error while processing update: %s", update, exc_info=context.error)
