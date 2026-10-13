"""Telegram update routing: allowlist enforcement, chat flow and commands."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import telegram.error
from telegram import Document, Message, Update
from telegram.ext import ContextTypes

from .agent import Agent, AgentError
from .attachments import AttachmentError, attachment_note, save_document, save_photo
from .config import Config
from .conversation_io import ConversationIOError, messages_to_json, parse_conversation
from .history import History
from .splitter import split_for_telegram

logger = logging.getLogger(__name__)

# Serialize turns per chat so rapid messages queue instead of racing.
LOCKS: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
SEND_ATTEMPTS = 3
PROGRESS_EDIT_INTERVAL = 2.0  # seconds between placeholder progress edits
DEFAULT_IMAGE_PROMPT = "Describe what is in this image, briefly."
WORKSPACE_SUBDIRS = ("inbox", "scripts", "out")
IMPORT_PENDING_TTL_S = 600  # how long /import waits for the conversation file
PENDING_IMPORTS: dict[int, float] = {}  # chat_id -> deadline (time.monotonic())


def list_workspace_files(workspace: Path, max_entries: int = 40) -> str:
    """Human-readable listing of workspace files (no model involved, zero API cost)."""
    lines: list[str] = []
    for p in sorted(workspace.rglob("*")):
        if not p.is_file():
            continue
        if len(lines) >= max_entries:
            lines.append(f"... (more files, truncated at {max_entries})")
            break
        rel = p.relative_to(workspace).as_posix()
        lines.append(f"{rel} ({p.stat().st_size / 1024:.0f} KB)")
    return "\n".join(lines) or "(workspace is empty)"


def clear_workspace(workspace: Path) -> tuple[int, int]:
    """Delete everything inside the workspace and recreate the conventional subdirs.

    Returns (files_removed, bytes_freed).
    """
    removed = 0
    freed = 0
    for child in workspace.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                for inner in child.rglob("*"):
                    if inner.is_file():
                        removed += 1
                        freed += inner.stat().st_size
                shutil.rmtree(child)
            else:
                if child.is_file():
                    removed += 1
                    freed += child.stat().st_size
                child.unlink()
        except OSError:
            logger.warning("Could not remove %s", child, exc_info=True)
    for sub in WORKSPACE_SUBDIRS:
        (workspace / sub).mkdir(parents=True, exist_ok=True)
    return removed, freed


def _config(context: ContextTypes.DEFAULT_TYPE) -> Config:
    return context.bot_data["config"]


def _agent(context: ContextTypes.DEFAULT_TYPE) -> Agent:
    return context.bot_data["agent"]


def _history(context: ContextTypes.DEFAULT_TYPE) -> History:
    return context.bot_data["history"]


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
        "Send me text, a photo, or a file (as a document). Commands:\n"
        "/new — start a fresh conversation (clear context)\n"
        "/status — current model and context stats\n"
        "/files — list files in my workspace\n"
        "/clear_files — delete all workspace files (needs confirmation)\n"
        "/import — load a conversation from a file (Claude Code .jsonl or JSON)\n"
        "/export — download this conversation as a JSON file",
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
    """Handle /status: show the model and context statistics."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        return
    message = update.effective_message
    if message is None:
        return
    stats = _history(context).stats(message.chat_id)
    await _reply_safe(
        message,
        f"Model: {cfg.glm_model}\n"
        f"Messages in context: {stats['messages']}\n"
        f"Context ≈ {stats['est_tokens']} / {cfg.context_token_budget} tokens\n"
        f"Workspace: {cfg.workspace_dir}",
    )


async def cmd_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /files: list workspace files (local, no model call)."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        return
    message = update.effective_message
    if message is None:
        return
    text = await asyncio.to_thread(list_workspace_files, cfg.workspace_dir)
    chunks = split_for_telegram(text)
    await _reply_safe(message, chunks[0])
    for chunk in chunks[1:]:
        await asyncio.sleep(1.0)
        await _reply_safe(message, chunk)


async def cmd_clear_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear_files: delete all workspace files; requires 'yes' confirmation."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        return
    message = update.effective_message
    if message is None:
        return
    args = context.args or []
    if not args or args[0].lower() not in ("yes", "tak"):
        preview = await asyncio.to_thread(list_workspace_files, cfg.workspace_dir, 15)
        await _reply_safe(
            message,
            "⚠️ This will delete ALL files in my workspace (inbox, scripts, out):\n"
            f"{preview}\n\n"
            "Run /clear_files yes to confirm.",
        )
        return
    removed, freed = await asyncio.to_thread(clear_workspace, cfg.workspace_dir)
    await _reply_safe(
        message,
        f"🗑 Removed {removed} files ({freed / 1e6:.1f} MB). "
        "inbox/, scripts/ and out/ were recreated empty.",
    )


def _import_pending(chat_id: int) -> bool:
    """True if this chat is waiting for a conversation file after /import."""
    deadline = PENDING_IMPORTS.get(chat_id)
    if deadline is None:
        return False
    if time.monotonic() > deadline:
        PENDING_IMPORTS.pop(chat_id, None)
        return False
    return True


async def _run_import(
    message: Message, context: ContextTypes.DEFAULT_TYPE, document: Document
) -> None:
    """Download a conversation file, parse it and load it into this chat's history."""
    chat_id = message.chat_id
    PENDING_IMPORTS.pop(chat_id, None)
    cfg = _config(context)
    try:
        rel = await save_document(context.bot, document, cfg.workspace_dir)
    except AttachmentError as exc:
        await _reply_safe(message, f"⚠️ {exc}")
        return
    path = cfg.workspace_dir / rel
    try:
        # utf-8-sig strips a BOM if the editor added one.
        text = await asyncio.to_thread(path.read_text, "utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        await _reply_safe(message, f"⚠️ Could not read the file: {exc}")
        return
    try:
        parsed = await asyncio.to_thread(parse_conversation, text)
    except ConversationIOError as exc:
        await _reply_safe(message, f"⚠️ {exc}")
        return
    _history(context).import_messages(chat_id, parsed)
    stats = _history(context).stats(chat_id)
    await _reply_safe(
        message,
        f"✅ Imported {len(parsed)} messages (~{stats['est_tokens'] / 1000:.0f}k tokens) from "
        f"{document.file_name or rel}.\n"
        f"Context keeps the newest ~{cfg.context_token_budget // 1000}k tokens. "
        "Continue the conversation — I remember it now.",
    )


async def cmd_import(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /import: load a conversation from a Claude Code .jsonl or JSON file."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        return
    message = update.effective_message
    if message is None:
        return
    args = context.args or []
    if args and args[0].lower() == "cancel":
        PENDING_IMPORTS.pop(message.chat_id, None)
        await _reply_safe(message, "Import cancelled.")
        return
    # Stateless shortcut: /import sent as a reply to a document imports that file.
    replied = message.reply_to_message
    if replied is not None and replied.document is not None:
        await _run_import(message, context, replied.document)
        return
    PENDING_IMPORTS[message.chat_id] = time.monotonic() + IMPORT_PENDING_TTL_S
    await _reply_safe(
        message,
        "📥 Send me the conversation file now — a Claude Code session (.jsonl) "
        "or a JSON array of messages.\n"
        "⚠️ It will REPLACE the current conversation in this chat "
        f"(pending for {IMPORT_PENDING_TTL_S // 60} min; /import cancel aborts).",
    )


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /export: send the whole conversation as a JSON document (local, no model call)."""
    cfg = _config(context)
    if not is_allowed(update, cfg.allowed_ids):
        return
    message = update.effective_message
    if message is None:
        return
    messages = _history(context).all_messages(message.chat_id)
    if not messages:
        await _reply_safe(message, "Nothing to export — this chat has no history yet.")
        return
    payload = await asyncio.to_thread(messages_to_json, messages)
    path = cfg.workspace_dir / "out" / f"conversation-{datetime.now():%Y%m%d-%H%M%S}.json"
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, payload, "utf-8")
    try:
        await message.reply_document(document=path, filename=path.name)
    except telegram.error.TelegramError:
        logger.warning("reply_document failed", exc_info=True)
        await _reply_safe(
            message, f"Saved the export to {path.name}, but sending it failed — see /files."
        )


async def _collect_user_text(
    message: Message, context: ContextTypes.DEFAULT_TYPE
) -> tuple[str, list[Path]] | None:
    """Turn an incoming message into (user_text, image_paths), or None if not usable."""
    cfg = _config(context)
    workspace = cfg.workspace_dir
    images: list[Path] = []

    if message.voice is not None or message.video_note is not None or message.audio is not None:
        await _reply_safe(message, "I don't support voice or audio messages — please send text.")
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

    if _import_pending(message.chat_id):
        if message.document is not None:
            await _run_import(message, context, message.document)
            return
        # Any other message cancels the pending import and is processed normally.
        PENDING_IMPORTS.pop(message.chat_id, None)

    try:
        collected = await _collect_user_text(message, context)
    except AttachmentError as exc:
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
