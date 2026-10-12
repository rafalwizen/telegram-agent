"""Saving Telegram attachments into the workspace inbox."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from telegram import Bot, Document

logger = logging.getLogger(__name__)

MAX_TG_DOWNLOAD = 20 * 1024 * 1024  # Telegram Bot API hard limit for bot downloads
_UNSAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")


class AttachmentError(Exception):
    """User-facing attachment problem (too large, download failed)."""


def sanitize_name(name: str) -> str:
    """Reduce a client-supplied filename to a safe single path component."""
    name = name.replace("\\", "/").split("/")[-1] or "file"
    name = _UNSAFE_RE.sub("_", name).strip("._") or "file"
    return name[:80]


async def save_document(bot: Bot, document: Document, workspace: Path) -> str:
    """Download a Telegram document into ``workspace/inbox/<date>/``; return relative path."""
    if document.file_size and document.file_size > MAX_TG_DOWNLOAD:
        raise AttachmentError(
            f"File is {document.file_size / 1e6:.1f} MB — Telegram bots can download at most 20 MB."
        )
    try:
        tg_file = await bot.get_file(document.file_id)
    except Exception as exc:
        raise AttachmentError(f"Could not fetch the file from Telegram: {exc}") from exc

    stamp = datetime.now()
    target_dir = workspace / "inbox" / stamp.strftime("%Y-%m-%d")
    target_dir.mkdir(parents=True, exist_ok=True)
    name = sanitize_name(document.file_name or "file")
    path = target_dir / f"{stamp.strftime('%H%M%S')}_{name}"
    # Avoid overwriting when two files with the same name arrive within one second.
    counter = 1
    while path.exists():
        path = target_dir / f"{stamp.strftime('%H%M%S')}_{counter}_{name}"
        counter += 1
    await tg_file.download_to_drive(path)

    rel = path.relative_to(workspace).as_posix()
    logger.info("Saved attachment %s (%.0f KB)", rel, (document.file_size or 0) / 1024)
    return rel


def attachment_note(rel_path: str, meta: str) -> str:
    """Model-facing note appended to the user turn when a file arrives."""
    return (
        f"[Attachment saved to workspace: {rel_path} — {meta}. "
        "Use list_files / read_file / extract_text to inspect it and answer the user.]"
    )
