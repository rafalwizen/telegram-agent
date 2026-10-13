"""Import/export of conversations: Claude Code session JSONL and plain JSON arrays."""

from __future__ import annotations

import json
import re

# Hard cap so a pathological file cannot flood the history table; the context
# builder trims by token budget anyway, so the newest tail is what matters.
MAX_MESSAGES = 500

# User texts that are slash-command plumbing, not real prompts.
_SKIP_PREFIXES = ("<command-name>", "<local-command", "<command-message")

# Harness-injected context blocks; strip them from imported messages.
_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>\s*", re.DOTALL)


class ConversationIOError(Exception):
    """User-facing conversation import/export problem."""


def _strip_reminders(text: str) -> str:
    return _REMINDER_RE.sub("", text)


def _blocks_text(content: object) -> str:
    """Extract human-readable text from a string or a Claude content-block list."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n\n".join(p for p in parts if p)


def _is_noise_user_text(text: str) -> bool:
    return not text.strip() or text.lstrip().startswith(_SKIP_PREFIXES)


def _merge_consecutive(messages: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: list[tuple[str, str]] = []
    for role, content in messages:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + "\n\n" + content)
        else:
            merged.append((role, content))
    return merged


def parse_conversation(text: str) -> list[tuple[str, str]]:
    """Detect the file format and return [(role, content), ...] dialogue turns."""
    stripped = text.lstrip()
    if not stripped:
        raise ConversationIOError("The file is empty.")
    if stripped.startswith("["):
        return parse_json_messages(text)
    if stripped.startswith("{"):
        return parse_claude_jsonl(text)
    raise ConversationIOError(
        "Unrecognized format — send a Claude Code session (.jsonl) or a JSON array of messages."
    )


def parse_claude_jsonl(text: str) -> list[tuple[str, str]]:
    """Parse a Claude Code session transcript (.jsonl), keeping only the dialogue.

    The JSONL layout is internal to Claude Code and may change between versions,
    so unknown line types, block types and malformed lines are skipped, not rejected.
    """
    messages: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("isSidechain") or entry.get("isMeta"):
            continue
        role = entry.get("type")
        if role not in ("user", "assistant"):
            continue
        content = _blocks_text((entry.get("message") or {}).get("content"))
        content = _strip_reminders(content).strip()
        if role == "user" and _is_noise_user_text(content):
            continue
        if not content:
            continue
        messages.append((role, content))
    merged = _merge_consecutive(messages)
    if not merged:
        raise ConversationIOError(
            "No user/assistant dialogue found — is this a Claude Code .jsonl session file?"
        )
    return merged[-MAX_MESSAGES:]


def parse_json_messages(text: str) -> list[tuple[str, str]]:
    """Parse a JSON array of {"role", "content"} dicts (the /export format)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConversationIOError(f"Not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ConversationIOError("Expected a JSON array of messages.")
    messages: list[tuple[str, str]] = []
    for item in data:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            raise ConversationIOError(
                "Each message must be an object with role 'user' or 'assistant'."
            )
        content = _strip_reminders(_blocks_text(item.get("content"))).strip()
        if not content:
            continue
        messages.append((item["role"], content))
    merged = _merge_consecutive(messages)
    if not merged:
        raise ConversationIOError("The file contains no messages.")
    return merged[-MAX_MESSAGES:]


def messages_to_json(messages: list[tuple[str, str]]) -> str:
    """Serialize [(role, content)] into the JSON array accepted by parse_conversation."""
    payload = [{"role": role, "content": content} for role, content in messages]
    return json.dumps(payload, ensure_ascii=False, indent=2)
