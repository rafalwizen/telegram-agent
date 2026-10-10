"""Utilities for Telegram message limits and rough token estimation."""

TELEGRAM_LIMIT = 4096
DEFAULT_CHUNK_LIMIT = 3900  # headroom below the hard 4096-char limit


def est_tokens(text: str) -> int:
    """Rough token estimate for Latin-script languages (PL/EN): ~4 chars per token."""
    return len(text) // 4


def _find_break(s: str, limit: int) -> int:
    """Index just past the best break point within limit (paragraph > line > sentence > word)."""
    window = s[:limit]
    for sep in ("\n\n", "\n", ". ", " "):
        idx = window.rfind(sep)
        if idx >= limit // 2:
            return idx + len(sep)
    return limit


def _split_chunks(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = _find_break(remaining, limit)
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    return [c for c in chunks if c]


def _balance_code_fences(chunks: list[str]) -> list[str]:
    """Close unbalanced ``` fences at chunk boundaries and reopen them in the next chunk."""
    fixed: list[str] = []
    open_fence = False
    for chunk in chunks:
        piece = chunk
        if open_fence:
            piece = "```\n" + piece
        open_fence = open_fence != (piece.count("```") % 2 == 1)
        if open_fence:
            piece += "\n```"
            open_fence = False
        fixed.append(piece)
    return fixed


def split_for_telegram(text: str, limit: int = DEFAULT_CHUNK_LIMIT) -> list[str]:
    """Split a long answer into chunks that each fit Telegram's message limit."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    return _balance_code_fences(_split_chunks(text, limit))
