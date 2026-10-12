"""Agent tool definitions (OpenAI function specs) and their dispatch."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .extract import extract_text
from .sandbox import PathOutsideWorkspace, safe_path

MAX_LIST_ENTRIES = 400
READ_LINE_LIMIT = 2000


@dataclass
class ToolContext:
    chat_id: int
    workspace: Path
    cfg: Config


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... (truncated)"


def _path_arg(args: dict[str, Any]) -> str:
    return str(args.get("path") or "")


async def _tool_list_files(args: dict[str, Any], ctx: ToolContext) -> str:
    def run() -> str:
        root = safe_path(ctx.workspace, _path_arg(args))
        lines: list[str] = []
        for p in sorted(root.rglob("*")):
            if len(lines) >= MAX_LIST_ENTRIES:
                lines.append(f"... (truncated at {MAX_LIST_ENTRIES} entries)")
                break
            if not p.is_file():
                continue
            rel = p.relative_to(ctx.workspace).as_posix()
            stat = p.stat()
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
            lines.append(f"{rel}\t{stat.st_size} B\t{mtime}")
        return "\n".join(lines) or "(workspace is empty)"

    return await asyncio.to_thread(run)


async def _tool_read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    def run() -> str:
        path = safe_path(ctx.workspace, _path_arg(args))
        try:
            offset = max(int(args.get("offset") or 0), 0)
        except (TypeError, ValueError):
            offset = 0
        try:
            limit = min(max(int(args.get("limit") or READ_LINE_LIMIT), 1), READ_LINE_LIMIT)
        except (TypeError, ValueError):
            limit = READ_LINE_LIMIT
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        picked = lines[offset : offset + limit]
        out = (
            "\n".join(f"{i + offset + 1}: {line}" for i, line in enumerate(picked))
            or "(empty file)"
        )
        if offset + limit < len(lines):
            out += f"\n... ({len(lines) - offset - limit} more lines; use offset={offset + limit})"
        return _truncate(out, ctx.cfg.tool_result_limit)

    return await asyncio.to_thread(run)


async def _tool_extract_text(args: dict[str, Any], ctx: ToolContext) -> str:
    path = safe_path(ctx.workspace, _path_arg(args))
    pages = str(args.get("pages") or "")
    text = await asyncio.to_thread(extract_text, path, pages, ctx.cfg.tool_result_limit)
    return _truncate(text, ctx.cfg.tool_result_limit)


TOOLS: dict[str, Any] = {
    "list_files": _tool_list_files,
    "read_file": _tool_read_file,
    "extract_text": _tool_extract_text,
}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files in the workspace (recursive), with sizes and modification times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional subdirectory to list, relative to the workspace root."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace as numbered lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "offset": {"type": "integer", "description": "0-based line to start from."},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text",
            "description": (
                "Extract text from a document (PDF, DOCX, XLSX, CSV, TXT and similar). "
                "For PDFs, 'pages' can select ranges like '1-5,8'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "pages": {
                        "type": "string",
                        "description": "PDF pages to extract, e.g. '1-5,8'.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


async def dispatch(name: str, args: dict[str, Any], ctx: ToolContext) -> str:
    """Run a tool by name; failures come back as 'ERROR: ...' strings, never as exceptions."""
    try:
        handler = TOOLS.get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}'"
        return await handler(args, ctx)
    except (PathOutsideWorkspace, FileNotFoundError, IsADirectoryError, PermissionError) as exc:
        return f"ERROR: {exc}"
    except Exception as exc:  # noqa: BLE001 - a tool failure must not crash the agent loop
        return f"ERROR: {type(exc).__name__}: {exc}"
