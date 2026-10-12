"""Tests for agent tool dispatch (list_files, read_file, extract_text)."""

from __future__ import annotations

import pytest

from tgagent.tools import ToolContext, dispatch


@pytest.fixture
def ctx(make_cfg) -> ToolContext:
    cfg = make_cfg()
    return ToolContext(chat_id=1, workspace=cfg.workspace_dir, cfg=cfg)


async def test_list_files_lists_workspace(ctx):
    (ctx.workspace / "inbox").mkdir(parents=True, exist_ok=True)
    (ctx.workspace / "inbox" / "a.txt").write_text("aaa")
    (ctx.workspace / "b.py").write_text("print('hi')")
    result = await dispatch("list_files", {}, ctx)
    assert "inbox/a.txt" in result
    assert "b.py" in result


async def test_list_files_empty_workspace(ctx):
    assert await dispatch("list_files", {}, ctx) == "(workspace is empty)"


async def test_read_file_numbered_lines(ctx):
    (ctx.workspace / "lines.txt").write_text("first\nsecond\nthird\n")
    result = await dispatch("read_file", {"path": "lines.txt"}, ctx)
    assert "1: first" in result and "3: third" in result


async def test_read_file_offset_and_more_note(ctx):
    (ctx.workspace / "lines.txt").write_text("\n".join(f"line{i}" for i in range(10)))
    result = await dispatch("read_file", {"path": "lines.txt", "offset": 8, "limit": 1}, ctx)
    assert "9: line8" in result
    assert "more lines" in result and "offset=9" in result


async def test_extract_text_tool(ctx):
    (ctx.workspace / "note.md").write_text("# Title\nBody text")
    result = await dispatch("extract_text", {"path": "note.md"}, ctx)
    assert "Body text" in result


async def test_path_escape_returns_error_string(ctx):
    result = await dispatch("read_file", {"path": "../secrets.txt"}, ctx)
    assert result.startswith("ERROR:")


async def test_unknown_tool_returns_error_string(ctx):
    result = await dispatch("destroy_everything", {}, ctx)
    assert result.startswith("ERROR: unknown tool")
