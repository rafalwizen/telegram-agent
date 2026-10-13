"""Tests for the /files and /clear_files workspace helpers."""

from __future__ import annotations

from pathlib import Path

from tgagent.handlers import clear_workspace, list_workspace_files


def _seed(workspace: Path) -> None:
    for sub in ("inbox", "scripts", "out"):
        (workspace / sub).mkdir(parents=True, exist_ok=True)
    (workspace / "inbox" / "a.pdf").write_bytes(b"x" * 100)
    (workspace / "scripts" / "s.py").write_text("print('hi')")
    (workspace / "out" / "result.txt").write_text("done")
    (workspace / "stray.tmp").write_text("loose file in root")


def test_list_workspace_files(tmp_path):
    _seed(tmp_path)
    listing = list_workspace_files(tmp_path)
    assert "inbox/a.pdf (0 KB)" in listing
    assert "scripts/s.py" in listing
    assert "out/result.txt" in listing
    assert "stray.tmp" in listing


def test_list_empty_workspace(tmp_path):
    assert list_workspace_files(tmp_path) == "(workspace is empty)"


def test_list_truncates_many_files(tmp_path):
    (tmp_path / "inbox").mkdir(parents=True)
    for i in range(50):
        (tmp_path / "inbox" / f"f{i}.txt").write_text("x")
    listing = list_workspace_files(tmp_path, max_entries=10)
    assert "truncated at 10" in listing


def test_clear_workspace_removes_and_recreates(tmp_path):
    _seed(tmp_path)
    removed, freed = clear_workspace(tmp_path)
    assert removed == 4
    assert freed == 100 + len(b"print('hi')") + len("done") + len("loose file in root")
    assert list_workspace_files(tmp_path) == "(workspace is empty)"
    for sub in ("inbox", "scripts", "out"):
        assert (tmp_path / sub).is_dir()
