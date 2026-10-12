"""Tests for workspace path confinement and the run_python sandbox."""

from __future__ import annotations

from pathlib import Path

import pytest

from tgagent.sandbox import PathOutsideWorkspace, run_python, safe_path


def test_path_inside_workspace(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file.txt").write_text("x")
    resolved = safe_path(tmp_path, "sub/file.txt")
    assert resolved == (tmp_path / "sub" / "file.txt").resolve()


def test_root_itself_is_allowed(tmp_path):
    assert safe_path(tmp_path, "") == tmp_path.resolve()


def test_traversal_rejected(tmp_path):
    with pytest.raises(PathOutsideWorkspace):
        safe_path(tmp_path, "../outside.txt")


def test_absolute_path_rejected(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    with pytest.raises(PathOutsideWorkspace):
        safe_path(tmp_path, str(outside))
    with pytest.raises(PathOutsideWorkspace):
        safe_path(tmp_path, "/etc/passwd")


# --- run_python ---


def _write_script(workspace: Path, rel: str, code: str) -> None:
    target = workspace / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding="utf-8")


async def test_run_python_success(tmp_path):
    _write_script(tmp_path, "scripts/hello.py", "print('hello from sandbox')")
    out = await run_python(
        "scripts/hello.py", "", tmp_path, timeout_s=15, mem_mb=256, output_limit=1000
    )
    assert "exit_code=0" in out
    assert "hello from sandbox" in out


async def test_run_python_passes_args(tmp_path):
    _write_script(tmp_path, "echo.py", "import sys; print(' '.join(sys.argv[1:]))")
    out = await run_python(
        "echo.py", "alpha beta", tmp_path, timeout_s=15, mem_mb=256, output_limit=1000
    )
    assert "alpha beta" in out


async def test_run_python_reports_stderr_and_exit_code(tmp_path):
    _write_script(tmp_path, "fail.py", "import sys; print('boom', file=sys.stderr); sys.exit(3)")
    out = await run_python("fail.py", "", tmp_path, timeout_s=15, mem_mb=256, output_limit=1000)
    assert "exit_code=3" in out
    assert "boom" in out


async def test_run_python_env_has_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "super-secret-key")
    _write_script(
        tmp_path,
        "env_probe.py",
        "import os; print('KEY=' + os.environ.get('GLM_API_KEY', 'ABSENT'))",
    )
    out = await run_python(
        "env_probe.py", "", tmp_path, timeout_s=15, mem_mb=256, output_limit=1000
    )
    assert "ABSENT" in out
    assert "super-secret-key" not in out


async def test_run_python_timeout_kills_script(tmp_path):
    _write_script(tmp_path, "loop.py", "while True:\n    pass")
    out = await run_python("loop.py", "", tmp_path, timeout_s=3, mem_mb=256, output_limit=1000)
    assert out.startswith("ERROR:")
    assert "limit" in out


async def test_run_python_missing_script(tmp_path):
    with pytest.raises(FileNotFoundError):
        await run_python("nope.py", "", tmp_path, timeout_s=5, mem_mb=256, output_limit=100)


async def test_run_python_rejects_path_escape(tmp_path):
    (tmp_path.parent / "outside.py").write_text("print('x')")
    with pytest.raises(PathOutsideWorkspace):
        await run_python("../outside.py", "", tmp_path, timeout_s=5, mem_mb=256, output_limit=100)


async def test_run_python_rejects_non_python(tmp_path):
    _write_script(tmp_path, "data.txt", "hello")
    with pytest.raises(ValueError):
        await run_python("data.txt", "", tmp_path, timeout_s=5, mem_mb=256, output_limit=100)
