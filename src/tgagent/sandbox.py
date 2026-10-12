"""Sandboxed Python execution and workspace path confinement."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

IS_POSIX = os.name == "posix"
RLIMIT_FSIZE_MB = 200  # max file size a sandboxed script may write
RLIMIT_NPROC = 64  # max processes for the sandboxed user
WALL_CLOCK_GRACE_S = 0  # wall timeout == configured timeout (deterministic across platforms)
# Environment passed to sandboxed scripts — deliberately tiny: no API keys, no bot token.
ENV_WHITELIST = ("PATH", "HOME", "LANG", "TMPDIR", "PYTHONIOENCODING")


class PathOutsideWorkspace(Exception):
    """Raised when a requested path resolves outside the workspace root."""


def safe_path(workspace: Path, rel: str) -> Path:
    """Resolve ``rel`` inside ``workspace``; reject anything that escapes the root.

    Handles absolute paths, ``..`` segments and symlink escapes (resolve() follows them).
    """
    root = workspace.resolve()
    candidate = (root / rel).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise PathOutsideWorkspace(f"path escapes the workspace: {rel!r}")
    return candidate


def _sandbox_env(workspace: Path) -> dict[str, str]:
    env = {key: os.environ[key] for key in ENV_WHITELIST if key in os.environ}
    env["HOME"] = str(workspace)
    env.setdefault("LANG", "C.UTF-8")
    env["PYTHONIOENCODING"] = "utf-8"
    if not IS_POSIX:
        # Windows Python refuses to start without the system basics.
        for key in ("SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "TEMP", "TMP"):
            if key in os.environ:
                env[key] = os.environ[key]
    return env


if IS_POSIX:

    def _make_preexec(cpu_s: int, mem_mb: int):  # type: ignore[misc]
        import resource

        def apply() -> None:
            # CPU limit sits above the wall clock so asyncio's timeout fires first.
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (RLIMIT_FSIZE_MB << 20, RLIMIT_FSIZE_MB << 20)
            )
            resource.setrlimit(resource.RLIMIT_NPROC, (RLIMIT_NPROC, RLIMIT_NPROC))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        return apply


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    if IS_POSIX:
        # Scripts run in their own session, so the pgid equals the pid.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    else:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=5)


def _format_result(exit_code: int, stdout: bytes, stderr: bytes, output_limit: int) -> str:
    out = stdout.decode("utf-8", errors="replace").strip() or "(empty)"
    err = stderr.decode("utf-8", errors="replace").strip() or "(empty)"
    note = ""
    if exit_code < 0:
        # Negative return codes mean the process was terminated by a signal (e.g. OOM).
        try:
            note = f" (terminated by {signal.Signals(-exit_code).name})"
        except ValueError:
            note = " (terminated by signal)"
    text = f"exit_code={exit_code}{note}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
    if len(text) > output_limit:
        half = output_limit // 2
        text = f"{text[:half]}\n... [output truncated] ...\n{text[-half:]}"
    return text


async def run_python(
    script_rel: str,
    args: str,
    workspace: Path,
    *,
    timeout_s: int,
    mem_mb: int,
    output_limit: int,
) -> str:
    """Run a workspace Python script in a restricted subprocess; return formatted output."""
    script = safe_path(workspace, script_rel)
    if script.suffix != ".py":
        raise ValueError(f"not a Python script: {script_rel!r}")
    if not script.is_file():
        raise FileNotFoundError(f"script not found: {script_rel}")

    # -I: isolated mode — no user site-packages, PYTHON* env vars are ignored.
    argv = [sys.executable, "-I", script_rel.replace("\\", "/"), *shlex.split(args or "")]
    popen_kwargs: dict[str, Any] = {
        "cwd": str(workspace),
        "env": _sandbox_env(workspace),
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
    }
    if IS_POSIX:
        popen_kwargs["start_new_session"] = True
        popen_kwargs["preexec_fn"] = _make_preexec(timeout_s + 10, mem_mb)
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = await asyncio.create_subprocess_exec(*argv, **popen_kwargs)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        await _kill_process_tree(proc)
        return f"ERROR: script exceeded the {timeout_s}s time limit and was killed."
    return _format_result(proc.returncode or 0, stdout, stderr, output_limit)
