"""Tests for the Docker healthcheck script."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "healthcheck.py"


def _run(heartbeat: Path) -> int:
    env = {**os.environ, "HEARTBEAT": str(heartbeat)}
    return subprocess.run(
        [sys.executable, str(SCRIPT)], env=env, capture_output=True, check=False
    ).returncode


def test_fresh_heartbeat_passes(tmp_path):
    beat = tmp_path / "heartbeat"
    beat.touch()
    assert _run(beat) == 0


def test_missing_heartbeat_fails(tmp_path):
    assert _run(tmp_path / "nonexistent") == 1


def test_stale_heartbeat_fails(tmp_path):
    beat = tmp_path / "heartbeat"
    beat.touch()
    old = time.time() - 10_000
    os.utime(beat, (old, old))
    assert _run(beat) == 1
