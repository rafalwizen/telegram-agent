"""Path confinement for workspace file operations (subprocess sandboxing lands in Phase 4)."""

from __future__ import annotations

from pathlib import Path


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
