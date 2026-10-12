"""Tests for workspace path confinement."""

import pytest

from tgagent.sandbox import PathOutsideWorkspace, safe_path


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
