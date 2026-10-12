"""Tests for image preparation (data URLs, downscaling)."""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

from tgagent.vision import MAX_BYTES, prepare_image


def _png(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (200, 30, 30)).save(path, format="PNG")


def test_prepare_image_returns_jpeg_data_url(tmp_path):
    path = tmp_path / "img.png"
    _png(path, (100, 80))
    url = prepare_image(path)
    assert url is not None
    assert url.startswith("data:image/jpeg;base64,")
    data = base64.b64decode(url.split(",", 1)[1])
    assert data[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_prepare_image_downscales_huge_image(tmp_path):
    path = tmp_path / "big.png"
    _png(path, (5000, 4000))
    url = prepare_image(path)
    assert url is not None
    assert len(base64.b64decode(url.split(",", 1)[1])) < MAX_BYTES


def test_prepare_image_invalid_file_returns_none(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"definitely not an image")
    assert prepare_image(path) is None


def test_prepare_image_missing_file_returns_none(tmp_path):
    assert prepare_image(tmp_path / "missing.png") is None
