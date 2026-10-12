"""Image preparation for the GLM vision model: base64 data URLs within size limits."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

MAX_BYTES = 4_500_000  # GLM accepts up to 5 MB per image; keep headroom
MAX_EDGE = 3000  # GLM accepts up to 6000 px per side; keep headroom


def _encode_jpeg(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def prepare_image(path: Path) -> str | None:
    """Return a JPEG data URL for the image, downscaled to fit API limits.

    Returns None when the file cannot be read as an image or stays too large.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if max(img.size) > MAX_EDGE:
                img.thumbnail((MAX_EDGE, MAX_EDGE))
            data = _encode_jpeg(img)
            ratio = 0.7
            while len(data) > MAX_BYTES and ratio > 0.05:
                img.thumbnail((max(int(img.width * ratio), 1), max(int(img.height * ratio), 1)))
                data = _encode_jpeg(img)
                ratio *= 0.7
            if len(data) > MAX_BYTES:
                return None
            encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None
