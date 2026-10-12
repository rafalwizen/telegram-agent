"""Tests for voice-note segmentation and transcription (requires ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tgagent.voice import VoiceError, voice_to_text

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_ogg(path: Path, seconds: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:a",
            "libopus",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


class FakeASR:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, audio_path: Path) -> str:
        text = f"chunk{self.calls}"
        self.calls += 1
        return text


@needs_ffmpeg
async def test_voice_to_text_joins_segments(tmp_path):
    ogg = tmp_path / "voice.ogg"
    _make_ogg(ogg, seconds=65)  # 65 s → 3 segments (29 + 29 + 7)
    fake = FakeASR()
    text = await voice_to_text(
        ogg, duration_s=65, glm=fake, tmp_dir=tmp_path / "seg", max_minutes=5
    )
    assert text == "chunk0 chunk1 chunk2"
    assert fake.calls == 3


@needs_ffmpeg
async def test_short_voice_single_segment(tmp_path):
    ogg = tmp_path / "voice.ogg"
    _make_ogg(ogg, seconds=5)
    fake = FakeASR()
    text = await voice_to_text(ogg, duration_s=5, glm=fake, tmp_dir=tmp_path / "seg", max_minutes=5)
    assert text == "chunk0"


async def test_voice_too_long_rejected_without_ffmpeg(tmp_path):
    # Duration check happens before ffmpeg is needed, so this passes everywhere.
    with pytest.raises(VoiceError, match="too long"):
        await voice_to_text(
            tmp_path / "x.ogg",
            duration_s=9999,
            glm=FakeASR(),
            tmp_dir=tmp_path / "seg",
            max_minutes=5,
        )
