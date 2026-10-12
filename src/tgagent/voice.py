"""Voice notes: OGG/Opus → segmented MP3 → GLM ASR → text."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Protocol

from .glm_client import GLMClient

# The GLM ASR endpoint accepts at most 30 s of audio per request.
SEGMENT_SECONDS = 29
FFMPEG_TIMEOUT_S = 120


class VoiceError(Exception):
    """User-facing voice processing problem."""


class _Transcriber(Protocol):
    async def transcribe(self, audio_path: Path) -> str: ...


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def segment_audio(ogg_path: Path, tmp_dir: Path) -> list[Path]:
    """Transcode an OGG/Opus voice note into <=29 s mono 16 kHz MP3 chunks."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(ogg_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-f",
        "segment",
        "-segment_time",
        str(SEGMENT_SECONDS),
        str(tmp_dir / "chunk_%03d.mp3"),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        returncode = await asyncio.wait_for(proc.wait(), timeout=FFMPEG_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        raise VoiceError("Voice conversion timed out.") from None
    if returncode != 0:
        raise VoiceError("Voice conversion failed (ffmpeg error).")
    return sorted(tmp_dir.glob("chunk_*.mp3"))


async def voice_to_text(
    ogg_path: Path,
    duration_s: int,
    glm: GLMClient | _Transcriber,
    tmp_dir: Path,
    max_minutes: int,
) -> str:
    """Convert a Telegram voice note file into transcribed text."""
    if duration_s > max_minutes * 60:
        raise VoiceError(f"Voice message is too long (over {max_minutes} min).")
    if not ffmpeg_available():
        raise VoiceError("ffmpeg is not installed on the server — voice notes are disabled.")
    chunks = await segment_audio(ogg_path, tmp_dir)
    if not chunks:
        raise VoiceError("Voice message appears to be empty.")
    texts: list[str] = []
    for chunk in chunks:
        text = await glm.transcribe(chunk)
        if text and text.strip():
            texts.append(text.strip())
    return " ".join(texts)
