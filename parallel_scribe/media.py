"""ParallelScribe media normalization helpers built around FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .errors import TranscriptionError


TARGET_SAMPLE_RATE = 16_000


def ensure_ffmpeg() -> None:
    """Fail with an actionable message when FFmpeg is unavailable."""

    if shutil.which("ffmpeg") is None:
        raise TranscriptionError(
            "FFmpeg was not found. Install it on macOS with "
            "`brew install ffmpeg`, then try again."
        )


def normalize_media(input_path: Path, output_path: Path) -> Path:
    """Convert FFmpeg-compatible media into a mono 16 kHz PCM WAV."""

    ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise TranscriptionError(f"FFmpeg could not be started: {error}") from error

    if result.returncode != 0:
        detail = result.stderr.strip() or "Unknown FFmpeg error."
        raise TranscriptionError(
            "The selected file could not be converted to audio.\n\n"
            f"FFmpeg reported: {detail}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TranscriptionError(
            "FFmpeg finished without producing usable audio."
        )

    return output_path
