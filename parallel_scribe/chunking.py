"""Speech-aware creation of ordered ParallelScribe input chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .errors import TranscriptionError
from .media import TARGET_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One ordered portion of normalized audio."""

    index: int
    path: Path
    start_seconds: float
    end_seconds: float


def load_normalized_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a normalized WAV and verify the assumptions used downstream."""

    try:
        audio, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )
    except (OSError, RuntimeError) as error:
        raise TranscriptionError(
            f"The normalized audio could not be read: {error}"
        ) from error

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        raise TranscriptionError(
            f"Expected {TARGET_SAMPLE_RATE} Hz audio, received {sample_rate} Hz."
        )
    if audio.size == 0:
        raise TranscriptionError("The converted audio file is empty.")

    return np.ascontiguousarray(audio), sample_rate


def fixed_boundaries(
    duration_seconds: float,
    max_chunk_seconds: float,
) -> list[tuple[float, float]]:
    """Split the entire recording when voice detection is unavailable."""

    boundaries: list[tuple[float, float]] = []
    cursor = 0.0

    while cursor < duration_seconds:
        end = min(cursor + max_chunk_seconds, duration_seconds)
        boundaries.append((cursor, end))
        cursor = end

    return boundaries


def _split_long_boundary(
    start: float,
    end: float,
    max_chunk_seconds: float,
) -> list[tuple[float, float]]:
    boundaries: list[tuple[float, float]] = []
    cursor = start

    while cursor < end:
        next_end = min(cursor + max_chunk_seconds, end)
        boundaries.append((cursor, next_end))
        cursor = next_end

    return boundaries


def group_speech_boundaries(
    speech_segments: list[dict],
    duration_seconds: float,
    max_chunk_seconds: float,
    padding_seconds: float = 0.2,
) -> list[tuple[float, float]]:
    """Group adjacent VAD regions without exceeding the requested duration."""

    grouped: list[tuple[float, float]] = []
    current_start: float | None = None
    current_end: float | None = None

    for segment in speech_segments:
        start = max(0.0, float(segment["start"]) - padding_seconds)
        end = min(duration_seconds, float(segment["end"]) + padding_seconds)

        if end <= start:
            continue

        if end - start > max_chunk_seconds:
            if current_start is not None and current_end is not None:
                grouped.append((current_start, current_end))
                current_start = None
                current_end = None

            grouped.extend(
                _split_long_boundary(start, end, max_chunk_seconds)
            )
            continue

        if current_start is None:
            current_start, current_end = start, end
        elif end - current_start <= max_chunk_seconds:
            current_end = end
        else:
            if current_end is not None:
                grouped.append((current_start, current_end))
            current_start, current_end = start, end

    if current_start is not None and current_end is not None:
        grouped.append((current_start, current_end))

    return grouped


def detect_boundaries(
    audio: np.ndarray,
    sample_rate: int,
    max_chunk_seconds: float,
) -> tuple[list[tuple[float, float]], str]:
    """Prefer Silero VAD and fall back to deterministic timed chunks."""

    duration_seconds = len(audio) / sample_rate

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        vad_model = load_silero_vad()
        speech_segments = get_speech_timestamps(
            torch.from_numpy(audio),
            vad_model,
            sampling_rate=sample_rate,
            return_seconds=True,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
        )
        boundaries = group_speech_boundaries(
            speech_segments,
            duration_seconds,
            max_chunk_seconds,
        )
        if boundaries:
            return boundaries, "Silero voice activity detection"
    except Exception:
        # Voice detection improves boundaries but is not required to transcribe.
        pass

    return (
        fixed_boundaries(duration_seconds, max_chunk_seconds),
        "fixed-duration fallback",
    )


def write_audio_chunks(
    normalized_audio_path: Path,
    chunks_directory: Path,
    max_chunk_seconds: float,
) -> tuple[list[AudioChunk], str]:
    """Write detected segments as sequentially numbered WAV files."""

    if max_chunk_seconds <= 0:
        raise TranscriptionError("Chunk duration must be greater than zero.")

    audio, sample_rate = load_normalized_audio(normalized_audio_path)
    boundaries, strategy = detect_boundaries(
        audio,
        sample_rate,
        max_chunk_seconds,
    )
    chunks_directory.mkdir(parents=True, exist_ok=True)

    chunks: list[AudioChunk] = []
    for index, (start, end) in enumerate(boundaries):
        start_sample = max(0, round(start * sample_rate))
        end_sample = min(len(audio), round(end * sample_rate))
        chunk_audio = audio[start_sample:end_sample]

        if chunk_audio.size == 0:
            continue

        chunk_path = chunks_directory / f"chunk_{index:05d}.wav"
        sf.write(
            str(chunk_path),
            chunk_audio,
            sample_rate,
            subtype="PCM_16",
        )
        chunks.append(
            AudioChunk(
                index=index,
                path=chunk_path,
                start_seconds=start,
                end_seconds=end,
            )
        )

    if not chunks:
        raise TranscriptionError("No transcribable audio chunks were found.")

    return chunks, strategy
