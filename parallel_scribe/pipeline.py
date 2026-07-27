"""End-to-end ParallelScribe transcription orchestration."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunking import write_audio_chunks
from .engine import (
    ProgressCallback,
    transcribe_parallel_cpu,
    transcribe_sequential,
)
from .errors import TranscriptionError
from .media import normalize_media


@dataclass(frozen=True, slots=True)
class TranscriptionMetadata:
    """Useful processing details displayed after a completed run."""

    model: str
    language: str
    device: str
    requested_workers: int
    active_workers: int
    chunk_count: int
    maximum_chunk_seconds: float
    segmentation: str
    execution: str
    elapsed_seconds: float
    processing_location: str = "local device"

    def to_dict(self) -> dict:
        return asdict(self)


def stitch_transcript(chunk_texts: list[str]) -> str:
    """Join non-empty chunk results in their original sequence."""

    return "\n".join(text.strip() for text in chunk_texts if text.strip()).strip()


def transcribe_file(
    input_path: Path,
    work_directory: Path,
    model_name: str,
    language: str | None,
    device: str,
    workers: int,
    max_chunk_seconds: float,
    initial_prompt: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, dict]:
    """Convert, segment, transcribe, and reassemble one media file."""

    started_at = time.perf_counter()
    input_path = Path(input_path)
    work_directory = Path(work_directory)

    if not input_path.is_file():
        raise TranscriptionError(f"Input file does not exist: {input_path}")
    if device not in {"cpu", "mps"}:
        raise TranscriptionError("Device must be either 'cpu' or 'mps'.")
    if workers < 1:
        raise TranscriptionError("At least one transcription worker is required.")

    work_directory.mkdir(parents=True, exist_ok=True)
    normalized_path = work_directory / "normalized.wav"
    chunks_directory = work_directory / "chunks"

    if progress_callback:
        progress_callback("Converting media with FFmpeg", 0, 1)

    normalize_media(input_path, normalized_path)

    if progress_callback:
        progress_callback("Converting media with FFmpeg", 1, 1)
        progress_callback("Detecting speech and creating chunks", 0, 1)

    chunks, segmentation_strategy = write_audio_chunks(
        normalized_path,
        chunks_directory,
        max_chunk_seconds,
    )

    if progress_callback:
        progress_callback("Detecting speech and creating chunks", 1, 1)

    active_workers = 1
    if device == "cpu" and workers > 1:
        active_workers = min(workers, len(chunks), os.cpu_count() or 1)
        chunk_results = transcribe_parallel_cpu(
            chunks=chunks,
            model_name=model_name,
            language=language,
            workers=active_workers,
            initial_prompt=initial_prompt,
            progress_callback=progress_callback,
        )
        execution = f"CPU process pool with {active_workers} workers"
    else:
        chunk_results = transcribe_sequential(
            chunks=chunks,
            model_name=model_name,
            language=language,
            device=device,
            initial_prompt=initial_prompt,
            progress_callback=progress_callback,
        )
        execution = f"single {device.upper()} worker"

    transcript = stitch_transcript([result.text for result in chunk_results])
    if not transcript:
        raise TranscriptionError(
            "Whisper completed, but no transcript text was produced."
        )

    metadata = TranscriptionMetadata(
        model=model_name,
        language=language or "automatic detection",
        device=device,
        requested_workers=workers,
        active_workers=active_workers,
        chunk_count=len(chunks),
        maximum_chunk_seconds=max_chunk_seconds,
        segmentation=segmentation_strategy,
        execution=execution,
        elapsed_seconds=round(time.perf_counter() - started_at, 2),
    )
    return transcript, metadata.to_dict()
