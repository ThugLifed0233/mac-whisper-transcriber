"""Whisper execution strategies for CPU multiprocessing and Apple MPS."""

from __future__ import annotations

import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import torch
import whisper

from .chunking import AudioChunk
from .errors import TranscriptionError


ProgressCallback = Callable[[str, int, int], None]

_WORKER_MODEL = None
_WORKER_LANGUAGE: str | None = None
_WORKER_PROMPT: str | None = None


@dataclass(frozen=True, slots=True)
class ChunkTranscript:
    """Text returned for one original audio chunk."""

    index: int
    text: str


def clean_text(text: str) -> str:
    """Remove noisy whitespace without changing the transcript language."""

    text = re.sub(r"[ \t]+", " ", text.strip())
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _transcribe_options(
    language: str | None,
    initial_prompt: str | None,
) -> dict:
    return {
        "task": "transcribe",
        "language": language,
        "initial_prompt": initial_prompt or None,
        "condition_on_previous_text": False,
        "temperature": 0,
        "fp16": False,
        "verbose": False,
    }


def initialize_cpu_worker(
    model_name: str,
    language: str | None,
    initial_prompt: str | None,
) -> None:
    """Load one independent model in each spawned worker process."""

    global _WORKER_MODEL
    global _WORKER_LANGUAGE
    global _WORKER_PROMPT

    # Process-level parallelism controls concurrency. Limiting Torch's internal
    # pool avoids every worker competing for all CPU cores at the same time.
    torch.set_num_threads(1)
    _WORKER_MODEL = whisper.load_model(model_name, device="cpu")
    _WORKER_LANGUAGE = language
    _WORKER_PROMPT = initial_prompt


def transcribe_cpu_chunk(job: tuple[int, str]) -> ChunkTranscript:
    """Transcribe one chunk inside a spawned worker process."""

    index, chunk_path = job
    if _WORKER_MODEL is None:
        raise RuntimeError("The Whisper worker model was not initialized.")

    result = _WORKER_MODEL.transcribe(
        chunk_path,
        **_transcribe_options(_WORKER_LANGUAGE, _WORKER_PROMPT),
    )
    return ChunkTranscript(index=index, text=clean_text(result.get("text", "")))


def transcribe_parallel_cpu(
    chunks: list[AudioChunk],
    model_name: str,
    language: str | None,
    workers: int,
    initial_prompt: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[ChunkTranscript]:
    """Distribute ordered audio chunks across spawned CPU processes."""

    if not chunks:
        return []

    worker_count = min(
        max(1, workers),
        len(chunks),
        os.cpu_count() or 1,
    )
    results_by_index: dict[int, ChunkTranscript] = {}
    spawn_context = multiprocessing.get_context("spawn")

    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=spawn_context,
            initializer=initialize_cpu_worker,
            initargs=(model_name, language, initial_prompt),
        ) as executor:
            futures = {
                executor.submit(
                    transcribe_cpu_chunk,
                    (chunk.index, str(chunk.path)),
                ): chunk
                for chunk in chunks
            }

            for completed, future in enumerate(as_completed(futures), start=1):
                transcript = future.result()
                results_by_index[transcript.index] = transcript

                if progress_callback:
                    progress_callback(
                        "Transcribing audio in parallel",
                        completed,
                        len(chunks),
                    )
    except Exception as error:
        raise TranscriptionError(
            f"Parallel Whisper transcription failed: {error}"
        ) from error

    return [results_by_index[index] for index in sorted(results_by_index)]


def transcribe_sequential(
    chunks: list[AudioChunk],
    model_name: str,
    language: str | None,
    device: str,
    initial_prompt: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[ChunkTranscript]:
    """Run a single model over chunks on CPU or Apple Silicon MPS."""

    if device == "mps" and not torch.backends.mps.is_available():
        raise TranscriptionError(
            "Apple Silicon acceleration was selected, but MPS is unavailable."
        )

    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as error:
        raise TranscriptionError(
            f"The Whisper model '{model_name}' could not be loaded: {error}"
        ) from error

    results: list[ChunkTranscript] = []
    for completed, chunk in enumerate(chunks, start=1):
        try:
            result = model.transcribe(
                str(chunk.path),
                **_transcribe_options(language, initial_prompt),
            )
        except Exception as error:
            raise TranscriptionError(
                f"Whisper failed on audio chunk {chunk.index + 1}: {error}"
            ) from error

        results.append(
            ChunkTranscript(
                index=chunk.index,
                text=clean_text(result.get("text", "")),
            )
        )

        if progress_callback:
            progress_callback(
                f"Transcribing audio on {device.upper()}",
                completed,
                len(chunks),
            )

    return results
