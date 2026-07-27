"""Streamlit interface for the local Whisper transcription pipeline."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import streamlit as st
import torch
import whisper

from mac_whisper_transcriber import TranscriptionError, transcribe_file


SUPPORTED_EXTENSIONS = [
    "mp3",
    "wav",
    "m4a",
    "aac",
    "flac",
    "ogg",
    "opus",
    "aiff",
    "mp4",
    "mov",
    "mkv",
    "avi",
    "webm",
]

LANGUAGES = {
    "Detect automatically": None,
    "Hindi": "hi",
    "English": "en",
}


def available_model_choices() -> list[str]:
    """Present common models in increasing capability order."""

    installed_api_models = set(whisper.available_models())
    preferred_order = [
        "tiny",
        "base",
        "small",
        "medium",
        "large-v2",
        "large-v3",
        "turbo",
    ]
    choices = [name for name in preferred_order if name in installed_api_models]
    return choices or sorted(installed_api_models)


def stage_progress(stage: str, completed: int, total: int) -> float:
    """Map individual pipeline stages onto one continuous progress bar."""

    stage_fraction = completed / total if total else 0.0
    if stage.startswith("Converting"):
        return 0.1 * stage_fraction
    if stage.startswith("Detecting"):
        return 0.1 + (0.1 * stage_fraction)
    return 0.2 + (0.8 * stage_fraction)


def render_result() -> None:
    """Keep the latest transcript visible across Streamlit reruns."""

    result = st.session_state.get("latest_result")
    if not result:
        return

    st.divider()
    st.subheader("Transcript")
    st.text_area(
        "Generated transcript",
        value=result["transcript"],
        height=420,
        label_visibility="collapsed",
    )
    st.download_button(
        "Download transcript",
        data=result["transcript"],
        file_name=result["output_filename"],
        mime="text/plain",
        use_container_width=True,
    )

    with st.expander("Processing details"):
        st.json(result["metadata"])


st.set_page_config(
    page_title="Mac Whisper Transcriber",
    page_icon="🎙️",
    layout="wide",
)

st.title("Mac Whisper Transcriber")
st.caption(
    "Private audio and video transcription powered by Whisper. "
    "Your recording stays on this Mac."
)

uploaded_file = st.file_uploader(
    "Drop an audio or video file here",
    type=SUPPORTED_EXTENSIONS,
    help="FFmpeg extracts and normalizes the first audio track.",
)

models = available_model_choices()
default_model = models.index("medium") if "medium" in models else 0

settings_column, performance_column = st.columns(2)

with settings_column:
    st.subheader("Transcription")
    model_name = st.selectbox(
        "Whisper model",
        options=models,
        index=default_model,
        help="Larger models usually improve accuracy but take more memory.",
    )
    language_label = st.selectbox(
        "Spoken language",
        options=list(LANGUAGES),
        help=(
            "Automatic detection is useful for mixed Hindi-English recordings. "
            "Force a language when detection is unreliable."
        ),
    )

with performance_column:
    st.subheader("Performance")
    execution_options = ["CPU — parallel chunks"]
    if torch.backends.mps.is_available():
        execution_options.append("Apple Silicon GPU — single worker")

    execution_label = st.selectbox(
        "Execution mode",
        options=execution_options,
    )
    device = "mps" if execution_label.startswith("Apple") else "cpu"
    cpu_count = os.cpu_count() or 1

    if device == "cpu":
        workers = st.slider(
            "CPU workers",
            min_value=1,
            max_value=cpu_count,
            value=min(4, cpu_count),
            help=(
                "Chunks are distributed across separate processes. Each worker "
                "loads its own model, so large models need substantial memory."
            ),
        )
        if workers == cpu_count:
            st.caption("Using every logical CPU core.")
    else:
        workers = 1
        st.caption(
            "MPS runs one model over chunks sequentially. Choose CPU mode "
            "to distribute chunks across multiple workers."
        )

with st.expander("Advanced controls"):
    max_chunk_seconds = st.slider(
        "Maximum chunk length (seconds)",
        min_value=20,
        max_value=120,
        value=45,
        step=5,
        help="Silero groups detected speech without exceeding this duration.",
    )
    initial_prompt = st.text_input(
        "Vocabulary hint",
        placeholder="Names, acronyms, Hindi-English terms…",
        help="Optional context passed independently to every Whisper chunk.",
    )

if device == "cpu" and workers > 1 and model_name.startswith("large"):
    st.warning(
        "Every CPU worker loads a separate large model. Reduce the worker count "
        "if macOS reports memory pressure."
    )

start_transcription = st.button(
    "Transcribe locally",
    type="primary",
    use_container_width=True,
)

if start_transcription:
    if uploaded_file is None:
        st.warning("Choose an audio or video file first.")
    else:
        st.session_state.pop("latest_result", None)
        progress_bar = st.progress(0.0)
        status = st.empty()
        started_at = time.perf_counter()

        def update_progress(stage: str, completed: int, total: int) -> None:
            progress_bar.progress(
                min(1.0, max(0.0, stage_progress(stage, completed, total)))
            )
            elapsed = time.perf_counter() - started_at
            status.markdown(
                f"**{stage}** · {completed}/{total} · {elapsed:.1f}s elapsed"
            )

        try:
            with tempfile.TemporaryDirectory(
                prefix="mac_whisper_transcriber_"
            ) as temporary_directory:
                work_directory = Path(temporary_directory)
                safe_suffix = Path(uploaded_file.name).suffix or ".media"
                input_path = work_directory / f"input{safe_suffix}"
                input_path.write_bytes(uploaded_file.getvalue())

                transcript, metadata = transcribe_file(
                    input_path=input_path,
                    work_directory=work_directory,
                    model_name=model_name,
                    language=LANGUAGES[language_label],
                    device=device,
                    workers=workers,
                    max_chunk_seconds=max_chunk_seconds,
                    initial_prompt=initial_prompt.strip() or None,
                    progress_callback=update_progress,
                )

            progress_bar.progress(1.0)
            status.success(
                f"Finished {metadata['chunk_count']} chunks in "
                f"{metadata['elapsed_seconds']:.1f} seconds."
            )
            st.session_state["latest_result"] = {
                "transcript": transcript,
                "output_filename": (
                    f"{Path(uploaded_file.name).stem}_transcript.txt"
                ),
                "metadata": metadata,
            }
        except TranscriptionError as error:
            status.empty()
            st.error(str(error))
        except Exception as error:
            status.empty()
            st.error(f"Unexpected transcription error: {error}")

render_result()

