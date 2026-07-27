"""ParallelScribe's local, privacy-first transcription interface."""

from .errors import TranscriptionError
from .pipeline import transcribe_file

__all__ = ["TranscriptionError", "transcribe_file"]
