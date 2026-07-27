"""Local, privacy-first media transcription for macOS."""

from .errors import TranscriptionError
from .pipeline import transcribe_file

__all__ = ["TranscriptionError", "transcribe_file"]
