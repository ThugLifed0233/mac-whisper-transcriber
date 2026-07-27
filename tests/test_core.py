from __future__ import annotations

import runpy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from mac_whisper_transcriber.chunking import (
    fixed_boundaries,
    group_speech_boundaries,
    load_normalized_audio,
)
from mac_whisper_transcriber.pipeline import stitch_transcript
from mac_whisper_transcriber.media import normalize_media


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ChunkingTests(unittest.TestCase):
    def test_fixed_boundaries_cover_duration_in_order(self) -> None:
        self.assertEqual(
            fixed_boundaries(10.5, 4),
            [(0.0, 4.0), (4.0, 8.0), (8.0, 10.5)],
        )

    def test_speech_boundaries_respect_maximum_length(self) -> None:
        segments = [
            {"start": 0.5, "end": 2.0},
            {"start": 2.4, "end": 4.0},
            {"start": 7.0, "end": 9.0},
        ]
        boundaries = group_speech_boundaries(
            segments,
            duration_seconds=10,
            max_chunk_seconds=5,
            padding_seconds=0,
        )

        self.assertEqual(boundaries, [(0.5, 4.0), (7.0, 9.0)])
        self.assertTrue(
            all(end - start <= 5 for start, end in boundaries)
        )


class TranscriptTests(unittest.TestCase):
    def test_stitching_preserves_supplied_order_and_skips_blanks(self) -> None:
        self.assertEqual(
            stitch_transcript(["first", "", " second ", "third"]),
            "first\nsecond\nthird",
        )


class MediaTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_ffmpeg_normalizes_sample_rate_and_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "stereo.wav"
            output_path = directory / "normalized.wav"
            sample_rate = 44_100
            seconds = 0.25
            timeline = np.arange(round(sample_rate * seconds)) / sample_rate
            tone = np.sin(2 * np.pi * 440 * timeline).astype(np.float32)
            stereo = np.column_stack((tone, tone * 0.5))
            sf.write(input_path, stereo, sample_rate)

            normalize_media(input_path, output_path)
            audio, normalized_rate = load_normalized_audio(output_path)

            self.assertEqual(normalized_rate, 16_000)
            self.assertEqual(audio.ndim, 1)
            self.assertGreater(audio.size, 0)


class EntrypointTests(unittest.TestCase):
    def test_spawn_import_does_not_execute_streamlit_page(self) -> None:
        sys.modules.pop("mac_whisper_transcriber.ui", None)

        runpy.run_path(
            str(REPOSITORY_ROOT / "app.py"),
            run_name="__mp_main__",
        )

        self.assertNotIn("mac_whisper_transcriber.ui", sys.modules)


if __name__ == "__main__":
    unittest.main()

