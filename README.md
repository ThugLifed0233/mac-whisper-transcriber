# Mac Whisper Transcriber

A private, local-first audio and video transcription workspace for macOS.

Drop in a recording, choose a Whisper model, and decide how many CPU workers
should process it. The app converts the source with FFmpeg, finds speech,
creates ordered chunks, transcribes those chunks locally, and stitches the
results back together.

No paid transcription API is used, and the uploaded recording is not sent to
an external transcription service.

## Features

- Audio and video uploads through a Streamlit interface
- MP3, WAV, M4A, FLAC, OGG, MP4, MOV, MKV, WebM, and other common formats
- FFmpeg normalization to a 16 kHz mono WAV
- Speech-aware segmentation with Silero VAD
- Fixed-duration chunking if voice detection is unavailable
- Selectable Whisper model
- Automatic language detection or forced Hindi/English transcription
- Optional vocabulary hints for names, acronyms, and Hinglish terms
- Configurable multiprocessing across every logical CPU core
- Optional single-worker Apple Silicon MPS mode
- Ordered reconstruction even when chunks finish out of sequence
- Progress and elapsed-time reporting
- Plain-text transcript download
- Temporary source and chunk cleanup after each run
- `condition_on_previous_text=False` to reduce repetition leaking between chunks

## How it works

```text
Audio or video
      │
      ▼
FFmpeg normalization
      │
      ▼
16 kHz mono WAV
      │
      ▼
Silero speech detection
      │
      ▼
Ordered audio chunks
      │
      ├──► CPU worker 1 ──┐
      ├──► CPU worker 2 ──┤
      ├──► CPU worker 3 ──┤
      └──► CPU worker N ──┘
                          │
                          ▼
                Index-based reordering
                          │
                          ▼
                  Clean text transcript
```

Each CPU worker loads its own Whisper model. This makes independent chunks
genuinely parallel, while the original chunk index guarantees correct output
order.

## Requirements

- macOS
- Python 3.11
- FFmpeg
- Enough memory for the chosen model and worker count

Apple Silicon is recommended, but CPU mode can run on Intel Macs as well.

## Install

Install FFmpeg with Homebrew:

```bash
brew install ffmpeg
```

Create an environment and install the Python packages:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit normally opens the app at
[`http://localhost:8501`](http://localhost:8501).

The first use of a model downloads its weights. Once cached by Whisper, the
model can be reused locally.

## Test

After installing the dependencies:

```bash
python -m unittest discover -s tests -v
```

The suite checks media normalization, speech boundary grouping, transcript
ordering, and the spawn-safe Streamlit entry point used by macOS CPU workers.

## Choosing a model

| Model | Relative speed | Relative accuracy | Suggested CPU workers |
|---|---:|---:|---:|
| `tiny` | Fastest | Basic | 4–8 |
| `base` | Very fast | Better | 4–8 |
| `small` | Fast | Good | 3–6 |
| `medium` | Moderate | Very good | 2–4 |
| `large-v3` | Slow | Highest | 1–2 |
| `turbo` | Faster large model | High | 1–2 |

These are starting points, not hard limits. The app allows any value up to the
Mac's logical CPU count.

## CPU workers and memory

Parallel CPU mode uses separate processes because each chunk is independent.
Every process loads another copy of the selected Whisper model. More workers
can improve throughput, but memory use rises with the model size.

On a Mac with substantial unified memory, begin with four workers for
`medium`. For `large-v3` or `turbo`, begin with one or two and increase only if
memory pressure remains comfortable.

## Language choices

- **Detect automatically** works well when recordings can switch between Hindi
  and English.
- **Hindi** forces Whisper's Hindi language mode.
- **English** forces English language mode.
- **Vocabulary hint** gives every chunk the same spelling context for names,
  industry terms, or recurring Hinglish phrases.

## Privacy

The selected file is written only to a temporary local directory. Converted
audio and chunks live in the same temporary workspace and are deleted when
processing finishes. Whisper runs on the Mac; no transcription API key is
needed.

Whisper model weights are downloaded from the model provider on first use.

## Current scope

Included:

- One uploaded media file per run
- Parallel processing of that file's audio chunks
- Model, language, device, worker, and chunk-size controls
- Plain-text export

Intentionally not included yet:

- Speaker diarization
- Multi-file folder queues
- Subtitle formats
- Word-level editing
- Native `.app` packaging
- Side-by-side transcript playback

## Project structure

```text
.
├── .streamlit/
│   └── config.toml
├── mac_whisper_transcriber/
│   ├── __init__.py
│   ├── chunking.py
│   ├── engine.py
│   ├── errors.py
│   ├── media.py
│   ├── pipeline.py
│   └── ui.py
├── tests/
│   └── test_core.py
├── app.py
├── LICENSE
├── README.md
└── requirements.txt
```

## License

MIT
