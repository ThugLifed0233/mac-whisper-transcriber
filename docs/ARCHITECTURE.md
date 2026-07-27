# ParallelScribe architecture

ParallelScribe is deliberately split into small stages so media handling,
speech segmentation, model execution, and the interface can change
independently.

## Processing pipeline

1. **Media normalization**
   - FFmpeg selects the first audio stream.
   - Audio is converted to 16 kHz mono PCM WAV.
   - Every later stage receives one predictable format.

2. **Speech segmentation**
   - Silero VAD identifies regions that contain speech.
   - Nearby regions are grouped without exceeding the selected chunk length.
   - A fixed-duration splitter is used when VAD is unavailable or returns no
     speech regions.

3. **Whisper execution**
   - CPU mode creates a spawn-based process pool.
   - Each worker owns one Whisper model and one Torch thread.
   - MPS mode uses one sequential model because multiple processes cannot
     safely share the same Metal device model.

4. **Ordered reconstruction**
   - Every audio chunk receives a stable numeric index before execution.
   - Workers may finish in any order.
   - Results are collected by index and sorted before joining.
   - Empty results are skipped, while non-empty chunks remain one line apart.

5. **Local cleanup**
   - Uploaded media, normalized WAV files, and chunks live in a temporary
     directory.
   - The temporary workspace is removed when the run completes or fails.

## Concurrency model

ParallelScribe uses processes instead of threads for CPU transcription. This
allows independent Whisper jobs to run concurrently without relying on one
shared Python interpreter.

Each worker loads its own model, which creates a clear tradeoff:

- More workers can reduce elapsed time.
- Larger models multiply memory use quickly.
- The worker count is capped by the number of chunks and logical CPU cores.
- Torch receives one thread per worker to avoid nested CPU oversubscription.

Before the pool starts, the parent process loads and releases the model once.
This ensures that first-run model downloads complete before several workers
try to use the same cache file.

## Spawn-safe interface

macOS uses the `spawn` multiprocessing strategy. A spawned process imports the
application entry point again, so executing Streamlit at module-import time
would cause every worker to recreate the interface.

`app.py` is therefore a guarded launcher. The Streamlit page lives in
`parallel_scribe/ui.py` and is executed only by the primary script process.
Regression coverage verifies that importing the entry point as
`__mp_main__` does not import the UI module.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `parallel_scribe/media.py` | FFmpeg discovery and media normalization |
| `parallel_scribe/chunking.py` | Audio loading, VAD, boundary grouping, WAV chunks |
| `parallel_scribe/engine.py` | Whisper models, CPU pool, MPS execution |
| `parallel_scribe/pipeline.py` | End-to-end orchestration and metadata |
| `parallel_scribe/ui.py` | Streamlit controls, progress, and transcript download |
| `app.py` | Spawn-safe Streamlit launcher |

## Error boundaries

Expected failures are converted into `TranscriptionError` with user-readable
messages. These include missing FFmpeg, invalid media, empty audio, unavailable
MPS, model-loading errors, and worker failures. Unexpected errors remain
visible in the interface without exposing uploaded media.

## Current architectural limits

- CPU workers duplicate model memory.
- Chunk boundaries do not yet preserve word-level timestamps.
- The pipeline handles one uploaded file per run.
- Speaker diarization and subtitle generation are intentionally outside the
  current scope.

