# Development process

ParallelScribe was built as an iterative, agent-assisted project. The commit
history intentionally preserves product-sized steps instead of collapsing the
work into one generated-code commit.

## Build progression

1. **Repository foundation**
   - Established the Python 3.11 environment, dependency list, ignore rules,
     and error boundary.

2. **Media preparation**
   - Added FFmpeg normalization and Silero speech detection.
   - Added a deterministic fixed-duration fallback.

3. **Parallel transcription**
   - Added spawn-based CPU processes.
   - Limited Torch to one thread per process.
   - Restored results by original chunk index.

4. **Local interface**
   - Added Streamlit upload, model, language, worker, and chunk-size controls.
   - Added progress, processing metadata, and transcript download.

5. **First-run reliability**
   - Warmed the Whisper cache before creating worker processes so simultaneous
     model downloads cannot collide.

6. **Real smoke testing**
   - Generated a spoken M4A fixture locally.
   - Confirmed FFmpeg conversion, Silero segmentation, five chunks, two CPU
     workers, and ordered text reconstruction.

7. **macOS spawn debugging**
   - Browser testing revealed Streamlit context warnings inside workers.
   - The UI was separated from the guarded entry point.
   - Regression tests were added for the exact failure mode.

8. **ParallelScribe rebrand**
   - Replaced the generic Mac-specific name with a product name centered on
     the parallel-processing design.
   - Renamed the internal package and expanded project documentation.

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Run the regression suite:

```bash
python -m unittest discover -s tests -v
```

Check the installed dependency graph:

```bash
python -m pip check
```

## Manual smoke-test checklist

Use a short recording that contains two clearly ordered phrases.

1. Upload an audio or video format that requires FFmpeg conversion.
2. Select `tiny` for a fast validation run.
3. Select the known language.
4. Choose at least two CPU workers.
5. Reduce the chunk length when necessary to create multiple chunks.
6. Confirm the processing details report multiple chunks and workers.
7. Verify that phrases remain in their original order.
8. Download the transcript and confirm the filename and contents.
9. Inspect the application logs for worker or Streamlit warnings.

## Change discipline

- Keep media, chunking, execution, and UI changes in their respective modules.
- Add a regression test when fixing a reproducible failure.
- Do not commit recordings, model weights, normalized audio, or chunk files.
- Prefer small commits that explain how the product evolved.
- Document memory-impacting changes because every CPU worker owns a model.

## Release checklist

- Run unit and integration checks.
- Perform one real CPU-parallel transcription.
- Confirm the Streamlit page loads and downloads a transcript.
- Verify README links and images from GitHub.
- Confirm the GitHub visibility and default branch.
- Record any untested hardware path, especially Apple MPS.

