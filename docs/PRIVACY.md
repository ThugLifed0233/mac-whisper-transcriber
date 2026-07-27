# Privacy model

ParallelScribe is designed so transcription data stays on the computer running
the application.

## What remains local

- The uploaded audio or video file
- The FFmpeg-normalized WAV
- Speech-detection output
- Temporary audio chunks
- Whisper inference
- The generated transcript

Uploaded media and intermediate files are created inside a temporary
directory. The application removes that directory after a completed or failed
run.

## When network access occurs

ParallelScribe does not call a transcription API. Network access is required
only when Whisper needs to download a model that is not already cached or when
the user installs or updates project dependencies.

After a model is cached, the core transcription pipeline can run without
sending the recording to an external service.

## What the application does not collect

ParallelScribe has no application analytics, account system, advertising SDK,
or telemetry pipeline. It does not require an API key.

Streamlit and installed dependencies retain their own upstream behavior. Users
who require a formally audited or isolated environment should review and pin
those dependencies, restrict outbound network access, and validate the build
under their organization’s security controls.

## Safe usage recommendations

- Run the application on a trusted local account.
- Avoid exposing the Streamlit port to other devices.
- Keep model caches and downloaded transcripts within encrypted local storage
  when recordings are sensitive.
- Review transcripts before sharing them; speech recognition can introduce
  errors.

