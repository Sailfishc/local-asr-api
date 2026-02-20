# Changelog

All notable changes to the OpenAI-compatible API and model support are documented here.

---

## [1.3.0] – Repository cleanup and Git removed (minimal local folder)

### Summary

Repository was streamlined to a minimal codebase for POC and future development: removed demo/test scripts, development images, export tooling, and generated certs; updated README and .gitignore.

### Removed

- **Demo / testing scripts:** `demo1.py`, `demo2.py`, `demo_onnx.py`, `demo_libtorch.py`, `webui.py`
- **Export tooling:** `export.py`, `export_meta.py`; `utils/export_utils.py`, `utils/model_bin.py` (if present)
- **Development images:** all files in `image/` (benchmark and UI screenshots). Empty `image/` directory removed.
- **Generated certs:** `certs/key.pem`, `certs/cert.pem`. Empty `certs/` directory removed.
- **Example data:** `data/train_example.jsonl`, `data/val_example.jsonl`. Empty `data/` removed. For finetuning, create `data/` and add your own `train.jsonl` and `val.jsonl` (see FunASR format).
- **Git metadata:** `.git` and `.github` directories removed. This folder is now a plain local directory with no version control.

### Changed

- **README.md:** Removed image references and WebUI section; fixed broken links to removed demo/export files; simplified Community section.
- **.gitignore:** Added `.venv`, `tmp/`, `certs/*.pem`.

### Kept (minimal core)

- **API & model:** `openai_whisper_compatible_api.py`, `api.py`, `model.py`
- **Utils:** `utils/ctc_alignment.py` (used by model), `utils/frontend.py`, `utils/infer_utils.py`
- **Finetuning:** `finetune.sh`, `deepspeed_conf/`, `data/train_example.jsonl`, `data/val_example.jsonl`
- **Docs & config:** `README.md`, `CHANGELOG.md`, `LICENSE`, `requirements.txt`
- **Git removed:** `.git` and `.github` deleted; folder is a plain local directory (no pull requests).

---

## [1.2.0] – Progress bar and dependency fix

### Summary

Transcription requests now show a **terminal progress bar** so you can see that the system is processing. The MLX path no longer requires `torch` at request time; if you see `No module named 'torch'`, install `torch` and `torchaudio` (in `requirements.txt`).

### Added

- **Progress bar** during `POST /v1/audio/transcriptions`: a terminal progress bar (via `tqdm`) advances while the model transcribes, so you can tell whether the process is still running or has stalled.
- **`tqdm`** in `requirements.txt` for the progress bar.
- **`_get_audio_duration_no_torch()`**: Gets audio duration for the progress bar using the stdlib `wave` module for `.wav` files (no torch); other formats use a default duration so the bar still moves.

### Changed

- **Transcription flow**: Duration is always computed without torch at the start of the request. Torch/torchaudio are only imported inside the SenseVoice branch of `run_inference()`, so the MLX path does not require torch for the progress bar.
- **Dependencies**: `torch` and `torchaudio` are listed in `requirements.txt`; installing them resolves `500 Internal Server Error` / `No module named 'torch'` when using the transcription endpoint (e.g. with MLX in some environments).

### Fixed

- **`500 Internal Server Error` / `No module named 'torch'`**: The progress bar initially loaded the uploaded file with `torchaudio` for all backends, so MLX-only environments (without torch) failed. Fix: use `_get_audio_duration_no_torch()` for duration; import torch only inside the SenseVoice branch of `run_inference()`. If the error persists (e.g. pulled in by the MLX stack), install `torch` and `torchaudio` as in `requirements.txt`.

---

## [1.1.0] – Multi-backend model support (SenseVoice + MLX)

### Summary

The API can now serve **multiple backends** (SenseVoice or MLX) and **switch models** via environment variables. Models are downloaded once and cached; the same OpenAI-compatible endpoints are used.

### Added

- **`BACKEND`** env var: `"sensevoice"` (default) or `"mlx"`. Chooses which engine loads at startup.
- **`LOCAL_MODEL`** env var: Model identifier (e.g. Hugging Face repo id for MLX, or ModelScope id for SenseVoice). Defaults depend on `BACKEND`.
- **MLX backend** (`BACKEND=mlx`): Uses `mlx-audio` to load STT models from Hugging Face (e.g. `mlx-community/Qwen3-ASR-1.7B-8bit`). Download once to `~/.cache/huggingface/hub`.
- **`_load_sensevoice()` / `_load_mlx()`**: Lazy backend loading; only the selected backend is loaded.
- **`model_inference()`** dispatcher: Routes transcription to SenseVoice or MLX based on `BACKEND`. MLX uses file path; SenseVoice uses waveform tuple.
- **`/` and `/v1/models`** now return current `backend` and `model` (LOCAL_MODEL) for visibility.
- **`python-multipart`** in `requirements.txt`: Required by FastAPI for form/file uploads (`POST /v1/audio/transcriptions`).
- **`mlx-audio`** in `requirements.txt**: Optional for MLX backend; install when using `BACKEND=mlx`.

### Changed

- **Startup**: Single hardcoded SenseVoice load replaced by branch on `BACKEND` (SenseVoice or MLX).
- **Transcription**: `POST /v1/audio/transcriptions` branches on backend—MLX gets `audio_path`, SenseVoice gets `input_wav` as before.
- **Form handling**: `language` normalized so it is never `None` (clients may omit the field or send null).
- **SenseVoice paths**: Optional `SENSEVOICE_LOCAL_PATH` and `SENSEVOICE_VAD_PATH` for custom cache paths.

### Fixed

- **`'NoneType' object has no attribute 'lower'`**: Some clients (e.g. Spokenly) omit the `language` form field or send `null`. Backends (e.g. MLX) then received `None` and called `.lower()` on it. Fix: normalize `language` to `"auto"` when missing/empty in the endpoint, and in MLX inference pass a string (e.g. `"auto"`) to `generate()` instead of `None`.
- **`Form data requires "python-multipart" to be installed`**: FastAPI needs `python-multipart` for `Form()` and `File()`. Added to `requirements.txt`.

### Server behavior

- **HTTP only**: Server runs at `http://0.0.0.0:8000` (e.g. `http://127.0.0.1:8000` for local clients). No HTTPS or certificates in this release.

---

## Change history and lessons learned

### Errors we hit (and fixes)

| Error | Cause | Fix |
|-------|--------|-----|
| Saw SenseVoice loading instead of MLX | Default `BACKEND` is `sensevoice` | Set `BACKEND=mlx` (and optionally `LOCAL_MODEL=...`) when running the script. |
| `Library not loaded: libmlx.dylib` | MLX runtime not correctly installed | Use a dedicated venv, reinstall `mlx` and `mlx-audio`; MLX targets Apple Silicon. |
| `Form data requires "python-multipart"` | Dependency missing for FastAPI form/file handling | Add `python-multipart` to `requirements.txt` and install. |
| `'NoneType' object has no attribute 'lower'` | Client sent no `language` (or null); backend called `.lower()` on it | Normalize `language` in the endpoint and in MLX inference so a string is always passed (never `None`). |
| Connection failed / invalid certificate (Spokenly) | Attempted HTTPS with self-signed cert; ATS/cert trust issues | Reverted to HTTP-only; user confirmed HTTP works for their use. |
| `500` / `No module named 'torch'` | Transcription path used torch/torchaudio; MLX-only env had no torch | Use duration without torch for progress bar; import torch only in SenseVoice branch. Install `torch` and `torchaudio` if error persists. |

### Lessons learned

1. **Default backend**: Keep default `BACKEND=sensevoice` so existing behavior is unchanged unless the user opts in to MLX.
2. **Third-party APIs**: Never pass `None` for parameters that downstream code might use as a string (e.g. `.lower()`). Normalize at the boundary (e.g. `language or "auto"`).
3. **FastAPI file upload**: Any endpoint using `Form()` or `File()` needs `python-multipart` installed.
4. **HTTPS vs HTTP**: Don’t assume HTTPS is required. Prefer simple HTTP first; add HTTPS only if the client or environment requires it.
5. **Caching**: Both backends use standard caches (ModelScope for SenseVoice, Hugging Face for MLX); no extra “download once” logic needed beyond using the same model id.

---

## Switching to another model (minimal steps)

### Same backend, different model (e.g. another MLX model)

- **Code**: No change.
- **Run**: Set env and start server.
  ```bash
  export BACKEND=mlx
  export LOCAL_MODEL=mlx-community/your-other-model
  python openai_whisper_compatible_api.py
  ```
- **First run**: New model downloads to cache; later runs use cache.

### New backend (e.g. another STT library)

1. **Env/config**: Add a new `BACKEND` value (e.g. `"whisper"`).
2. **Load**: Add `_load_<backend>()` that loads the model (and uses its cache if any).
3. **Inference**: Add `model_inference_<backend>()` with the same contract: input (path or waveform), `language`, returns `str` text.
4. **Dispatch**: In startup, branch on `BACKEND` and set `model = "..."`. In `model_inference()` and in `transcriptions`, branch on `model` and call the right function.
5. **Dependencies**: Add the new library to `requirements.txt`.
6. **Errors to prevent**:
   - Normalize all request parameters (e.g. `language`) so backends never receive `None` where they expect a string.
   - If the new backend expects a file path vs in-memory audio, mirror the MLX vs SenseVoice pattern (path for MLX, waveform for SenseVoice) in the transcriptions endpoint.

### Updating this changelog

When adding a new backend or model option:

- Add an **Added** line under the current (or new) version.
- If you fixed a bug while integrating, add a **Fixed** line.
- Optionally add a short note under **Switching to another model** or **Lessons learned** if it’s reusable.

---

[1.3.0]: https://github.com/modelscope/SenseVoice/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/modelscope/SenseVoice/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/modelscope/SenseVoice/compare/v1.0.0...v1.1.0
