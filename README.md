# docker-step-audio-editx

Purpose-built CUDA 13 / Python 3.12 voice-cloning API for
[Step-Audio-EditX](https://github.com/stepfun-ai/Step-Audio-EditX), pinned to
`a652e87052c109e26f616d60971376ff47a829d4`. No model weights or credentials are
included. This replaces the approximately 51.4 GB SGLang-based local runtime;
it does **not** install SGLang, DeepSpeed, TRL, WandB, Gradio, quantization tools,
or the upstream training environment.

## Image and validation

`ghcr.io/unitvectory-labs/docker-step-audio-editx:current`

Native Linux ARM64 and AMD64 builds use GitHub-hosted runners, without QEMU.
Every build runs real vLLM/Step/FunASR/CosyVoice imports and model-free FastAPI
health, multipart validation, mocked synthesis, and 24 kHz WAV checks, with
network disabled. Required FunASR registry components, Step1 architecture
support, and the KV/Triton/Transformers compatibility wrappers are also checked.
These are **CPU checks, not GPU inference tests**.

The dependency baseline was inspected from the existing GB10 ARM64 image:
vLLM 0.31.0, PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130, Transformers 5.12.1,
FlashInfer 0.7.0.post1, GPU ONNX Runtime 1.31.0. The replacement has not yet been
validated on a GB10 GPU; AMD64 GPU inference is also untested. CPU-only inference
is unsupported. Use an NVIDIA driver supporting CUDA 13 and NVIDIA Container
Toolkit. Upstream recommends at least 16 GB GPU memory; GB10's unified memory
also serves the host, so do not treat the KV cap as a total memory limit.

The CUDA **devel** base is intentional: Triton/FlashInfer and PyTorch extensions
may compile at runtime. A multi-stage build keeps Git and source checkout
metadata out of the final image; Python packages live in `/opt/venv`, without
`--break-system-packages`. CUDA math libraries and necessary vLLM dependencies
are still substantial. Only FlashInfer's SM121a JIT cache is preinstalled rather
than every GPU architecture's cache. Other GPUs may JIT-compile into writable
cache storage. GPU ONNX support is mandatory and never replaced with CPU ONNX.

## External models: complete directory trees

Download models separately after reviewing their licenses. Mount materialized
snapshots, not Git LFS pointers. Preserve **all** nested files and directories:

```text
/model/                              stepfun-ai/Step-Audio-EditX
  config.json, tokenizer files, *.safetensors, custom Python code, ...
  CosyVoice-300M-25Hz/                complete decoder, YAML and ONNX/checkpoints
/tokenizer/                          stepfun-ai/Step-Audio-Tokenizer
  linguistic_tokenizer.npy
  speech_tokenizer_v1.onnx
  dengcunqin/
    speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online/
      configuration/config/model files, tokenizer resources, ...
```

The exact filenames are controlled by the model repositories; do not use an
allowlist that drops CosyVoice, FunASR, YAML, `.npy`, `.onnx`, or custom code.
HF cache snapshots often contain symlinks to sibling `blobs` directories:
either mount the **entire cache** additionally at the same absolute host path,
or export a materialized model directory. Mounting only a symlink-filled
snapshot at `/model` or `/tokenizer` breaks those links. The upstream loader
uses `trust_remote_code=True`; only use trusted, revision-pinned snapshots.

## Run (operator command; no GPU validation performed here)

Create `/srv/step-cache` on the host and give UID/GID `1000:1000` write access.
Ensure both model trees are readable/traversable by that UID. Then:

```bash
docker run --rm --gpus all --user 1000:1000 \
  --name step-audio-editx --shm-size 2g \
  -p 127.0.0.1:7860:7860 \
  -v /srv/models/Step-Audio-EditX:/model:ro \
  -v /srv/models/Step-Audio-Tokenizer:/tokenizer:ro \
  -v /srv/step-cache:/home/step/.cache \
  ghcr.io/unitvectory-labs/docker-step-audio-editx:current
```

The server binds `0.0.0.0:7860` inside the container. No built-in authentication,
TLS, rate limiting, or upload limit: keep it private or add an authenticated
proxy. HOME `/home/step`, HF, Torch, Triton, and extension caches are writable
for UID 1000. Temporary reference uploads use `/tmp` and are removed after each
request. For `--read-only`, supply writable `/tmp` and cache mounts; that mode
has not been validated. Weights stay read-only and are never auto-downloaded by
the wrapper. FunASR may perform upstream update checks; for strict offline use
restrict network access and prepopulate all resources.

### API

`GET /health`: 200 `{"status":"ok"}` after engine initialization. Startup loads
the models synchronously; the socket may not exist yet while loading. Model-free
tests can observe 503 `{"status":"loading"}` before engine initialization.

`POST /v1/audio/speech`: **multipart/form-data**, with required `text`,
`ref_text` (accurate transcript of the reference), and `ref_audio` file. This is
not the OpenAI JSON speech schema. Inference is serialized. Upstream recommends
audio shorter than 30 seconds per inference.

```bash
curl --fail-with-body http://127.0.0.1:7860/v1/audio/speech \
  -F 'text=Hello from Step Audio.' \
  -F 'ref_text=The words actually spoken in reference.wav.' \
  -F 'ref_audio=@reference.wav;type=audio/wav' \
  --output speech.wav
```

Returns `audio/wav`, 24,000 Hz from the upstream decoder. Missing fields return
422; blank text/transcript or an empty upload returns 400; unready engine returns
503; synthesis errors return 500. Reference WAV is the recommended input;
other formats depend on TorchCodec/FFmpeg support and are not validated.

### Preserved compatibility fixes

`server.py` is copied unchanged from the previous runtime:

- Explicit 4 GiB vLLM KV cache (`STEP_KV_CACHE_MEMORY_BYTES=4294967296`),
  rather than reserving half of GB10 unified memory.
- Explicit `AttentionBackendEnum.TRITON_ATTN` in loader arguments for Step1's
  sqrt-Alibi; setting an environment variable alone is insufficient.
- Transformers 5 `apply_chat_template` output converted to token IDs.
- BF16, eager execution, one sequence, disabled CosyVoice CUDA graphs.

Also configurable: `STEP_MODEL_PATH=/model`, `STEP_TOKENIZER_PATH=/tokenizer`,
`STEP_GPU_MEMORY_UTILIZATION=0.5`, `STEP_MAX_MODEL_LEN=8192`. A positive fixed
KV cache overrides vLLM's utilization-based allocation; setting the byte cap to
zero disables the fix and can consume excessive unified memory.

`entrypoint.py` imports Torch and preloads ONNX CUDA libraries before loading
the unmodified server. Missing compiled CUDA provider is a startup error.
Provider availability in CPU checks does not prove successful GPU execution.

## Maintenance and publication

Builds run on relevant `main` pushes, manual **Build CUDA runtime** dispatch,
and Saturdays at **02:00 UTC**. `current` updates only after both native builds
pass. Each run publishes `sha-<full-commit>-run-<run-id>-<attempt>`; the initial
successful source build also gets `sha-<full-commit>` without overwriting an
existing tag. OCI metadata records this repository and its revision, upstream
revision, and licenses. Pin a digest for reproducible deployment: scheduled
rebuilds intentionally resolve newer base/transitive packages.

Direct inference requirements are tracked in `requirements.txt` and
`requirements-flashinfer.txt`, based on observed versions. Dependabot checks
Python, Docker, and Actions weekly. Treat vLLM, PyTorch/audio/vision, CUDA, and
FlashInfer updates as a coordinated stack; incompatible proposals should not
be merged merely because CPU API tests pass. The upstream training `uv.lock`
is intentionally not used. FunASR's eager registry import makes some otherwise
optional inference packages necessary; imports are tested during the build.
Org project, repository indexer, Semgrep and zizmor workflows follow the shared
UnitVectorY-Labs conventions; their org runners/secrets must be available.

**GHCR visibility:** first publication defaults to private even for a public
repository. A package administrator must use Package settings → Change
visibility → Public. The current CLI lacks package scopes; no authentication
scopes are changed by this project. Until visibility is changed, anonymous pull
may fail. Repository source labels do not make packages public automatically.

## Licenses and responsible use

The Docker/API packaging is MIT (see `LICENSE`); upstream Step code is
Apache-2.0 and its original license remains in `/opt/step-audio-editx/LICENSE`.
The tokenizer model card declares Apache-2.0. The EditX model card currently
does **not** declare a license in Hub metadata: review the complete model
repository's license/terms before use rather than assuming the code license
also covers weights. Nested CosyVoice/FunASR models and all dependencies retain
their own licenses and obligations. No weights are distributed in this image.
Only clone voices with permission; do not impersonate, defraud, or violate
applicable laws.
