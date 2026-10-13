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

The final image uses Google's actual **`gcr.io/distroless/cc-debian13:nonroot`**
base, not an Ubuntu image relabeled as Distroless. A CUDA 13.4.2 Ubuntu builder
supplies CPython 3.12 and selected audio/JIT dependencies; its toolkit and
package manager are not copied. Debian 13's newer glibc supports the copied
Ubuntu 24.04 interpreter ABI. Python packages live in `/opt/venv`, without
`--break-system-packages`; pip, apt, Git and checkout metadata are absent.

**Distroless tradeoff:** this is a Distroless-**derived** runtime, not a fully
shell-free/compiler-free image. Triton compiles its CUDA driver launcher with
a C compiler at runtime, even when attention kernels are cached. FlashInfer
and PyTorch extension fallback compilation uses Ninja, GCC/C++, NVCC and
`/bin/sh`. These specific tools and headers are deliberately retained rather
than silently disabling GPU paths. CUDA libraries and NVCC come from the
CUDA 13 Python wheels, not the builder's CUDA 13.4 toolkit. The final-runtime
CPU check compiles a native shared object and tests TorchCodec WAV decoding,
FFmpeg, SoX and Ninja in addition to the imports/API checks.

CUDA math libraries and necessary vLLM dependencies remain substantial. Only
SM121a (ARM64 GB10) / SM120f (AMD64 Blackwell) JIT caches are preinstalled,
rather than every GPU architecture's cache. Other GPUs may JIT-compile into
writable storage. GPU ONNX support is mandatory, never replaced with CPU ONNX.

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
pass. `current-arm64` and `current-amd64` are named platform aliases, also
updated only after both required builds pass. Each run publishes
`sha-<full-commit>-run-<run-id>-<attempt>`; the initial
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

**GHCR access:** verify anonymous access after publication rather than assuming
visibility from the public source repository. If access is denied, a package
administrator can use Package settings → Change visibility → Public. No CLI
authentication scopes are changed by this project.

### Dependabot compatibility review

The initial PRs were reviewed individually and their applicable changes are
incorporated: CUDA builder 13.4.2, setup-uv 10.3.0, CodeQL SARIF action 4.38.3,
explicit CUDA 13 local versions for Torch/audio/vision/codec, Diffusers 0.38.0
security fixes. NumPy remains 2.3.5, the newest compatible 2.3 release: the
actual resolver rejected 2.4.6 because vLLM's Mistral-common dependency requires
NumPy <2.4 on Python 3.12.
vLLM's latest release is still 0.31.0 and explicitly requires Torch 2.13.0,
torchaudio 2.11.0, torchvision 0.28.0, FlashInfer 0.7.0.post1 and Numba 0.65.0.
Therefore proposed Torch 2.14.1 / torchvision 0.29.1 / FlashInfer 0.7.1 cannot
be combined with it. Numba also requires NumPy <2.5, so proposed NumPy 2.5.2
cannot be installed either. These are upstream dependency constraints, not blanket
Dependabot ignores. Python updates are grouped for coordinated review.

NVIDIA's cuSPARSELt 0.8.1 ARM wheel declares the nonstandard
`py3-none-manylinux2014_sbsa` tag. The dependency check permits **only** that
exact aarch64 tag warning; all dependency conflicts remain fatal. Final-image
Torch imports and ELF linking are checked. This is a packaging-tag workaround,
not evidence of GPU inference validation.

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
