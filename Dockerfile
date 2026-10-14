# syntax=docker/dockerfile:1
# Builder only: the published image is Google's genuine Distroless CC base.
FROM nvidia/cuda:13.4.2-devel-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev ca-certificates git g++ binutils \
    ffmpeg sox libsox-fmt-all libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
COPY requirements*.txt /tmp/requirements/
COPY check_dependencies.py /tmp/check_dependencies.py
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements/requirements.txt \
    && pip install -r /tmp/requirements/requirements-flashinfer.txt \
    && python /tmp/check_dependencies.py
COPY patch_jit_cache_shim.py /tmp/patch_jit_cache_shim.py
COPY requirements-shim/requirements.txt /tmp/requirements-shim.txt
RUN pip download --no-deps -r /tmp/requirements-shim.txt -d /tmp/shim-original \
    && python /tmp/patch_jit_cache_shim.py /tmp/shim-original/*.whl /tmp/shim-selected/$(basename /tmp/shim-original/*.whl) \
    && pip install --no-deps /tmp/shim-selected/*.whl \
    && python /tmp/check_dependencies.py
ARG STEP_REF=a652e87052c109e26f616d60971376ff47a829d4
RUN git init /opt/step-audio-editx \
    && git -C /opt/step-audio-editx remote add origin https://github.com/stepfun-ai/Step-Audio-EditX.git \
    && git -C /opt/step-audio-editx fetch --depth 1 origin "${STEP_REF}" \
    && git -C /opt/step-audio-editx checkout --detach FETCH_HEAD \
    && rm -rf /opt/step-audio-editx/.git
COPY assemble_runtime.py /tmp/assemble_runtime.py
RUN python /tmp/assemble_runtime.py \
    && rm -rf /opt/venv/lib/python3.12/site-packages/pip \
       /opt/venv/lib/python3.12/site-packages/pip-*.dist-info \
       /opt/venv/bin/pip /opt/venv/bin/pip3 /opt/venv/bin/pip3.12

FROM gcr.io/distroless/cc-debian13:nonroot AS runtime
ENV PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:/usr/bin:/bin \
    HOME=/home/step XDG_CACHE_HOME=/home/step/.cache \
    HF_HOME=/home/step/.cache/huggingface TORCH_HOME=/home/step/.cache/torch \
    TRITON_CACHE_DIR=/home/step/.cache/triton \
    TORCH_EXTENSIONS_DIR=/home/step/.cache/torch_extensions \
    CUDA_HOME=/opt/venv/lib/python3.12/site-packages/nvidia/cu13 \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:/opt/venv/lib/python3.12/site-packages/torch/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TOKENIZERS_PARALLELISM=false VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    STEP_MODEL_PATH=/model STEP_TOKENIZER_PATH=/tokenizer \
    STEP_GPU_MEMORY_UTILIZATION=0.5 STEP_MAX_MODEL_LEN=8192 \
    STEP_KV_CACHE_MEMORY_BYTES=4294967296
LABEL org.opencontainers.image.source="https://github.com/UnitVectorY-Labs/docker-step-audio-editx" \
    org.opencontainers.image.title="Step-Audio-EditX Distroless-derived CUDA runtime" \
    org.opencontainers.image.licenses="MIT AND Apache-2.0" \
    ai.step-audio.upstream.revision="a652e87052c109e26f616d60971376ff47a829d4"
# Selectively copied interpreter, audio libraries and runtime JIT toolchain;
# retain Google's glibc/libstdc++ and exclude apt, Git and builder CUDA toolkit.
COPY --from=build /runtime-root/ /
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/step-audio-editx /opt/step-audio-editx
COPY --from=build --chown=1000:1000 /runtime-home/ /home/step/
WORKDIR /opt/step-audio-editx
COPY server.py entrypoint.py smoke_test.py ./
USER 0:0
# Generate against the final Google-based filesystem, not Ubuntu builder paths.
RUN --network=none ["/usr/sbin/ldconfig"]
USER 1000:1000
# Tests execute in the FINAL Google-based runtime, not the builder or old image.
RUN --network=none ["/opt/venv/bin/python", "smoke_test.py"]
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"]
ENTRYPOINT ["/opt/venv/bin/python", "entrypoint.py"]
