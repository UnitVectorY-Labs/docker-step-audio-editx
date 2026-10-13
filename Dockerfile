# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS python-base
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev ca-certificates ffmpeg sox libsox-fmt-all \
    libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM python-base AS build
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
COPY requirements*.txt /tmp/requirements/
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements/requirements.txt \
    && pip install -r /tmp/requirements/requirements-flashinfer.txt \
    && pip check
ARG STEP_REF=a652e87052c109e26f616d60971376ff47a829d4
RUN git init /opt/step-audio-editx \
    && git -C /opt/step-audio-editx remote add origin https://github.com/stepfun-ai/Step-Audio-EditX.git \
    && git -C /opt/step-audio-editx fetch --depth 1 origin "${STEP_REF}" \
    && git -C /opt/step-audio-editx checkout --detach FETCH_HEAD \
    && rm -rf /opt/step-audio-editx/.git

FROM python-base AS runtime
ENV HOME=/home/step XDG_CACHE_HOME=/home/step/.cache \
    HF_HOME=/home/step/.cache/huggingface TORCH_HOME=/home/step/.cache/torch \
    TRITON_CACHE_DIR=/home/step/.cache/triton \
    TORCH_EXTENSIONS_DIR=/home/step/.cache/torch_extensions \
    TOKENIZERS_PARALLELISM=false VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    STEP_MODEL_PATH=/model STEP_TOKENIZER_PATH=/tokenizer \
    STEP_GPU_MEMORY_UTILIZATION=0.5 STEP_MAX_MODEL_LEN=8192 \
    STEP_KV_CACHE_MEMORY_BYTES=4294967296
LABEL org.opencontainers.image.source="https://github.com/UnitVectorY-Labs/docker-step-audio-editx" \
    org.opencontainers.image.title="Step-Audio-EditX CUDA runtime" \
    org.opencontainers.image.licenses="MIT AND Apache-2.0" \
    ai.step-audio.upstream.revision="a652e87052c109e26f616d60971376ff47a829d4"
# Ubuntu's CUDA base already contains UID 1000. Reuse it without assuming its name.
RUN if ! getent group 1000 >/dev/null; then groupadd --gid 1000 step; fi \
    && if ! getent passwd 1000 >/dev/null; then useradd --uid 1000 --gid 1000 --home-dir /home/step step; fi \
    && install -d -o 1000 -g 1000 /home/step /home/step/.cache /model /tokenizer
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/step-audio-editx /opt/step-audio-editx
WORKDIR /opt/step-audio-editx
COPY server.py entrypoint.py smoke_test.py ./
USER 1000:1000
# No GPU, network, model initialization, or weight downloads in this check.
RUN --network=none python smoke_test.py
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"]
ENTRYPOINT ["python", "entrypoint.py"]
