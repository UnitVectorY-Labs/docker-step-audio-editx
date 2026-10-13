"""HTTP wrapper around the official Step-Audio-EditX voice-cloning engine."""
import asyncio
import io
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from model_loader import ModelSource, model_loader
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("step-audio-api")

MODEL_PATH = os.environ.get("STEP_MODEL_PATH", "/model")
TOKENIZER_PATH = os.environ.get("STEP_TOKENIZER_PATH", "/tokenizer")
GPU_MEMORY_UTILIZATION = float(os.environ.get("STEP_GPU_MEMORY_UTILIZATION", "0.5"))
MAX_MODEL_LEN = int(os.environ.get("STEP_MAX_MODEL_LEN", "8192"))
KV_CACHE_MEMORY_BYTES = int(os.environ.get("STEP_KV_CACHE_MEMORY_BYTES", str(4 * 1024**3)))

engine = None
inference_lock = threading.Lock()


def load_engine():
    global engine
    # StepAudioTTS does not expose vLLM's kv_cache_memory_bytes argument.
    # Its loader accepts **kwargs and forwards them to vllm.LLM.
    original_load_model = model_loader.load_model

    def bounded_load_model(*args, **kwargs):
        if KV_CACHE_MEMORY_BYTES > 0:
            kwargs.setdefault("kv_cache_memory_bytes", KV_CACHE_MEMORY_BYTES)
        # Environment-based backend selection is not reliably honored by the
        # offline LLM API in vLLM 0.31. Step1 uses sqrt-Alibi, which its
        # FlashAttention backend rejects; pass Triton explicitly instead.
        attention_config = kwargs.setdefault("attention_config", {})
        if isinstance(attention_config, dict):
            attention_config.setdefault("backend", AttentionBackendEnum.TRITON_ATTN)
        return original_load_model(*args, **kwargs)

    model_loader.load_model = bounded_load_model
    logger.info("Loading Step-Audio tokenizer from %s", TOKENIZER_PATH)
    audio_tokenizer = StepAudioTokenizer(TOKENIZER_PATH, model_source=ModelSource.LOCAL)
    logger.info("Loading Step-Audio-EditX engine from %s", MODEL_PATH)
    engine = StepAudioTTS(
        MODEL_PATH,
        audio_tokenizer,
        model_source=ModelSource.LOCAL,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        dtype="bfloat16",
        max_num_seqs=1,
        cosyvoice_dtype="bfloat16",
        cosyvoice_cuda_graph=False,
    )
    # Transformers 5 returns BatchEncoding from apply_chat_template by
    # default, while this upstream Step-Audio revision expects list[int].
    # Keep the engine API compatible by returning only the prompt token IDs.
    tokenizer = engine.tokenizer
    original_apply_chat_template = tokenizer.apply_chat_template

    def apply_chat_template_token_ids(*args, **kwargs):
        result = original_apply_chat_template(*args, **kwargs)
        if hasattr(result, "input_ids"):
            return result.input_ids
        if isinstance(result, dict) and "input_ids" in result:
            return result["input_ids"]
        return result

    tokenizer.apply_chat_template = apply_chat_template_token_ids
    logger.info("Step-Audio-EditX engine ready")


def _run_clone(ref_path: str, ref_text: str, text: str):
    with inference_lock:
        return engine.clone(ref_path, ref_text, text)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_engine()
    yield


app = FastAPI(title="Step-Audio-EditX TTS", lifespan=lifespan)


@app.get("/health")
def health():
    if engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.post("/v1/audio/speech")
async def tts(text: str = Form(...), ref_text: str = Form(...), ref_audio: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(status_code=503, detail="engine is loading")
    if not text.strip() or not ref_text.strip():
        raise HTTPException(status_code=400, detail="text and ref_text are required")
    payload = await ref_audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="ref_audio is empty")

    suffix = os.path.splitext(ref_audio.filename or "reference.wav")[1] or ".wav"
    fd, ref_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        # vLLM and the audio pipeline are synchronous and share a single GPU
        # engine; serialize inference calls off the event loop.
        audio, sample_rate = await asyncio.to_thread(_run_clone, ref_path, ref_text, text)
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().float().cpu().numpy().squeeze()
        output = io.BytesIO()
        sf.write(output, audio, sample_rate, format="WAV")
        return Response(output.getvalue(), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(ref_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
