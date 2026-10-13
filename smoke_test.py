"""Real imports plus model-free API checks; deliberately does not run lifespan."""
import io
import os
from types import SimpleNamespace

import numpy as np
import onnxruntime
import soundfile as sf
import torch
from fastapi.testclient import TestClient

import server
from stepvocoder.cosyvoice2.flow.flow import CausalMaskedDiffWithXvec
import stepvocoder.cosyvoice2.flow.decoder_dit
from funasr_detach.register import tables
from vllm.model_executor.models import ModelRegistry

assert not torch.cuda.is_available(), "Build checks must not use a GPU"
assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()
assert os.access(os.environ["HOME"], os.W_OK)
assert server.KV_CACHE_MEMORY_BYTES == 4294967296
assert "Step1ForCausalLM" in ModelRegistry.get_supported_archs()
# FunASR swallows optional registry import failures; check the actual local
# tokenizer architecture explicitly rather than trusting package import alone.
for table, name in [(tables.model_classes, "ParaformerStreaming"),
                    (tables.encoder_classes, "SANMEncoderChunkOpt"),
                    (tables.decoder_classes, "ParaformerSANMDecoder"),
                    (tables.predictor_classes, "CifPredictorV2"),
                    (tables.frontend_classes, "WavFrontendOnline")]:
    assert name in table, f"Required FunASR component did not import: {name}"
client = TestClient(server.app)
assert client.get("/health").status_code == 503
assert client.post("/v1/audio/speech").status_code == 422
server.engine = SimpleNamespace(clone=lambda *_: (np.zeros(2400, dtype=np.float32), 24000))
assert client.get("/health").json() == {"status": "ok"}
response = client.post("/v1/audio/speech", data={"text": "hello", "ref_text": "reference"},
                       files={"ref_audio": ("ref.wav", b"mock-reference", "audio/wav")})
assert response.status_code == 200, response.text
assert response.headers["content-type"] == "audio/wav"
audio, rate = sf.read(io.BytesIO(response.content))
assert rate == 24000 and len(audio) == 2400
assert client.post("/v1/audio/speech", data={"text": " ", "ref_text": "reference"},
                   files={"ref_audio": ("ref.wav", b"x")}).status_code == 400
assert client.post("/v1/audio/speech", data={"text": "hello", "ref_text": "reference"},
                   files={"ref_audio": ("ref.wav", b"")}).status_code == 400
print("CPU imports/API/WAV checks passed; GPU inference NOT tested")

# Exercise the compatibility wrappers with fakes, never constructing an LLM.
seen = {}
original_loader = server.model_loader.load_model
original_audio_tokenizer = server.StepAudioTokenizer
original_tts = server.StepAudioTTS
def fake_loader(*args, **kwargs):
    seen.update(kwargs)
    return None
class FakeTTS:
    def __init__(self, *args, **kwargs):
        server.model_loader.load_model("/model")
        self.tokenizer = SimpleNamespace(apply_chat_template=lambda *_args, **_kwargs:
                                         SimpleNamespace(input_ids=[1, 2, 3]))
server.model_loader.load_model = fake_loader
server.StepAudioTokenizer = lambda *_args, **_kwargs: None
server.StepAudioTTS = FakeTTS
try:
    server.load_engine()
    assert seen["kv_cache_memory_bytes"] == 4294967296
    assert seen["attention_config"]["backend"] == server.AttentionBackendEnum.TRITON_ATTN
    assert server.engine.tokenizer.apply_chat_template([]) == [1, 2, 3]
finally:
    server.model_loader.load_model = original_loader
    server.StepAudioTokenizer = original_audio_tokenizer
    server.StepAudioTTS = original_tts
print("KV cap/Triton/Transformers compatibility checks passed")
