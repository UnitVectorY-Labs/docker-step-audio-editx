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

assert not torch.cuda.is_available(), "Build checks must not use a GPU"
assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()
assert os.access(os.environ["HOME"], os.W_OK)
assert server.KV_CACHE_MEMORY_BYTES == 4294967296
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
