"""Tests for the public/GPU-less deployment path: code policy, hosted-model
generator, rate limiting, and the sandbox environment."""

import json

import pytest

from verification.policy import check_code
from verification.verify_kernel import _sandbox_env

KERNEL = """import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
"""


@pytest.mark.parametrize(
    "source, fragment",
    [
        ("import os", "`os`"),
        ("import subprocess as sp", "`subprocess`"),
        ("from socket import socket", "`socket`"),
        ("from . import x", "imports from"),
        ("eval('1+1')", "`eval()`"),
        ("f = open", "`open`"),
        ("x = ().__class__.__bases__[0].__subclasses__()", "accesses `.__"),
        ("__builtins__['exec']('1')", "`__builtins__`"),
        ("import torch\ntorch.load('weights.pt')", "`torch.load`"),
        ("import torch\ntorch.utils.cpp_extension.load_inline()", "`torch.utils`"),
        ("import numpy as np\nnp.fromfile('/etc/passwd')", "`np.fromfile`"),
        ("def f(:\n  pass", "syntax error"),
    ],
)
def test_policy_rejects_escape_routes(source, fragment):
    reason = check_code(source)
    assert reason is not None and fragment in reason


def test_policy_allows_ordinary_kernels():
    assert check_code(KERNEL) is None
    assert check_code("import math\nfrom typing import Optional\nx = math.sqrt(2)") is None


def test_policy_accepts_every_catalog_reference_and_kernel():
    pytest.importorskip("fastapi")
    from app.backend.catalog import Catalog

    for t in Catalog().templates:
        d = t.detail()
        assert check_code(d["pytorch_reference"]) is None, t.id
        assert check_code(d["smoke_reference"]) is None, t.id


def test_sandbox_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("KERNELFORGE_LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = _sandbox_env()
    assert "KERNELFORGE_LLM_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "PATH" in env


# ---------------- hosted-model generator ----------------

httpx = pytest.importorskip("httpx")

from kernelforge.remote import RemoteChatGenerator, RemoteGenerationError  # noqa: E402


def _sse(*chunks):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks]
    return ("\n\n".join(lines + ["data: [DONE]"]) + "\n\n").encode()


def _generator(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return RemoteChatGenerator("https://llm.example/v1/", "key", "some-model", max_tokens=64, client=client)


def test_remote_streams_parallel_candidates_with_the_request_contract():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((str(request.url), request.headers["authorization"], body))
        return httpx.Response(200, content=_sse("kernel", ":", " ok"),
                              headers={"content-type": "text/event-stream"})

    gen = _generator(handler)
    out: dict[int, str] = {}
    for i, text in gen.stream_chat([{"role": "user", "content": "hi"}], num_sequences=3, temperature=0.8):
        out[i] = out.get(i, "") + text

    assert out == {0: "kernel: ok", 1: "kernel: ok", 2: "kernel: ok"}
    assert len(seen) == 3
    url, auth, body = seen[0]
    assert url == "https://llm.example/v1/chat/completions"
    assert auth == "Bearer key"
    assert body["stream"] is True and body["temperature"] == 0.8 and body["model"] == "some-model"
    assert body["max_tokens"] == 64


def test_remote_raises_when_every_candidate_fails():
    gen = _generator(lambda request: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(RemoteGenerationError, match="HTTP 401"):
        list(gen.stream_chat([{"role": "user", "content": "hi"}], num_sequences=2))


def test_remote_marks_a_single_failed_candidate_and_keeps_the_rest():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, content=_sse("fine"), headers={"content-type": "text/event-stream"})

    gen = _generator(handler)
    out: dict[int, str] = {}
    for i, text in gen.stream_chat([{"role": "user", "content": "hi"}], num_sequences=2):
        out[i] = out.get(i, "") + text
    texts = sorted(out.values())
    assert "fine" in texts
    assert any("generation failed" in t and "429" in t for t in texts)


# ---------------- public mode ----------------

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.backend import main  # noqa: E402


@pytest.fixture
def public_client(monkeypatch):
    for var in ("KERNELFORGE_ADAPTER_PATH", "KERNELFORGE_PRELOAD", "KERNELFORGE_MOCK_MODEL", "KERNELFORGE_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KERNELFORGE_PUBLIC", "1")
    monkeypatch.setattr(main, "_generator", None)
    main._rate_limiter.clear()
    main._verification_cache.clear()
    return TestClient(main.app)


def _template(client):
    tid = client.get("/templates").json()[0]["id"]
    return client.get(f"/templates/{tid}").json()


def test_public_without_llm_falls_back_to_mock(public_client):
    body = public_client.get("/health").json()
    assert body["model_variant"] == "mock"
    assert body["public"] is True
    assert body["limits"]["rate_limit_per_min"] == main.DEFAULT_RATE_LIMIT


def test_llm_env_selects_the_remote_variant(public_client, monkeypatch):
    monkeypatch.setenv("KERNELFORGE_LLM_MODEL", "provider/coder-32b")
    body = public_client.get("/health").json()
    assert body["model_variant"] == "remote"
    assert body["model_name"] == "provider/coder-32b"
    assert "not KernelForge's fine-tuned adapter" in body["banner"]
    assert isinstance(main._make_generator("remote"), RemoteChatGenerator)


def test_public_rejects_references_that_break_policy(public_client):
    resp = public_client.post("/generate", json={
        "description": "x", "pytorch_reference": "import os\nos.system('id')", "verify": False})
    assert resp.status_code == 422
    assert "code policy" in resp.json()["detail"]


def test_public_blocks_policy_violating_kernels_without_running_them(public_client, monkeypatch):
    ran = []
    monkeypatch.setattr(main, "verify_model_kernel", lambda *a, **k: ran.append(1))
    monkeypatch.setattr(main, "_triton_version", lambda: "3.5.0")
    t = _template(public_client)
    resp = public_client.post("/verify", json={
        "pytorch_reference": t["smoke_reference"],
        "triton_kernel": "import subprocess\nclass ModelNew: pass",
        "device": "cpu",
    })
    body = resp.json()
    assert resp.status_code == 200
    assert body["passed"] is False and "code policy" in body["error"]
    assert ran == []


def test_public_rate_limit_returns_429_per_client(public_client, monkeypatch):
    monkeypatch.setenv("KERNELFORGE_RATE_LIMIT", "2")
    t = _template(public_client)
    payload = {"description": t["description"], "pytorch_reference": t["pytorch_reference"], "verify": False}
    headers = {"X-Forwarded-For": "203.0.113.7"}
    assert public_client.post("/generate", json=payload, headers=headers).status_code == 200
    assert public_client.post("/generate", json=payload, headers=headers).status_code == 200
    limited = public_client.post("/generate", json=payload, headers=headers)
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers
    other = public_client.post("/generate", json=payload, headers={"X-Forwarded-For": "198.51.100.1"})
    assert other.status_code == 200


def test_rate_limit_is_off_outside_public_mode(public_client, monkeypatch):
    monkeypatch.delenv("KERNELFORGE_PUBLIC")
    monkeypatch.setenv("KERNELFORGE_MOCK_MODEL", "1")
    monkeypatch.setenv("KERNELFORGE_RATE_LIMIT", "1")
    t = _template(public_client)
    payload = {"description": t["description"], "pytorch_reference": t["pytorch_reference"], "verify": False}
    for _ in range(3):
        assert public_client.post("/generate", json=payload).status_code == 200


def test_queue_cap_returns_503(public_client, monkeypatch):
    monkeypatch.setattr(main, "_queued", main.MAX_QUEUED_REQUESTS)
    t = _template(public_client)
    resp = public_client.post("/generate/stream", json={
        "description": t["description"], "pytorch_reference": t["pytorch_reference"], "verify": False})
    assert resp.status_code == 503
