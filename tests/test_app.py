"""Tests for the demo backend's request path (no GPU or model download required)."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from app.backend import main
from kernelforge.dataset import CATEGORIES
from verification.verify_kernel import VerifyResult


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("KERNELFORGE_ADAPTER_PATH", raising=False)
    monkeypatch.delenv("KERNELFORGE_PRELOAD", raising=False)
    monkeypatch.delenv("KERNELFORGE_PUBLIC", raising=False)
    monkeypatch.delenv("KERNELFORGE_LLM_MODEL", raising=False)
    monkeypatch.setenv("KERNELFORGE_MOCK_MODEL", "1")
    monkeypatch.setattr(main, "_generator", None)
    main._rate_limiter.clear()
    return TestClient(main.app)


def _first_template(client):
    template_id = client.get("/templates").json()[0]["id"]
    return client.get(f"/templates/{template_id}").json()


def test_health_defaults_to_base_model_without_loading_it(client, monkeypatch):
    monkeypatch.delenv("KERNELFORGE_MOCK_MODEL")
    body = client.get("/health").json()
    assert body["model_variant"] == "base"
    assert body["adapter_path"] is None
    assert body["model_loaded"] is False
    assert "base model" in body["banner"]
    assert main._generator is None


def test_health_reports_fine_tuned_when_adapter_set(client, monkeypatch):
    monkeypatch.delenv("KERNELFORGE_MOCK_MODEL")
    monkeypatch.setenv("KERNELFORGE_ADAPTER_PATH", "/tmp/adapter")
    body = client.get("/health").json()
    assert body["model_variant"] == "fine-tuned"
    assert "/tmp/adapter" in body["banner"]


def test_health_flags_mock_mode(client):
    body = client.get("/health").json()
    assert body["model_variant"] == "mock"
    assert "MOCK" in body["banner"]


def test_categories(client):
    assert client.get("/categories").json()["categories"] == list(CATEGORIES)


def test_templates_list_and_detail(client):
    templates = client.get("/templates").json()
    assert templates
    detail = client.get(f"/templates/{templates[0]['id']}").json()
    assert "class Model" in detail["pytorch_reference"]
    assert client.get("/templates/does-not-exist").status_code == 404


def test_generate_parsed_output_with_verification_disabled(client):
    t = _first_template(client)
    resp = client.post(
        "/generate",
        json={"description": t["description"], "pytorch_reference": t["pytorch_reference"], "verify": False},
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "ok"
    assert body["model_variant"] == "mock"
    assert "ModelNew" in body["triton_kernel"]
    assert body["optimization_explanation"]
    assert body["verification"]["ran"] is False
    assert "disabled" in body["verification"]["skipped_reason"]


def test_generate_unparseable_output_is_a_clean_response(client):
    resp = client.post(
        "/generate",
        json={"description": "custom op", "pytorch_reference": "import torch\nclass Model: pass\n", "verify": True},
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "unparseable"
    assert body["triton_kernel"] is None
    assert body["raw_output"]
    assert body["verification"]["ran"] is False
    assert "No kernel" in body["verification"]["skipped_reason"]


def test_generate_skips_verification_without_cuda(client):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA present; the no-GPU skip path doesn't apply")
    t = _first_template(client)
    body = client.post(
        "/generate",
        json={"description": t["description"], "pytorch_reference": t["pytorch_reference"], "device": "cuda"},
    ).json()
    assert body["status"] == "ok"
    assert body["verification"]["ran"] is False
    assert "CUDA" in body["verification"]["skipped_reason"]


def _boom(*args, **kwargs):
    raise RuntimeError("CUDA out of memory")
    yield  # pragma: no cover


def test_generate_model_failure_returns_500_with_detail(client, monkeypatch):
    monkeypatch.setattr(main.MockGenerator, "stream_chat", staticmethod(_boom))
    resp = client.post("/generate", json={"description": "x", "pytorch_reference": "y"})
    assert resp.status_code == 500
    assert "CUDA out of memory" in resp.json()["detail"]


def _stream_events(client, payload):
    with client.stream("POST", "/generate/stream", json=payload) as resp:
        assert resp.status_code == 200
        return [json.loads(line) for line in resp.iter_lines() if line]


def test_stream_emits_full_pipeline_in_order(client):
    t = _first_template(client)
    events = _stream_events(
        client, {"description": t["description"], "pytorch_reference": t["pytorch_reference"], "verify": False}
    )
    kinds = [e["event"] for e in events]
    assert kinds[0] == "meta" and kinds[1] == "started" and kinds[-1] == "done"
    assert "token" in kinds
    assert kinds.index("generated") < kinds.index("parsed") < kinds.index("verification")

    streamed = "".join(e["text"] for e in events if e["event"] == "token").strip()
    parsed = next(e for e in events if e["event"] == "parsed")
    assert parsed["status"] == "ok"
    assert parsed["raw_output"] == streamed
    verification = next(e for e in events if e["event"] == "verification")
    assert verification["ran"] is False
    assert not main._inference_lock.locked()


def test_stream_unparseable_output(client):
    events = _stream_events(client, {"description": "custom", "pytorch_reference": "import torch\n"})
    parsed = next(e for e in events if e["event"] == "parsed")
    assert parsed["status"] == "unparseable"
    assert "verifying" not in [e["event"] for e in events]


def test_stream_generation_failure_emits_error_and_releases_lock(client, monkeypatch):
    monkeypatch.setattr(main.MockGenerator, "stream_chat", staticmethod(_boom))
    events = _stream_events(client, {"description": "x", "pytorch_reference": "y"})
    assert events[-1]["event"] == "error"
    assert "CUDA out of memory" in events[-1]["detail"]
    assert not main._inference_lock.locked()


def test_web_ui_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "KernelForge" in resp.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_generate_rejects_oversized_input(client):
    resp = client.post(
        "/generate",
        json={"description": "x", "pytorch_reference": "a" * (main.MAX_REFERENCE_CHARS + 1)},
    )
    assert resp.status_code == 422


def test_generate_rejects_out_of_range_search_settings(client):
    base = {"description": "x", "pytorch_reference": "y"}
    assert client.post("/generate", json={**base, "num_candidates": main.MAX_CANDIDATES + 1}).status_code == 422
    assert client.post("/generate", json={**base, "repair_rounds": main.MAX_REPAIR_ROUNDS + 1}).status_code == 422


def test_catalog_exposes_all_fifteen_templates_with_both_shapes(client):
    templates = client.get("/templates").json()
    assert len(templates) == 15
    assert {t["category"] for t in templates} == set(CATEGORIES)
    detail = client.get(f"/templates/{templates[0]['id']}").json()
    assert "class Model" in detail["smoke_reference"]
    assert detail["smoke_reference"] != detail["pytorch_reference"]
    assert len(client.get("/templates", params={"category": "rope"}).json()) == 3


def test_stream_best_of_n_tags_tokens_per_candidate_and_dedupes(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_verification_skip_reason", lambda req, kernel: None if kernel else "no kernel")
    monkeypatch.setattr(
        main, "verify_model_kernel",
        lambda *a, **k: calls.append(1) or VerifyResult(status="ok", correct=True, speedup=1.7),
    )
    main._verification_cache.clear()
    t = _first_template(client)
    events = _stream_events(client, {"description": t["description"], "pytorch_reference": t["pytorch_reference"],
                                     "num_candidates": 3})
    tagged = {e["candidate"] for e in events if e["event"] == "token"}
    assert tagged == {"r0c0", "r0c1", "r0c2"}
    # The mock returns the same kernel three times: one harness run, two duplicates.
    assert len(calls) == 1
    verdicts = [e for e in events if e["event"] == "verification"]
    assert [v["duplicate_of"] for v in verdicts] == [None, "r0c0", "r0c0"]
    selected = next(e for e in events if e["event"] == "selected")
    assert selected["verification"]["passed"] and selected["num_candidates"] == 3
    assert [e["event"] for e in events][-1] == "done"


def test_verification_cache_is_reused_across_requests_and_counted(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_verification_skip_reason", lambda req, kernel: None if kernel else "no kernel")
    monkeypatch.setattr(
        main, "verify_model_kernel",
        lambda *a, **k: calls.append(1) or VerifyResult(status="ok", correct=True, speedup=2.0),
    )
    main._verification_cache.clear()
    main._stats.reset()
    t = _first_template(client)
    payload = {"description": t["description"], "pytorch_reference": t["pytorch_reference"]}
    first = client.post("/generate", json=payload).json()
    second = client.post("/generate", json=payload).json()
    assert len(calls) == 1
    assert first["verification"]["cached"] is False and second["verification"]["cached"] is True
    stats = client.get("/stats").json()
    assert stats["requests"] == 2 and stats["verification_cache_hits"] == 1
    assert stats["searches_passed"] == 2 and stats["best_speedup"] == 2.0


def test_repair_round_feeds_harness_error_back_to_the_model(client, monkeypatch):
    prompts = []
    real = main.MockGenerator.stream_chat

    def recording(self, messages, num_sequences=1, temperature=None):
        prompts.append(messages)
        text = "".join(chunk for _, chunk in real(self, messages, 1, temperature))
        if len(prompts) > 1:
            text = text.replace("import", "# repaired\nimport", 1)
        yield 0, text

    results = iter([
        VerifyResult(status="error", error="IndexError: out of bounds in candidate.py"),
        VerifyResult(status="ok", correct=True, speedup=1.2),
    ])
    monkeypatch.setattr(main.MockGenerator, "stream_chat", recording)
    monkeypatch.setattr(main, "_verification_skip_reason", lambda req, kernel: None if kernel else "no kernel")
    monkeypatch.setattr(main, "verify_model_kernel", lambda *a, **k: next(results))
    main._verification_cache.clear()
    t = _first_template(client)
    body = client.post("/generate", json={"description": t["description"], "pytorch_reference": t["pytorch_reference"],
                                          "repair_rounds": 2}).json()
    assert len(prompts) == 2
    assert "IndexError" in prompts[1][-1]["content"] and prompts[1][-2]["role"] == "assistant"
    assert body["selected"] == "r1c0" and "repair round 1" in body["selection_reason"]
    assert [c["verdict"] for c in body["candidates"]] == ["error", "passed"]


def test_no_repair_when_verification_is_skipped(client):
    t = _first_template(client)
    body = client.post("/generate", json={"description": t["description"], "pytorch_reference": t["pytorch_reference"],
                                          "verify": False, "repair_rounds": 2}).json()
    assert len(body["candidates"]) == 1
    assert body["selection_reason"].startswith("not verified")
