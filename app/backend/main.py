"""FastAPI backend + web UI host for the KernelForge demo.

Request path: POST /generate[/stream] -> kernelforge.search.run_search, which
samples N candidates from KernelGenerator (base model unless
KERNELFORGE_ADAPTER_PATH is set), verifies each with verify_model_kernel (the
same harness build_dataset.py and evaluation/ use), and optionally repairs the
best failure using the harness's error message. The single-page UI in app/web/
is served at "/".

Verification devices:
    cuda  real GPU run: correctness + timed speedup
    cpu   Triton interpreter on CPU: correctness only, no timings

Environment variables:
    KERNELFORGE_MODEL_CONFIG   GenerationConfig JSON/YAML (default: Qwen2.5-Coder-1.5B config)
    KERNELFORGE_ADAPTER_PATH   LoRA adapter dir; unset = base model
    KERNELFORGE_MAX_NEW_TOKENS cap generation length (demo latency on a T4)
    KERNELFORGE_PRELOAD=1      load the model at startup instead of on first request
    KERNELFORGE_MOCK_MODEL=1   no model; echoes hand-written template kernels (UI plumbing only)

  Hosted model instead of a local one (for GPU-less hosts such as Railway):
    KERNELFORGE_LLM_MODEL      model id; setting it switches generation to the API
    KERNELFORGE_LLM_BASE_URL   OpenAI-compatible base URL (default https://api.openai.com/v1)
    KERNELFORGE_LLM_API_KEY    bearer token for that API

  Public deployment:
    KERNELFORGE_PUBLIC=1       code policy on references and kernels, per-IP rate limit,
                               bounded queue; without an LLM configured, falls back to mock
    KERNELFORGE_RATE_LIMIT     searches per IP per minute in public mode (default 6)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Optional

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _load_dotenv(path: Path) -> None:
    """Fill unset variables from a local .env (real environment wins)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value:
            os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).parent.parent.parent / ".env")

from app.backend.catalog import Catalog
from kernelforge.dataset import CATEGORIES
from kernelforge.model import GenerationConfig, KernelGenerator, load_generation_config
from kernelforge.remote import RemoteChatGenerator
from kernelforge.search import SearchConfig, SearchStats, kernel_key, run_search
from verification.policy import check_code
from verification.verify_kernel import verify_model_kernel

DEFAULT_MODEL_CONFIG = Path(__file__).parent.parent.parent / "training" / "configs" / "qwen2.5-coder-1.5b.json"
WEB_DIR = Path(__file__).parent.parent / "web"
MAX_REFERENCE_CHARS = 20_000
MAX_DESCRIPTION_CHARS = 2_000
MAX_CANDIDATES = 4
MAX_REPAIR_ROUNDS = 2
HARNESS_WARMUP = 10
HARNESS_ITERS = 50
GPU_TIMEOUT_S = 60.0
INTERPRETER_TIMEOUT_S = 180.0
VERIFICATION_CACHE_SIZE = 128
MAX_QUEUED_REQUESTS = 4
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_RATE_LIMIT = 6

BANNERS = {
    "base": "Running the base model ({model}) — not fine-tuned. The fine-tuned adapter is in progress "
    "and will drop into this same pipeline.",
    "fine-tuned": "Running the fine-tuned adapter ({adapter}) on {model}.",
    "remote": "Running {model} through a hosted API — a general-purpose model, not KernelForge's "
    "fine-tuned adapter. The search loop and verification harness are the real ones.",
    "mock": "MOCK MODE — no model is loaded. Outputs are the hand-written template kernels, not "
    "model-generated. For testing the UI plumbing only.",
}

_generator = None
_generator_lock = threading.Lock()
_inference_lock = threading.Lock()
_catalog: Optional[Catalog] = None
_catalog_lock = threading.Lock()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _generation_config() -> GenerationConfig:
    config = load_generation_config(os.environ.get("KERNELFORGE_MODEL_CONFIG", str(DEFAULT_MODEL_CONFIG)))
    adapter = os.environ.get("KERNELFORGE_ADAPTER_PATH")
    if adapter:
        config.adapter_path = adapter
    max_new_tokens = os.environ.get("KERNELFORGE_MAX_NEW_TOKENS")
    if max_new_tokens:
        config.max_new_tokens = int(max_new_tokens)
    return config


def _public() -> bool:
    return _env_flag("KERNELFORGE_PUBLIC")


def _model_variant() -> str:
    if _env_flag("KERNELFORGE_MOCK_MODEL"):
        return "mock"
    if os.environ.get("KERNELFORGE_LLM_MODEL"):
        return "remote"
    if _public():
        return "mock"
    return "fine-tuned" if _generation_config().adapter_path else "base"


def _model_name(variant: str, config: GenerationConfig) -> str:
    if variant == "remote":
        return os.environ["KERNELFORGE_LLM_MODEL"]
    if variant == "mock":
        return "mock (template echo)"
    return config.model_name


def _banner(variant: str, config: GenerationConfig) -> str:
    return BANNERS[variant].format(model=_model_name(variant, config), adapter=config.adapter_path)


def _make_generator(variant: str):
    if variant == "mock":
        return MockGenerator()
    config = _generation_config()
    if variant == "remote":
        return RemoteChatGenerator(
            base_url=os.environ.get("KERNELFORGE_LLM_BASE_URL") or DEFAULT_LLM_BASE_URL,
            api_key=os.environ.get("KERNELFORGE_LLM_API_KEY", ""),
            model=os.environ["KERNELFORGE_LLM_MODEL"],
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    return KernelGenerator(config)


def _get_catalog() -> Catalog:
    global _catalog
    with _catalog_lock:
        if _catalog is None:
            _catalog = Catalog()
        return _catalog


class MockGenerator:
    """Stands in for KernelGenerator so the UI can be exercised without a GPU."""

    backend = "mock"

    def load(self) -> None:
        pass

    def _reply(self, messages: list[dict]) -> str:
        entry = _get_catalog().match_prompt(messages)
        if entry is None:
            return "Mock mode only echoes built-in templates; this input has no template match."
        return f"kernel:\n{entry['triton_kernel']}\n\nexplanation:\n{entry['optimization_explanation']}"

    def stream_chat(
        self, messages: list[dict], num_sequences: int = 1, temperature: Optional[float] = None
    ) -> Iterator[tuple[int, str]]:
        text = self._reply(messages)
        for i in range(0, len(text), 24):
            time.sleep(0.004)
            for seq in range(num_sequences):
                yield seq, text[i : i + 24]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _get_generator():
    global _generator
    with _generator_lock:
        if _generator is None:
            _generator = _make_generator(_model_variant())
        return _generator


def _cuda_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "gpu_name": None}
    available = torch.cuda.is_available()
    return {"cuda_available": available, "gpu_name": torch.cuda.get_device_name(0) if available else None}


def _triton_version() -> Optional[str]:
    try:
        import triton
    except ImportError:
        return None
    return getattr(triton, "__version__", "unknown")


class VerificationCache:
    """LRU of harness results keyed on (reference, kernel, device, tolerances).

    Re-running the same kernel is common (resubmits, duplicate samples, repair
    rounds that converge), and a GPU/interpreter run costs seconds.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class ServerStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.reset()

    def reset(self) -> None:
        self.requests = 0
        self.candidates = 0
        self.tokens = 0
        self.generation_time_s = 0.0
        self.verifications_run = 0
        self.verification_cache_hits = 0
        self.searches_passed = 0
        self.repairs_that_passed = 0
        self.best_speedup: Optional[float] = None

    def record(self, search: SearchStats, selected: Optional[dict]) -> None:
        with self._lock:
            self.requests += 1
            self.candidates += search.candidates
            self.tokens += search.tokens
            self.generation_time_s += search.generation_time_s
            self.verifications_run += search.verifications_run
            self.verification_cache_hits += search.verification_cache_hits
            v = (selected or {}).get("verification") or {}
            if v.get("passed"):
                self.searches_passed += 1
                if selected["round"] > 0:
                    self.repairs_that_passed += 1
                if v.get("speedup") is not None:
                    self.best_speedup = max(self.best_speedup or 0.0, v["speedup"])

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self.started_at, 1),
                "requests": self.requests,
                "candidates_generated": self.candidates,
                "tokens_generated": self.tokens,
                "tokens_per_s": round(self.tokens / self.generation_time_s, 1) if self.generation_time_s else None,
                "verifications_run": self.verifications_run,
                "verification_cache_hits": self.verification_cache_hits,
                "verification_cache_entries": len(_verification_cache),
                "searches_passed": self.searches_passed,
                "pass_rate": round(self.searches_passed / self.requests, 3) if self.requests else None,
                "repairs_that_passed": self.repairs_that_passed,
                "best_speedup": self.best_speedup,
            }


class RateLimiter:
    """Sliding one-minute window per client IP."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_s: float = 60.0) -> Optional[float]:
        """Record a hit; return seconds to wait if `key` is over `limit`, else None."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > window_s:
                hits.popleft()
            if len(hits) >= limit:
                return round(window_s - (now - hits[0]), 1)
            hits.append(now)
            if len(self._hits) > 10_000:
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] <= window_s}
            return None

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


_verification_cache = VerificationCache(VERIFICATION_CACHE_SIZE)
_stats = ServerStats()
_rate_limiter = RateLimiter()
_queue_lock = threading.Lock()
_queued = 0


def _rate_limit() -> int:
    try:
        return max(1, int(os.environ.get("KERNELFORGE_RATE_LIMIT", DEFAULT_RATE_LIMIT)))
    except ValueError:
        return DEFAULT_RATE_LIMIT


def _client_ip(request: Request) -> str:
    # Railway and most PaaS proxies put the original client first in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    if not _public():
        return
    wait = _rate_limiter.check(_client_ip(request), _rate_limit())
    if wait is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {_rate_limit()} runs per minute on this public demo. Try again in {wait:.0f}s.",
            headers={"Retry-After": str(max(1, int(wait)))},
        )


def _reject_disallowed_reference(reference: str) -> None:
    if _public():
        reason = check_code(reference, "PyTorch reference")
        if reason:
            raise HTTPException(status_code=422, detail=f"Blocked by the public-server code policy: {reason}.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _env_flag("KERNELFORGE_PRELOAD"):
        _get_generator().load()
    yield


app = FastAPI(
    title="KernelForge",
    description="Generate, verify, and repair Triton kernels from PyTorch ops.",
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    description: str = Field(..., max_length=MAX_DESCRIPTION_CHARS, description="Plain-English op description")
    pytorch_reference: str = Field(
        ...,
        max_length=MAX_REFERENCE_CHARS,
        description="PyTorch reference source (Model + get_inputs/get_init_inputs)",
    )
    verify: bool = Field(True, description="Run the verification harness on generated kernels")
    device: str = Field("cuda", pattern="^(cuda|cpu)$", description="cuda = GPU timing; cpu = Triton interpreter")
    tolerance_atol: float = Field(0.01, gt=0, le=1)
    tolerance_rtol: float = Field(0.01, gt=0, le=1)
    num_candidates: int = Field(1, ge=1, le=MAX_CANDIDATES, description="Samples drawn in the first round")
    repair_rounds: int = Field(0, ge=0, le=MAX_REPAIR_ROUNDS, description="Harness-feedback repair attempts")
    few_shot: Optional[bool] = Field(
        None, description="Show a solved template for a different op plus Triton API notes; "
        "default: on for general-purpose models, off for the fine-tuned adapter",
    )


class VerificationInfo(BaseModel):
    ran: bool
    skipped_reason: Optional[str] = None
    status: Optional[str] = None
    correct: Optional[bool] = None
    passed: Optional[bool] = None
    speedup: Optional[float] = None
    reference_time_s: Optional[float] = None
    candidate_time_s: Optional[float] = None
    error: Optional[str] = None
    device: Optional[str] = None
    interpreted: bool = False
    cached: bool = False
    atol: Optional[float] = None
    rtol: Optional[float] = None
    warmup: Optional[int] = None
    iters: Optional[int] = None
    wall_time_s: Optional[float] = None


class CandidateInfo(BaseModel):
    candidate: str
    round: int
    status: str
    passed: bool
    speedup: Optional[float] = None
    verdict: str
    duplicate_of: Optional[str] = None


class GenerateResponse(BaseModel):
    status: str  # "ok" | "unparseable"
    model_variant: str
    model_name: str
    triton_kernel: Optional[str]
    optimization_explanation: Optional[str]
    raw_output: str
    generation_time_s: float
    verification: VerificationInfo
    selected: str
    selection_reason: str
    candidates: list[CandidateInfo]
    example: Optional[str] = None


class TemplateInfo(BaseModel):
    id: str
    category: str
    op_name: str
    description: str
    interpreter_checked: bool
    gpu_verified: int
    median_speedup: Optional[float] = None


def _verification_skip_reason(req: GenerateRequest, kernel: Optional[str]) -> Optional[str]:
    if not req.verify:
        return "Verification disabled for this request."
    if not kernel:
        return "No kernel could be parsed from the model output."
    if req.device == "cuda" and not _cuda_info()["cuda_available"]:
        return "No CUDA device on this server; switch to the interpreter for a correctness-only check."
    if req.device == "cpu" and _triton_version() is None:
        return "Triton isn't installed on this server, so the interpreter check can't run."
    return None


def _static_rejection(kernel: str) -> Optional[str]:
    """Why the kernel can't be run at all (syntax, or public code policy), else None."""
    try:
        compile(kernel, "candidate.py", "exec")
    except SyntaxError as exc:
        line = (exc.text or "").rstrip()
        return f"SyntaxError on line {exc.lineno}: {exc.msg}" + (f"\n    {line.strip()}" if line else "")
    violation = check_code(kernel, "The kernel") if _public() else None
    if violation:
        return (f"Not executed — blocked by the public-server code policy: {violation}. "
                "Use only torch/triton/math and no file, process or dunder access.")
    return None


def _run_verification(req: GenerateRequest, kernel: str) -> dict:
    """Harness result as a VerificationInfo dict, memoized across requests."""
    interpret = req.device == "cpu"
    key = kernel_key(req.pytorch_reference, kernel, req.device, req.tolerance_atol, req.tolerance_rtol)
    hit = _verification_cache.get(key)
    if hit is not None:
        return {**hit, "cached": True}

    rejection = _static_rejection(kernel)
    if rejection:
        return VerificationInfo(
            ran=True, status="error", correct=False, passed=False, device=req.device,
            interpreted=interpret, atol=req.tolerance_atol, rtol=req.tolerance_rtol, error=rejection,
        ).model_dump()

    start = time.perf_counter()
    result = verify_model_kernel(
        req.pytorch_reference,
        kernel,
        device=req.device,
        atol=req.tolerance_atol,
        rtol=req.tolerance_rtol,
        warmup=HARNESS_WARMUP,
        iters=HARNESS_ITERS,
        timeout_s=INTERPRETER_TIMEOUT_S if interpret else GPU_TIMEOUT_S,
        interpret=interpret,
    )
    info = VerificationInfo(
        ran=True,
        status=result.status,
        correct=result.correct,
        passed=result.passed,
        speedup=result.speedup,
        reference_time_s=result.reference_time_s,
        candidate_time_s=result.candidate_time_s,
        error=result.error,
        device=req.device,
        interpreted=result.interpreted,
        atol=req.tolerance_atol,
        rtol=req.tolerance_rtol,
        warmup=None if interpret else HARNESS_WARMUP,
        iters=None if interpret else HARNESS_ITERS,
        wall_time_s=round(time.perf_counter() - start, 3),
    ).model_dump()
    # Timeouts/crashes can be transient (another process on the GPU); don't pin them.
    if result.status in ("ok", "error"):
        _verification_cache.put(key, info)
    return info


def _few_shot_example(req: GenerateRequest) -> Optional[dict]:
    enabled = req.few_shot if req.few_shot is not None else _model_variant() != "fine-tuned"
    return _get_catalog().example_for(req.pytorch_reference, req.description) if enabled else None


def _search(req: GenerateRequest, generator, stats: SearchStats, example: Optional[dict] = None) -> Iterator[dict]:
    return run_search(
        req.description,
        req.pytorch_reference,
        SearchConfig(req.num_candidates, req.repair_rounds, _generation_config().temperature,
                     example=example, triton_notes=example is not None),
        generate=generator.stream_chat,
        verify=lambda kernel: _run_verification(req, kernel),
        skip_reason=lambda kernel: _verification_skip_reason(req, kernel),
        count_tokens=generator.count_tokens,
        stats=stats,
    )


@app.get("/health")
def health():
    config = _generation_config()
    variant = _model_variant()
    generator = _generator
    triton_version = _triton_version()
    return {
        "status": "ok",
        "model_variant": variant,
        "model_name": _model_name(variant, config),
        "adapter_path": config.adapter_path,
        "max_new_tokens": config.max_new_tokens,
        "model_loaded": generator is not None and (variant in ("mock", "remote") or generator.backend is not None),
        "inference_backend": generator.backend if generator is not None else None,
        "triton_available": triton_version is not None,
        "triton_version": triton_version,
        "interpreter_available": triton_version is not None,
        "busy": _inference_lock.locked(),
        "banner": _banner(variant, config),
        "public": _public(),
        "limits": {
            "max_candidates": MAX_CANDIDATES,
            "max_repair_rounds": MAX_REPAIR_ROUNDS,
            "rate_limit_per_min": _rate_limit() if _public() else None,
            "max_queued": MAX_QUEUED_REQUESTS,
        },
        **_cuda_info(),
    }


@app.get("/stats")
def stats():
    return _stats.snapshot()


@app.get("/categories")
def list_categories():
    return {"categories": list(CATEGORIES)}


@app.get("/templates", response_model=list[TemplateInfo])
def list_templates(category: Optional[str] = None):
    templates = _get_catalog().templates
    if category:
        templates = [t for t in templates if t.category == category]
    return [TemplateInfo(**t.info()) for t in templates]


@app.get("/templates/{template_id}")
def get_template(template_id: str):
    template = _get_catalog().get(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return template.detail()


class _QueueSlot:
    """Bounds how many requests may wait on the device lock at once."""

    def __enter__(self):
        global _queued
        with _queue_lock:
            if _queued >= MAX_QUEUED_REQUESTS:
                raise HTTPException(
                    status_code=503,
                    detail=f"Server busy: {_queued} runs already queued. Try again shortly.",
                    headers={"Retry-After": "15"},
                )
            _queued += 1
        return self

    def __exit__(self, *exc):
        global _queued
        with _queue_lock:
            _queued -= 1
        return False


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(enforce_rate_limit)])
def generate(req: GenerateRequest):
    _reject_disallowed_reference(req.pytorch_reference)
    generator = _get_generator()
    config = _generation_config()
    variant = _model_variant()
    search_stats = SearchStats()
    example = _few_shot_example(req)

    # One device, one model instance: serialize generation + verification so concurrent
    # visitors queue instead of contending for VRAM/CPU.
    with _QueueSlot(), _inference_lock:
        start = time.perf_counter()
        try:
            selected = next(e for e in _search(req, generator, search_stats, example) if e["event"] == "selected")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {type(exc).__name__}: {exc}") from exc
        _stats.record(search_stats, selected)

    return GenerateResponse(
        status=selected["status"],
        model_variant=variant,
        model_name=_model_name(variant, config),
        triton_kernel=selected["triton_kernel"],
        optimization_explanation=selected["optimization_explanation"],
        raw_output=selected["raw_output"],
        generation_time_s=round(time.perf_counter() - start, 3),
        verification=VerificationInfo(**selected["verification"]),
        selected=selected["candidate"],
        selection_reason=selected["reason"],
        candidates=[CandidateInfo(**c) for c in selected["candidates"]],
        example=example["id"] if example else None,
    )


def _event(kind: str, **data) -> bytes:
    return (json.dumps({"event": kind, **data}) + "\n").encode()


@app.post("/generate/stream", dependencies=[Depends(enforce_rate_limit)])
async def generate_stream(req: GenerateRequest):
    """Same search as /generate, streamed as NDJSON events.

    Events: meta, queued, started, then per round: round, token (tagged with a
    candidate id like "r0c1"), generated, parsed, verifying, verification;
    then selected, done -- or error. Blocking work runs in worker threads; if
    the client disconnects, generation is stopped before the device lock is released.
    """
    _reject_disallowed_reference(req.pytorch_reference)
    if _queued >= MAX_QUEUED_REQUESTS:
        raise HTTPException(status_code=503, detail=f"Server busy: {_queued} runs already queued. "
                            "Try again shortly.", headers={"Retry-After": "15"})
    generator = _get_generator()
    config = _generation_config()
    variant = _model_variant()

    async def events():
        yield _event("meta", model_variant=variant, model_name=_model_name(variant, config),
                     max_new_tokens=config.max_new_tokens, num_candidates=req.num_candidates,
                     repair_rounds=req.repair_rounds, device=req.device)

        if not _inference_lock.acquire(blocking=False):
            try:
                slot = _QueueSlot().__enter__()
            except HTTPException as exc:
                yield _event("error", detail=exc.detail)
                return
            try:
                yield _event("queued", detail="Another run is in progress on this server; waiting for it to finish.")
                try:
                    await anyio.to_thread.run_sync(_inference_lock.acquire)
                except BaseException:
                    # run_sync isn't abandoned on cancel, so the lock was acquired before this raised.
                    _inference_lock.release()
                    raise
            finally:
                slot.__exit__(None, None, None)
        search_stats = SearchStats()
        selected = None
        try:
            yield _event("started")
            start = time.perf_counter()
            search = _search(req, generator, search_stats, _few_shot_example(req))
            try:
                while True:
                    event = await anyio.to_thread.run_sync(next, search, None)
                    if event is None:
                        break
                    if event["event"] == "selected":
                        selected = event
                    yield (json.dumps(event) + "\n").encode()
            except Exception as exc:
                yield _event("error", detail=f"Generation failed: {type(exc).__name__}: {exc}")
                return
            finally:
                # Shielded: on client disconnect, generation must stop before the lock is released.
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(search.close)
            _stats.record(search_stats, selected)
            yield _event("done", total_time_s=round(time.perf_counter() - start, 3))
        finally:
            _inference_lock.release()

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class VerifyRequest(BaseModel):
    pytorch_reference: str = Field(..., max_length=MAX_REFERENCE_CHARS)
    triton_kernel: str = Field(..., max_length=MAX_REFERENCE_CHARS)
    device: str = Field("cuda", pattern="^(cuda|cpu)$")
    atol: float = Field(0.01, gt=0, le=1)
    rtol: float = Field(0.01, gt=0, le=1)


@app.post("/verify", response_model=VerificationInfo, dependencies=[Depends(enforce_rate_limit)])
def verify_only(req: VerifyRequest):
    _reject_disallowed_reference(req.pytorch_reference)
    as_generate = GenerateRequest(
        description="", pytorch_reference=req.pytorch_reference, device=req.device,
        tolerance_atol=req.atol, tolerance_rtol=req.rtol,
    )
    reason = _verification_skip_reason(as_generate, req.triton_kernel)
    if reason:
        return VerificationInfo(ran=False, skipped_reason=reason)
    with _QueueSlot(), _inference_lock:
        return VerificationInfo(**_run_verification(as_generate, req.triton_kernel))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
