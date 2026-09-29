# KernelForge — Documentation

LLM that turns a naive PyTorch op into an optimized Triton GPU kernel (fine-tuning
in progress; the demo currently runs the base model), with automated correctness
verification and speedup benchmarking.
This document covers what's been built so far, how it works, and how to run
and extend it. The roadmap, study track and next steps live in
[MASTER_PLAN.md](MASTER_PLAN.md); this file tracks implementation.

## Current status

| Phase | State |
|---|---|
| 0 — Foundations | Partial: 3 hand-written example kernels done (`examples/kernels/`) |
| **0.5 — Demo + kernel search** | **Built.** `app/` (FastAPI + web studio) runs `kernelforge/search.py` (best-of-N + harness-feedback repair) on the base model; see [Demo app](#demo-app-app). First live GPU run + GIF pending |
| 1 — Dataset | Templates fixed (see [Template fixes](#template-fixes-sep-29-2026)); **15/15 pass Triton's CPU interpreter**; GPU verification of the 300 candidates pending |
| 2 — Verification harness | **Done**, GPU-validated; function + Model/ModelNew contracts; GPU mode and interpreter mode; precise mismatch reports |
| 3 — Model selection & baseline | Script written — `evaluation/baseline_eval.py`, `training/configs/`; not yet run on GPU |
| 4 — Fine-tuning | Script written — `training/finetune.py` (Unsloth QLoRA + PEFT fallback); **no adapter trained yet** |
| 5 — Evaluation | Scripts written — `evaluation/finetuned_eval.py`, `evaluation/explanation_review.py`; no results yet |
| 6 — Tool/UI | Folded into Phase 0.5; will serve the fine-tuned adapter once it exists |
| 7 — Write-up & positioning | Draft — `docs/project_writeup.md`; results tables empty until eval runs |

## Getting started: see it working

The fastest way to see the project end to end is the demo, before touching the
dataset or training pipeline. On a Linux + NVIDIA machine (Kaggle/Colab):

```bash
git clone https://github.com/Arsalan-05/KernelForge.git && cd KernelForge
pip install -q -r requirements-demo.txt   # torch + triton are preinstalled on Kaggle/Colab
python app/launch.py --share
```

This loads the base model, starts the server (API + UI), and prints a public
`*.trycloudflare.com` link. Pick an example op, hit generate, and the generated
kernel streams in live, then runs through the same verification harness
described below. Details in
[Demo app](#demo-app-app).

## Demo app (`app/`)

> Runs the **base model (Qwen2.5-Coder, not fine-tuned)**; no fine-tuned
> adapter exists yet.

**One harness, everywhere.** The demo's generate endpoints verify kernels
by calling the exact same `verification.verify_kernel.verify_model_kernel`
function that `data/build_dataset.py` uses to filter the training set and
`evaluation/common.py` uses to score baseline and fine-tuned runs. There is no
demo-specific correctness check: "passed" in the UI means the same thing as
"verified" in `dataset.jsonl` and "correct" in the eval tables.

### Request path

```
Web studio (app/web/, served by FastAPI at /)   badge from GET /health: base / fine-tuned / mock
        │  POST /generate/stream {description, pytorch_reference, verify, device, atol, rtol,
        │                         num_candidates 1-4, repair_rounds 0-2}
        ▼
FastAPI (app/backend/main.py)                   NDJSON event stream back to the browser
        │  _inference_lock: one request at a time on the single GPU ("queued" event while waiting)
        ▼
run_search()                 kernelforge/search.py — one "round" event per round
        │
        ├─ KernelGenerator.stream_chat(messages, N)   one batched generate(), N sequences;
        │     "token" events tagged with a candidate id (r0c0, r0c1, …)
        ├─ parse_model_output()   per candidate → "parsed"
        ├─ verify (per candidate) → "verifying" / "verification"
        │     identical kernels in one search: verified once ("duplicate_of")
        │     across requests: LRU cache in main.py ("cached": true)
        │     verify_model_kernel() → subprocess sandbox_runner.py (GPU, or TRITON_INTERPRET=1 on CPU)
        │     skipped with a reason if verify=false, no CUDA (GPU), or no Triton (interpreter)
        └─ any pass → stop; else repair the best failure with its harness error (next round)
        ▼
"selected" (winner + reason + every candidate's verdict) → "done" ──► UI
```

`POST /generate` runs the same search without streaming and returns the
selected candidate plus a `candidates` summary, for scripts and `curl`.

### `app/backend/main.py` — FastAPI

| Endpoint | Purpose |
|---|---|
| `POST /generate` | `description` + `pytorch_reference` (+ `verify`, `device`, tolerances, `num_candidates`, `repair_rounds`) in → the selected candidate's `triton_kernel`, `optimization_explanation`, `raw_output`, `verification` block (`status`, `correct`, `passed`, `speedup`, timings, `error`, `interpreted`, `cached`, or `skipped_reason`), plus `selected`, `selection_reason`, and a `candidates` list (id, round, verdict, speedup, `duplicate_of`). Returns `status: "unparseable"` if the selected output has no kernel. |
| `POST /generate/stream` | Same input as `/generate`; NDJSON events: `meta`, `queued` (only if another request holds the GPU), `started`, then per round `round`, `token`… (with `candidate`), `generated`, `parsed`, `verifying` (only if the harness will run), `verification`; then `selected`, `done` — or `error`. If the client disconnects, generation is stopped before the GPU lock is released. |
| `POST /verify` | Run the harness (cached) on a user-supplied reference + kernel, no generation. |
| `GET /` , `/static/*` | The web studio UI (`app/web/`). `/docs` is the interactive OpenAPI page. |
| `GET /health` | Model variant, model name, adapter path, whether the model is loaded, CUDA/GPU name, Triton/interpreter availability, search limits, and the banner text the UI shows. |
| `GET /stats` | In-memory server counters: requests, pass rate, candidates, tokens/s, harness runs, cache hits, repairs that passed, best speedup. |
| `GET /categories` | The 5 kernel categories. |
| `GET /templates` | The 15 dataset templates (optionally `?category=`) with `interpreter_checked` / `gpu_verified` status read from real artifacts. `GET /templates/{id}` returns the realistic GPU-shape reference and a small interpreter-shape reference of the same op (`app/backend/catalog.py`). |

Generation and verification are serialized behind a lock (one GPU, one model
instance), so concurrent visitors on a share link queue rather than contend for
VRAM.

### Kernel search engine (`kernelforge/search.py`)

The engine is pure orchestration. It takes `generate`, `verify`,
`skip_reason`, and `count_tokens` as arguments, so the demo, the mock
generator in tests, and evaluation can all drive the same loop.

**Prompting for general-purpose models.** `few_shot` on a request is
automatic by default: on for the base and hosted models, off for the
fine-tuned adapter, which learned the format in training. When it's on, the
prompt gets two additions.

- **A solved example.** `Catalog.example_for` picks the solved smoke
  template, as a prior user/assistant turn:
  - It is never the requested op.
  - It is the shortest interpreter-checked sibling in the same category. For a
    custom op, the category is guessed from keywords in the description and
    reference.
- **`TRITON_NOTES`:** Triton 3.x pitfalls, such as hand-written softmax, no
  `tl.isnan`, power-of-two `tl.arange`, `tl.dot` shape rules and NaN-safe
  causal masking.

The example is disclosed in the `round` event and in the response's
`example` field, and the UI shows it on the round. Repair feedback also gets
**targeted hints** (`prompts.repair_hints`) when the harness error matches a
known pattern: a missing `tl.*` function, an unexpected keyword argument,
NaN/inf output, `tl.arange`/`tl.dot` shape errors, syntax errors or policy
blocks. These were the failure modes observed live with `qwen3-coder`.

- **Round 0 (sample):** `num_candidates` completions from one batched
  `generate()` (`num_return_sequences`, streamed per row by
  `BatchTextStreamer`). With N > 1, temperature is raised to at least 0.7;
  otherwise best-of-N would draw near-identical samples.
- **Verify:** each parseable candidate goes through the harness. An
  identical kernel in the same search reuses the first result
  (`duplicate_of`).
- **Select:** the passing candidate with the highest speedup wins. With no
  timings (interpreter), it's the first distinct passing kernel, and the
  reason says there were no timings to rank by.
- **Repair (rounds 1..`repair_rounds`):** if nothing passed, the best failure
  is repaired. Wrong numerics outrank a harness error, which outranks
  unparseable output. `build_repair_messages` sends the original prompt, that
  attempt as the assistant turn, and the harness feedback as the next user
  turn: the mismatch description, or the last 25 traceback lines with temp
  paths rewritten to `candidate.py` / `reference.py`. Only the latest attempt
  is kept, so context stays bounded. There's no repair when verification was
  skipped, because there's no signal to repair from.
- **Cache:** `main.py` wraps the harness in a 128-entry LRU keyed on
  sha256(reference, kernel, device, atol, rtol). Timeouts and crashes aren't
  cached, since they can be transient.

Environment variables:

| Variable | Effect |
|---|---|
| `KERNELFORGE_MODEL_CONFIG` | Generation config (default: `training/configs/qwen2.5-coder-1.5b.json`) |
| `KERNELFORGE_ADAPTER_PATH` | LoRA adapter dir; unset = base model |
| `KERNELFORGE_PRELOAD=1` | Load the model at startup instead of on first request |
| `KERNELFORGE_MAX_NEW_TOKENS` | Cap generation length (`launch.py` sets 2048; the 4096 training default is slow on a T4) |
| `KERNELFORGE_MOCK_MODEL=1` | No model; echoes hand-written template kernels (UI plumbing only) |

### `app/web/` — web studio UI

A dependency-free single page (`index.html`, `styles.css`, `app.js`; only
highlight.js and fonts from a CDN), served by the backend so the API and UI
share one origin and one tunnel.

- **Op library** sidebar: the 15 templates grouped by category, with
  `interp ✓` / `GPU n` status badges, search, a "Custom op" starter, and
  `?op=<id>` deep links. Server stats (from `/stats`) sit below it.
- **Reference editor** with line numbers, plus a verify toggle, **Harness**
  (GPU, or Interpreter; switching swaps in the matching shape of the same op
  if the reference is unedited), **Samples** 1–4, **Repair** off/1/2, and
  atol/rtol. It picks the interpreter automatically when the server has no GPU.
- **Live pipeline**: GPU slot → Generate (tokens/s) → Parse → Verify →
  Select, each with its own state (active, done, failed, skipped with reason).
- **Kernel search card** (shown when Samples > 1 or Repair is on): one row per
  round, one tile per candidate (streaming, verdict, speedup, `= r0c0` for
  duplicates, cached), the harness feedback sent for each repair round, and
  the selected candidate. Click any tile to inspect its kernel, raw output,
  and verification.
- **Output tabs**: highlighted Triton kernel, side-by-side with the PyTorch
  reference, and raw streaming output (shown automatically when the output
  isn't parseable, which is common with the base model since it hasn't
  learned the `kernel:` / `explanation:` format). Copy and download `.py`.
- **Verification card**: passed (speedup plus reference vs. kernel timing
  bars), "correct in the interpreter" (explicitly not GPU-verified),
  incorrect (with the mismatch description), error/timeout/crash (with log),
  or "not run" with the harness's skip reason. The benchmark settings chip
  only appears when a result was actually measured.
- **"Why it's faster"** card for the model-generated explanation, shown
  only when a kernel was parsed.
- **Session runs**: the last 20 runs stored in `localStorage`; click to
  restore. Stop with Esc, run with ⌘/Ctrl+Enter.

All model output is inserted as text, never as HTML.

**Persistent model-variant badge.** A pill in the header and a banner under
it always state which model is serving: amber "Running the base model — not
fine-tuned", green for a fine-tuned adapter, red for mock mode. It's populated
from `/health` (polled every 20 s), and the banner defaults to the base-model
message if the backend can't be reached. It exists so nobody watching the demo mistakes
base-model output for fine-tuned results — the honesty disclaimer lives in the
UI itself, not just in the README.

### Running it

**Locally** (UI development; macOS can't run Triton, so use mock mode or skip
verification):

```bash
python app/launch.py --mock                 # open http://localhost:8000, no model
# or run the server directly:
KERNELFORGE_MOCK_MODEL=1 uvicorn app.backend.main:app --port 8000
```

**On Kaggle/Colab** (real model + GPU verification, public link):

```bash
python app/launch.py --share                                   # base model
python app/launch.py --share --adapter training/checkpoints/run_x/adapter   # once trained
```

`launch.py` starts uvicorn as a subprocess, waits (up to 20 min by default)
for the model to download and load, then with `--share` opens a Cloudflare
quick tunnel (`cloudflared tunnel --url`, binary auto-downloaded on Linux) and
prints the temporary public `*.trycloudflare.com` link. Stopping the cell shuts
both down. (Gradio's `share=True` only works for Gradio apps, which is why the
tunnel replaced it.)

Tests: `tests/test_app.py` covers health/variant reporting, the 15-template
catalog, parsed and unparseable generations, verification skip reasons,
generation failures, input and search-setting caps, the streaming event order
(including lock release on success and on error), candidate-tagged tokens and
dedupe, the cross-request cache and `/stats`, a repair round that receives the
harness error, and that the UI is served, all using the mock generator (no
GPU needed). `tests/test_search.py` tests the engine with fake generators and
verifiers (selection, temperature, repair targeting, format feedback, dedupe,
skip handling), plus `BatchTextStreamer` and the harness mismatch messages.
`tests/test_deploy.py` covers the deployment path:
- the code policy: escape routes rejected; every catalog reference and kernel accepted
- sandbox environment scrubbing
- the hosted-model client against a mock HTTP transport: parallel candidates, request contract, one failed candidate vs all failing
- public mode: mock fallback, remote variant, 422 and blocked kernels, per-IP 429, the 503 queue cap

### Design decisions

- **Search is training-free and verifier-driven.** Best-of-N plus repair
  uses the harness as the only judge, so its gains are measured with the same
  definition of "correct" as the dataset and the eval. It's the inference-time
  version of what Kevin-32B trains with RL.
- **Verification is a flag, not mandatory.** A harness run costs warmup + 50
  timed iterations in a fresh subprocess on top of generation, and it's
  meaningless on a machine without CUDA. When it doesn't run, the response
  says why (`skipped_reason`) instead of reporting a fake failure. Without
  that, a CPU-only server would show every kernel as ❌ because of the
  harness's "no CUDA device" guard.
- **Unparseable output is a normal response, not an error.** The base model
  hasn't been trained on the `kernel:` / `explanation:` format, so sometimes
  it won't produce a parseable kernel. That returns HTTP 200 with
  `status: "unparseable"`; the UI explains it and shows the raw output. Only an
  actual generation failure (e.g. CUDA OOM) is an HTTP 500, which the UI also
  renders as a readable message.
- **Model variant comes from config, not assumptions.** `base` unless
  `KERNELFORGE_ADAPTER_PATH` is set, so flipping to the fine-tuned adapter is
  one flag (`app/launch.py --adapter ...`) and the banner follows
  automatically.
- **Template selection never fills output fields.** The hand-written
  reference explanation is not shown in the "model-generated" box, so nothing
  hand-written can be mistaken for model output.
- **Mock mode is for plumbing only.** It never loads a model, shows a red
  banner, and is never used for screenshots presented as model output.
- **Input caps** (20k chars for the reference, 2k for the description) and a
  single-request inference lock keep a share link from being trivially
  abused or OOMing the GPU with concurrent generations.

### Safety scope (be precise about this)

The demo lets a visitor's request trigger execution of LLM-generated code
through the harness. What protects it: the code runs in a **separate
subprocess with a timeout** (hangs are killed, crashes don't take down the
server), inside a **temporary Kaggle/Colab VM** that isn't an always-on public
server and holds nothing sensitive. What does **not** protect it: there's no
filesystem/network isolation, seccomp, or container boundary around the
subprocess. That's acceptable for a short-lived demo session; it would not be
acceptable as a production service.

Every deployment also gets:

- **A scrubbed sandbox environment.** Only PATH/HOME/locale and the
  `CUDA_*`/`TRITON_*`/`TORCH_*`/`OMP_*` variables reach the subprocess, so API
  keys never do.
- **rlimits:** CPU seconds (4× the wall timeout, since torch is
  multithreaded), a 256 MB file-size cap, and no core dumps.

### Public deployment (Railway)

`Dockerfile` + `railway.json` at the repo root build a GPU-less web image. It
contains:

- CPU torch 2.9.0
- Triton 3.5.0 (interpreter mode)
- `requirements-server.txt`: the server only, no training stack

The image runs as a non-root user and binds `$PORT`. Railway's health check is
`/health`. With no GPU, the UI switches the harness to the interpreter
automatically, and the footer says there are no speedups.

**Generation** comes from `kernelforge/remote.py`, used when
`KERNELFORGE_LLM_MODEL` is set:

- It is an OpenAI-compatible streaming client.
- N candidates are N parallel requests, merged into the same
  `(sequence, text)` stream the local model produces. The search engine, repair
  loop and harness are therefore unchanged.
- If one candidate's request fails, the failure is shown inside that
  candidate. If all of them fail, the run errors.
- It's labelled "Hosted model" with a blue banner. It is not the fine-tuned
  model.
- Without `KERNELFORGE_LLM_MODEL`, public mode falls back to mock mode, which has
  its red banner.

**`KERNELFORGE_PUBLIC=1`** (set in the image) adds three protections:

- **Code policy** (`verification/policy.py`), an AST check on the reference and on
  every generated kernel:
  - Imports are limited to torch, triton, math, numpy, typing, functools and a
    few other stdlib modules.
  - It rejects exec/eval/open/`__import__`/getattr, dunder names and
    attributes (except `__init__`/`__name__`), and torch/numpy loaders
    (`torch.load`, `torch.utils`, `torch.ops`, `np.fromfile`, …).
  - A blocked reference returns 422.
  - A blocked kernel is reported as a harness error without running. The repair
    round sees that error and can fix it.
- **Rate limiting:** a per-IP sliding window (`KERNELFORGE_RATE_LIMIT`, default
  6/min; client from `X-Forwarded-For`) returns 429 with `Retry-After`.
- **Bounded queue:** more than 4 waiting runs returns 503.

The layers together: an AST policy, a subprocess, rlimits, a scrubbed
environment, a non-root container and Railway's container boundary. That is
reasonable for a portfolio demo. It is still not a proof of isolation: Python
has no airtight in-process sandbox, and a hardened service would add
gVisor/Firecracker or a separate no-network worker.

## Why the harness came first

Everything downstream depends on being able to answer one question
automatically: *does this candidate kernel produce the same output as
PyTorch, and how much faster is it?* Dataset curation needs it to filter
LLM-generated (op → kernel) pairs down to ones that actually work. Baseline
and fine-tuned evaluation need it to score model outputs, using the exact
same definitions of "correct" and "faster" in both places so the numbers
are comparable. Building this first (Phase 2, before Phase 1's dataset and
Phase 3's model selection) avoids the failure mode where a dataset gets
built, then it turns out the correctness/speed evaluation was inconsistent
between build-time and eval-time.

## Platform constraint

Triton only ships wheels for **Linux + NVIDIA GPU**. There is no macOS or
Windows build, regardless of local hardware (including Apple Silicon).
Consequences for how this repo is used:

- All local development on macOS/Windows is for **scripting and harness
  logic only** — `torch` (CPU build) is installed locally for this.
- Real Triton kernels (`examples/kernels/*/candidate_triton.py`) cannot be
  executed or tested locally. They're written by hand, checked with
  `ast.parse` for syntax, and must be run on Kaggle or Colab to actually
  verify correctness/speed.
- The harness itself (`verification/`) is fully testable locally by
  swapping in plain-Python/PyTorch "candidate" functions instead of real
  Triton kernels — this validates the *mechanics* (subprocess isolation,
  timeout handling, correctness comparison, benchmark timing) independent
  of whether Triton is installed. See `tests/test_verify_kernel.py`.
- **Triton's interpreter** (`TRITON_INTERPRET=1`) runs real Triton kernels
  on CPU tensors inside a Linux container. `docker/verify.Dockerfile` pins
  torch 2.9 (CPU), triton 3.5, and `numpy<2.3` (Triton 3.5's interpreter
  calls `int()` on 1-element arrays for runtime loop bounds, which numpy
  2.3+ rejects). This checks correctness on small shapes; it can't measure
  speed.

```bash
docker build -f docker/verify.Dockerfile -t kernelforge-verify .
docker run --rm -v "$PWD":/work kernelforge-verify python data/build_dataset.py --interpret --smoke --jobs 8
docker run --rm -v "$PWD":/work kernelforge-verify python -m pytest -q tests/test_verify_kernel.py
```

## The verification harness

### Two contracts

**Function contract** (`verify_kernel`) — a "reference" and a "candidate"
are each a Python source file that defines a top-level callable (default
name `solution`) with the same signature: takes N `torch.Tensor` positional
arguments, returns a `torch.Tensor` or a tuple of them.

```python
# reference.py
import torch

def solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
```

```python
# candidate_triton.py
import torch, triton, triton.language as tl

@triton.jit
def _kernel(...): ...

def solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ...  # launches the Triton kernel, returns a tensor
```

This is what `examples/kernels/` (Phase 0) uses, and what
`tests/test_verify_kernel.py`'s first six tests exercise.

**Model contract** (`verify_model_kernel`) — what the dataset (Phase 1)
actually stores, matching KernelBench's own convention: reference defines
`Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`; candidate defines
`ModelNew(nn.Module)` with the same `__init__`/`forward` signature as
`Model`. `get_init_inputs()` handles ops needing constructor-time state
(quantization scales, norm gain, cache shapes) that the bare function
contract can't express. See `data/SCHEMA.md` for the full rationale.

Both contracts share the same subprocess isolation, timing, and
correctness-comparison machinery — they differ only in how the reference/
candidate callables get loaded and constructed.

### Files

- **`verification/verify_kernel.py`** — the public API. Both
  `verify_kernel(reference_code, candidate_code, input_spec, ...)` (function
  contract) and `verify_model_kernel(reference_code, candidate_code, ...)`
  (model contract) write code strings to temp files, build a JSON payload,
  and spawn `sandbox_runner.py` as a subprocess with a timeout, parsing its
  single JSON line of output into a `VerifyResult`. Also exposes a CLI for
  the function contract:
  `python verification/verify_kernel.py <reference.py> <candidate.py> --shape ... --device cuda`.

- **`verification/sandbox_runner.py`** — runs *inside* the subprocess.
  Dispatches on `payload["contract"]` to `run_function_contract` or
  `run_model_contract`. The model-contract path loads both modules, seeds
  the RNG **immediately before each of `Model(*init)` and `ModelNew(*init)`
  construction** (not just before `get_inputs()`) so ops with random
  internal state (e.g. quantized weights) compare against identical random
  state rather than two independent draws, then times and compares
  `.forward` on both exactly as the function contract does. Both paths
  build identical input tensors for reference and candidate (same
  shape/dtype/seed, so results are directly comparable — see below), time
  with warmup + N repeated iterations, check numerical equality, compute
  `speedup = reference_time / candidate_time`, and always emit exactly one
  JSON object on stdout — including on internal exceptions (caught and
  reported as `{"status": "error", "error": <traceback>}`).

### Why a subprocess, not an in-process call

A generated kernel can hang (bad grid/block math causing a spin), crash the
process, or in principle wedge GPU state. Running it via
`subprocess.run(..., timeout=...)` means:
- A hang is killed at the timeout and reported as `status: "timeout"`,
  not a wedged parent process — critical once this same harness is
  called thousands of times unattended to filter LLM-generated dataset
  candidates or evaluate a fine-tuned model.
- A crash (segfault, unhandled fatal error) surfaces as `status: "crash"`
  with captured stderr, rather than taking down whatever loop is calling
  the harness.
- An optional soft memory cap (`memory_limit_bytes`, via `resource.setrlimit`
  on POSIX) can bound a runaway allocation; it's best-effort and silently
  skipped where unsupported (e.g. Windows).

### Correctness and timing details worth knowing

- **Same seed, independent tensor construction.** Reference and candidate
  each get their own freshly-constructed input tensors from the same
  `input_spec` + `seed`, built inside the same subprocess call via
  `torch.manual_seed(seed)` before each. They are not the *same* tensor
  object passed to both — this catches a candidate that mutates its input
  in place, which sharing a single tensor would hide.
- **Numerical comparison** uses `torch.allclose(atol=..., rtol=...)` on
  `.float()`-cast outputs, elementwise across tuple outputs, with
  `equal_nan=False` (a NaN in the candidate's output is always a failure,
  never treated as "matching" a NaN in the reference).
- **Timing** uses `torch.cuda.Event` pairs (correct for async CUDA kernel
  launches) when `device="cuda"`, and `time.perf_counter()` on CPU. Reports
  the **median** of `iters` timed runs (after `warmup` untimed runs), which
  is more robust to one-off scheduling noise than the mean.
- **`VerifyResult.passed`** is `True` only when `status == "ok" and correct`
  — a kernel that's fast but wrong, or one that times out, is never "passed".
- **Mismatch reports.** When outputs differ, `error` describes how: output
  count, non-tensor output, shape, NaN count, or "N/M elements outside
  atol, rtol; max abs error X at index I (candidate c, reference r)". The
  repair loop feeds this text back to the model.
- **Interpreter mode** (`verify_model_kernel(..., interpret=True)`) sets
  `TRITON_INTERPRET=1` for the subprocess only, forces CPU, runs one
  untimed iteration, and returns `interpreted=True` with no timings or
  speedup.

### Result shape

```python
@dataclass
class VerifyResult:
    status: str              # "ok" | "error" | "timeout" | "crash"
    correct: bool | None
    speedup: float | None    # reference_time_s / candidate_time_s
    reference_time_s: float | None
    candidate_time_s: float | None
    error: str | None        # traceback, or the mismatch description when correct=False
    interpreted: bool        # True when run through Triton's interpreter (no timings)
```

### Running it

```bash
# Harness mechanics — runs anywhere, no GPU/Triton needed
pytest tests/ -v

# A real kernel — Linux + NVIDIA only (Kaggle/Colab)
python verification/verify_kernel.py \
    examples/kernels/softmax/reference.py \
    examples/kernels/softmax/candidate_triton.py \
    --shape 4096 4096 --device cuda --atol 1e-2 --rtol 1e-2
```

### Test coverage (`tests/test_verify_kernel.py`)

All run on CPU with plain-PyTorch stand-ins (no Triton required), covering
the mechanics the harness has to get right regardless of what the
candidate actually is:

| Test | Verifies |
|---|---|
| `test_correct_candidate_passes` | happy path returns `status="ok"`, `correct=True`, positive speedup |
| `test_incorrect_candidate_fails_correctness` | wrong output → `correct=False`, not a crash |
| `test_candidate_that_raises_is_reported_as_error` | exception inside candidate → `status="error"` with traceback, not an unhandled subprocess crash |
| `test_candidate_missing_entry_point_is_reported_as_error` | missing `solution` function is a clear error, not an `AttributeError` traceback dump |
| `test_candidate_that_hangs_times_out` | infinite loop / long sleep → `status="timeout"` at the configured limit, subprocess killed |
| `test_identical_seed_yields_identical_inputs_across_processes` | same seed → same tensors for reference and candidate, so the correctness comparison is meaningful |
| `test_model_contract_correct_candidate_passes` | model-contract happy path |
| `test_model_contract_incorrect_candidate_fails_correctness` | model-contract wrong output → `correct=False` |
| `test_model_contract_missing_modelnew_is_reported_as_error` | missing `ModelNew` class → clear error |
| `test_model_contract_seeds_random_init_identically` | `Model`/`ModelNew` each calling `torch.randn` in `__init__` still compare correctly under identical pre-construction seeding |
| `test_model_contract_handles_tuple_outputs` | multi-tensor `forward()` return (e.g. RoPE's Q+K, cache-write's K+V) compares correctly |
| `test_model_contract_reports_where_outputs_differ` | a wrong output carries a "max abs error" description in `error` |
| `test_interpreter_mode_checks_a_triton_kernel_on_cpu` | a real `@triton.jit` kernel passes through the interpreter with `interpreted=True` and no speedup (skipped where Triton isn't installed; runs in the Docker image) |

## Example kernels (`examples/kernels/`)

Three hand-written Triton kernels, each with a PyTorch `reference.py` and a
`notes.md` explaining the optimization rationale (per Phase 0's goal of
being able to explain *why* a kernel is faster, not just produce one):

- **`vector_add`** — elementwise add; a bandwidth-bound baseline that exists
  to prove out the launch/grid/mask mechanics.
- **`relu`** — elementwise max; same baseline role, one more Triton
  primitive.
- **`softmax`** — row-wise softmax, fused into one kernel launch instead of
  eager's ~5 (max, subtract, exp, sum, divide). This is the first example
  where fusion is the actual source of speedup.

These are written for Kaggle/Colab (Linux + CUDA); on this dev machine they
are only syntax-checked (`ast.parse`), never executed.

## Phase 1 — dataset (in progress)

Full field-by-field schema lives in `data/SCHEMA.md`; this section covers
how the bulk dataset gets generated and where it currently stands.

### Scope: 5 inference-serving categories

`quantized_matmul`, `attention`, `kv_cache`, `norm`, `rope` — chosen over a
generic elementwise/reduction/matmul grab-bag specifically because they're
the ops that dominate LLM *serving* cost (as opposed to training), which is
the framing that resonates with NVIDIA/Cerebras/Cohere. `norm_rope` from
the original schema proposal was later split into `norm` and `rope` as
separate categories/template libraries.

### Template-based generation (`data/templates/`, `data/build_dataset.py`)

Hand-writing 300+ genuinely distinct, correct Triton kernels isn't
tractable (or a good use of effort — most of KernelBench's own "100
problems per level" are shape variations of similar operation families,
too). Instead: **3 hand-written templates per category x 20 shape
variants = 60 entries/category, 300 total.**

- `data/templates/common.py` — `make_entry(...)` assembles the
  schema-conformant dict; `FP16_TOLERANCE` shared constant.
- `data/templates/{quantized_matmul,attention,kv_cache,norm,rope}.py` — one
  module per category, each with 3 template functions (e.g.
  `quantized_matmul.py` has W8A16 per-tensor, W8A16 per-channel, and W8A8
  linear layers) and a `generate()` that cross-produces every template
  with every shape config in that module's grid.
- `data/templates/__init__.py` — `GENERATORS: dict[str, Callable[[], list[dict]]]`,
  one entry per category.
- `data/build_dataset.py` — CLI orchestrator: for each category, calls its
  generator, runs every entry through `verify_model_kernel()`, writes
  passing entries to `data/verified/dataset.jsonl` and everything else
  (with the failure reason attached under `verification.status`/`.error`)
  to `data/raw/rejected.jsonl`, and writes a `data/raw/generation_report.json`
  summary. Supports `--categories`, `--limit` (for smoke tests), `--device`,
  `--warmup`, `--iters`, `--timeout`, and `--interpret --smoke --jobs N`
  (one small instance per template through Triton's interpreter, in
  parallel; writes `data/raw/interpret_report.json` and never touches the
  dataset files).

### Template fixes (Sep 29, 2026)

The first GPU runs failed on some templates. Running every template through
the interpreter locally found why, and all of them were fixed at the
template level:

- **Attention/GQA references were broken.** They added an fp32 `-inf` mask
  inside an fp16 matmul, which raises "expected m1 and m2 to have the same
  dtype". The *reference* failed, so no kernel could pass. They now use a
  boolean `triu` mask with `masked_fill`.
- **The W8A8 reference used an int32 matmul**, which CUDA doesn't support. It
  now multiplies the int8 values in fp32 (exact for this range). The kernel
  runs `tl.dot` on the int8 values cast to fp16 (also exact), then applies
  both scales.
- **Attention kernels didn't scale.** One program held a whole sequence.
  They're now FlashAttention-style: 64-row query blocks, 64-column KV tiles,
  online softmax with log2(e) folded into the scale so the inner loop uses
  `exp2`, and tensor-core `tl.dot`. Causal and GQA loops stop at the diagonal.
- **Decode attention** is now a tiled loop with online softmax. GQA decode
  packs a query group (padded to 16 rows) per KV head into one `tl.dot`.
- **W8A16** dequantizes to fp16 before `tl.dot`, matching the reference's
  rounding exactly (fp32/TF32 accumulation of the unrounded product
  wouldn't). **RoPE** processes 16-row tiles instead of one program per
  row. **Norms** pick `num_warps` from the block size.
- `is_cuda` asserts were removed so the same kernels run in the interpreter.

**Interpreter result: 15/15 templates pass** (`data/raw/interpret_report.json`).
The gate is real: an off-by-one causal mask fails with "output contains NaN",
and dropping the per-channel scale fails with max abs error 1416. An
interpreter pass means the kernel computes the right thing on a small shape.
It says nothing about speed, so none of this counts as GPU-verified.

### GPU verification status: pending (0/300 on this machine)

Running `python data/build_dataset.py --device cuda` on this machine
produces:

| category | generated | verified | rejected |
|---|---|---|---|
| quantized_matmul | 60 | 0 | 60 |
| attention | 60 | 0 | 60 |
| kv_cache | 60 | 0 | 60 |
| norm | 60 | 0 | 60 |
| rope | 60 | 0 | 60 |
| **TOTAL** | **300** | **0** | **300** |

Every single rejection has the identical status/error:
`status: "error"`, `"device 'cuda' requested but no CUDA device is
available"`. This is `sandbox_runner.py`'s device guard firing before any
`Model`/`ModelNew` code runs — it is **not** a correctness signal on any of
the 300 kernels, it's this machine lacking a CUDA GPU (see Platform
constraint, above). The uniform, single-error-message pattern across all
300 is itself the evidence that this is an environment gate, not 300
independent kernel bugs.

**To get a real pass/fail verdict**, run the identical command on Kaggle or
Colab:

```bash
pip install -r requirements.txt   # picks up triton on Linux automatically
python data/build_dataset.py --device cuda
```

This will populate `data/verified/dataset.jsonl` with whatever fraction
actually compiles, runs, and matches the reference within tolerance — the
harness's entire job is to be the thing that tells you that number, rather
than trusting that hand-written (or eventually LLM-generated) kernels are
correct by inspection.

The template bugs those GPU runs surfaced are fixed (see above); the table
above is still the local no-GPU run and will be replaced with GPU numbers
after the rerun.

## Next steps

1. **Deploy the demo** — `python app/launch.py --share` on Kaggle, run a
   Samples 4 + Repair 1 search on GPU, record `docs/assets/demo.gif`.
2. Rerun `python data/build_dataset.py --device cuda` on Kaggle/Colab for
   per-category pass/fail/speedup numbers.
3. Run `evaluation/baseline_eval.py --device cuda` for the category-level
   baseline, and report pass@1 vs best-of-4 + repair.
4. Run `training/finetune.py` then `evaluation/finetuned_eval.py` for before/after.
5. Use `evaluation/explanation_review.py` for manual explanation rubric scoring.
6. Swap the trained adapter into the demo (`app/launch.py --share --adapter ...`).
