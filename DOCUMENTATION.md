# KernelForge — Documentation

Fine-tuned LLM that turns a naive PyTorch op into an optimized Triton GPU
kernel, with automated correctness verification and speedup benchmarking.
This document covers what's been built so far, how it works, and how to run
and extend it. For the full multi-phase project plan (dataset strategy,
fine-tuning, evaluation, app, positioning), see the plan shared at project
kickoff — this file tracks implementation, not the roadmap.

## Current status

| Phase | State |
|---|---|
| 0 — Foundations | Partial: 3 hand-written example kernels done (`examples/kernels/`) |
| 1 — Dataset | In progress: schema approved, 300 candidate entries generated across 5 categories, harness-verified pipeline built and run — **0/300 GPU-verified so far, blocked on lacking a local CUDA GPU, not on kernel correctness** (see below) |
| 2 — Verification harness | **Done**, locally testable, now supports both the function contract and KernelBench's Model/ModelNew contract |
| 3 — Model selection & baseline | Not started |
| 4 — Fine-tuning | Not started |
| 5 — Evaluation | Not started |
| 6 — Tool/UI | Not started |
| 7 — Write-up & positioning | Not started |

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

### Result shape

```python
@dataclass
class VerifyResult:
    status: str              # "ok" | "error" | "timeout" | "crash"
    correct: bool | None
    speedup: float | None    # reference_time_s / candidate_time_s
    reference_time_s: float | None
    candidate_time_s: float | None
    error: str | None
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
  `--warmup`, `--iters`, `--timeout`.

All 300 generated entries are syntax-checked (`ast.parse`) as part of
building them; templates were also manually reviewed for the RoPE/GQA/
quantization math they implement, but **none have executed on a real GPU
yet** (see below) — the math being *plausible* is not the same as it being
*verified*, which is the whole reason this harness exists.

### Current verification status: 0/300, blocked on local hardware

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

## Next steps

1. Run `python data/build_dataset.py --device cuda` on Kaggle/Colab to get
   the real per-category pass/fail/speedup numbers.
2. For any template with a low pass rate, fix the template (not each
   individual failing shape variant) and regenerate — the shape-grid
   design means one fix propagates to all 20 variants of that template.
3. Once a solid verified set exists, decide whether 300 is enough or
   whether to widen shape grids / add templates for under-represented
   categories.
4. Run the three Phase 0 example kernels (`examples/kernels/`) through the
   harness too, closing out that deliverable.
