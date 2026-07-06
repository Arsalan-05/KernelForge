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
| 1 — Dataset | Not started |
| 2 — Verification harness | **Done**, locally testable |
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

### Contract

A "reference" and a "candidate" are each a Python source file that defines
a top-level callable (default name `solution`) with the same signature:
takes N `torch.Tensor` positional arguments, returns a `torch.Tensor` or a
tuple of them.

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

This mirrors KernelBench's op-level structure (Phase 1's planned dataset
source) and keeps the harness agnostic to *how* the candidate is
implemented — it could be a hand-written Triton kernel today, or an
LLM-generated one later, without changing the interface.

### Files

- **`verification/verify_kernel.py`** — the public API.
  `verify_kernel(reference_code, candidate_code, input_spec, ...)` writes
  both code strings to temp files, builds a JSON payload describing how to
  reconstruct inputs, and spawns `sandbox_runner.py` as a subprocess with a
  timeout. Parses the subprocess's single JSON line of output into a
  `VerifyResult`. Also exposes a CLI:
  `python verification/verify_kernel.py <reference.py> <candidate.py> --shape ... --device cuda`.

- **`verification/sandbox_runner.py`** — runs *inside* the subprocess.
  Loads both modules via `importlib`, builds identical input tensors for
  each (same shape/dtype/seed, so results are directly comparable — see
  below), times both with warmup + N repeated iterations, checks numerical
  equality, computes `speedup = reference_time / candidate_time`, and
  always emits exactly one JSON object on stdout — including on internal
  exceptions (caught and reported as `{"status": "error", "error": <traceback>}`).

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

## Next steps

1. Run the three example kernels on Kaggle/Colab against the harness to get
   real correctness/speedup numbers (closes out Phase 0's deliverable).
2. Start Phase 1: pull KernelBench's task structure, decide the exact
   dataset schema, and begin generating + verifying (op → kernel) pairs
   through this same harness.
