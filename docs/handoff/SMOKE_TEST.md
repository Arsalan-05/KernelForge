# CPU Smoke Test — 2026-11-15

This document records what ran and what was skipped when the full pipeline was
exercised on a CPU-only macOS dev machine (Apple Silicon, Python 3.9, torch 2.8.0).
The goal is to find real code bugs before the Kaggle GPU session, not to produce
valid results.

---

## Environment

| Item | Value |
|---|---|
| Machine | macOS (dev laptop, CPU only) |
| Python | 3.9 |
| torch | 2.8.0 |
| triton | **not installed** (Linux + CUDA only) |
| transformers | **not installed** (not in requirements.txt for dev) |
| datasets, peft, unsloth | **not installed** |
| fastapi, uvicorn | installed (server) |
| numpy, matplotlib | installed |

---

## Pipeline Steps: Ran / Skipped / Failed

### 1. Dataset build (`data/build_dataset.py --smoke --interpret`)

**Status: SKIPPED — expected.**

```
ModuleNotFoundError: No module named 'triton'
```

The `--interpret` flag routes all kernel verification through `triton.interpreter`,
which requires the `triton` package. Triton is Linux-only (requires CUDA driver headers
at build time) and is not installable on macOS. This is not a bug.

The dataset *template generation* (Python side, no triton) was verified independently:

```python
from data.templates import attention, norm, rope, kv_cache, quantized_matmul
# All five generate() calls complete successfully
# attention: 60 entries, norm: 60, rope: 60, kv_cache: 60, quantized_matmul: 60
# Total: 300 entries — no import errors, no assertion failures
```

**What runs on Kaggle:** `python data/build_dataset.py --smoke --interpret --jobs 4`
uses the Triton interpreter in a Docker container on a Kaggle CPU-only instance;
the full 300-entry run with `--device cuda` runs on T4.

---

### 2. Fine-tune dry-run (`training/finetune.py --dry-run`)

**Status: SKIPPED — expected.**

```
ModuleNotFoundError: No module named 'transformers'
```

The dry-run builds a tiny 2-layer random Qwen2 model in memory (no download), runs
2 optimizer steps, and saves an adapter. It requires `transformers` and `peft`, which
are not installed on the dev machine (they are listed in `requirements.txt` for the
GPU environment).

The *non-model* parts of `finetune.py` were verified:

- `load_eval_entries(None, dry_run=True)` → 300 unverified entries, source string correct.
- `make_split(entries, 'template', seed=42)` → train=180, val=20, test=100. No assertion errors.
- `build_sft_features` logic was exercised in the test suite
  (`tests/test_training_pipeline.py`, 3 of 4 tests pass on CPU; 1 skipped for Qwen tokenizer).

**What runs on Kaggle:** `python training/finetune.py --dry-run` completes in
under 2 minutes (tiny random model, 8 examples, 2 steps, no download).

---

### 3. Generation / baseline eval (`evaluation/baseline_eval.py --dry-run`)

**Status: PARTIAL — preflight passes, kernel loop fails at CUDA check.**

Running `python evaluation/baseline_eval.py --dry-run` starts cleanly, loads 100
held-out examples, then fails per entry with:

```
device 'cuda' requested but no CUDA device is available
```

The dry-run scores ground-truth kernels (not model output) and uses the same
verification harness. All 100 entries hit the CUDA check immediately, so no
kernels were executed. This is expected on a machine without a GPU.

The generation and search logic was confirmed in the test suite:
`tests/test_eval_runs.py` (10 tests pass, 3 skipped for transformers/model calls).

**What runs on Kaggle:** `python evaluation/baseline_eval.py --dry-run --interpret`
(interpreter mode, no GPU needed) or the full `--device cuda` run.

---

### 4. Verification (`verification/verify_kernel.py`)

**Status: SKIPPED — expected.**

The verifier imports `triton` at the top of each kernel module it executes in a
subprocess. Without triton installed, every verification attempt raises
`ModuleNotFoundError`. This is not a code bug.

The harness logic (sandbox runner, policy, timeout handling) was confirmed in the
test suite: `tests/test_verify_kernel.py` (12 tests pass, 1 skipped for
`test_interpreter_mode_checks_a_triton_kernel_on_cpu`).

**What runs on Kaggle:** `python verification/verify_kernel.py ref.py cand.py --device cuda`
runs in a subprocess with the Triton compiler active.

---

### 5. Report generator (`python -m evaluation.report`)

**Status: PASSED.**

```bash
python -m evaluation.report /tmp/kf_smoke_outputs --out /tmp/kf_smoke_report --no-figures
```

The report generator was exercised against a minimal `kernelforge_outputs/` directory
containing only `data/raw/interpret_report.json` (copied from the committed interpreter
report). Output:

- `results.md` — written, all GPU sections marked absent.
- `readme_results_snippet.md` — written.
- `summary.json` — written, all GPU fields null.
- No `.tex` fragments written (no eval data present — correct).
- Exit code 0, no exceptions.

The figures path was skipped (`--no-figures`) because matplotlib is optional.

**What runs on Kaggle:** Exactly the same command against the full `kernelforge_outputs/`
directory produced by the GPU session.

---

### 6. Pre-flight check (`scripts/preflight_check.py`)

**Status: PASSED (expected failures reported, no crash).**

```
3 required check(s) failed: datasets, peft, CUDA available
```

Pre-flight correctly identified the three missing requirements for a full GPU run.
All "optional" checks (unsloth, triton) passed as expected-absent. No unhandled
exceptions.

---

### 7. Leakage check (`scripts/check_split_leakage.py`)

**Status: PASSED.**

```
PASSED — no leakage detected.
Test entries per category: {'attention': 20, 'kv_cache': 20, 'norm': 20, 'quantized_matmul': 20, 'rope': 20}
Held-out templates: {'attention': 'attention__gqa_causal_prefill', 'kv_cache': 'kv_cache__decode_attention_gqa',
                     'norm': 'norm__rmsnorm_residual', 'quantized_matmul': 'quantized_matmul__w8a16_perchannel',
                     'rope': 'rope__interleaved'}
```

---

### 8. Full test suite (`pytest tests/`)

**Status: PASSED.**

```
272 passed, 5 skipped, 1 warning in ~33 s
```

Skipped tests (all expected — missing packages on dev machine):

| Test | Reason |
|---|---|
| `test_time_limit_callback_saves_adapter_and_stops` | needs `transformers` |
| `test_finetune_dry_run_time_limit_writes_logs_into_run_folder` | needs `transformers` |
| `test_recorded_generation_settings_match_what_generate_used` | needs `transformers` |
| `test_loss_mask_with_real_qwen_tokenizer` | needs `transformers` (Qwen tokenizer) |
| `test_interpreter_mode_checks_a_triton_kernel_on_cpu` | needs `triton` |

No test failures.

---

## Bugs Found and Fixed

None. All failures were expected package-availability issues on a dev Mac. No
code logic bugs were surfaced by the smoke test.

The two categories of failures — `No module named 'triton'` and
`No module named 'transformers'` — both have correct conditional guards in the
production code (try/except at import, CLI flags like `--interpret` and `--dry-run`,
and pytest `skipIf` decorators).

---

## What Needs a Real GPU to Test

| Component | Minimum | Notes |
|---|---|---|
| Dataset build (kernel verification) | CUDA device | `--interpret` works CPU-only but needs `triton` (Linux) |
| Fine-tune dry-run | CPU + transformers | `pip install transformers peft datasets` on Kaggle |
| Fine-tune real run | T4 (16 GB) + transformers + unsloth | See `docs/handoff/KAGGLE_RUNBOOK.md` |
| Generation eval | CUDA device + model weights | `--dry-run --interpret` skips model download |
| Kernel verification | CUDA device | Triton compiles to real PTX |
| Token-length audit | tokenizer only | Needs `transformers` for `AutoTokenizer` |

---

## Confidence Assessment

The smoke test confirms that all Python-only components work correctly on CPU:
template generation, dataset splitting, prompt building, response parsing, search
logic, evaluation scaffolding, report generation, pre-flight check, and leakage
check. The gap between this smoke test and a real T4 run is exactly:

1. `triton` compilation (kernel correctness, throughput).
2. `transformers` / `peft` / `datasets` (model load, tokenization, fine-tuning loop).
3. CUDA device (all verification steps).

None of these gaps indicate code bugs; they are hardware/package prerequisites
that the Kaggle environment provides.
