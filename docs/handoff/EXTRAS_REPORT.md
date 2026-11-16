# Extras Report — agent/extras

Branch: `agent/extras` (from `agent/merged`)  
Committed: 2026-11-15  
Test result: **269 passed, 5 skipped** (all skips are GPU/transformers-dependent)

---

## 1. Experiment Matrix

**Files:**
- `evaluation/ablation_matrix.json` — 7 conditions, shared flags
- `scripts/run_ablations.py` — iterates conditions, builds CLI commands, writes `ablations_summary.json`
- `tests/test_ablations.py` — 13 tests

**What it does:**

`ablation_matrix.json` defines the full ablation grid:

| condition | base/ft | few-shot | LoRA rank |
|---|---|---|---|
| `base_zero_shot` | base | none | — |
| `base_few_shot_other` | base | other-category sibling | — |
| `base_few_shot_same` | base | same-template sibling | — |
| `ft_r8_zero_shot` | fine-tuned | none | 8 |
| `ft_r16_zero_shot` | fine-tuned | none | 16 |
| `ft_r32_zero_shot` | fine-tuned | none | 32 |
| `ft_r16_few_shot_other` | fine-tuned | other-category sibling | 16 |

`scripts/run_ablations.py` builds the `baseline_eval.py` CLI call for each condition, writes a temporary QLoRA config for fine-tuned conditions, and aggregates results into `ablations_summary.json`. Flags: `--dry-run`, `--conditions` (comma-separated filter), `--extra-flags`.

**What remains untested on a real GPU:**
- Actual subprocess execution (tests mock subprocess). The dry-run path is covered; the live path needs a Kaggle T4 session.
- Fine-tuned conditions require a saved adapter in `training/outputs/`.

---

## 2. Extra Report Metrics

**Files:**
- `evaluation/report/stats.py` — `pass_at_k()`, `pass_at_k_aggregate()`, `FAILURE_TYPE_LABELS`, `failure_breakdown()`
- `tests/test_report.py` — 7 new tests (added to existing file)

**What it does:**

`pass_at_k(n, c, k)` — Chen et al. 2021 unbiased estimator:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Computed in log-domain to avoid overflow for large n. Boundary conditions:
- `c == 0` → `0.0`
- `n - c < k` → `1.0` (fewer than k incorrect samples; must pick ≥ 1 correct)
- `n < k` → `None` (not enough samples)

`pass_at_k_aggregate(counts, k)` — applies `pass_at_k` to a list of `(n, c)` pairs and returns the macro-average over entries where the estimator is defined.

`failure_breakdown(outcome_counts)` — maps raw outcome strings (`no_kernel`, `error`, `timeout`, `crash`, `oom`) to display categories and returns a sorted list of `(label, count)` pairs.

**What remains untested:**
- Integration with actual eval reports (requires GPU run). Stats functions are unit-tested against hand-verified values.

---

## 3. Harness Hardening

**File:** `tests/test_verify_kernel.py` (extended)

**What it does:**

Added parametrized tests to `test_verify_kernel.py`:

- **Shape coverage:** `[1]`, `[7]` (non-power-of-2), `[512]`, `[1024]`, `[4, 128]`, `[2, 3, 64]` — the harness must handle tensors of varying rank and size.
- **Dtype coverage:** `float32`, `float16`. Integer dtypes (`int32`, `int64`) are not tested because the harness uses `torch.rand()` to build inputs, which does not support integer types. This is a documented limitation, not a defect.
- **Tolerance test:** Demonstrates that the `atol`/`rtol` parameters are respected (fp16 roundtrip test).
- **Error message quality:** When outputs differ, the error message must mention magnitude (`max abs error`, `differ`, `mismatch`, or `abs`).
- **`VerifyResult.passed` property:** Verifies the property returns `True` iff `status == "ok"` and `correct == True`.

**What remains untested:**
- Triton interpreter mode on real Triton kernels (the `test_interpreter_mode_checks_a_triton_kernel_on_cpu` test covers this but is skipped when `triton` is not installed).
- GPU device path (`device="cuda"`).

---

## 4. Split Leakage Checker

**Files:**
- `scripts/check_split_leakage.py` — `check_leakage(train, val, test) -> list[str]`, `main()`
- `tests/test_split_leakage.py` — 7 tests

**What it does:**

`check_leakage()` runs 6 checks on a train/val/test split:

1. **Template isolation** — no template appears on both sides of the split boundary.
2. **ID isolation** — no entry id appears in both train/val and test.
3. **Kernel source isolation** — no kernel source string appears on both sides. Within a template all shapes share the same kernel logic, so a shape-level split would leak the kernel.
4. **One held-out template per category** — test must have exactly one template per category (the structural requirement for the held-out-template split).
5. **Category balance** — all categories must have the same number of test entries.
6. **Explanation leakage** — verbatim explanation text must not appear on both sides (a training answer present verbatim in test means the model only needs to copy).

`main()` CLI: `python scripts/check_split_leakage.py` (uses unverified template candidates), or `--dataset data/verified/dataset.jsonl` for the GPU-verified dataset. Exits 0 if clean, 1 if leakage found.

The `test_check_leakage_no_false_positives_on_clean_split` test runs against the real template candidates and asserts `check_leakage(...) == []`.

**What remains untested:**
- Against the GPU-verified `data/verified/dataset.jsonl` (file does not exist yet; needs GPU run).

---

## 5. Pre-flight Check

**File:** `scripts/preflight_check.py`

**What it does:**

Runs all checks that do not require a GPU unconditionally, and GPU-dependent checks unless `--no-gpu` is passed. Exits 0 if all required checks pass, 1 if any fail.

Checks:

| # | Check | Required / Warning |
|---|---|---|
| 1 | Python ≥ 3.9 | Required |
| 2 | `torch`, `datasets`, `peft` installed | Required |
| 2 | `triton`, `unsloth` installed | Warning (optional) |
| 3 | CUDA available | Required (unless --no-gpu) |
| 3 | GPU memory ≥ 14 GB | Warning |
| 3 | GPU free memory > 1 GB | Required (unless --no-gpu) |
| 4 | `data/verified/dataset.jsonl` present | Warning (falls back to unverified candidates) |
| 4 | Unverified template candidates loadable | Required if verified dataset missing |
| 5 | Config files parseable as valid JSON | Required |
| 6 | OOM shape reminder (from interpret_report) | Warning (informational) |
| 7 | pytest tests/ pass (--run-tests only) | Required if requested |

Usage:
```
python scripts/preflight_check.py           # all checks
python scripts/preflight_check.py --no-gpu  # skip GPU checks
python scripts/preflight_check.py --run-tests  # also run test suite
```

**What remains untested:**
- GPU path (checks 3, 3b, 3c). The script is written to handle `torch.cuda.is_available() == False` gracefully; tested implicitly by the fact that the test machine has no GPU and `--no-gpu` is the implied default.
- The `--run-tests` path is covered by code reading alone; it uses `subprocess.run` and relies on pytest being installed.

---

## Test count breakdown

| File | Tests added |
|---|---|
| `tests/test_ablations.py` | 13 (new file) |
| `tests/test_report.py` | 7 (added to existing) |
| `tests/test_verify_kernel.py` | 14 (added to existing) |
| `tests/test_split_leakage.py` | 7 (new file) |
| **Total new** | **41** |

Final suite: **269 passed, 5 skipped**.
