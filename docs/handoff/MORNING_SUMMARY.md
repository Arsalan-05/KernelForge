# Morning Summary — 2026-11-15

This document is the handoff from an overnight integration session. It covers what was done, what is verified, what is blocked on GPU, and exactly what to do next.

---

## What is in each branch

### `agent/merged` ← start here

The main integration branch. All four agent branches (a/b/c/d) were merged cleanly (no conflicts — disjoint file sets). All work from the overnight session that is complete and tested lives here.

**231 tests pass, 5 skip** (the 5 skips require a GPU or the `transformers` package, which is not installed on this Mac).

**What is on this branch:**

| Area | Status | Key files |
|---|---|---|
| Docker interpreter report | ✅ Committed | `data/raw/interpret_report_2026-11-15.json` |
| GPU session scripts | ✅ Committed | `scripts/gpu_session.py`, `notebooks/kaggle_gpu_session.ipynb` |
| Dataset tooling | ✅ Committed | `data/build_dataset.py`, `data/templates/` |
| Verification harness | ✅ Committed | `verification/verify_kernel.py` |
| Evaluation pipeline | ✅ Committed | `evaluation/baseline_eval.py`, `evaluation/common.py` |
| Report generator | ✅ Committed | `evaluation/report/`, 5 LaTeX fragments |
| Fine-tuning | ✅ Committed | `training/finetune.py` (+ `peak_memory_gb`) |
| Paper skeleton | ✅ Committed | `docs/paper/paper.tex` (all table cells `\todo{}`) |
| README | ✅ Updated | Accurate numbers, `--budget-hours` flag, KAGGLE_RUNBOOK link |
| KAGGLE_RUNBOOK | ✅ Written | `docs/handoff/KAGGLE_RUNBOOK.md` |
| INTEGRATION_REPORT | ✅ Written | `docs/handoff/INTEGRATION_REPORT.md` |
| BLOCKERS | ✅ Written | `docs/handoff/BLOCKERS.md` |

### `agent/extras`

Experiment infrastructure and test hardening on top of `agent/merged`.

**269 tests pass, 5 skip.**

| Item | File(s) |
|---|---|
| Experiment matrix (7 conditions) | `evaluation/ablation_matrix.json`, `scripts/run_ablations.py` |
| `pass_at_k()` (Chen et al. 2021) + failure breakdown | `evaluation/report/stats.py` |
| Harness hardening: shapes, dtypes, tolerances | `tests/test_verify_kernel.py` |
| Split leakage checker | `scripts/check_split_leakage.py`, `tests/test_split_leakage.py` |
| Pre-flight check | `scripts/preflight_check.py` |
| Handoff doc | `docs/handoff/EXTRAS_REPORT.md` |

### `agent/ui`

Studio UI upgrade on top of `agent/merged`.

**231 tests pass, 5 skip** (backend tests only; no browser automation).

| Item | File(s) |
|---|---|
| `GET /results` endpoint | `app/backend/main.py` |
| Results tab (coverage data) | `app/web/index.html`, `app/web/app.js` |
| Saved settings (localStorage) | `app/web/app.js` |
| Retry button after error | `app/web/index.html`, `app/web/app.js` |
| Accessibility (focus rings, ARIA) | `app/web/styles.css`, `app/web/index.html` |
| No-layout-jump (min-height) | `app/web/styles.css` |
| Compare pane model label | `app/web/app.js` |
| Handoff doc | `docs/handoff/UI_REPORT.md` |

---

## Verified facts (non-negotiable, cite from the committed report)

**Source file:** `data/raw/interpret_report_2026-11-15.json`

| Fact | Value |
|---|---|
| Harness mode | Triton interpreter, CPU, correctness only — **no GPU, no speedups** |
| Total shapes tested | 300 (15 templates × 20 shapes × 5 categories) |
| Shapes that pass | **285 / 300** |
| Shapes that OOM (crash exit −9) | **15** |
| Templates that pass all 20 shapes | **10 / 15** |
| Templates with OOM shapes | **5 / 15**: `attention/causal_prefill` (1 OOM), `attention/gqa_causal_prefill` (1 OOM), `norm/rmsnorm` (4 OOM), `norm/layernorm` (4 OOM), `norm/rmsnorm_residual` (5 OOM) |
| OOM category | All 15 OOM shapes are the largest shapes in attention and norm templates |
| T4 fit | **Unknown** — not tested. The 8 GB VM OOMed; T4 has 16 GB; may or may not fit |

**Do not say "11 of 15 pass" or "4 templates have OOM shapes".** The correct numbers are 10 and 5.

---

## What has never happened (GPU blockers)

Nothing that depends on a real GPU has run. Specifically:

- **G1** — No GPU-verified dataset entries. `data/verified/dataset.jsonl` is empty. All entries come from the interpreter; the template kernels compute correctly but have never been benchmarked.
- **G2** — The 15 OOM shapes (8 GB VM) have never been tested on a 16 GB T4.
- **G3** — No baseline eval results. All paper table cells are `\todo{}`.
- **G4** — No fine-tuned adapter. `training/outputs/` is empty.
- **G5** — `peak_memory_gb` in `split_info.json` is `null`; will be set after a real training run.
- **G6** — No speedup numbers, anywhere.

The LaTeX fragments in `docs/results/` do not exist yet — they are created by `evaluation/report/__main__.py` after a GPU run.

---

## Immediate next steps

### Step 1 — Run the pre-flight check

```bash
python scripts/preflight_check.py --no-gpu   # CPU checks only (this machine)
```

This will catch any missing configs or broken imports before you go to Kaggle.

### Step 2 — Kaggle T4 session

Follow `docs/handoff/KAGGLE_RUNBOOK.md` exactly. The minimum viable run:

```bash
python scripts/gpu_session.py --budget-hours 4
```

This builds the dataset, runs the baseline eval, and writes:
- `data/verified/dataset.jsonl`
- `evaluation/results/<run_name>.json`
- LaTeX fragments to `docs/results/`

### Step 3 — After the GPU run

1. Confirm `data/verified/dataset.jsonl` has ~270–300 entries (some may be rejected for being slow or wrong).
2. Run `python -m evaluation.report` to regenerate LaTeX fragments.
3. Run `python scripts/check_split_leakage.py --dataset data/verified/dataset.jsonl` to confirm no leakage.
4. Update the paper table cells (they will fill from the LaTeX fragments automatically).
5. Update the Results tab in the Studio: the `GET /results` endpoint will show GPU eval data once `evaluation/results/` has a file.

### Step 4 — Fine-tuning (after a verified dataset exists)

```bash
python training/finetune.py --config training/configs/finetune_default.json \
    --dataset data/verified/dataset.jsonl --output training/outputs/ft_r16
```

Then restart the studio with `KERNELFORGE_ADAPTER_PATH=training/outputs/ft_r16` to flip the pill to green.

---

## Test counts at a glance

| Branch | Passing | Skipped | Notes |
|---|---|---|---|
| `agent/merged` | 231 | 5 | GPU + transformers skips |
| `agent/extras` | 269 | 5 | +38 new tests (ablations, stats, harness, leakage) |
| `agent/ui` | 231 | 5 | UI changes have no browser-automation tests |

---

## Files to read before touching anything

| Read first | Why |
|---|---|
| `MASTER_PLAN.md` | Full architecture, roadmap, honest-scope rules |
| `docs/handoff/BLOCKERS.md` | Exact list of GPU-blocked items and what unblocks each |
| `docs/handoff/KAGGLE_RUNBOOK.md` | Step-by-step GPU session guide |
| `docs/handoff/INTEGRATION_REPORT.md` | Per-task record of what the overnight session did |
| `data/raw/interpret_report_2026-11-15.json` | Canonical source for the 285/300 and 15 OOM facts |
| `evaluation/report/README.md` | Exact format of the LaTeX fragments and how to regenerate them |
