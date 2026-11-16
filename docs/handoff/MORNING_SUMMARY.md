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
| `agent/ui` | 234 | 5 | +3 tests for `GET /results` (present, missing, malformed report) |

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

---

## How to merge into main (safest order)

Merge in this order: `agent/merged` → `agent/extras` → `agent/ui`. Each merge is a fast-forward or a pure-additive merge with no shared files between the branch being merged and the ones already in.

```bash
git checkout main

# 1 — merge agent/merged first (the integration base; all prior agents are already in here)
git merge --no-ff agent/merged -m "Merge agent/merged: GPU session, dataset, verification, eval, fine-tuning, paper skeleton"

# 2 — merge agent/extras (experiment matrix, pass@k, harness hardening, leakage checker, pre-flight)
git merge --no-ff agent/extras -m "Merge agent/extras: ablation matrix, pass@k estimator, split leakage checker, pre-flight check"

# 3 — merge agent/ui last (Studio UI upgrade, GET /results, accessibility, saved settings)
git merge --no-ff agent/ui -m "Merge agent/ui: results tab, honest labels, accessibility, no-layout-jumps, saved settings"
```

**Which branch to review most carefully: `agent/ui`.**

`agent/ui` is the only branch that modifies files served live by the Railway deploy (`app/backend/main.py`, `app/web/`). A mistake here is immediately visible to anyone hitting the public URL. Review every change in `app/` before merging.

**Likely conflicts between agent/extras and agent/ui:**

There are **no expected conflicts**. The two branches were both cut from `agent/merged` at the same commit and touch entirely disjoint file sets:

- `agent/extras` touches: `evaluation/`, `scripts/`, `tests/test_report.py`, `tests/test_verify_kernel.py`, `tests/test_ablations.py`, `tests/test_split_leakage.py`
- `agent/ui` touches: `app/backend/main.py`, `app/web/`, `tests/test_app.py`, `docs/handoff/UI_REPORT.md`

The `/results` tests were placed in `tests/test_app.py` (not `tests/test_report.py`) specifically to keep the test files disjoint. If `git merge` does raise a conflict, it will be in `tests/test_app.py` only (if extras somehow touched it — it does not).

After all three merges, run `pytest` to confirm 234+ tests pass.

---

## Railway deploy risk list

The following items in `agent/ui` could affect the live Railway deploy. Review each before merging or pushing.

| Risk | Detail | Severity |
|---|---|---|
| **New `GET /results` endpoint** | Added to `app/backend/main.py`. No new dependencies; reads a local file. Safe if the file is absent (returns `{interpreter: null, gpu: null}`). | Low |
| **Report file exists only locally** | `data/raw/interpret_report_2026-11-15.json` is git-ignored but force-added locally. Railway will not have it unless it is committed to the repo or placed in the Docker image. The endpoint handles this gracefully (returns `null`). | Low — handled |
| **`_ROOT` path constant** | `app/backend/main.py` now defines `_ROOT = Path(__file__).parent.parent.parent` and derives `_RESULTS_REPORT` from it. If Railway's working directory or Docker `WORKDIR` differs, the path may resolve incorrectly. Verify with a one-shot `python -c "from app.backend.main import _RESULTS_REPORT; print(_RESULTS_REPORT)"` in the Railway shell. | Low |
| **No new Python dependencies** | The `/results` endpoint uses only `json` and `pathlib` (stdlib). No new entries in `requirements.txt`. | None |
| **No new environment variables** | The UI changes add `localStorage` state client-side only. No new `KERNELFORGE_*` vars are read server-side. | None |
| **Hosted-model label change** | `renderKernel()` now reads `model_variant` from the `/generate` response and updates the compare-pane label. Railway uses `KERNELFORGE_LLM_MODEL`; the `/health` endpoint already returns `model_variant: "base"` or `"fine-tuned"`. The label will display whatever the backend reports — no silent wrong label. | Low |
| **localStorage settings key** | `kernelforge.settings.v1` is written to the browser's localStorage. No server-side effect. If a returning user has stale data under a different key it is silently ignored. | None |
| **CSS `min-height` additions** | `min-height: 400px` on tab panels prevents layout jump. Pure additive CSS; no breakage risk. | None |

**Summary:** No new packages, no new environment variables, no schema migrations. The only runtime change that touches the server is the new `GET /results` route, which is safe to deploy before the report file exists.

---

## Session timing

All times are US/Eastern (UTC-5).

| Event | Timestamp |
|---|---|
| First commit tonight | `2026-11-15 17:33:16 -0500` — "build_dataset: incremental progress log, --resume, solo OOM retry, time budget" |
| Last commit tonight | `2026-11-15 21:04:45 -0500` — "Add tests for GET /results: present report, missing report, malformed report" (on `agent/ui`) |
| **Total session** | **3 hours 31 minutes** |

61 commits were made across all branches during this session.

---

## Worktree and branch state

Output of `git worktree list` at end of session:

```
/Users/Arsalan Amir Ali/Developer/GitHub/KernelForge 58e2588 [agent/merged]
```

(No separate worktrees — all branch work was done by switching branches in the main worktree.)

Output of `git branch -a` at end of session:

```
  agent/a
  agent/b
  agent/c
  agent/d
  agent/extras
* agent/merged
  agent/ui
  main
  remotes/origin/HEAD -> origin/main
  remotes/origin/main
```

---

## Merge into main — results

Performed after the overnight session. All three merges succeeded with no conflicts (disjoint file sets).

| Merge | Commit | Strategy |
|---|---|---|
| `agent/merged` → `main` | `ee802ae` | ort, no conflicts |
| `agent/extras` → `main` | `d1656a2` | ort, no conflicts |
| `agent/ui` → `main` | `cabef55` | ort, no conflicts |

**Test suite on main after all merges:** `272 passed, 5 skipped, 0 failed`

The 3 extra tests (vs. 269 on `agent/extras`) come from the `GET /results` tests added on `agent/ui`.

**Deploy verification:**
- `Dockerfile` line 27: `COPY . .` — report file will be included
- `.dockerignore` excludes `rejected.jsonl` and `generation_report.json` but NOT `interpret_report_2026-11-15.json` — file will reach the container
- `data/raw/interpret_report_2026-11-15.json` is tracked and present on main
- No `.env` files or secrets tracked (`git ls-files | grep -iE "\.env$|secret|key"` → nothing)
- No wording says "11 of 15" or "4 templates have OOM shapes" anywhere on main
- All interpreter results correctly labelled "correctness only, no GPU result"

**Blockers:** none — all three merges completed cleanly.

**`agent/phase2` branch** created from this point on main (commit `cabef55`).

### To push when ready

```bash
git remote set-url --push origin https://github.com/Arsalan-05/KernelForge.git
git push origin main
## Phase 2 — 2026-11-15

**Branch:** `agent/phase2` (cut from `main` after all Phase 1 merges)

**Commit:** `fb588a1` — "Phase 2: T4 fit review, smoke test, Kaggle notebook, paper sections, quality pass"

**Tests:** 272 passed, 5 skipped (same as end of Phase 1; no test count change in this session).

---

### What was done

#### 1. T4 Fit Review (`docs/handoff/T4_FIT.md`)

Computed peak tensor-memory estimates for all 300 (template × shape) entries using the kernel input/output tensor sizes (fp16, int8 as appropriate). All 300 fit on a 16 GB T4:

- Largest kernel: `norm/rmsnorm_residual r65536 h12288` = 4.50 GB
- Fine-tuning overhead (QLoRA 1.5B): ~2.5 GB → 13.5 GB remaining for kernels
- Generation overhead (inference): ~2.0 GB → 14.0 GB remaining for kernels
- **All 300 fit under both fine-tuning and generation scenarios.**
- The 15 interpreter OOMs all hit the 8 GB CPU RAM cap of the Docker VM; their tensor footprints (0.27–4.50 GB) comfortably fit T4 GPU VRAM.

**Labelled as estimates throughout. T4 fit is untested.**

#### 2. CPU Smoke Test (`docs/handoff/SMOKE_TEST.md`)

Exercised the full pipeline on the dev Mac (no GPU, no triton, no transformers):

| Step | Result |
|---|---|
| Dataset build (`--smoke --interpret`) | SKIPPED — `No module named 'triton'` (expected) |
| Fine-tune dry-run | SKIPPED — `No module named 'transformers'` (expected) |
| Baseline eval `--dry-run` | PARTIAL — starts cleanly, fails at CUDA check (expected) |
| Verification harness | SKIPPED — needs triton (expected) |
| Report generator | PASSED — exit 0, wrote results.md, summary.json |
| Pre-flight check | PASSED — 3 expected failures reported, no crash |
| Leakage check | PASSED — no leakage detected |
| Full test suite | PASSED — 272 passed, 5 skipped |

**No code bugs found.** All failures are package-availability issues on macOS.

#### 3. Kaggle Notebook (`notebooks/kaggle_gpu_session.ipynb`)

Added three new cells between setup and main run:
- **Cell 2b:** `python scripts/preflight_check.py` — must show 0 required failures
- **Cell 2c:** `python scripts/check_split_leakage.py` — must show PASSED
- Updated settings cell with inline time-budget comment (est. 6–7 h without fine-tuning, 8–9 h with)
- Expanded fallback zip cell to also collect `evaluation/results/` and `training/checkpoints/`

Updated `docs/handoff/KAGGLE_RUNBOOK.md` with a time-budget table and cross-references to `T4_FIT.md` and `SMOKE_TEST.md`.

#### 4. Paper Sections (`docs/paper/paper.tex`)

Wrote/completed these sections:

- **Method (§3):** Fixed incorrect claim about loss masking (prompt tokens ARE masked with IGNORE\_INDEX = -100; loss is completion-only). Clarified that the standard evaluation uses zero-shot prompt. Updated interpreter results to: **285/300 pass, 0 incorrect, 15 OOM, 10 of 15 templates fully pass**.
- **Experimental Setup (§4):** Filled decoding cap (2048 tokens, above the longest reference at 1,302 tokens). Fixed n per category (20 prompts, 100 total). Removed fast\_1 TODO and stated that raw numerator is reported alongside conditional mean. GPU hardware section left as `\todo{}` pending the actual GPU session.
- **Threats to Validity (§7, new section):** Wrote seven distinct threats covering construct validity (tolerance choice, forward-pass only), internal validity (single GPU / no variance, template diversity, explanation rater, loss-masking choice), and external validity (hardware, operator coverage). Separated from the existing Limitations section.
- All GPU result cells remain `\todo{}` placeholders. No numbers were invented.

#### 5. Quality Pass

- `kernelforge/model.py`: removed unused `Any` import from `typing`
- `README.md`: updated stale "84 tests" count to "272 tests pass on macOS"
- `MASTER_PLAN.md`: same test count update

---

### Blockers (unchanged from Phase 1)

All GPU-dependent items remain blocked. See `docs/handoff/BLOCKERS.md`.

| Item | Blocked on |
|---|---|
| GPU-verified dataset | Kaggle T4 session |
| Baseline / fine-tuned eval results | GPU session |
| Peak memory in training | GPU session |
| Fine-tuning backend (Unsloth vs transformers) | GPU session |
| Table cells in paper.tex | GPU session |
| Token count audit (`scripts/check_token_lengths.py`) | `transformers` package |

---

### Next step

Run the Kaggle GPU session using `notebooks/kaggle_gpu_session.ipynb` (now updated with pre-flight and leakage checks). Follow `docs/handoff/KAGGLE_RUNBOOK.md`.

---

## Overnight session — 2026-11-16

**Branch:** `agent/overnight` (cut from `agent/data` @ `242509d`)
**Session:** 01:14–05:45 EST
**Tests:** 283 passed, 5 skipped (unchanged — no test-count regressions).

### What was done

#### Task 1 — Mutation test expanded to 32 mutations / all 15 templates

- **Script:** `scripts/mutation_test.py`
- **Mutations:** 32 total (all 15 templates, 2–3 per template); types: wrong arithmetic, wrong reduction axes, missing masks, missing scale factors, wrong indices, missing operands, dtype errors.
- **Results as reported (overclaim — see correction in agent/overnight2):** 27 caught / 27 denominator = 100%.
  Denominator excluded 2 `documented_gap` mutations (`norm_rmsnorm__missing_weight`, `layernorm__missing_bias`)
  that are real blind spots where the harness fails to catch the mutation. These were removed from the
  denominator by labelling them `documented_gap`, inflating the score.
- **Corrected results (agent/overnight2):** 29 caught / 29 non-equivalent = 100% (denominator = 29).
  Both norm blind spots fixed with random non-identity init; now caught.
- **Post-hoc reclassification disclosure:** `cross_attn__wrong_reduction_axis` started as type
  `wrong_reduction` (expected caught), survived the run, and was reclassified to `harmless_change`
  post-run. The math is correct (softmax shift-invariance) but the reclassification was not declared
  before the run.
- 3 actual survivors (all pre-declared `harmless_change`): `attention__zero_max_init`,
  `attention__full_loop_no_op`, `cross_attn__wrong_reduction_axis` — all softmax shift-invariant.
- **No new real gaps found.**

#### Task 2 — Streaming vs non-streaming newline comparison

- **Script:** `scripts/newline_comparison.py`
- **1 successful streaming call** ($0.00145828) on `quantized_matmul__w8a16_pertensor__smoke_b5`.
- **Non-streaming comparison blocked** — daily OpenRouter credit limit exhausted after the streaming call; 5 × 130-second retries all returned HTTP 402 `openrouter_credits`.
- **Conclusion:** This run did not reproduce a SyntaxError in the extracted kernel. Non-streaming comparison was blocked by daily API credit limit. Root cause of missing newlines is **not established**: the streaming inspection is consistent with a model artifact (space ord=32 where `\n` ord=10 was expected) but a streaming-specific cause cannot be ruled out without the non-streaming comparison.

#### Task 3 — `scripts/reproduce_all.sh`

- CPU-only end-to-end reproducibility: pytest (35s), Docker mutation test (22s), report generation (<1s). Total ~57s.
- Fixed bash 3 compatibility (macOS ships bash 3.2; removed `declare -A`).
- Exit code 0 if all pass, 1 if any fail. Idempotent, safe to re-run.

#### Task 4 — Kaggle dry-run + KAGGLE_RUNBOOK rewrite

- CPU-verified 23 unit tests in `tests/test_gpu_session.py`: stage failures, budget enforcement, timeout watchdog, atomic zip, resume-after-disconnect.
- Resume verified end-to-end: simulated 2-entry progress log in zip → `restore_progress` → extracted correctly.
- Rewrote `docs/handoff/KAGGLE_RUNBOOK.md`: fixed section numbering (was 0, 0, 1, 2, …), added `reproduce_all.sh` to pre-flight, added atomic zip timing note, added resume flow explanation, added troubleshooting row for "report shows all stages absent".

#### Task 5 — Paper strengthening

- **Harness quality paragraph** added to §2.3 (Verification harness): documents 32 mutations with corrected count (29/29, denominator=29 after fixing norm blind spots; 3 softmax-invariant survivors). Note: the paragraph was initially written with overclaimed numbers; corrected in agent/overnight2.
- **Threats to Validity** (§6): added new paragraph "Construct validity: harness sensitivity limits" naming the two documented gaps (RMSNorm weight $\equiv\mathbf{1}$, LayerNorm bias $\equiv\mathbf{0}$) and the softmax shift-invariance class.
- **`[CITATION NEEDED]`** marker added to the explanation-uniqueness claim in the contributions list (item 2): `\todo{[CITATION NEEDED] Verified against abstracts and model cards only...}`.
- All GPU result cells remain `\todo{}` — no numbers invented.

#### Task 6 — README and Studio honesty pass

- Updated stale test count in README: "272 tests" → "283 tests".
- No other stale claims found; all GPU-blocked items are labelled as such.

### Files changed in this session

| File | Change |
|---|---|
| `scripts/mutation_test.py` | 32 mutations; `cross_attn__wrong_reduction_axis` reclassified; `cross_attn__l_i_accumulation_reset` added |
| `scripts/newline_comparison.py` | New — streaming vs non-streaming comparison script |
| `scripts/reproduce_all.sh` | New — CPU-only reproducibility script |
| `docs/handoff/KAGGLE_RUNBOOK.md` | Rewritten: fixed numbering, added pre-flight, resume explanation, troubleshooting |
| `docs/handoff/OVERNIGHT_LOG.md` | Task log entries for Tasks 1–4 |
| `docs/paper/paper.tex` | Mutation testing paragraph, harness sensitivity threats, `[CITATION NEEDED]` |
| `README.md` | Test count: 272 → 283 |

### Tests at end of session

**283 passed, 5 skipped** (same as start — no regressions, no new tests in this session).

### Blockers (unchanged — all GPU-dependent)

See `docs/handoff/BLOCKERS.md`. Nothing new unblocked; nothing new blocked.

### Next step

Run the Kaggle GPU session:
```bash
bash scripts/reproduce_all.sh   # pre-flight: confirms ~57s end-to-end
# Then follow docs/handoff/KAGGLE_RUNBOOK.md
```

---

## Overnight session 2 — 2026-11-16

**Branch:** `agent/overnight2` (cut from `agent/overnight` @ `80961da`)
**Session:** ~02:30–07:00 EST
**Tests at end of session:** 288 passed, 2 skipped (was 285/5; 3 transformers-guarded tests now run; 1 test skips on missing `datasets`).

### What was done

#### Task 1 — Norm blind-spot fix and overclaim removal

**Problem:** `mutation_test.py` had two blind spots (`norm_rmsnorm__missing_weight`, `layernorm__missing_bias`) that were excluded from the denominator under a `documented_gap` label, inflating the reported score from 27/29 to 27/27 = 100%. Additionally, `cross_attn__wrong_reduction_axis` was reclassified post-hoc (after observing it survived), which is a methodological issue even if the math is correct.

**Fix:** Both norm entries replaced with custom template entries using `torch.randn` initialization. With random init:
- `norm_rmsnorm__missing_weight` → CAUGHT (2036/2100 elements wrong; max error 7.714)
- `layernorm__missing_bias` → CAUGHT (2092/2100 elements wrong; max error 3.411)

**Corrected score:** 29/29 = 100% (denominator = 29, excludes only 3 pre-declared `harmless_change` softmax-invariants).

**Text corrections:** `docs/paper/paper.tex`, `docs/handoff/OVERNIGHT_LOG.md`, `docs/handoff/MORNING_SUMMARY.md`, `docs/handoff/DATA_REPORT.md` — all overclaims and "definitive" language removed.

**Tests added:** `test_missing_bias_caught_with_random_init`, `test_missing_bias_survives_with_zeros_init`.

**Commit:** `d55bcd7`

#### Task 3 — Local base-model evaluation (Qwen2.5-Coder-1.5B-Instruct, MPS, zero-shot)

**Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (BASE model, not fine-tuned)
**Device:** MPS float16 (Apple Silicon)
**Script:** `scripts/local_baseline_eval.py`
**Entries:** 60 (12 per category: 3 smoke + 9 regular); categories: quantized_matmul, attention, kv_cache, norm, rope
**Method:** generation → free model completely → Docker interpreter verification

**Pilot results (3 entries, 1 per category):**
- Tokens/second: ~10 tok/s (median), max_new_tokens=768
- Parse rate: 100% (kernel: section found in all 3 outputs)
- Correct: 0/3 — all 3 had SyntaxErrors in the generated Triton code (base model, zero-shot)

**Full run results (60 entries):**
See `evaluation/results/local_baseline_report.json` for complete numbers.

| Metric | Value |
|---|---|
| Total entries | 60 |
| Parse OK | 60 / 60 (100.0%) |
| Correct (pass@1) | 0 / 60 (0.0%) |
| Wilson 95% CI | [0.0%, 6.0%] |
| Failure types | syntax_error: 23, runtime_error: 37 |

**Note:** These are BASE model results. The fine-tuned model is expected to perform significantly better.
Tokens per second: median=13.2, mean=12.2 tok/s on MPS float16 (useful for planning Kaggle generation time estimates).

**Commits:** `614f777` (script), `bd645cd` (results + doc update)

#### Task 4 — Verified bibliography

**New finding:** Reflexion (Shinn et al.) confirmed at **NeurIPS 2023** via
`papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html`.
Updated `references.bib` from `@article` to `@inproceedings` (NeurIPS 2023). Author list corrected to
the 5-author proceedings version (Berman absent from proceedings).

Other checks confirmed: `baronio2025kevin`, `liu2026drkernel`, `woo2025tritonrl`, `su2021roformer`,
`chen2023selfdebug` — all present/unchanged at their arXiv URLs.

**Commit:** `0a5d106`

#### Task 2 — AST-based mutation campaign (≥200 mutations, 90-min time-box)

**Script:** `scripts/ast_mutation_campaign.py`
**Mutations generated:** 353 across all 15 smoke templates (28–31 per template)
**Classification rule (declared before running):**
- `harmless_change`: mutant produces IDENTICAL output on ALL 5 screening seeds {0,1,2,3,4}
- `expected_caught`: mutant produces DIFFERENT output on ANY screening seed
- Official score seed: 42 (not in screening set)

**Types generated:** flip_add_to_sub, flip_sub_to_add, flip_mul_to_div, flip_div_to_mul,
flip_cmp_lt/gt/lte/gte, flip_axis_{0,1}, swap_sqrt/rsqrt/exp/log, remove_binop_right,
replace_const_{127,0.0,1.0}

**Results (90-minute time-box):**
See `evaluation/results/ast_mutation_report.json` for complete numbers.

| Metric | Value |
|---|---|
| Mutations run (in time-box) | 353 / 353 (all completed) |
| harmless_change | 24 |
| expected_caught | 329 |
| Caught (by seed=42) | 329 / 329 |
| Mutation score | 329/329 = 100.0% |
| Wilson 95% CI | [98.8%, 100.0%] |

**Commits:** `614f777` (script), `28c8a47` (results + doc update)

### Test counts

| Event | Count |
|---|---|
| Start of session (`agent/overnight` state) | 285 passed, 5 skipped |
| After Task 1 (norm blind spots) | 285 passed, 5 skipped |
| After installing `transformers` (Task 3) | 288 passed, 2 skipped |
| End of session | 288 passed, 2 skipped |

The skip count change: `transformers` now installed → 3 previously-skipped tests run and pass. 1 test skips on missing `datasets` (pre-existing gap; guarded with `pytest.importorskip`).

### Commits this session

```
28c8a47 Task 2: AST mutation campaign complete; 329/329 = 100% (Wilson CI [98.8%, 100%])
bd645cd Task 3: local base-model eval complete; 0/60 pass@1 (100% parse, 23 syntax_err, 37 runtime_err)
3bd3221 Fix test skipping: guard finetune dry-run test with datasets importskip
614f777 Add Task 3+4+2 scripts; log Task 4 (bibliography) in OVERNIGHT_LOG
0a5d106 Task 4: verify bibliography; promote Reflexion to NeurIPS 2023 proceedings
d55bcd7 Task 1: fix norm blind spots, correct overclaims, add LayerNorm bias tests
```

### Blockers (unchanged — all GPU-dependent)

See `docs/handoff/BLOCKERS.md`. Nothing new unblocked.

### Next step

Run the Kaggle GPU session:
```bash
bash scripts/reproduce_all.sh   # pre-flight: confirms ~57s end-to-end
# Then follow docs/handoff/KAGGLE_RUNBOOK.md
```
