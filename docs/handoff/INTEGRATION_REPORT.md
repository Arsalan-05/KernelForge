# Integration Report — agent/merged

_Date: 2026-11-15/16_
_Agent: Claude Sonnet (overnight integration)_

---

## Pre-merge Baseline

- Branch: `agent/merged`
- Tests: 83 passed, 1 skipped

---

## Task 1 — Handoff Notes and Branch Survey

### Branches

| Branch | Worktree | Commits ahead of main | Files changed |
|--------|----------|-----------------------|---------------|
| agent/a | ../KernelForge-a | 9 | 10 |
| agent/b | ../KernelForge-b | 10 | 12 |
| agent/c | ../KernelForge-c | 7 | 10 |
| agent/d | ../KernelForge-d | 6 | 9 |

### File overlaps across branches

**None.** All four branches touch completely disjoint file sets. Expected clean four-way merge.

### Files per branch

**A:** `DOCUMENTATION.md`, `MASTER_PLAN.md`, `data/build_dataset.py`, `data/raw/interpret_report_2026-11-15.json`, `notebooks/kaggle_gpu_session.ipynb`, `scripts/check_token_lengths.py`, `scripts/gpu_session.py`, `tests/test_build_dataset.py`, `tests/test_check_token_lengths.py`, `tests/test_gpu_session.py`

**B:** `docs/handoff/B_integration_notes.md`, `evaluation/baseline_eval.py`, `evaluation/common.py`, `evaluation/explanation_review.py`, `evaluation/finetuned_eval.py`, `kernelforge/dataset.py`, `kernelforge/model.py`, `kernelforge/parsing.py`, `tests/test_eval_runs.py`, `tests/test_parsing_edge_cases.py`, `tests/test_training_pipeline.py`, `training/finetune.py`

**C:** `evaluation/report/README.md`, `evaluation/report/__init__.py`, `evaluation/report/__main__.py`, `evaluation/report/artifacts.py`, `evaluation/report/figures.py`, `evaluation/report/latex.py`, `evaluation/report/render.py`, `evaluation/report/requirements.txt`, `evaluation/report/stats.py`, `tests/test_report.py`

**D:** `docs/paper/CHECKLIST.md`, `docs/paper/DOC_CORRECTIONS.md`, `docs/paper/README.md`, `docs/paper/RESULTS_INTERFACE.md`, `docs/paper/SOURCES.md`, `docs/paper/UNVERIFIED.md`, `docs/paper/paper.tex`, `docs/paper/references.bib`, `docs/project_writeup.md`

### Handoff notes read

- `docs/handoff/B_integration_notes.md` (agent/b) — full interface spec between B and C, gpu_session flags, zip manifest, report key schema.
- `docs/paper/RESULTS_INTERFACE.md` (agent/d) — 5 LaTeX fragment names and exact column/row specs.
- `docs/paper/DOC_CORRECTIONS.md` (agent/d) — 7 corrections to apply post-merge to README, DOCUMENTATION.md, MASTER_PLAN.md, project_writeup.md.

No handoff notes from agent/c or agent/a beyond their commit messages.

---

## Task 2 — Merges

| Order | Branch | Strategy | Conflicts | Tests after |
|-------|--------|----------|-----------|-------------|
| 1 | agent/b | ort (no-ff) | None | 2 failed (transformers not installed) |
| 2 | agent/c | ort (no-ff) | None | 13 failed (save_eval_report missing) |
| 3 | agent/a | ort (no-ff) | None | 230 passed, 5 skipped |
| 4 | agent/d | ort (no-ff) | None | 230 passed, 5 skipped |

**Fixes applied:**
- `tests/test_eval_runs.py`: Added `pytest.importorskip("transformers")` guards to 2 tests that import directly from `transformers` at function level. These tests are correctly skipped when `transformers` is not installed.
- `evaluation/common.py`: Added `save_eval_report()` function that test_report.py expected. The function was implicitly referenced by C's report tests but had not been extracted from `run_evaluation()`.

---

## Task 3 — interpret_report verification

`data/raw/interpret_report_2026-11-15.json` is present (44 KB). Top-level fields confirm:
- `total: 300`, `passed: 285`, `partial: false`
- `mode: "triton-interpreter (correctness only, CPU; no timings)"`
- 15 OOM shapes (exit code -9) across attention and norm templates

---

## Task 4 — B↔C interface check

**Result: exact match, no mismatches.**

- `generation_settings` dict keys: match (`do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens`)
- `few_shot_source` field: B writes at top level; C reads via `EvalRun.few_shot_source` and `_settings()`
- `prompt_mode` field: B writes `"zero_shot"` or `"few_shot"`; C reads via `_prompt_mode(meta)`
- Output filename patterns: `baseline_few_shot_<source>.jsonl` / `baseline_few_shot_<source>_<timestamp>.json` match C's `INCREMENTAL_LOG` regex

---

## Task 5 — C↔D interface check

**Result: exact match, no mismatches.**

C's `FRAGMENT_NAMES` in `latex.py`: `("table_dataset", "table_categories", "table_search", "table_training", "table_rubric")`

D's `RESULTS_INTERFACE.md` expects exactly those 5 filenames under `docs/results/`.

`paper.tex` loads them with `\ifhave<name> \resultrows{<name>}` — using `../results/table_<name>.tex` relative to `docs/paper/`. Correct.

---

## Task 6 — repetition_penalty enforcement

**Fully enforced in all generation paths.**

- `evaluation/common.py:34`: `EVAL_REPETITION_PENALTY = 1.0`
- `evaluation/baseline_eval.py:52`: pins it before creating generator
- `evaluation/finetuned_eval.py:70`: same
- `evaluation/common.py:488-489`: defensive `raise ValueError` guard if missed
- `evaluation/common.py:523`: written to report JSON
- `kernelforge/model.py`: default `None` (keeps model bundled value); eval scripts must override explicitly (and do)

---

## Task 7 — OOM/GPU wording audit

Fixed in commit `c866fe8`:
- `MASTER_PLAN.md`: replaced unverified 261/300 pre-flight with actual result (285/300, 15 OOM in 8 GB VM, no GPU result) from the committed report
- `README.md`: Updated "15/15 pass the interpreter" to "285/300 shapes pass, 15 OOM in 8 GB VM"
- `docs/paper/paper.tex`: Updated dataset section to cite 285/300 from `data/raw/interpret_report_2026-11-15.json`

Applied all 7 corrections from `docs/paper/DOC_CORRECTIONS.md`:
1. TritonBench speedup claim corrected to actual numbers
2. Kevin-32B identified as CUDA model in README, project_writeup
3. KernelLLM identified as SFT-only
4. AutoTriton/TritonRL corrected to SFT-then-RL
5. Flash-decoding claim corrected
6. 261/300 pre-flight replaced with 285/300 real result
7. "None output a natural-language rationale" softened to reflect reasoning-trace models

---

## Task 8 — OOM shapes in dataset docs

The 15 OOM shape IDs are documented in `MASTER_PLAN.md`. They are kept in the dataset. The note states: "Whether these 15 shapes fit on a 16 GB T4 is untested." `docs/paper/paper.tex` also carries a comment listing the OOM shapes with their IDs.

**OOM shapes:**
- `attention__causal_prefill__b16_s2048`
- `attention__gqa_causal_prefill__b16_s2048`
- `norm__rmsnorm__r65536_h{4096,8192,11008,12288}` (4 shapes)
- `norm__layernorm__r65536_h{4096,8192,11008,12288}` (4 shapes)
- `norm__rmsnorm_residual__r16384_h12288` (1 shape)
- `norm__rmsnorm_residual__r65536_h{4096,8192,11008,12288}` (4 shapes)

---

## Task 9 — Final test suite

**231 passed, 5 skipped, 0 failed**

Skipped (5): 2 tests require `transformers` (not installed), 1 requires Qwen tokenizer in HF cache, 2 require matplotlib (skipped in some environments).

---

## Task 10 — peak_memory_gb

### 10b — training/finetune.py

After `trainer.train()`, `peak_memory_gb = torch.cuda.max_memory_allocated() / 1024^3` is recorded. Guarded by `if cuda and torch.cuda.is_available()` — returns `None` on CPU. Added to the return dict and therefore written to `split_info.json`.

**Test:** `tests/test_training_pipeline.py::test_peak_memory_gb_is_none_without_gpu` — stubs `train()` and `load_eval_entries`, verifies `peak_memory_gb: null` in split_info.json on a CPU-only run.

### 10c — paper table captions

All 5 table captions updated. Each now states:
- What the numbers measure
- Settings: greedy decoding, `repetition_penalty=1.0`, few-shot source (other-category sibling)
- "All cells show \todo{} until the GPU run completes."
- No made-up numbers.

---

## Task 11 — Pipeline smoke test

Ran report generator with a synthetic eval report:
```bash
python -m evaluation.report <tmp_dir> --out <tmp_out> --no-figures
```
Result: `table_categories.tex` written (data present), other 4 fragments correctly absent (no data). `tables.tex` (combined) also written.

Paper references `docs/results/table_<name>.tex` via `\resultsdir = ../results` from `docs/paper/`. Path alignment confirmed.

**LaTeX:** No LaTeX install found (`pdflatex` / `xelatex` not in PATH). Paper cannot be compiled locally. This is expected — paper is compiled in the GPU session or CI.

---

## Task 12 — Cleanup

All conditions met:
- Full test suite passes (231 passed, 5 skipped)
- `git branch --merged agent/merged` lists agent/a, agent/b, agent/c, agent/d ✓
- BLOCKERS.md has no open items needing worktrees ✓

Actions taken:
- Backed up `../KernelForge-a/data/raw` to `./backup_a_raw` (not committed; excluded via `.git/info/exclude`)
- Removed all 4 worktrees with `git worktree remove` (no --force needed)
- Ran `git worktree prune`
- Branches agent/a, agent/b, agent/c, agent/d preserved

---

## Task 13 — Safe project work

See commits after this report for README improvements, KAGGLE_RUNBOOK.md, and additional test coverage.

---

## What was NOT done

- GPU run (no GPU available)
- LaTeX compilation (no TeX install)
- Peak memory test on a real GPU (needs GPU)
- Paper text section fill-ins (marked \todo{} as intended, pending GPU results)
