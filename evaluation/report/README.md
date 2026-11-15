# Results report (`python -m evaluation.report`)

Turns the artifacts of a GPU session into tables, figures and write-up fragments.
Every number is computed from the artifacts it reads. A stage with no artifact is
reported as "not run", never estimated. Every key is optional: a missing or
malformed key leaves a cell empty or unknown, never filled in.

## Run it on the Kaggle zip

```bash
source .venv/bin/activate
pip install -r requirements.txt                    # repo requirements
pip install -r evaluation/report/requirements.txt  # matplotlib, for the figures

python -m evaluation.report ~/Downloads/kernelforge_outputs.zip --out docs/results/
```

The source can be the zip or the unzipped `kernelforge_outputs/` folder. The zip
is read in memory and never extracted. Only `.json` files are read, plus `.jsonl`
eval logs and dataset progress logs. The adapter weights and the dataset itself
are ignored. Use `--no-figures` to skip plotting.

Without matplotlib the report still runs. Every table is written, and
`results.md` lists each figure as "not drawn (matplotlib is not installed …)".

The console output lists every artifact as present or absent, every figure or
LaTeX fragment that was skipped and why, and every report that was excluded.

## What it reads

| Artifact (path inside the zip) | Written by | Used for |
|---|---|---|
| `stage_status.json` | `scripts/gpu_session.py` | session stage table: status, seconds, error, prompts scored |
| `session.json` | `scripts/gpu_session.py` | GPU and versions; stage records if `stage_status.json` is missing; `zip_files` |
| `data/raw/generation_report.json` | `data/build_dataset.py --device cuda` | dataset table; `partial` and `run_stats` |
| `data/raw/interpret_report.json` | `data/build_dataset.py --interpret` | interpreter-only dataset check (Markdown only) |
| `data/raw/progress_<mode>.jsonl` | `data/build_dataset.py` | "build did not finish" warning when there is no report for that mode |
| `evaluation/results/<run>_*.json` | `evaluation/baseline_eval.py`, `finetuned_eval.py` | all model results |
| `evaluation/results/<run>_*.jsonl` | an eval run that was killed | the run's results, marked PARTIAL |
| `training/**/run_*/split_info.json` | `training/finetune.py` | split and config in the training table |
| `training/**/run_*/train_log.json`, `trainer_state.json`, `checkpoint-*/trainer_state.json` | `training/finetune.py`, Hugging Face `Trainer` | loss, epochs, steps, runtime |
| `training/**/run_*/training_metrics.json` | `training/finetune.py` | wall-clock, steps, peak GPU memory |
| any JSON with `rubric` and `items` keys | `evaluation/explanation_review.py --output` | explanation rubric scores |

**Skipped or set aside:**

- Excluded, with the reason shown under Sources:
  - files named `dryrun_*`
  - reports with `"dry_run": true` or `"skip_verification": true`
- Superseded and listed, but not used:
  - an older report of the same model, search mode and prompt mode, when a newer one exists (by `generated_at`)
  - an incremental `.jsonl` log that a final report covers. That means the same file stem, the same run key, or a final report of the same model and search mode that scored all of the log's prompts.

**`training_metrics.json` keys accepted** (the first one present wins):

- wall-clock: `wall_clock_s`, `wall_clock_seconds`, `train_runtime_s`, `train_runtime`, `seconds`
- peak memory: `peak_memory_gb`, `peak_gpu_memory_gb`, `peak_memory_mb`, `peak_gpu_memory_mb`, `peak_memory_bytes`, `peak_gpu_memory_bytes`, `max_memory_allocated` (bytes, as `torch.cuda.max_memory_allocated()` returns). Reported in decimal GB.
- steps: `steps`, `global_step`, `optimizer_steps`

If the file has no wall-clock, the trainer's `train_runtime` is used.

## Explanation reviews

Scored reviews are not part of the GPU session zip. You make them afterwards
with `evaluation/explanation_review.py --output …` and fill in `rubric_score`
(0/1/2) by hand. To include them, put the review JSON inside the source you
pass, e.g. copy it into the unzipped outputs folder:

```bash
unzip kernelforge_outputs.zip -d kernelforge_outputs
cp evaluation/reviews/explanation_review_finetuned.json kernelforge_outputs/
python -m evaluation.report kernelforge_outputs --out docs/results/
```

A review is attributed to the base or fine-tuned model by the file name in its
`source_report` field (`baseline_…` or `finetuned_…`). Only items with a
`rubric_score` of 0, 1 or 2 are counted.

## What it writes (`--out`)

| File | Contents |
|---|---|
| `results.md` | Full report. First, a WARNING block if anything is partial or incomplete. Then: the session stage table, eval coverage, artifacts found, the Phase 5 category table, search on the same prompts, outcomes, token-limit truncations, dataset tables, training, explanation review, figures, limitations and sources. |
| `readme_results_snippet.md` | A short "Results" section for the top-level README, with the warning block when present. |
| `summary.json` | Every computed number: k, n, %, Wilson interval, mean, geomean, median, fast_1, coverage, token limits, the same-prompt search comparisons, warnings and stage records. |
| `figures/correctness_by_category.png` | % correct per category for each pass@1 run (base zero-shot and few-shot as separate series), with Wilson 95% intervals and n. |
| `figures/speedup_distribution.png` | One point per correct kernel, log scale, with a 1× reference line. |
| `figures/search_gain.png` | Per search group: pass@1 on the same prompts, best-of-N, and best-of-N + repair. |
| `table_dataset.tex` | GPU dataset build: pass / incorrect / OOM / timeout / other, per category and template. |
| `table_categories.tex` | The Phase 5 per-category table. |
| `table_search.tex` | The same-prompt search comparison. |
| `table_training.tex` | Training run: split, config, loss, epochs, steps, wall-clock and peak memory, for whatever exists. |
| `table_rubric.tex` | Explanation rubric scores. |
| `tables.tex` | All `table_*.tex` files that exist, concatenated. |

**LaTeX fragments are written only when their data exists.** The paper
`\input`s them by name and shows a TODO placeholder for any missing file. So a
missing stage means a missing file, never an empty table or placeholder
numbers. On re-runs, a fragment whose data is no longer in the source is
deleted. In particular:

- `table_search.tex` needs at least one search run; pass@1 alone is not a search result.
- `table_dataset.tex` needs the GPU dataset report; an interpreter check doesn't count.
- `table_rubric.tex` needs at least one scored item.

Each fragment states the packages it needs: `\usepackage{booktabs}` and
`\usepackage{graphicx}`. Tables wider than four columns are wrapped in
`\resizebox{\columnwidth}{!}{…}`.

## How to read the numbers

- **% correct**: k/n with a Wilson 95% interval. Cells with n < 20 get a † because
  they are too few to compare on.
- **Mean speedup**: the arithmetic mean of reference time / kernel time over correct
  kernels. It matches `evaluation/common.py`.
- **Geomean speedup**: the geometric mean over the same kernels.
- **fast_1**: attempts that are correct *and* faster than the reference, divided by
  *all* attempts.
- **Infinite speedups** (kernel timed at 0 s) are excluded from all three and counted
  in `summary.json`.
- **Prompt modes**: zero-shot and few-shot base runs are separate column groups,
  table rows and figure series. They are never pooled or averaged, and an
  ambiguous lookup returns nothing rather than picking one.
- **Search on the same prompts**: a search run scores only `search_limit` prompts.
  The search table compares it only with pass@1 *recomputed on those exact
  prompts*, from the pass@1 run's per-entry rows. If pass@1 missed some of them,
  the search rows are repeated on the shared prompts only, and a warning is raised.
  The "FT best-of-4 + repair" column of the category table is marked ‡: it is on
  that subset and is not comparable with the "% correct" column next to it.
- **Coverage**: each run shows prompts scored against the held-out total.
  - The held-out total comes from the session's eval stage record whose
    `eval_report` names the run's file. For fine-tuned runs it falls back to
    `split_info.json`'s `test` count.
  - A run that didn't score the full set, a report marked `partial`, or a run
    that left only an incremental log triggers a WARNING at the top of
    `results.md` and in the README snippet.
- **Token-limit truncations**: outputs with `hit_token_limit: true`, and how many of
  the failures they account for (with a Wilson interval). If no row records the
  flag, the table says nothing rather than reporting 0.

## Interface for the eval scripts

Eval reports are the JSON written by `evaluation/common.py::save_eval_report`.
Every key is optional; these are the ones the report uses.

**Search runs.** Whoever wires best-of-N + repair into `evaluation/*.py` writes
the search as its own report, with this top-level key (e.g. via
`save_eval_report(..., extra={...})`):

```json
"search": {"num_candidates": 4, "repair_rounds": 1, "search_limit": 20, "prompts_scored": 20}
```

- `num_candidates`: N in best-of-N (1 = a single sample).
- `repair_rounds`: harness-feedback repair rounds after the first round (0 = none).
- `search_limit`: how many held-out prompts the search was asked to score.
- `prompts_scored`: how many it actually scored.

`results` holds one row per scored prompt, with the **winning candidate's**
verification: the fastest correct one, or the best failure if none passed. That
way one prompt is still one attempt. The prompt `id`s must match the pass@1
run's ids, because that is how the same-prompt pass@1 is recomputed.

Leave out the key, or use `{"num_candidates": 1, "repair_rounds": 0}`, for plain
pass@1. The tables use `(1, 0)` for pass@1, `(4, 0)` for best-of-4 and `(4, 1)`
for best-of-4 + repair. Other settings appear in the search table and Sources.

**Other top-level keys:**

- `run_name`: `baseline…` or `finetuned…`. A non-null `adapter_path` also marks a run as fine-tuned.
- `generated_at`
- `prompt_mode`: `"zero_shot"` or `"few_shot"`. Runs are keyed by model, search mode *and* prompt mode.
- `partial` (bool)
- `max_new_tokens`, `data_source`, `backend`, `baseline_report`: shown under Sources.
- `outcomes`: counts; used only if rows lack `outcome`.
- `hit_token_limit`: a top-level count, used only if rows lack the flag.
- `split`, `held_out_templates`
- `dry_run`, `skip_verification`: if true, the report is excluded.
- `per_category`: fallback when `results` is missing.

**Per-row keys:** `id`, `category`, `outcome`, `hit_token_limit` (bool),
`verification` (`status`, `correct`, `speedup`, `interpreted`),
`generated_explanation`, `generated_text`.

**Incremental logs (assumed format, to confirm).** A run that is killed before
writing its report can leave `evaluation/results/<run>_*.jsonl`:

- Each line with an `id` is a result row in the format above.
- Any line without an `id` is run metadata (`run_name`, `prompt_mode`, `search`, …), merged in order.
- The file stem is matched against final reports, so use the same stem as the final `.json`.
- A truncated last line is skipped and reported.

Such a run is always reported as PARTIAL.
