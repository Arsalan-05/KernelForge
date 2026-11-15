# Results report (`python -m evaluation.report`)

Turns the artifacts of a GPU session into tables, figures and write-up fragments.
Every number is computed from the artifacts it reads. A stage with no artifact is
reported as "not run", never estimated.

## Run it on the Kaggle zip

```bash
source .venv/bin/activate
pip install -r requirements.txt                    # repo requirements
pip install -r evaluation/report/requirements.txt  # matplotlib, for the figures

python -m evaluation.report ~/Downloads/kernelforge_outputs.zip --out docs/results/
```

The source can be the zip or the unzipped `kernelforge_outputs/` folder. The zip
is read in memory and never extracted. Only `.json` files are read, so the
adapter weights are ignored. Use `--no-figures` to skip plotting.

Without matplotlib the report still runs. Every table is written, and
`results.md` lists each figure as "not drawn (matplotlib is not installed …)".

The console output lists every stage as present or absent, every figure or
LaTeX fragment that was skipped and why, and every report that was excluded
(dry runs, parse-only runs).

## What it reads

| Artifact (path inside the zip) | Written by | Used for |
|---|---|---|
| `session.json` | `scripts/gpu_session.py` | stage statuses, GPU name, versions |
| `data/raw/generation_report.json` | `data/build_dataset.py --device cuda` | dataset table |
| `data/raw/interpret_report.json` | `data/build_dataset.py --interpret` | interpreter-only dataset check (Markdown only) |
| `evaluation/results/baseline_*.json` | `evaluation/baseline_eval.py` | base model results |
| `evaluation/results/finetuned_*.json` | `evaluation/finetuned_eval.py` | fine-tuned results |
| `training/run_*/split_info.json` | `training/finetune.py` | split and config in the training table |
| `training/run_*/checkpoint-*/trainer_state.json` | Hugging Face `Trainer` | loss, epochs, steps, wall-clock (newest checkpoint) |
| `training/run_*/training_metrics.json` | nothing yet (see below) | `peak_memory_gb`, `wall_clock_s` |
| any JSON with `rubric` and `items` keys | `evaluation/explanation_review.py --output` | explanation rubric scores |

Eval reports with `"dry_run": true` (scored the dataset's own kernels) or
`"skip_verification": true` (parse only) are excluded, with the reason shown in
the Sources section. When several reports describe the same model and search
mode, the newest by `generated_at` is used and the others are listed as
superseded.

**Training artifacts.** `scripts/gpu_session.py` currently copies only
`adapter/`, `split_info.json` and `held_out_test.jsonl` out of the run folder.
So the training table holds the split and config but no loss or wall-clock
until `checkpoint-*/trainer_state.json` is collected too. Nothing records peak
GPU memory yet. To get it into the table, write
`training/run_*/training_metrics.json` containing
`{"peak_memory_gb": <torch.cuda.max_memory_allocated() / 1e9>}`.

## Explanation reviews

Scored reviews are not part of the GPU session zip. You make them afterwards
with `evaluation/explanation_review.py --output …` and fill in `rubric_score`
(0/1/2) by hand. To include them, put the review JSON inside the source you
pass: either copy it into the unzipped `kernelforge_outputs/` folder, or point
the tool at a folder that holds both the session artifacts and the reviews.

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
| `results.md` | Full report: stage status, the Phase 5 category table, the search table, dataset tables, training, explanation review, figures, auto-generated limitations, and the source file of every number. |
| `readme_results_snippet.md` | A short "Results" section to paste into the top-level README. |
| `summary.json` | Every computed number (k, n, %, Wilson interval, mean, geomean, median, fast_1, infinite speedups excluded) so you can check them by hand. |
| `figures/correctness_by_category.png` | % correct per category, base vs fine-tuned pass@1, with Wilson 95% intervals and n. |
| `figures/speedup_distribution.png` | One point per correct kernel, log scale, with a 1× reference line. |
| `figures/search_gain.png` | Overall % correct for pass@1, best-of-4 and best-of-4 + repair. |
| `table_dataset.tex` | GPU dataset build: pass / incorrect / OOM / timeout / other, per category and template. |
| `table_categories.tex` | The Phase 5 per-category table. |
| `table_search.tex` | The search comparison. |
| `table_training.tex` | Training run: split, config, loss, epochs, wall-clock and memory, for whatever exists. |
| `table_rubric.tex` | Explanation rubric scores. |
| `tables.tex` | All `table_*.tex` files that exist, concatenated. |

**LaTeX fragments are written only when their data exists.** The paper
`\input`s them by name and shows a TODO placeholder for any missing file. So a
missing stage means a missing file, never an empty table or placeholder
numbers. On re-runs, a fragment whose data is no longer in the source is
deleted. In particular:

- `table_search.tex` needs at least one best-of-4 run; pass@1 alone is not a search result.
- `table_dataset.tex` needs the GPU dataset report; an interpreter check doesn't count.
- `table_rubric.tex` needs at least one scored item.

Each fragment states the packages it needs: `\usepackage{booktabs}` and
`\usepackage{graphicx}`. Tables wider than four columns are wrapped in
`\resizebox{\columnwidth}{!}{…}`.

## How to read the numbers

- **% correct**: k/n with a Wilson 95% interval. Wilson stays sensible at 0/5 or
  5/5, where the textbook interval collapses to a single point. Cells with n < 20
  get a † because they are too few to compare on.
- **Mean speedup**: the arithmetic mean of reference time / kernel time over
  correct kernels. It matches `evaluation/common.py`.
- **Geomean speedup**: the geometric mean over the same kernels. This is the right
  average for ratios: a 2× kernel and a 0.5× kernel average to 1×.
- **fast_1**: attempts that are correct *and* faster than the reference, divided by
  *all* attempts (KernelBench's metric). It is the only speedup view that also
  penalises failures.
- **Infinite speedups** (kernel timed at 0 s) are excluded from all three and
  counted in `summary.json` as `infinite_speedups_excluded`.
- **Interpreter runs** report correctness only; their speedup cells read
  "n/a (interpreter)".

## Interface for the eval scripts

Search results (best-of-N sampling and repair rounds) use the same JSON as
`evaluation/common.py::save_eval_report`. Whoever wires best-of-N + repair into
`evaluation/*.py` must add this top-level key, for example through
`save_eval_report(..., extra={...})`:

```json
"search": {"num_candidates": 4, "repair_rounds": 1}
```

- `num_candidates`: N in best-of-N (1 = a single sample).
- `repair_rounds`: harness-feedback repair rounds after the first round (0 = none).
- Leave out the key, or use `{"num_candidates": 1, "repair_rounds": 0}`, for plain pass@1.

The tables use `(1, 0)` for pass@1, `(4, 0)` for best-of-4 and `(4, 1)` for
best-of-4 + repair. Other settings are parsed and listed, but they get no
column of their own.

Everything else is what `save_eval_report` already writes:

- `run_name`: starts with `baseline` or `finetuned`. A non-null `adapter_path` also marks the run as fine-tuned.
- `generated_at`: the newest report wins when several describe the same model and search mode.
- `split` and `held_out_templates`: shown in the limitations.
- `dry_run` and `skip_verification`: if true, the report is excluded.
- `results`: one item per test entry, with `id`, `category`, and `verification` holding `status`, `correct`, `speedup` and `interpreted`.

For a search run, `verification` must be that of the **winning candidate**: the
fastest correct one, or the best failure if none passed. That way one entry
still counts as one attempt. If `results` is missing, the report falls back to
`per_category`; in that case the geomean, fast_1 and the speedup figure are
unavailable.
