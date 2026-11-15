# Results report (`python -m evaluation.report`)

Turns the artifacts of a GPU session into tables, figures and the paper's LaTeX
fragments. Every number is computed from the artifacts it reads. A stage with no
artifact is reported as "not run", never estimated. Every key is optional: a
missing or malformed key leaves a cell unknown, never filled in.

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

## What it reads

The formats below are what `scripts/gpu_session.py`, `data/build_dataset.py`
(agent/a) and `evaluation/common.py`, `training/finetune.py`,
`evaluation/explanation_review.py` (agent/b) write.

| Artifact (path inside the zip) | Used for |
|---|---|
| `stage_status.json` | session stage table: status, seconds, error, prompts scored, held-out total |
| `session.json` | GPU and versions; stage records if `stage_status.json` is missing; `zip_files` |
| `data/raw/generation_report.json` | dataset tables and Table 1; `partial`, `run_stats` |
| `data/raw/interpret_report.json` | interpreter-only dataset check (Markdown only, never Table 1) |
| `data/raw/progress_<mode>.jsonl` | "build did not finish" warning when there is no report for that mode |
| `evaluation/results/<run>_<timestamp>.json` | eval report: pass@1 rows in `results`, the search in an embedded `search` block |
| `evaluation/results/baseline.jsonl`, `baseline_few_shot.jsonl`, `finetuned.jsonl` | incremental logs, used when a run was killed before writing its report |
| `training/run_*/split_info.json` | split, config, `partial`, `stopped_at_step`, `steps_planned`, `dry_run` |
| `training/run_*/train_log.json` | Trainer `log_history`: losses, `train_runtime` |
| `training/run_*/trainer_state.json`, `checkpoint-*/trainer_state.json` | `global_step`, `epoch`, `log_history` |
| `<review>.json` (has `rubric` and `items`) + `<review>.key.json` | explanation rubric scores |

**Eval reports.**

- The report's own `results` are greedy pass@1 rows.
- A `search` block (`num_candidates`, `repair_rounds`, `search_limit`,
  `prompts_planned`, `prompts_scored`, `seed`, `round0_temperature`,
  `repair_temperature`, `per_category`, `outcomes`, `results`) becomes a second
  run: best-of-N + repair on the search prompts.
- If every search row has `selected_round`, a third run is derived from it:
  best-of-N without repair, counted as correct only when the selected candidate
  came from round 0. This works because the search stops at round 0 whenever any
  candidate passes (`kernelforge/search.py`). The derived run is labelled
  "(round 0 of the repair run)".
- Fallback for the older format: a report whose `search` dict has no `results`,
  or a `run_name` like `baseline_best4_repair1`, is read as a search run on its own.

**Incremental logs** are read from the three names above only:

- A `{"type": "meta", …}` line holds the run settings. Then come
  `{"type": "row", "phase": "pass@1" | "search", "id", …}` rows; any line without
  an `id` counts as metadata.
- The two phases become separate runs, both marked PARTIAL.
- A final report wins over its log, unless the log holds every prompt the report
  has and more (a resumed run that was killed again).
- Logs that `ResultsLog` moved aside (`baseline.<timestamp>.jsonl`) and `dryrun_*`
  files are listed as excluded.

**Training.**

- Final loss is the last logged step `loss`. Best validation loss is the minimum
  `eval_loss`.
- Wall-clock is the Trainer's `train_runtime`. If that's missing, it is the
  session's `finetune` stage seconds, which also include model loading; the
  report says which source was used.
- **No script records peak GPU memory.** It is always reported as `n/r`. The
  session's `gpu_memory_gb` is the card's total memory, not a peak, so it is not
  used.
- Runs whose `split_info.json` says `dry_run: true` are excluded.

**Reviews.**

- Each item's model comes from its `source` (non-blind review) or from the
  `<review>.key.json` answer key (blind review).
- A blind review without its key is reported as "Unknown model".
- Base or fine-tuned is decided by `run_name` or `adapter_path`; the prompt mode
  comes from `source.prompt_mode`.
- Older reviews with a top-level `source_report` still work.
- Reviews are not in the session zip. Copy the review file and its key into the
  unzipped outputs folder before running the tool.

**Skipped:** `dryrun_*` files, and reports with `dry_run` or `skip_verification` set.

## What it writes (`--out`)

| File | Contents |
|---|---|
| `results.md` | Full report. First, a WARNING block if anything is partial, incomplete, or uses mismatched `repetition_penalty`. Then: session stages, eval coverage, run settings, artifacts found, the category table, search on the same prompts, outcomes, token-limit truncations, dataset, training, explanation review, figures, which LaTeX fragments were written or refused, limitations and sources. |
| `readme_results_snippet.md` | A short "Results" section for the top-level README. |
| `summary.json` | Every computed number, plus settings, coverage, the same-prompt search comparisons, training runs and warnings. |
| `figures/*.png` | Correctness by category, speedup distribution, and search gain on the same prompts. |
| `table_dataset.tex` … `table_rubric.tex` | The paper's five fragments (below). |
| `tables.tex` | All fragments that exist, concatenated, for reading only. The paper `\input`s the separate files. |

**Run settings** shows, per run:

- prompt mode and few-shot source;
- decoding, do_sample, temperature, top_k, top_p, repetition_penalty and max_new_tokens;
- for search runs: seed and the sampling temperatures.

Values are read from the report's top level, or from a nested
`generation_settings`, `effective_generation_settings`, `generation`, `sampling`
or `decoding` dict. Anything missing shows "not recorded".

Few-shot source: the report's `few_shot_source` if present. Otherwise it is
derived from each row's `example_id`: whether the example's category matches the
prompt's, labelled "(from example_id)". If the runs being compared record
different `repetition_penalty` values, a WARNING is raised.

## LaTeX fragments (docs/paper/RESULTS_INTERFACE.md)

The fragments follow the interface agreed with the paper exactly:

- Body rows only. The first line is the comment `% generated <UTC time> from <artifacts>`; further comment lines carry notes such as PARTIAL warnings.
- Exact cell counts, row order and first-cell labels.
- Percentages have one decimal and no `%` sign; speedups look like `1.37$\times$`; counts are integers.
- `--` means undefined; `n/r` means not run.

| Fragment | Cells | Rows | Source |
|---|---|---|---|
| `table_dataset.tex` | 6 | 5 categories, `\midrule`, `Total` | GPU `generation_report.json` (timeout + error + crash → "Other error") |
| `table_categories.tex` | 7 | 5 categories | zero-shot base and FT pass@1, FT best-of-4 + repair, n |
| `table_search.tex` | 4 | `Base`, `Fine-tuned` | pass@1, best-of-4, best-of-4 + repair, **all on the same prompts** |
| `table_training.tex` | 5 | one per training run, newest last (`QLoRA run`, or `QLoRA run 1` … `QLoRA run k (reported)`) | train_log / trainer_state; memory always `n/r` |
| `table_rubric.tex` | 5 | `Base`, `Fine-tuned` | review scores; "copied" is `--` for Base, `n/r` for FT |

A fragment is written only when its data exists, and stale ones are deleted on
re-runs. A fragment is **refused** (not written; the reason appears in
`results.md` and the console) when:

- Table 1: GPU-verified + incorrect + OOM + other error ≠ generated for any category;
- Table 2: base and fine-tuned have different prompt counts in any category.

The richer metrics stay in `results.md` and `summary.json`: intervals, geomean,
fast_1, few-shot base runs and outcomes.

## How to read the numbers

- **% correct**: k/n with a Wilson 95% interval in `results.md`. † marks n < 20.
- **Mean speedup** (matches `evaluation/common.py`) and **geomean speedup** are over
  correct kernels only. **fast_1** counts correct *and* faster than the reference,
  over all attempts. Infinite speedups are excluded and counted separately.
- **Prompt modes**: zero-shot and few-shot base runs are never pooled. The paper's
  Base is the zero-shot run (or a run that records no prompt mode).
- **Search on the same prompts**: search runs score only `search_limit` prompts.
  They are compared only with pass@1 recomputed on those prompts, and on the shared
  prompts if pass@1 missed some.
- **Coverage**: prompts scored, prompts planned (`prompts_planned`) and the
  held-out total (from the session's stage record). A run that is partial, killed,
  or short of its planned prompts or of the held-out set gets a WARNING.
- **Token-limit truncations**: rows with `hit_token_limit: true`, and the share of
  failures they explain.

## Still assumed (to confirm with agent/b)

- The report key that will carry the effective generation settings and the
  few-shot source. agent/b's `effective_settings()` exists but isn't written to the
  report yet. The tool accepts top-level keys or the nested dicts listed above.
- That search rows always carry `selected_round`, and that round 0 stops whenever a
  candidate passes. The best-of-4 column depends on this; without `selected_round`
  it is `n/r`.
- That the paper's Base and FT rows mean the zero-shot runs.
