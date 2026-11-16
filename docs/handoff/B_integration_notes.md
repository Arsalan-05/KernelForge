# Workstream B: integration notes

For the sessions that own `scripts/gpu_session.py` (A), `evaluation/report/` (C),
and the docs. Everything below describes code on branch `agent/b`. No GPU run has
used it yet, and no numbers in this file are results.

## (a) Flags `scripts/gpu_session.py` should pass

**Drop `--limit` from both eval stages.** pass@1 must cover every held-out prompt.
`--limit` is now a dev-only cap; best-of-N search already runs on its own
stratified subset (`--search-limit`, default 25 = 5 per category).

`<M>` is a per-stage time budget in minutes. Choose it to fit the Kaggle session;
stage durations on a T4 have not been measured.

Baseline (zero-shot):

```
python evaluation/baseline_eval.py --device cuda --config <eval_config> \
    --max-new-tokens 2048 --num-candidates 4 --repair-rounds 1 --search-limit 25 \
    --max-minutes <M> --resume
```

Few-shot baseline (optional, separate invocation; the example comes from the
TRAIN split and, by default, from a different category):

```
python evaluation/baseline_eval.py --device cuda --config <eval_config> \
    --max-new-tokens 2048 --few-shot --few-shot-source other_category \
    --max-minutes <M> --resume
```

Fine-tune:

```
python training/finetune.py --config training/configs/finetune_default.json --max-minutes <M>
```

Fine-tuned eval (`--adapter` is optional and defaults to `<run>/adapter`):

```
python evaluation/finetuned_eval.py --device cuda --config <eval_config> \
    --checkpoint-run <run> --max-new-tokens 2048 \
    --num-candidates 4 --repair-rounds 1 --search-limit 25 \
    --max-minutes <M> --resume
```

Notes:
- `--resume` is safe on a fresh session: with no results log it starts from
  scratch. If a log exists with different settings, the script exits with an error
  instead of mixing rows.
- A run stopped by `--max-minutes` exits with code 0 and writes a report with
  `"partial": true`; check that key instead of the exit code.
- Every eval pins `repetition_penalty=1.0` itself; no flag is needed.
- `--max-new-tokens 2048` is already the default; passing it explicitly keeps
  the session log self-describing.

## (b) Doc changes needed

`DOCUMENTATION.md`:
- Phase 3/5 sections: pass@1 is greedy (do_sample off, repetition_penalty 1.0)
  on all held-out prompts. Best-of-N + repair runs on a stratified 25-prompt
  subset with a fixed seed (default 1234).
- New flags: `--max-new-tokens` (default 2048), `--results-jsonl`, `--resume`,
  `--max-minutes`, `--num-candidates`, `--repair-rounds`, `--search-limit`,
  `--seed`, `--interpret`, plus `--few-shot` / `--few-shot-source` on
  `baseline_eval.py`.
- Incremental results: one JSONL per run (`evaluation/results/baseline.jsonl`,
  `baseline_few_shot_<source>.jsonl`, `finetuned.jsonl`); the JSON report is
  rebuilt from it.
- `training/finetune.py`: `--dry-run` (tiny random model, CPU), `--max-minutes`
  (saves the adapter and stops), `--backend`; `train_log.json`,
  `trainer_state.json` and `split_info.json` in the run folder.
- Training details: loss only on the assistant answer, examples longer than
  `max_seq_length` dropped (none at 8192; the longest is 2,060 tokens), and no
  fallback to the schema examples on real runs.
- Explanation review is now three steps: build (`--report ... --output review.json
  [--blind]`), score (`--score review.json`, interactive), summarize
  (`--summarize review.json`, joins `review.key.json` when blind). Remove any
  wording about hand-editing `rubric_score` in the JSON.

`MASTER_PLAN.md`:
- Phase 3 / Phase 5 tables: add a prompt-mode column (or footnote few-shot
  rows), state that pass@1 covers all held-out prompts while "best-of-4 + repair"
  covers a 25-prompt subset, and give n for every cell.
- Phase F drill: `baseline_eval.py --dry-run --device cpu` still works; it now
  writes `dryrun_baseline.jsonl` and `dryrun_baseline_*.json`. Add
  `python training/finetune.py --dry-run` as the fine-tuning drill.
- Part III step 0: the `gpu_session.py` stages use the flags in section (a).
- Write-up framing: the held-out template in each category is a close relative
  of a training template (e.g. GQA causal prefill held out, causal prefill
  trained), and few-shot rows must name their source (`other_category` or
  `same_category`).

## (c) Report keys `evaluation/report/` needs to read

Report files: `evaluation/results/<run_name>_<UTC timestamp>[_<n>].json`, with
`run_name` in `baseline`, `baseline_few_shot_other_category`,
`baseline_few_shot_same_category`, `finetuned`. Skip any `dryrun_*.json`, and any
report with `"dry_run": true`.

Top level:

| key | meaning |
|---|---|
| `prompt_mode` | `"zero_shot"` or `"few_shot"` |
| `few_shot_source` | `"other_category"`, `"same_category"`, or `null` for zero-shot |
| `partial` | `true` if stopped by `--max-minutes` or not all planned prompts were scored |
| `prompts_planned` | pass@1 prompts the run was meant to score |
| `prompts_scored` | pass@1 prompts actually scored (the n for pass@1 cells) |
| `decoding` | `"greedy"` for pass@1 |
| `max_new_tokens` | generation cap |
| `repetition_penalty` | `1.0` for every eval |
| `generation_settings` | effective pass@1 settings: `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens` (sampling-only keys are `null` under greedy) |
| `results_jsonl` | path of the per-prompt log the report was built from |
| `per_category` | pass@1 metrics per category |
| `outcomes` | pass@1 outcome counts (`correct`, `incorrect`, `no_kernel`, `error`, `timeout`, ...) |
| `hit_token_limit` | pass@1 generations that stopped at `max_new_tokens` |
| `results` | pass@1 rows |
| `model_name`, `adapter_path`, `dry_run`, `split`, `held_out_templates` | provenance |
| `baseline_report`, `baseline_per_category` | fine-tuned reports only: the zero-shot baseline compared against |

`search` block (present only when search ran):

| key | meaning |
|---|---|
| `num_candidates` | N samples in round 0 |
| `repair_rounds` | R repair rounds |
| `search_limit` | K requested |
| `prompts_scored` | prompts actually scored (the n for search cells) |
| `prompts_planned` | prompts the search was meant to score |
| `seed` | torch seed set before each prompt's search |
| `round0_temperature`, `repair_temperature` | sampling temperatures |
| `generation_settings` | `{"round0": {...}, "repair": {...}}`, same keys as the top-level one |
| `per_category`, `outcomes`, `results` | search metrics and rows |

## (d) Files to collect into the zip

- `evaluation/results/*.json`: reports.
- `evaluation/results/*.jsonl`: per-prompt logs, including partial runs and any
  `*.<timestamp>.jsonl` logs that a rerun moved aside.
- From the latest `training/checkpoints/run_*/`: `adapter/`, `split_info.json`,
  `held_out_test.jsonl`, `train_log.json`, `trainer_state.json`.
- Not needed: `checkpoint-*` folders (optimizer state, large) and any
  `training/checkpoints/dryrun_*` folder.
