# Paper checklist: what to fill in after the GPU run

Work top to bottom. Nothing in `paper.tex` may be posted while a red `[TODO: …]` remains
(search the PDF for "TODO"). Every number must come from an artifact in the session archive
(`/kaggle/working/kernelforge_outputs.zip`). Record which file each number came from.

## A. Before the GPU run (decisions that change what gets measured)

These need code changes in files this workstream doesn't own. They're listed so the owner can
decide before spending GPU hours.

| # | Decision | Why it matters | Where |
|---|---|---|---|
| A1 | **Evaluate all 100 held-out prompts or only 25?** `gpu_session.py --baseline-limit` defaults to 25 (5 per category). | At n=5 per category, one success moves the percentage by 20 points. Wilson intervals will be very wide. | `scripts/gpu_session.py` |
| A2 | **Does the 1536-token generation cap truncate kernels?** The longest reference response is about 4.3k characters (GQA prefill attention), which may be close to 1.5k tokens. | A truncated kernel is scored as a failure of the model when it is really a failure of the setup. | `scripts/gpu_session.py --max-new-tokens`; check with the Qwen tokenizer |
| A3 | **Best-of-4 + repair column.** `baseline_eval.py` and `finetuned_eval.py` run zero-shot pass@1 only and don't call `kernelforge/search.py`. | Table 2's last column and all of Table 3 need a search-based eval. Without one, drop those columns. | `evaluation/` |
| A4 | **Speedup metric.** `avg_speedup` is the arithmetic mean over *passing* kernels only. | It is conditional on correctness and sensitive to outliers. Consider also reporting fast_1 (KernelBench) and a geometric mean. | `evaluation/common.py` |
| A5 | **Base-model prompt.** Evals are zero-shot, with no solved example and no `TRITON_NOTES`. The base model hasn't learned the `kernel:`/`explanation:` format. | Much of the base model's failure may be formatting. Report % parsed, and consider a few-shot base row as a stronger baseline. | `evaluation/baseline_eval.py` |
| A6 | **Loss masking.** Training loss covers the prompt tokens too (no completion-only masking). | It is a legitimate choice, but it must be stated. Completion-only loss may train better on a ~200-example set. | `training/finetune.py` |
| A7 | **Rubric rater.** Blind the explanation scoring (shuffle base/FT, hide model identity)? Add a second rater? | Without either, the rubric is easy to dismiss. | `evaluation/explanation_review.py` |

## B. Fill-ins after the run (in paper order)

| Where in `paper.tex` | What | Source artifact |
|---|---|---|
| Abstract | X% → Y% correct, A× → B× speedup, Z% with search, N/M rubric, weakest category | `evaluation/results/*.json`, rubric JSON |
| Related Work, KernelLLM paragraph | final verified dataset size | `data/raw/generation_report.json` |
| §3.3 / Table 1 | per-category verified / incorrect / OOM / other error; median template speedup | `generation_report.json`, `dataset.jsonl` |
| §3.5 | actual held-out templates and split counts | `training/checkpoints/run_*/split_info.json` |
| §3.6 | loss-masking statement (A6) | — |
| §3.7 | which prompt aids (few-shot, notes, hints) were on for each row | eval configs |
| §4 Fine-tuning | backend actually used (Unsloth or PEFT) and the optimizer | training log |
| §4 Hardware | GPU, driver, CUDA, PyTorch, Triton versions; wall clock per stage; peak memory | session `env` report and logs |
| §4 Decoding | final max-new-tokens (A2) | `eval_model_config.json` |
| §4 Metrics | n per category (A1); fast_1 / geomean (A4) or delete that sentence | — |
| Table 2 | base / FT % correct and mean speedup; FT + search column (A3) | `baseline_*.json`, `finetuned_*.json` |
| Table 3 | pass@1 / best-of-4 / +1 repair, plus extra tokens and harness runs | search eval output (A3) |
| Table 4 | final train loss, best val loss, wall clock, peak memory | trainer logs |
| §5 Failure modes | counts by failure type; 3–5 examples | eval reports |
| §6 / Table 5 | rubric counts, sample seed, copied-explanation count, rater setup (A7) | rubric JSON |
| §7 Limitations | rater wording; baseline-strength item (A5) | — |
| §9 Reproducibility | repo URL and commit hash; released artifacts; timing-variance note | — |
| Title page | author, affiliation, date | — |

## C. Claims that depend on results (re-read each once the numbers are in)

- **Abstract and intro: "fine-tuning changes correctness…"** If FT doesn't beat base, say so plainly. The
  paper is still valid as a small-budget reference point; rewrite the abstract sentence, don't bury it.
- **Contribution 2 (explanations).** Only claim the model "explains" if rubric 2-scores are a clear
  majority *and* explanations aren't mostly copied from training templates. Otherwise report it as a
  negative result.
- **"Search reaches Z%".** Only if A3 is implemented and run. Report the extra cost next to it.
- **Any per-category conclusion.** Each category's test set is one held-out operator at up to 20 shapes.
  Phrase conclusions as "on the held-out GQA prefill template", not "on attention".
- **Speedups.** These are vs PyTorch eager on a T4 only. Don't generalize to other GPUs or to
  `torch.compile`.
- **"All numbers come from GPU mode"** (§3.4). Confirm that no table includes interpreter results.
- **Novelty sentence on explanations** (Contribution 2, Related Work). Before posting, re-read the full
  KernelLLM, AutoTriton, TritonRL, Kevin and Dr. Kernel papers for any explanation/rationale
  evaluation (see `UNVERIFIED.md`).

## D. Pre-posting checks

- [ ] `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper` builds with no undefined
      references. (Not compiled yet: no TeX on the dev Mac. Overleaf works.)
- [ ] No `TODO` remains in the PDF.
- [ ] Every number in the text matches its table, and every table cell matches an artifact.
- [ ] Re-check arXiv versions of cited preprints. Several may have gained a venue by January 2027.
      Update `references.bib` and `SOURCES.md`.
- [ ] Fix MASTER_PLAN's TritonBench sentence ("speedup rarely exceeds 1.5×") so the repo and paper agree.
- [ ] arXiv category (likely cs.LG with cs.PF/cs.SE cross-lists), and endorsement if this is a first submission.
