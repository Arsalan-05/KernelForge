# Results interface: table fragments read by `paper.tex`

`paper.tex` has five results tables. Each one reads its **body rows** from a fragment under
`docs/results/` if that file exists. If it doesn't, the table shows red `[TODO]` placeholder rows. The
paper compiles with any subset of the fragments present, including none.

The paper owns each table's header, column spec, caption and rules. A fragment supplies **only the
rows between the header's `\midrule` and the closing `\bottomrule`**.

## Rules that apply to every fragment

1. **Rows only.** No `\begin{tabular}`, `\toprule`, `\bottomrule`, `\caption` or `\label`. A
   `\midrule` *between* rows is allowed (Table 1 uses one before its Total row).
2. **Exact column count.** Every row has exactly the number of `&`-separated cells listed below and
   ends with `\\`. A wrong count fails the build ("Extra alignment tab") or silently shifts columns.
3. **Fixed row order and labels**, as listed below. The first cell must match the listed label exactly,
   so the tables line up with the prose.
4. **Plain LaTeX text cells.** Escape `%` as `\%`, `_` as `\_`, `&` as `\&`. Use `$\times$` for ×.
   Don't use packages beyond `booktabs` and `xcolor`.
5. **Missing values.** Write `--` when a value is undefined (e.g. mean speedup with no passing kernels).
   Write `n/r` when that run was not done. Never leave a cell empty, and never write a number that
   didn't come from an artifact.
6. **Numbers.** Percentages have one decimal place and no `%` sign inside the cell; the header says
   it's a percentage. Speedups have two decimal places followed by `$\times$`, e.g. `1.37$\times$`.
   Counts are integers.
7. **Provenance comment.** The first line of every fragment is a LaTeX comment naming the source
   artifact(s) and the generation time, e.g.
   `% generated 2027-01-05T12:00Z from evaluation/results/finetuned_20270105.json`. Comments don't
   affect the table.
8. **Line endings.** UTF-8 (ASCII is safest), ending with a newline after the final `\\`.

Category labels, in this order, wherever a table has one row per category:

| Row | First cell (exact) | Category id in code |
|---|---|---|
| 1 | `Quantized matmul` | `quantized_matmul` |
| 2 | `Attention` | `attention` |
| 3 | `KV-cache` | `kv_cache` |
| 4 | `Norm` | `norm` |
| 5 | `RoPE` | `rope` |

---

## 1. `docs/results/table_dataset.tex` (Table 1, dataset construction)

**6 cells per row.** 5 category rows, then `\midrule`, then one `Total` row (7 lines of rows in all).

| # | Column header in paper | Content | Source |
|---|---|---|---|
| 1 | Category | label from the list above / `Total` | — |
| 2 | Generated | candidates generated (integer; 60 per category, 300 total if the grid is unchanged) | `data/raw/generation_report.json` |
| 3 | GPU-verified | entries that passed on the GPU and entered `dataset.jsonl` | same |
| 4 | Incorrect | ran but `correct == false` | same |
| 5 | OOM | out-of-memory failures (counted separately by the report) | same |
| 6 | Other error | everything else: error, timeout, crash | same |

Columns 3–6 must add up to column 2 in every row. The generator should assert this.

Example of the **shape only** (synthetic values; never commit this as a result):
```latex
% SYNTHETIC EXAMPLE -- format only
Quantized matmul & 60 & 0 & 0 & 0 & 60 \\
...
\midrule
Total & 300 & 0 & 0 & 0 & 300 \\
```

## 2. `docs/results/table_categories.tex` (Table 2, main held-out-template results)

**7 cells per row.** Exactly 5 category rows.

| # | Column header in paper | Content |
|---|---|---|
| 1 | Category | label |
| 2 | Base: % correct | % of the category's test prompts whose kernel passed the harness (base model, pass@1) |
| 3 | Base: mean speedup | mean of `speedup` over **passing** kernels; `--` if none passed |
| 4 | FT: % correct | as column 2, fine-tuned adapter |
| 5 | FT: mean speedup | as column 3, fine-tuned adapter |
| 6 | FT best-of-4 + repair: % correct | best-of-4 + 1 repair round with the adapter; `n/r` if the search eval wasn't run |
| 7 | $n$ | number of test prompts in the category (integer) |

Sources: `pct_correct` and `avg_speedup` per category in the baseline and fine-tuned reports under
`evaluation/results/` (computed by `evaluation/common.py`). Column 7 is that category's `total`. If base
and FT were evaluated on different prompt counts, stop: the table assumes the same prompts for both.

## 3. `docs/results/table_search.tex` (Table 3, training-free search)

**4 cells per row.** Exactly 2 rows, in this order: `Base`, `Fine-tuned`.

| # | Column header in paper | Content |
|---|---|---|
| 1 | Model | `Base` / `Fine-tuned` |
| 2 | pass@1 | % correct over all test prompts, one sample |
| 3 | best-of-4 | % correct with 4 samples, the harness picking any passing one |
| 4 | best-of-4 + 1 repair | % correct after one repair round on the best failure |

Use `n/r` for any setting that wasn't run. The caption asks for extra cost (mean generated tokens and
harness runs per column). Put those in the prose, not in this fragment.

## 4. `docs/results/table_training.tex` (Table 4, training run)

**5 cells per row.** One row per training run. Normally a single row; if there were several
iterations, put the reported run last.

| # | Column header in paper | Content |
|---|---|---|
| 1 | (run label) | short label, e.g. `QLoRA run` or `QLoRA run 2 (reported)` |
| 2 | Train loss (final) | last logged training loss, 3 decimals |
| 3 | Val loss (best) | lowest validation loss across epochs, 3 decimals; `--` if there was no val set |
| 4 | Wall clock | training time as `h:mm` |
| 5 | Peak GPU memory | in GB with one decimal followed by ` GB`, e.g. `11.2 GB`; `n/r` if not logged |

Sources: trainer logs and the session log in `kernelforge_outputs.zip`.

## 5. `docs/results/table_rubric.tex` (Table 5, explanation rubric)

**5 cells per row.** Exactly 2 rows, in this order: `Base`, `Fine-tuned`.

| # | Column header in paper | Content |
|---|---|---|
| 1 | Model | `Base` / `Fine-tuned` |
| 2 | Score 2 | count of samples scored 2 |
| 3 | Score 1 | count scored 1 |
| 4 | Score 0 | count scored 0 |
| 5 | Copied from a training template | count of explanations copied (nearly) verbatim from a training template; `--` for `Base` |

Source: the rubric JSON from `evaluation/explanation_review.py` after the scores are filled in. Columns
2–4 must add up to the sample size stated in the caption.

---

## Checking a fragment

From `docs/paper/`, run `pdflatex -interaction=nonstopmode paper`. The table must show your rows and
no red TODO rows. Then delete or rename the fragment and rebuild: the TODO rows must come back. Both
builds must succeed.
