# BLOCKERS

All open items that need human attention or a GPU run to resolve.

_Last updated: 2026-11-15_

---

## GPU-dependent (needs Kaggle T4 session)

| # | Item | File / Location | Detail |
|---|------|----------------|--------|
| G1 | GPU dataset verification | `data/verified/dataset.jsonl` | File is empty. Run `python data/build_dataset.py --device cuda`. |
| G2 | 15 OOM shapes on T4 | `data/raw/interpret_report_2026-11-15.json` | Whether the 15 largest shapes (attention b16×s2048, norm r65536) fit on a 16 GB T4 is untested. |
| G3 | Baseline eval results | `evaluation/results/` | No JSON reports exist yet. Run `python evaluation/baseline_eval.py` after dataset is verified. |
| G4 | Fine-tuned adapter | `training/checkpoints/` | No adapter exists. Run `python training/finetune.py` after dataset is verified. |
| G5 | peak_memory_gb | `training/checkpoints/run_*/split_info.json` | Will be `null` until a GPU training run writes the real value. |
| G6 | Paper tables | `docs/results/table_*.tex` | All five LaTeX fragments are `\todo{}` until the GPU run. |

## LaTeX

| # | Item | File / Location | Detail |
|---|------|----------------|--------|
| L1 | Paper compilation | `docs/paper/paper.tex` | No `pdflatex` or `xelatex` found on this machine. Compile on a machine with TeX Live, or use a CI runner. |

## Manual review

| # | Item | File / Location | Detail |
|---|------|----------------|--------|
| M1 | Explanation rubric scoring | `evaluation/explanation_review.py` | Rubric scoring is manual (20–30 samples per model). Cannot be automated. |
| M2 | Paper abstract fill-in | `docs/paper/paper.tex` | Abstract cites `\todo{}` for all result numbers. Requires GPU results. |

---

_Items without a GPU requirement and without a human decision are resolved._
