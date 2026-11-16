# Kaggle T4 GPU Session Runbook

This runbook covers running a full KernelForge session on a Kaggle T4 GPU (16 GB VRAM, 12 h wall clock) from setup to artifact download.

**Related docs:**
- `docs/handoff/T4_FIT.md` — per-shape VRAM estimates for all 300 kernel entries (estimate only; T4 fit is untested).
- `docs/handoff/SMOKE_TEST.md` — CPU smoke-test results: what ran, what was skipped, and why.

---

## 0. Time Budget (Estimates — Untested on T4)

| Stage | Estimated time | Notes |
|---|---|---|
| `env` | < 1 min | GPU check, python/triton version |
| `harness` | 1–2 min | Verifier smoke check, one kernel |
| `dataset` | 4–5 h | 300 kernels × (compile + verify); 15 may OOM (counted, not fatal) |
| `baseline` | 1–2 h | 100 prompts, greedy, 2048 max tokens |
| `finetune` | ~1 h | QLoRA, 1.5B, 3 epochs, batch=1, grad_accum=8 (`FINETUNE=True` only) |
| `eval` | ~1 h | 100 prompts on the fine-tuned adapter (`FINETUNE=True` only) |
| **Total (no FT)** | **~6–7 h** | Well within 12 h |
| **Total (+ FT)** | **~8–9 h** | Still within 12 h |

These are rough estimates based on typical Triton compile times and 1.5B model inference speed.
Actual times will vary. If running short on time, use `--eval-limit 25` to score 25 prompts
instead of 100.

---

## 0. Pre-flight (local, no GPU needed)

Before starting a Kaggle session, verify everything works on CPU:

```bash
# 1. Run the full test suite (must be 0 failures)
python -m pytest tests/ -q --tb=short

# 2. Smoke-check the interpreter path (needs Docker + Triton image)
docker build -f docker/verify.Dockerfile -t kernelforge-verify .
docker run --rm -v "$PWD":/work kernelforge-verify \
    python data/build_dataset.py --interpret --smoke --jobs 4

# 3. Confirm the token-length check passes
python scripts/check_token_lengths.py
```

If any step fails, fix it before spending GPU time.

---

## 1. Kaggle Setup

### 1.1 Create a new Kaggle notebook

1. Go to kaggle.com → New Notebook
2. Set accelerator to **GPU T4 x2** (or **GPU T4**, whichever is available)
3. Set **Internet** to **On** (needed for `pip install` and model download)
4. Set **Persistence** to **Files** so `/kaggle/working` survives between sessions (optional but useful for resuming)

### 1.2 Attach the previous session's output (for resume)

If you have a `kernelforge_outputs.zip` from an earlier session:
1. Upload it as a Kaggle Dataset
2. Attach it to the notebook (Add Data → Your Datasets)
3. Its path will be `/kaggle/input/<dataset-name>/kernelforge_outputs.zip`
4. Pass it to the runner with `--resume-from` (see §2)

---

## 2. Session Commands

Run these in a Kaggle code cell. Use the notebook at `notebooks/kaggle_gpu_session.ipynb` or run from a terminal cell.

### 2.1 Setup

```bash
!git clone https://github.com/Arsalan-05/KernelForge.git /kaggle/working/KernelForge
%cd /kaggle/working/KernelForge
!pip install -q -r requirements.txt
```

### 2.2 Standard run (dataset + baseline eval)

```bash
!python scripts/gpu_session.py \
    --budget-hours 11 \
    --out /kaggle/working/kernelforge_outputs.zip
```

Runs stages: `env` → `harness` → `dataset` → `baseline`

### 2.3 Full run (+ QLoRA fine-tune + fine-tuned eval)

```bash
!python scripts/gpu_session.py \
    --finetune \
    --budget-hours 11 \
    --out /kaggle/working/kernelforge_outputs.zip
```

Runs all stages: `env` → `harness` → `dataset` → `baseline` → `finetune` → `eval`

### 2.4 Resume from a previous session

```bash
!python scripts/gpu_session.py \
    --finetune \
    --budget-hours 11 \
    --resume-from /kaggle/input/<dataset-name>/kernelforge_outputs.zip \
    --out /kaggle/working/kernelforge_outputs.zip
```

### 2.5 Single-stage run (for debugging)

```bash
# Re-run just the dataset stage
!python scripts/gpu_session.py --stages dataset --budget-hours 11

# Re-run just the baseline eval
!python scripts/gpu_session.py --stages baseline --budget-hours 11
```

### 2.6 Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--finetune` | off | Also run QLoRA fine-tune and fine-tuned eval |
| `--budget-hours N` | (none) | Stop cleanly before wall-clock expires |
| `--out PATH` | `kernelforge_outputs.zip` | Where to write the output zip |
| `--resume-from PATH` | (none) | Dataset progress zip from a previous run |
| `--stages s1 s2 ...` | all | Run only these stages |
| `--eval-limit N` | None (all 100) | Cap the number of eval prompts |

---

## 3. Eval flags (passed internally by gpu_session.py)

The session script passes these to `baseline_eval.py` and `finetuned_eval.py`:

- `--max-new-tokens 2048` — required; do not lower (longest template target is ~1302 tokens)
- `--resume` — safe on fresh sessions; skips prompts already in the JSONL log
- `repetition_penalty=1.0` — pinned internally; not an external flag

Few-shot mode:

```bash
# To run the few-shot baseline (other-category sibling, default):
!python evaluation/baseline_eval.py \
    --few-shot-source other_category \
    --max-new-tokens 2048

# To run few-shot with same-category sibling:
!python evaluation/baseline_eval.py \
    --few-shot-source same_category \
    --max-new-tokens 2048
```

---

## 4. Expected Output Files

After a successful run, `kernelforge_outputs.zip` contains:

```
data/verified/dataset.jsonl          # GPU-verified triples (300 entries if all pass)
data/raw/generation_report.json      # Pass/fail counts per template and category
data/raw/progress_cuda.jsonl         # Incremental progress log (for resume)
evaluation/results/baseline_*.json  # Baseline eval report
evaluation/results/baseline_*.jsonl # Incremental baseline log
evaluation/results/finetuned_*.json # Fine-tuned eval report (if --finetune)
training/checkpoints/run_*/
  adapter/                           # LoRA adapter weights
  split_info.json                    # Train/val/test split + peak_memory_gb
  held_out_test.jsonl                # Held-out test set
  train_log.json                     # Per-step loss history
  trainer_state.json                 # HuggingFace Trainer state
session.json                         # Session summary + env + artifact manifest
stage_status.json                    # Per-stage: status, seconds, error
```

---

## 5. Post-Session Checklist

After downloading `kernelforge_outputs.zip`:

- [ ] Check `stage_status.json` — every stage should be `"ok"` or `"skipped (no data)"` (not `"failed"`)
- [ ] Check `session.json` → `dataset.passed` — should be 285 (the 15 OOM shapes may still OOM on T4)
- [ ] Extract the zip to the repo root or a sibling directory
- [ ] Copy result files to their repo locations:
  ```bash
  cp -f data/verified/dataset.jsonl data/verified/
  cp -f data/raw/generation_report.json data/raw/
  cp -f evaluation/results/*.json evaluation/results/
  cp -rf training/checkpoints/run_* training/checkpoints/
  ```
- [ ] Generate the report fragments:
  ```bash
  python -m evaluation.report . --out docs/results/
  ```
- [ ] Verify 5 fragment files appear in `docs/results/`:
  - `table_dataset.tex`, `table_categories.tex`, `table_search.tex`, `table_training.tex`, `table_rubric.tex`
- [ ] Compile the paper (requires TeX Live):
  ```bash
  cd docs/paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
  ```
- [ ] Commit the results and fragments

---

## 6. If the Session Dies Mid-Run

1. Download `kernelforge_outputs.zip` from `/kaggle/working/` before the session expires
2. Upload it as a Kaggle Dataset
3. Start a new session and use `--resume-from /kaggle/input/<name>/kernelforge_outputs.zip`
4. The dataset stage will skip entries already in the progress log
5. The eval stages will skip prompts already in their JSONL logs (via `--resume`)

---

## 7. Memory and Performance Notes

- The largest attention and norm shapes (15 total) OOM'd in an 8 GB VM with Triton interpreter.
  Whether they fit on a 16 GB T4 is untested — `oom` is counted separately from `incorrect`.
- QLoRA peak memory on T4 is not yet measured (shown as `null` in `split_info.json` until run).
- If a stage times out inside `--budget-hours`, it is marked `"failed"` and the session continues.
- Use `--eval-limit 25` to do a quick smoke check on 25 prompts instead of 100.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `env` stage fails | No CUDA GPU visible | Check accelerator setting in notebook |
| `harness` stage fails | Triton or torch version mismatch | Check `session.json` → env |
| `dataset` stage produces 0 entries | Triton install issue | Check `data/raw/generation_report.json` for errors |
| Eval reports `0 prompts scored` | dataset.jsonl is empty | Fix dataset stage first |
| `split_info.json` has `peak_memory_gb: null` | GPU OOM during training | Lower `per_device_train_batch_size` in `training/configs/finetune_default.json` |
| `--resume-from` path not found | Dataset not attached | Attach the dataset zip in Kaggle UI before running |
