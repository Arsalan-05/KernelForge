# Kaggle T4 GPU Session Runbook

This runbook covers running a full KernelForge session on a Kaggle T4 GPU
(16 GB VRAM, 12 h wall clock) from setup to artifact download.

**Related docs:**
- `docs/handoff/T4_FIT.md` — per-shape VRAM estimates for all 300 entries.
- `docs/handoff/SMOKE_TEST.md` — CPU interpreter results: 285/300 passed, 15 OOM.
- `docs/handoff/BLOCKERS.md` — exact list of GPU-blocked items.

**What is CPU-verified vs GPU-estimated:**
- Stage runner, resume logic, budget enforcement, artifact zipping — all tested
  (23 unit tests in `tests/test_gpu_session.py`, all pass on CPU).
- Time estimates — rough; actual T4 wall-clock has never been measured.

---

## 1. Pre-flight (local, no GPU needed)

Before spending Kaggle GPU time, confirm everything works locally:

```bash
# 1. Full CPU reproducibility check: tests + mutation test + report
bash scripts/reproduce_all.sh
# Expected: "All steps PASSED" in ~60s

# 2. Pre-flight check (Python packages, configs, dataset status)
python scripts/preflight_check.py --no-gpu
# Expected: "All required checks passed."

# 3. Split-leakage check
python scripts/check_split_leakage.py
# Expected: "PASSED — no leakage detected."
```

If any step fails, fix it before going to Kaggle.

---

## 2. Time Budget (Estimates — not yet measured on T4)

| Stage | Estimated time | Notes |
|---|---|---|
| `env` | < 1 min | GPU check, python/triton version |
| `harness` | 1–2 min | Verifier smoke check, one kernel |
| `dataset` | 4–5 h | 300 kernels × (compile + verify); 15 may OOM (counted, not fatal) |
| `baseline` | 1–2 h | 100 prompts, greedy, 2048 max tokens |
| `finetune` | ~1 h | QLoRA, 1.5B, 3 epochs, batch=1, grad_accum=8 (`--finetune` only) |
| `eval` | ~1 h | 100 prompts on the fine-tuned adapter (`--finetune` only) |
| **Total (no FT)** | **~6–7 h** | Well within 12 h |
| **Total (+ FT)** | **~8–9 h** | Still within 12 h |

If running short on time, use `--eval-limit 25` to score 25 prompts instead of 100.

---

## 2b. Expected Base-Model Numbers (Verified 2026-11-16)

These numbers were measured locally on `Qwen/Qwen2.5-Coder-1.5B-Instruct` in fp32
on MPS, using `notes=False` (the same prompt as finetune.py and the Kaggle baseline).
They are the pre-fine-tuning floor to compare against.

| Metric | Value | Notes |
|---|---|---|
| No-header pass@1 | **0/60 = 0.000** | 95% CI [0.000, 0.060] |
| With-header pass@1 | **0/50 = 0.000** | NameError entries only; still 0 |
| Truncation rate | 4/60 = 6.7% | Large layernorm + kv_cache entries hit 1536-token cap |
| Degenerate output | 0/60 | fp32 on MPS; no fp16 NaN `!!!!` repetition |
| tok/s (MPS, fp32) | median 5.9, mean 6.1 | T4 with 4-bit will be faster |

**What to expect from the Kaggle baseline stage:**  
The base model (before fine-tuning) will very likely score 0/N on T4 as well.
The 4-bit NF4 baseline on T4 may differ slightly (quantization vs. fp32), but
the structural failures — missing imports, missing `class ModelNew`, hallucinated
Triton API — are independent of precision. Any pass@1 > 0.000 from the fine-tuned
adapter would confirm the adapter learned the response format.

**Dominant failure modes (no fine-tuning):**
- 50/60: `NameError: name 'triton' is not defined` — model omits all imports
- 10/60: `SyntaxError` — invalid default-arg expressions or truncation
- After adding import header: model still missing `class ModelNew` in 37/50 cases

---

## 3. Kaggle Setup

### 3.1 Create a new Kaggle notebook

1. Go to kaggle.com → **New Notebook**
2. Set Accelerator → **GPU T4** (or **GPU T4 x2**)
3. Set **Internet** → **On** (needed for `pip install` and model download)
4. Set **Persistence** → **Files** (optional; keeps `/kaggle/working` between sessions)

### 3.2 Attach the previous session's output (for resume)

If you have a `kernelforge_outputs.zip` from an earlier session:
1. Upload it as a Kaggle Dataset
2. Attach it to the notebook: **Add Data → Your Datasets**
3. Its path will be `/kaggle/input/<dataset-name>/kernelforge_outputs.zip`
4. Pass it to the runner with `--resume-from` (see §4)

---

## 4. Session Commands

Run these in a Kaggle code cell or use `notebooks/kaggle_gpu_session.ipynb`.

### 4.1 Setup

```bash
!git clone --depth 1 -b main https://github.com/Arsalan-05/KernelForge.git
%cd KernelForge
!pip install -q -r requirements-demo.txt
!python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.cuda.is_available())"
```

### 4.2 Standard run (dataset + baseline eval)

```bash
!python scripts/gpu_session.py \
    --budget-hours 11 \
    --out /kaggle/working/kernelforge_outputs.zip
```

Runs stages: `env` → `harness` → `dataset` → `baseline`

### 4.3 Full run (+ QLoRA fine-tune + fine-tuned eval)

```bash
!python scripts/gpu_session.py \
    --finetune \
    --budget-hours 11 \
    --out /kaggle/working/kernelforge_outputs.zip
```

Runs all stages: `env` → `harness` → `dataset` → `baseline` → `finetune` → `eval`

### 4.4 Resume from a previous session

```bash
!python scripts/gpu_session.py \
    --finetune \
    --budget-hours 11 \
    --resume-from /kaggle/input/<dataset-name>/kernelforge_outputs.zip \
    --out /kaggle/working/kernelforge_outputs.zip
```

**How resume works (CPU-verified):**
1. `restore_progress(zip)` extracts `data/raw/progress_cuda.jsonl` from the zip
   into the new repo root.
2. `build_dataset.py --resume` reads that log and skips entries that already have
   a result. Only the unfinished entries are compiled and verified.
3. The eval stages use `--resume` to skip prompts already in their JSONL logs.

This was verified end-to-end on CPU:
- Simulated 2-entry progress log in a zip → restored correctly to a fresh repo root
- Counts: `test_restore_progress_from_zip_and_folder` and `test_restore_progress_reports_missing_log`
  pass (see `tests/test_gpu_session.py`)

### 4.5 Single-stage run (for debugging)

```bash
# Re-run just the dataset stage
!python scripts/gpu_session.py --stages dataset --budget-hours 11

# Re-run just the baseline eval
!python scripts/gpu_session.py --stages baseline --budget-hours 11
```

### 4.6 Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--finetune` | off | Also run QLoRA fine-tune and fine-tuned eval |
| `--budget-hours N` | (none) | Stop cleanly before wall-clock expires |
| `--out PATH` | `kernelforge_outputs.zip` | Where to write the output zip |
| `--resume-from PATH` | (none) | Dataset progress zip from a previous session |
| `--resume` | off | Resume from `data/raw/progress_cuda.jsonl` in-place (no zip) |
| `--stages s1 s2 ...` | all | Run only these stages |
| `--eval-limit N` | None (all) | Cap the number of eval prompts per eval |

---

## 5. What the Output Zip Contains

After a successful run, `kernelforge_outputs.zip` contains:

```
data/verified/dataset.jsonl          # GPU-verified triples (up to 300 entries)
data/raw/generation_report.json      # Pass/fail counts per template and category
data/raw/progress_cuda.jsonl         # Incremental progress log (for resume)
evaluation/results/baseline_*.json  # Baseline eval report
evaluation/results/baseline_*.jsonl # Incremental baseline log
evaluation/results/finetuned_*.json # Fine-tuned eval report (if --finetune)
training/run_*/
  adapter/                           # LoRA adapter weights (latest run only)
  split_info.json                    # Train/val/test split + peak_memory_gb
  held_out_test.jsonl                # Held-out test set
  train_log.json                     # Per-step loss history
  trainer_state.json                 # HuggingFace Trainer state
session.json                         # Session summary + env + artifact manifest
stage_status.json                    # Per-stage: status, seconds, error
```

**Note:** Checkpoint optimizer states are excluded from the zip to keep it small.
Only the final adapter weights are included.

---

## 6. Post-Session Checklist

After downloading `kernelforge_outputs.zip`:

- [ ] Check `stage_status.json` — every stage should be `"ok"` or `"skipped (no data)"`, not `"failed"`
- [ ] Check `session.json` → `dataset.passed` — should be 285–300 (15 OOM shapes may still OOM on T4)
- [ ] Extract the zip or pass it directly to the report generator:
  ```bash
  python -m evaluation.report kernelforge_outputs.zip --out docs/results/
  ```
- [ ] Verify the LaTeX fragments appear in `docs/results/`:
  - `table_dataset.tex`, `table_categories.tex`, `table_search.tex`,
    `table_training.tex`, `table_rubric.tex`
- [ ] Run the leakage check on the new dataset:
  ```bash
  python scripts/check_split_leakage.py --dataset data/verified/dataset.jsonl
  # Expected: "PASSED — no leakage detected."
  ```
- [ ] Compile the paper (requires TeX Live):
  ```bash
  cd docs/paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
  ```
- [ ] Commit the results and fragments

---

## 7. If the Session Dies Mid-Run

1. **Download `kernelforge_outputs.zip`** from `/kaggle/working/` before the
   session expires. The zip is refreshed every 10 minutes and after each stage.
2. **Upload it as a Kaggle Dataset** ("Add data" → "Create dataset")
3. **Start a new session**, attach the dataset, then run:
   ```bash
   !python scripts/gpu_session.py \
       --resume-from /kaggle/input/<name>/kernelforge_outputs.zip \
       --budget-hours 11
   ```
4. The dataset stage will skip entries already in `progress_cuda.jsonl`.
5. The eval stages will skip prompts already in their JSONL logs.

**Important timing:** The zip is written atomically (rename, not copy) to
`kernelforge_outputs.zip.tmp` → `kernelforge_outputs.zip`. Even if Kaggle kills
the session mid-zip, the previous complete zip is kept.

---

## 8. Memory and Performance Notes

- 15 of the 300 entries OOM'd during CPU interpreter testing (8 GB VM). Whether
  they fit on a 16 GB T4 is unknown — `oom` is counted separately from `incorrect`.
- QLoRA peak memory on T4 is not yet measured (shown as `null` in `split_info.json`
  until a real training run).
- If a stage times out inside `--budget-hours`, it is marked `"failed"` and the
  session continues to later stages.
- Use `--eval-limit 25` for a quick smoke check on 25 prompts instead of 100.

---

## 10. Precision and NaN Guard

### Generation precision on T4

Generation (`KernelGenerator`) uses 4-bit NF4 quantization with fp16 activations
(`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)`).
This is the standard QLoRA setup and is stable on T4 CUDA.

Training uses `torch_dtype=torch.float16` and `fp16=True` in `TrainingArguments`
(T4 supports fp16 natively; it does not support bf16).

The fp16 NaN issue (repeating `!` output) was observed on MPS (Apple Silicon) with
*full* fp16 weights — not on T4. On T4 with 4-bit quantization the NaN risk is low.

### Degenerate-output detection

The eval pipeline (`evaluation/common.py`) and local baseline script
(`scripts/local_baseline_eval.py`) now detect degenerate output before attempting
verification. If more than 50% of the last 300 generated characters are a single
repeated character, the result is recorded as `degenerate_output` (not `syntax_error`).

If you see a high `degenerate_output` count in results, check:
1. `load_in_4bit` is `True` in `training/configs/finetune_default.json` (default)
2. No `torch_dtype` override is bypassing quantization in `KernelGenerator`

---

## 11. Pre-flight Notebook Static Check (verified 2026-11-16)

Checked `notebooks/kaggle_gpu_session.ipynb` statically:
- **Hard-coded local paths:** none (no `/Users/...` or other non-Kaggle paths)
- **Non-Kaggle absolute paths:** Cell 6 reads `/kaggle/working/kernelforge_outputs/session.json` — correct for Kaggle
- **Imports:** `shlex` (Cell 5), `json, os` (Cell 6), `os, shutil` (Cell 7) — all standard library, always available
- **Cell-order dependencies:** `BRANCH`, `BUDGET_HOURS`, `FINETUNE`, `EVAL_LIMIT`, `RESUME_FROM` defined in Cell 1 and used in Cell 2 and Cell 5 — correct order
- **No forward references** — no cell reads a variable set in a later cell

### Training pipeline CPU dry-run (verified 2026-11-16)

```bash
python training/finetune.py --dry-run
```

Result: **complete** (exit code 0).

What was exercised:
- Template entry loading from all 5 categories (GENERATORS, unverified — dry-run mode)
- `build_messages` + `render_prompt` tokenization, prompt+answer assembly
- Loss masking: 59% supervised tokens (answer-only), 41% prompt tokens masked to -100
- Tiny random Qwen2 (2 layers, hidden=64) created from configured model's tokenizer
- LoRA config applied via `get_peft_model` (32,768 trainable params / 9.8M total)
- `DataCollatorForSeq2Seq` padding
- 2 optimizer steps (HuggingFace `Trainer`, transformers backend, CPU fp32)
- `split_info.json`, `held_out_test.jsonl`, `train_log.json`, `trainer_state.json` written
- LoRA adapter saved to `training/checkpoints/dryrun_*/adapter/`

What was **not** exercised:
- Real model weights (tiny random weights → loss is noise, ~11.9)
- CUDA / T4 GPU path (CPU only)
- Unsloth backend (uses `transformers` backend on CPU)
- 4-bit quantization (CPU path loads full fp32)
- `TimeLimitCallback` (`--max-minutes` not passed)
- `datasets.Dataset` load from JSONL (dry-run uses template candidates directly)

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `env` stage fails "no CUDA GPU visible" | No GPU accelerator selected | Settings → Accelerator → GPU T4 |
| `harness` stage fails | Triton or torch version mismatch | Check `session.json` → env |
| `dataset` stage produces 0 entries | Triton install issue | Check `data/raw/generation_report.json` |
| Eval reports `0 prompts scored` | `dataset.jsonl` is empty | Fix dataset stage first |
| `split_info.json` has `peak_memory_gb: null` | GPU OOM during training | Lower `per_device_train_batch_size` in `training/configs/finetune_default.json` |
| `--resume-from` path not found | Dataset not attached | Attach the dataset zip in Kaggle UI before running |
| `report` shows all stages "absent" | Zip contains only `progress_cuda.jsonl` | The dataset stage may not have written `generation_report.json` — check `stage_status.json` |
| Eval shows many `degenerate_output` entries | fp16 NaN (unlikely on T4+4bit) | Verify `load_in_4bit=true` in finetune_default.json |
