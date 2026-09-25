# KernelForge — Complete Learning Plan

**Goal:** Own every layer of this project so you can explain architecture, tradeoffs, failure modes, and code paths in an interview — not just “I used AI to build it.”

**How to use this doc:** Follow phases in order. Each phase has: *what to learn*, *which files to read*, *hands-on drills*, and *interview questions you must answer without looking*. Do not skip drills — reading alone will not stick.

**Time budget (realistic, university-safe):**
- Weekdays: 30–45 min → one “Concept block” or one drill
- Weekend: 1 deep session (1.5–3 hrs) → one full phase + whiteboard practice
- Target total: ~4–6 weeks to interview-ready ownership

---

## 0. The 60-second pitch (memorize this first)

> KernelForge is a **data-efficient, explainable LLM specialist** for **LLM-serving GPU kernels**. Given a PyTorch reference op, it generates an optimized **Triton** kernel **plus** a short natural-language explanation of *why* the optimization helps. Correctness and speedup are checked by a **subprocess-isolated verification harness** that matches KernelBench’s `Model`/`ModelNew` contract. It is deliberately narrow — five serving-relevant categories, supervised QLoRA on a small curated dataset, **no RL** — and is positioned against KernelLLM / Kevin-32B / AutoTriton, not as a claim of beating them on broad benchmarks.

**Two contributions you must always name:**
1. **Data-efficient specialization** on LLM-*serving* ops (not generic KernelBench breadth).
2. **Explainability** — kernel + optimization rationale as a first-class output.

**Honest limitations (say these unprompted):**
- Small model (1.5B) + ~300-example target vs 8B–32B RL systems
- Narrow categories vs KernelBench/TritonBench
- No RL; explanations scored by manual rubric
- Triton requires Linux + NVIDIA (macOS = scripting/harness only)

---

## 1. Mental model of the system

```
PyTorch reference (Model + get_inputs/get_init_inputs)
        │
        ▼
┌───────────────────┐     prompts + parsing      ┌────────────────┐
│  Fine-tuned LLM   │ ─────────────────────────► │ Triton ModelNew │
│  (Qwen + LoRA)    │                            │ + explanation  │
└───────────────────┘                            └───────┬────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────────┐
                                              │ Verification harness│
                                              │ (subprocess sandbox)│
                                              │ correct? + speedup? │
                                              └─────────────────────┘
                                                         │
                    ┌────────────────────────────────────┼────────────────┐
                    ▼                                    ▼                ▼
             Dataset filter                      Baseline / FT eval    Demo API
           (build_dataset.py)                    (evaluation/)       (FastAPI+Gradio)
```

**Same harness everywhere.** Dataset build, baseline eval, fine-tuned eval, and the demo API all use the same definition of “correct” and “faster.” That consistency is a core design claim — be ready to defend it.

---

## 2. Repo map (know what lives where)

| Path | Role | Interview one-liner |
|---|---|---|
| `verification/` | Correctness + timing sandbox | “Subprocess isolation so bad kernels can’t hang the parent.” |
| `data/` | Schema, templates, dataset builder | “3 templates × 20 shapes × 5 categories = 300 candidates.” |
| `kernelforge/` | Shared prompts, parsing, dataset I/O, model wrapper | “The library layer used by train/eval/app.” |
| `training/` | QLoRA fine-tune | “Unsloth preferred; transformers+PEFT fallback.” |
| `evaluation/` | Baseline, FT eval, explanation rubric | “Category-level metrics, not one aggregate number.” |
| `app/` | FastAPI + Gradio demo | “Generate → parse → optionally verify.” |
| `examples/kernels/` | Hand-written Triton (Phase 0) | “Prove fusion > eager multi-launch (softmax).” |
| `docs/project_writeup.md` | Positioning + prior work | “Cite KernelBench, KernelLLM, Kevin, AutoTriton.” |
| `DOCUMENTATION.md` | Implementation truth | “How the harness actually works.” |
| `PROJECT PLAN.md` | Roadmap + decisions | “Why we narrowed scope and skipped RL.” |

---

## Phase A — Foundations (GPU / Triton / why serving ops)

**Duration:** 3–5 weekday sessions + 1 weekend  
**Outcome:** You can whiteboard fusion vs eager, memory-bound vs compute-bound, and why the 5 categories matter for inference.

### Concepts to learn
1. **GPU basics:** HBM vs SRAM, kernel launch overhead, warp/thread block (enough to talk, not CUDA C mastery).
2. **Triton:** `@triton.jit`, `tl.load`/`tl.store`, masks, `program_id`, grid/block sizes.
3. **Fusion:** fewer launches + fewer global-memory round-trips.
4. **Serving vs training ops:** prefill vs decode; why quantized matmul, attention, KV-cache, norm, RoPE dominate serving cost.
5. **Prior work names:** KernelBench, TritonBench, KernelLLM, Kevin-32B, AutoTriton, Liger Kernel — one sentence each.

### Files to read (in order)
1. `docs/project_writeup.md` (whole file)
2. `PROJECT PLAN.md` § Positioning + Prior Work
3. `examples/kernels/vector_add/{reference.py,candidate_triton.py,notes.md}`
4. `examples/kernels/relu/...` (same trio)
5. `examples/kernels/softmax/...` (**most important** — fusion story)
6. `data/SCHEMA.md` § Scope table

### Drills
- [ ] On paper: redraw softmax eager path (max → sub → exp → sum → div) vs single fused Triton pass.
- [ ] Explain why vector_add can be *slower* in Triton than PyTorch (memory-bound, no fusion win).
- [ ] For each of the 5 categories, say in one sentence *when it runs in an LLM forward* (prefill vs decode) and *what cost it hits*.

### Must-answer interview questions
1. Why Triton instead of writing CUDA?
2. Why not target all of KernelBench?
3. What’s the difference between prefill attention and KV-cache decode attention?
4. What does “memory-bandwidth bound” mean, and how does fusion help?

---

## Phase B — Verification harness (the project’s spine)

**Duration:** 4–6 weekday sessions + 1 weekend  
**Outcome:** You can walk `verify_kernel` → `sandbox_runner` line-by-line and defend every design choice.

### Concepts to learn
1. **Two contracts**
   - Function: `solution(*tensors)` — Phase 0 examples
   - Model: `Model` / `ModelNew` + `get_inputs` / `get_init_inputs` — dataset + KernelBench
2. **Why subprocess:** hang → timeout; crash → isolated; optional memory limit
3. **Same seed, separate tensors:** catches in-place mutation bugs
4. **Seeding before each Model construction:** quantized weights / random init must match
5. **Timing:** CUDA Events + median of N iters after warmup (not mean)
6. **Correctness:** `torch.allclose` on float-cast; `equal_nan=False`; status vs `passed`
7. **`VerifyResult` fields** and when `passed` is True

### Files to read (in order)
1. `DOCUMENTATION.md` § The verification harness (entire section)
2. `verification/verify_kernel.py` — public API + CLI
3. `verification/sandbox_runner.py` — both contracts end-to-end
4. `tests/test_verify_kernel.py` — every test name is a design decision

### Drills
- [ ] Run locally: `pytest tests/ -v` and explain every failing/passing test in your own words.
- [ ] Trace one call path on paper: `verify_model_kernel(code, code)` → temp files → JSON payload → subprocess → JSON line → `VerifyResult`.
- [ ] Without looking: list the four `status` values and what causes each.
- [ ] Explain why seeding *only* before `get_inputs()` would break quantized matmul templates.

### Must-answer interview questions
1. Why not call the candidate in-process?
2. Why median instead of mean for timing?
3. Why match KernelBench’s Model contract instead of inventing your own?
4. How do you prevent a generated kernel from hanging your eval loop for hours?
5. What’s the difference between `status="ok", correct=False` and `status="error"`?

---

## Phase C — Dataset & templates

**Duration:** 4–5 weekday sessions + 1 weekend  
**Outcome:** You can describe how 300 entries are produced, what fields matter for training, and why unverified data never enters the train set.

### Concepts to learn
1. **Schema fields** (especially `optimization_explanation`, `verification.*`, `tolerance`)
2. **Template strategy:** 3 hand-written templates × 20 shapes = 60/category — tractable vs hand-writing 300 unique kernels
3. **Instruction-tuning projection:** description + pytorch_reference → kernel + explanation
4. **Filter pipeline:** generate → `verify_model_kernel` → `verified/dataset.jsonl` vs `raw/rejected.jsonl`
5. **Current status honesty:** 0/300 GPU-verified on this Mac because no CUDA — environment gate, not 300 kernel bugs

### Files to read (in order)
1. `data/SCHEMA.md` (full)
2. `data/examples/*.json` — three concrete examples
3. `data/templates/common.py`
4. Skim **one** full template file deeply (`norm.py` is easiest), then skim headers of the other four
5. `data/build_dataset.py`
6. `data/raw/generation_report.json` (what rejection looks like)

### Drills
- [ ] Open one JSON example and map every field to: “used at train time / used at verify time / provenance only.”
- [ ] Pick `norm` template: explain what the Triton kernel fuses and why that helps serving.
- [ ] Write the ID naming pattern from memory (e.g. `kv_cache__decode_attention_mha__b4_s1024`).
- [ ] Explain the difference between “syntax-checked via `ast.parse`” and “GPU-verified.”

### Must-answer interview questions
1. How did you get 300 examples without writing 300 kernels by hand?
2. Why is explanation part of the *training target*, not just a UI string?
3. What happens to a kernel that is fast but numerically wrong?
4. Why fp16 tolerances (~1e-2) vs fp32 (~1e-4)?

---

## Phase D — Shared library (`kernelforge/`)

**Duration:** 2–3 weekday sessions  
**Outcome:** You can explain the train/infer/parse contract the rest of the stack depends on.

### Files to read
1. `kernelforge/prompts.py` — system prompt + `kernel:` / `explanation:` format
2. `kernelforge/parsing.py` — how raw LLM text becomes structured fields
3. `kernelforge/dataset.py` — load / split / categories
4. `kernelforge/model.py` — `KernelGenerator`, Unsloth vs transformers fallback, 4-bit load

### Drills
- [ ] From memory: write the exact two-section output format the model must produce.
- [ ] Explain what breaks if parsing is fragile (demo + eval both depend on it).
- [ ] Why Unsloth on Linux+GPU and PEFT fallback on macOS?

### Must-answer interview questions
1. Walk me through one training example from schema dict → chat messages → tokenized text.
2. What model did you choose and why (size, free-tier T4, code specialization)?
3. How do you load a LoRA adapter at inference time?

---

## Phase E — Fine-tuning

**Duration:** 3–4 weekday sessions + half weekend  
**Outcome:** You can explain QLoRA, hyperparameters, and what the model is learning (code *and* explanations).

### Concepts to learn
1. **LoRA / QLoRA:** what `r`, `alpha`, `dropout` mean; why target q/k/v/o + MLP projections
2. **4-bit base + LoRA adapters** — memory math at a high level (fit T4)
3. **SFTTrainer** + chat template
4. **Split:** 85/10/5 train/val/test
5. **Hyperparams in** `training/configs/finetune_default.json` — know the numbers and *why* they’re conservative
6. **Failure mode:** generic/templated explanations — how you’d catch that (spot-check every epoch)

### Files to read
1. `training/finetune.py` (both Unsloth and transformers paths)
2. `training/configs/finetune_default.json`
3. `training/configs/qwen2.5-coder-1.5b.json`

### Drills
- [ ] Recite config values: model name, `lora_r=16`, `lora_alpha=32`, lr `2e-4`, epochs `3`, batch `1` × accum `8`.
- [ ] Explain effective batch size = 8.
- [ ] Why train on structured `kernel:` + `explanation:` instead of kernel-only?

### Must-answer interview questions
1. Why QLoRA instead of full fine-tune?
2. Why not RL (Kevin-style)? What’s the tradeoff?
3. How would you detect that the model is overfitting to template wording in explanations?
4. What would you change if quantized_matmul stayed near-zero correct after FT?

---

## Phase F — Evaluation

**Duration:** 3–4 weekday sessions  
**Outcome:** You can present category-level before/after and explain the explanation rubric.

### Concepts to learn
1. **Baseline first:** un-fine-tuned model establishes the “before”
2. **Same harness** for scoring generated kernels
3. **Category-level table** (not one accuracy number) — different ops have different difficulty
4. **Explanation rubric:** 0 / 1 / 2 on technique specificity
5. **Dry-run mode** for local pipeline testing without GPU/model

### Files to read
1. `evaluation/common.py`
2. `evaluation/baseline_eval.py`
3. `evaluation/finetuned_eval.py`
4. `evaluation/explanation_review.py`
5. Any file under `evaluation/results/` (format of reports)

### Drills
- [ ] Run: `python evaluation/baseline_eval.py --dry-run --device cpu` and explain what it exercises.
- [ ] Draw the empty category comparison table from memory and say what each cell means.
- [ ] Score 3 sample explanations yourself with the 0/1/2 rubric (use template explanations from `data/examples/`).

### Must-answer interview questions
1. Why break results down by category?
2. How do you evaluate explanations if there’s no BLEU/ROUGE that works?
3. What does “passed” mean in eval vs “model produced something that looks like Triton”?
4. What would a failed eval look like if parsing broke vs if the kernel was wrong?

---

## Phase G — Demo app

**Duration:** 2 weekday sessions  
**Outcome:** You can diagram the request path and say what is optional vs required.

### Files to read
1. `app/backend/main.py`
2. `app/frontend/gradio_app.py`

### Request path to memorize
```
Gradio UI → POST /generate
  → KernelGenerator.generate(description, pytorch_reference)
  → parse_model_output(raw)
  → (optional) verify_model_kernel(...)
  → {triton_kernel, optimization_explanation, verification}
```

Also know: `/health`, `/categories`, `/templates`, env vars `KERNELFORGE_MODEL_CONFIG`, `KERNELFORGE_ADAPTER_PATH`.

### Drills
- [ ] List every field on `GenerateRequest` / `GenerateResponse`.
- [ ] Explain why verification is a flag (CPU-only machines / latency).

### Must-answer interview questions
1. Where does the differentiating feature (explanation) show up in the UI/API?
2. How would you harden this for production (auth, rate limits, queue GPU jobs)?

---

## Phase H — End-to-end ownership (interview simulation)

**Duration:** 1–2 weekend sessions  
**Outcome:** You can give a 10-minute architecture talk and survive deep-dives.

### Whiteboard scripts (practice out loud)
1. **Problem → prior work → our gap → architecture** (5 min)
2. **Harness deep-dive** (5 min) — subprocess, contracts, seeding, timing
3. **Data → train → eval loop** (5 min)
4. **Tradeoffs & limitations** (3 min) — say them before they ask

### Self-test checklist (must all be “yes”)
- [ ] I can name all 5 categories and why each matters for serving.
- [ ] I can explain both verification contracts.
- [ ] I can explain why the harness was built *before* trusting the dataset.
- [ ] I can walk LoRA config and the train format.
- [ ] I can cite 4+ prior works and our two gaps vs them.
- [ ] I can describe current project status honestly (GPU verification pending on Mac).
- [ ] I can point to the exact file for: harness, dataset build, finetune, baseline eval, API.

### Mock interview prompt (give to a friend or AI)
> “Walk me through KernelForge. Then I’ll pick one file and ask you to explain a design decision. Then I’ll ask what you’d do if attention correctness stayed low after fine-tuning.”

---

## 3. Suggested weekly schedule

| Week | Focus | Done when… |
|---|---|---|
| 1 | Phase A + start B | Pitch + softmax fusion story solid; harness overview clear |
| 2 | Finish B + Phase C | Can teach the harness; schema + template strategy solid |
| 3 | Phases D + E | Train format + QLoRA story solid |
| 4 | Phases F + G + H | Full mock interview without notes |

**Daily micro-habit (15 min):** pick one “Must-answer” question and answer it out loud, then check against the file.

---

## 4. Hands-on “prove you own it” milestones

Do these in order; each produces evidence you actually ran/understood the system:

1. **Harness ownership:** `pytest tests/ -v` green; write 2–3 sentences on what each model-contract test protects.
2. **Template ownership:** pick one category; explain its 3 templates’ optimization techniques without reading notes.
3. **Pipeline ownership:** dry-run baseline eval; explain the JSON report shape.
4. **GPU milestone (Kaggle/Colab):** run `python data/build_dataset.py --device cuda` and interpret per-category pass rates (this is also the real project next step).
5. **Story ownership:** record yourself giving the 5-minute architecture pitch; re-record until clean.

---

## 5. Interview answer bank (short form)

**“What does KernelForge do?”**  
Op → Triton kernel + explanation, verified for correctness/speedup, specialized for LLM serving ops.

**“What’s novel?”**  
Not broad SOTA — data-efficient supervised specialization + explainability on serving kernels under solo compute.

**“How do you know a kernel is correct?”**  
Subprocess sandbox runs reference Model vs ModelNew on identically seeded inputs; `allclose` with dtype-aware tolerances; timeout/crash isolation.

**“How was the dataset built?”**  
Hand-authored templates × shape grids → harness filter → only verified triples train the model (kernel + explanation).

**“How did you fine-tune?”**  
Qwen2.5-Coder-1.5B-Instruct, QLoRA (r=16, α=32), Unsloth on T4, structured SFT output with kernel and explanation sections.

**“How do you measure success?”**  
Category-level % correct and avg speedup vs baseline; separate manual rubric for explanation quality.

**“What didn’t work / what’s blocked?”**  
Local Mac can’t run Triton; 300 candidates need CUDA verification. Vector-add style ops often don’t beat PyTorch — fusion is where wins appear.

**“What would you do next?”**  
GPU-verify dataset → fix low-pass templates → baseline → FT → category eval; later best-of-N via harness, `@triton.autotune`, wider grids.

---

## 6. File-order reading list (one continuous path)

If you only want a single linear reading order:

1. `README.md`
2. `docs/project_writeup.md`
3. `PROJECT PLAN.md` (Positioning → Phase 2 → Phase 4–5)
4. `DOCUMENTATION.md`
5. `examples/kernels/softmax/*`
6. `verification/verify_kernel.py` + `sandbox_runner.py`
7. `tests/test_verify_kernel.py`
8. `data/SCHEMA.md` + `data/examples/*`
9. `data/templates/common.py` + one category template
10. `data/build_dataset.py`
11. `kernelforge/prompts.py` → `parsing.py` → `dataset.py` → `model.py`
12. `training/finetune.py` + configs
13. `evaluation/*.py`
14. `app/backend/main.py` + `app/frontend/gradio_app.py`

---

## 7. Concepts cheat-sheet (keep short notes beside this)

| Topic | One-line rule |
|---|---|
| Fusion | Fewer launches + fewer HBM trips → speedup on memory-bound multi-op chains |
| Prefill vs decode | Full-seq attention vs single-token attn against KV cache |
| QLoRA | 4-bit base frozen; small low-rank adapters trained |
| Harness first | Same pass/fail definition for dataset and eval |
| Explanation | Trained output field, not post-hoc captioning |
| Scope | Serving ops, not generic KernelBench sweep |

---

## 8. After you’re interview-ready — project execution (don’t confuse with learning)

Learning ownership ≠ finishing GPU runs. Once Phases A–H are solid, the *project* next steps remain:

1. Kaggle/Colab: `python data/build_dataset.py --device cuda`
2. Fix templates with low pass rates; regenerate
3. `baseline_eval.py` → `finetune.py` → `finetuned_eval.py` → `explanation_review.py`
4. Fill the results table in `docs/project_writeup.md`
5. Demo with a real adapter path set

You can interview on architecture *before* those numbers exist — just be explicit that verification/metrics are pending GPU runs.

---

*Source of truth for implementation details: `DOCUMENTATION.md`. Source of truth for positioning: `docs/project_writeup.md`. This file is the study path that ties them to interview readiness.*
