# KernelForge — Master Plan
### A data-efficient, explainable LLM specialist for LLM-serving kernel optimization

This is the single plan for the project. It replaces `PROJECT PLAN.md` (build roadmap), `LEARNING_PLAN.md` (study path v1), and `KERNELFORGE_LEARNING_PLAN_v2.md` (UI-first resequencing). Part I is **what gets built**, Part II is **what you must own for interviews**, and Part III is **what to do next**.

*Source of truth for implementation details: [`DOCUMENTATION.md`](DOCUMENTATION.md). Source of truth for positioning: [`docs/project_writeup.md`](docs/project_writeup.md).*

---

## Status at a glance (Sep 29, 2026)

| Phase | Status |
|---|---|
| 0 — Foundations | ✅ Complete |
| 0.5 — Demo studio + kernel search engine | ✅ Built and tested locally; always-on Railway deployment (interpreter-only, hosted-model option, code policy + rate limits). Remaining: first live GPU run, demo GIF |
| 1 — Dataset | 🔄 All 15 templates fixed and **15/15 pass the Triton interpreter** (CPU, correctness only). Remaining: GPU run for real verification + speedups |
| 2 — Verification harness | ✅ Complete, GPU-validated. Now also has an interpreter mode and precise mismatch reports |
| 3 — Baseline | Script written, not run |
| 4 — Fine-tuning | Script written, no adapter trained |
| 5 — Evaluation | Scripts written, no results |
| 6 — Tool | Folded into 0.5; swap in the adapter after Phase 4 |
| 7 — Write-up | Draft |
| 8 — Efficiency & scaling | Inference-time best-of-N + repair done (pulled forward); the rest is post-v1 |

**What has not happened yet, stated plainly:** no GPU-verified dataset entries (`data/verified/dataset.jsonl` is empty), no baseline numbers, no fine-tuned adapter, no evaluation results. The interpreter pass means the kernels compute the right thing; it says nothing about speed.

**Tests:** 84 tests. 83 pass on macOS; the 84th (interpreter mode on a real Triton kernel) needs Triton and passes inside `docker/verify.Dockerfile`.

---

# Part I — Build roadmap

## Positioning (read this before anything else)

LLM-driven GPU kernel generation is an active, crowded research area. Prior work is better resourced than a solo student project can match (see **Prior work**). This project deliberately does **not** try to beat it on broad coverage or raw speedup. It targets gaps that are still open:

1. **Data-efficient specialization instead of broad, RL-heavy coverage.** Kevin-32B, AutoTriton and TritonRL rely on large-scale RL and heavy compute to chase generic operator coverage. Few ask how good a narrowly specialized model can get with a small, curated supervised dataset, zero RL, and a free-tier compute budget.
2. **Explainability.** Every surveyed system outputs a kernel and a benchmark number. None explain *why* the optimization works (which tiling, fusion, or memory-access decision was made). This is unclaimed and cheap to add.
3. **One verification harness, everywhere.** The same `verify_model_kernel` gates the training dataset, scores the evaluation, drives the demo's kernel search, and produces what the live demo shows. Nothing in the demo is presented as verified unless the harness ran.

**Scope narrowing:** instead of generic PyTorch ops (the KernelBench/TritonBench territory), the project covers **LLM-inference-serving kernels**, the ops that actually decide serving cost (relevant to Cohere's business and to NVIDIA's and Cerebras' efficiency focus). Fewer categories, deeper coverage, each annotated with an explanation.

**Say this in any write-up or interview:** *"a small, data-efficient model specialized for LLM-serving kernel patterns, with explanation generation and verifier-guided search, built without RL or large-scale compute — positioned against and citing prior work, not claiming to surpass it on raw benchmarks."*

## Prior work (cite in the write-up)

- **KernelBench** (Stanford Scaling Intelligence Lab, Feb 2025): the standard benchmark for LLM-generated GPU kernels; 250 PyTorch workloads. We adopt its `Model`/`ModelNew` contract.
- **TritonBench** (Li et al., Feb 2025): Triton-specific benchmark; frontier models peak below 24% correctness on real-world kernels, and speedup rarely exceeds 1.5×.
- **KernelLLM** (Fisches et al., 2025): 8B Llama-3.1 fine-tuned for PyTorch→Triton. The closest prior work to the original framing.
- **Kevin-32B** (Baronio et al., 2025): multi-turn RL for kernel self-refinement. Our repair loop uses the same multi-turn feedback idea **at inference time, without training**.
- **AutoTriton / TritonRL** (Li et al., Woo et al., 2025): RL-trained Triton models.
- **AI CUDA Engineer** (Sakana AI, 2025): agentic, training-free PyTorch→CUDA with iterative optimization.
- **Liger Kernel**: production hand-written Triton kernels for LLM training; prior art for the patterns we target.

None of these output a natural-language rationale. RL approaches need large domain-specific datasets and compute, so nobody reports what is achievable without that budget.

## Architecture

```
                 Web studio (app/web) — model badge, strategy controls, candidate timeline
                                   │ POST /generate/stream  (NDJSON events)
                                   ▼
            FastAPI (app/backend/main.py) — one GPU lock, verification LRU cache, /stats
                                   │
                                   ▼
        kernelforge/search.py — verifier-guided kernel search
        ┌──────────────────────────────────────────────────────────────────┐
        │ round 0: sample N candidates in one batched generate()           │
        │          (KernelGenerator.stream_chat, tokens tagged r0c0..r0cN) │
        │ parse each → verify each (identical kernels verified once)       │
        │ any pass? → select the fastest correct kernel                    │
        │ else      → best failure + harness error → next user turn        │
        │ round k:  repair (up to 2 rounds, context bounded to 1 attempt)  │
        └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
            verification/verify_kernel.py — subprocess sandbox
            GPU: seeded inputs · allclose · median CUDA-event timing · timeout
            Interpreter: TRITON_INTERPRET=1 on CPU · correctness only
                                   │
          ┌────────────────────────┼─────────────────────────┐
          ▼                        ▼                         ▼
   Dataset filter           Baseline / FT eval          Live demo
 (data/build_dataset.py)      (evaluation/)          (same function)
```

## Repo map

| Path | Role | Interview one-liner |
|---|---|---|
| `app/` | FastAPI + web studio + Kaggle launcher | "The harness that gates training data verifies what you see live." |
| `kernelforge/search.py` | Best-of-N + repair engine | "Training-free verifier-guided search; the model sees the exact mismatch and retries." |
| `kernelforge/` (rest) | Prompts, parsing, dataset I/O, model wrapper | "The library layer train, eval, and the app all consume." |
| `verification/` | Correctness + timing sandbox | "Subprocess isolation so bad kernels can't hang or crash the parent." |
| `data/` | Schema, 15 templates, dataset builder | "3 templates × 20 shapes × 5 categories = 300 candidates, each harness-filtered." |
| `docker/verify.Dockerfile` | Triton interpreter on CPU | "Correctness-check Triton kernels on a Mac with no GPU." |
| `training/` | QLoRA fine-tune | "Unsloth preferred; transformers + PEFT fallback." |
| `evaluation/` | Baseline, FT eval, explanation rubric | "Category-level metrics, not one aggregate number." |
| `examples/kernels/` | Hand-written Triton (Phase 0) | "Fusion beats eager multi-launch (softmax)." |

---

## Phase 0 — Foundations — ✅ Complete
Triton basics, KernelBench structure, dev environment (Kaggle GPU notebook, dependencies).

## Phase 0.5 — Demo studio + kernel search — ✅ Built, GPU run pending

**Why out of order:** the project needed to be demoable for an NVIDIA recruiting conversation before the dataset and fine-tuning finish. The harness was already GPU-validated, so a UI on the **base model + the real harness** shows the whole pipeline honestly today. Phase 6's tool is pulled forward here; the adapter swaps in via config. This changes *what gets built first*, not the dataset → baseline → fine-tune → eval order.

**Built:**
1. **Backend** (`app/backend/main.py`): `POST /generate` and `POST /generate/stream` both run the search engine. Also `POST /verify`, `GET /health`, `/stats`, `/categories`, `/templates`, `/templates/{id}`. Request fields: `num_candidates` (1–4), `repair_rounds` (0–2), `device` (`cuda` = GPU timing, `cpu` = Triton interpreter), tolerances. Base model by default; `KERNELFORGE_ADAPTER_PATH` switches to an adapter; `KERNELFORGE_MOCK_MODEL=1` for GPU-less UI testing.
2. **Search engine** (`kernelforge/search.py`): batched best-of-N sampling (`num_return_sequences` with a per-row streamer), per-candidate parse + verify, selection of the fastest correct kernel, and a repair loop that feeds the harness error back as the next turn. Failure ranking for the repair target: wrong numerics > harness crash > unparseable. No repair when verification was skipped (there's no signal to repair from).
3. **Efficiency:** identical kernels share one harness run within a search; a 128-entry LRU caches results across requests (keyed on reference + kernel + device + tolerances; timeouts and crashes aren't cached because they can be transient); `/stats` reports requests, pass rate, tokens/s, harness runs, cache hits, repairs that passed, and best speedup.
4. **Catalog:** all 15 templates, each with a realistic serving shape (GPU) and a small shape (interpreter), with status badges read from real artifacts (`data/raw/interpret_report.json`, `data/verified/dataset.jsonl`).
5. **Frontend** (`app/web/`): op library with status badges; strategy controls (Harness GPU/Interpreter, Samples 1–4, Repair off/1/2); a 5-step pipeline (GPU slot → Generate → Parse → Verify → Select); a Kernel search card showing every candidate per round with its verdict, speedup, and dedup link, plus the exact feedback sent to the model; per-candidate inspection; server stats; session history. It switches to the interpreter automatically when the server has no GPU.
6. **Honesty:** persistent base-model badge; interpreter passes are labelled "correctness only, not GPU-verified"; mock mode has a red banner.
7. **Launch:** `python app/launch.py --share` (Kaggle/Colab, Cloudflare quick tunnel) or `--mock` locally.

**Acceptance criteria**
- [ ] `python app/launch.py --share` on a Kaggle T4 produces a working public link with the base model loaded.
- [ ] A generate with Samples 4 + Repair 1 on GPU returns a real verdict + speedup per candidate.
- [x] Harness failures (error/timeout/crash/incorrect) show as readable results, never as a UI crash.
- [x] The base-model badge is always visible, including when the backend is unreachable.
- [x] `python app/launch.py --mock` works on macOS with no GPU.
- [x] Full search path (15 templates, 2 candidates each, interpreter harness) verified end to end in Docker.
- [x] `pytest tests/` passes.
- [ ] Demo GIF at `docs/assets/demo.gif`, honestly labelled as the base model.

## Phase 1 — Dataset — 🔄 Templates fixed, GPU verification pending

**1.1 Task (locked):** input a PyTorch reference for an LLM-serving op; output (a) a Triton kernel and (b) a 1–3 sentence explanation of the optimization.

**1.2 Categories (locked at 5):**
- `quantized_matmul`: W8A16 per-tensor, W8A16 per-channel, W8A8
- `attention`: causal self-attention, non-causal cross-attention, GQA (prefill)
- `kv_cache`: decode-step attention (MHA, GQA), KV-cache write
- `norm`: RMSNorm, LayerNorm, fused residual-add + RMSNorm
- `rope`: rotate-half, interleaved-pair, fused Q+K

60 candidates per category (3 templates × 20 shapes), the low end of the 60–150 target; widen after the GPU run.

**1.3 Schema:** see [`data/SCHEMA.md`](data/SCHEMA.md). Fields follow KernelBench (`pytorch_reference` / `triton_kernel`), plus `optimization_explanation`, `dtype`, `tolerance`, `test_shapes`, `provenance`, and a nested `verification` object. Three reviewed examples live in `data/examples/`.

**1.4 Sourcing:** the template library plus `data/build_dataset.py` generates every entry and runs it through `verify_model_kernel`. Only passing entries enter `dataset.jsonl`; the rest go to `data/raw/rejected.jsonl`.

**Template fixes (done Sep 29, 2026):**
- Attention/GQA references mixed an fp32 mask into an fp16 matmul, so the *reference itself* raised. They now use a boolean `triu` mask with `masked_fill`.
- The W8A8 reference used an int32 matmul that CUDA doesn't support. It now matmuls int8 values in fp32 (exact).
- Attention kernels were whole-sequence single blocks. They're now FlashAttention-style: query-block programs, KV tiles, online softmax with exp2, tensor-core `tl.dot`, and causal/GQA loops that stop at the diagonal.
- Decode attention is now a tiled loop with online softmax (flash-decoding style). GQA packs a whole query group per KV head into one `tl.dot`.
- W8A16 dequantizes to fp16 before `tl.dot`, matching the reference's rounding. RoPE processes 16-row tiles. Norms pick `num_warps` from the block size.
- `is_cuda` asserts were removed so the interpreter can run them.

**Local verification:** `python data/build_dataset.py --interpret --smoke --jobs 8` in the Docker image runs one small instance per template: **15/15 pass**. A negative check confirms the gate is real: an off-by-one causal mask fails with "output contains NaN", and a missing per-channel scale fails with max abs error 1416.

**Deliverable:** `data/verified/dataset.jsonl` with 300+ GPU-verified triples and a per-category pass/speedup report.

## Phase 2 — Verification harness — ✅ Complete, GPU-validated

`verification/verify_kernel.py` + `sandbox_runner.py`:
- **Function contract** (`verify_kernel`): bare `solution(*tensors)`, used by the Phase 0 examples.
- **Model contract** (`verify_model_kernel`): KernelBench `Model`/`ModelNew` + `get_inputs`/`get_init_inputs`. It seeds the RNG before *each* construction, which ops with random internal state (quantized weights) need.
- **Interpreter mode** (`interpret=True`): runs with `TRITON_INTERPRET=1` on CPU; returns correctness with `interpreted=True` and no timings.
- **Mismatch reports:** for example "6599/20480 elements outside atol=0.0001, rtol=0.0001; max abs error 0.001953 at index (1, 1, 1, 33)". Also reports shape, output-count, and NaN failures. These are what the repair loop feeds back to the model.

Manually GPU-validated with a real vector-add kernel: correct, 0.48× (expected: memory-bound with nothing to fuse).

## Phase 3 — Baseline — script written (`evaluation/baseline_eval.py`), not run
Candidates (all fit a T4 with 4-bit + LoRA): Qwen2.5-Coder 1.5B/7B (default), DeepSeek-Coder 1.3B/6.7B, StarCoder2 3B. Record per category: % correct, average speedup, and whether the base model produces any explanation at all.

**New:** also report **pass@1 vs best-of-4 vs best-of-4 + 1 repair**, using `kernelforge/search.py` with the same harness. This quantifies how much the training-free search buys before any fine-tuning.

**Deliverable:** baseline report broken down by the 5 categories.

## Phase 4 — Fine-tuning — script written (`training/finetune.py`), not trained
QLoRA via Unsloth, or transformers + PEFT + bitsandbytes 4-bit on Kaggle, where Unsloth fights the preinstalled torch. The training target contains both sections (`kernel:` … `explanation:` …). Split by template: one held-out op per category for test, 10% of the remaining shapes for val. LoRA r=16, α=32, lr 2e-4, 3 epochs, batch 1 × accumulation 8. Expect 2–4 iterations; spot-check explanations every time for generic boilerplate. Once an adapter exists: `python app/launch.py --share --adapter <path>` flips the badge automatically.

**Deliverable:** LoRA adapter + training logs.

## Phase 5 — Evaluation — scripts written, no results

| Category | Base % correct | Base avg speedup | FT % correct | FT avg speedup | FT best-of-4 + repair % correct |
|---|---|---|---|---|---|
| Quantized matmul | | | | | |
| Attention | | | | | |
| KV-cache | | | | | |
| Norm | | | | | |
| RoPE | | | | | |

Explanations are scored separately with a manual 0/1/2 rubric on 20–30 samples (`evaluation/explanation_review.py`). Report weak categories and failure modes as they are.

**Deliverable:** category table + rubric review + 3–5 concrete good and bad examples.

## Phase 6 — Tool — folded into Phase 0.5
Remaining work: swap in the fine-tuned adapter and record the before/after.

## Phase 7 — Write-up
README structure: problem → prior work → narrow contribution → dataset method → category results → search results → explanation review → limitations → future work. Put the honest framing in the first paragraph.

Limitations to state: small model and dataset vs 8B–32B RL systems; 5 narrow categories; no RL (search is inference-time only); explanations scored by manual rubric; the interpreter checks correctness only; demo-grade sandbox.

## Phase 8 — Efficiency & scaling (post-v1)
- **Done early:** inference-time best-of-N + harness-feedback repair, dedupe, and a result cache.
- **Next best step:** rejection-sampling fine-tuning. Run the search over the training prompts, keep kernels that pass *and* beat a speedup threshold, and fine-tune on them. A lightweight alternative to RL.
- Teach `@triton.autotune`-decorated kernels instead of fixed block sizes.
- More fused multi-op examples; that's where speedups come from.
- Serving the model: merge LoRA, quantize (AWQ/GPTQ), serve via vLLM or TensorRT-LLM.

---

# Part II — Learning track (own every layer)

**Goal:** explain architecture, tradeoffs, failure modes, and code paths without notes — not just "I used AI to build it."

**Time budget (university first):** weekdays 30–45 min (one concept block or one drill); one weekend deep session of 1.5–3 h (one phase + whiteboard practice). About 4–6 weeks to interview-ready.

**Rule:** reading alone won't stick. Do every drill and answer every must-answer question out loud before checking the file.

## The 60-second pitch (memorize first)

> KernelForge is a **data-efficient, explainable LLM specialist** for **LLM-serving GPU kernels**. Given a PyTorch reference op, it generates an optimized **Triton** kernel **plus** an explanation of *why* the optimization helps. A **subprocess-isolated verification harness** on KernelBench's `Model`/`ModelNew` contract checks correctness and speedup. The same harness gates the training data, scores the eval, and drives a **training-free kernel search**: sample several candidates, keep the fastest correct one, or feed the exact mismatch back to the model for a repair attempt. There's a working demo on the base model today, with fine-tuning in progress on the same pipeline. It's deliberately narrow — five serving categories, supervised QLoRA, **no RL** — and is positioned against KernelLLM / Kevin-32B / AutoTriton, not as a claim of beating them.

**Say these limitations unprompted:** the demo runs the base model; 1.5B model and ~300-example target vs 8B–32B RL systems; narrow categories; no RL; explanations are rubric-scored; Triton needs Linux + NVIDIA (the Mac does the interpreter check only, and GPU work runs on Kaggle/Colab).

## Phase A — Foundations (GPU, Triton, why serving ops)
**Outcome:** whiteboard fusion vs eager, memory-bound vs compute-bound, and why the 5 categories matter.

- **Concepts:** HBM vs SRAM, launch overhead, warps/blocks; `@triton.jit`, `tl.load`/`tl.store`, masks, `program_id`, grids; fusion; prefill vs decode; the prior-work names in one sentence each.
- **Read:** `docs/project_writeup.md`; Positioning and Prior work above; `examples/kernels/{vector_add,relu,softmax}/` (softmax is the fusion story); `data/SCHEMA.md` scope table.
- **Drills:**
  - [ ] Draw eager softmax (max → sub → exp → sum → div) vs one fused pass.
  - [ ] Explain why vector add can be slower in Triton.
  - [ ] For each category: when it runs (prefill/decode) and what cost it hits.
- **Must answer:** Why Triton over CUDA? Why not all of KernelBench? Prefill vs decode attention? What is memory-bandwidth bound, and how does fusion help?

## Phase B — Verification harness (the spine)
**Outcome:** walk `verify_model_kernel` → `sandbox_runner` line by line and defend each choice.

- **Concepts:** the two contracts; subprocess isolation (hangs → timeout, crashes isolated, optional memory limit); same seed, separate tensors (catches in-place bugs); seeding before each construction; CUDA events + median after warmup; `allclose` on float casts with `equal_nan=False`; `status` vs `passed`; interpreter mode and why it can't time anything; mismatch messages as repair signal.
- **Read:** `DOCUMENTATION.md` harness section; `verification/verify_kernel.py`; `verification/sandbox_runner.py`; `tests/test_verify_kernel.py`.
- **Drills:**
  - [ ] `pytest tests/ -v` and explain every test.
  - [ ] Trace `verify_model_kernel(ref, cand)` → temp files → JSON payload → subprocess → JSON line → `VerifyResult`.
  - [ ] List the four statuses and their causes.
  - [ ] Explain why seeding only before `get_inputs()` breaks the quantized templates.
  - [ ] Run the interpreter check in Docker and break a template on purpose to see the mismatch.
- **Must answer:** Why not in-process? Why median? Why KernelBench's contract? How do you stop a kernel hanging the eval for hours? `status="ok", correct=False` vs `status="error"`? A stranger's browser triggers generated-code execution — what actually protects you? (Subprocess + timeout + a temporary session you own; not a hardened sandbox. Don't overclaim.)

## Phase C — Dataset & templates
**Outcome:** describe how 300 entries are made, which fields train the model, and why unverified data never enters training.

- **Concepts:** schema fields; 3 templates × 20 shapes; instruction-tuning projection; generate → verify → `dataset.jsonl` / `rejected.jsonl`; the template bugs and how they were found; interpreter smoke vs GPU verification.
- **Read:** `data/SCHEMA.md`; `data/examples/*.json`; `data/templates/common.py`; `norm.py` deeply, then `attention.py` (FlashAttention tiling); `data/build_dataset.py`.
- **Drills:**
  - [ ] Map every example field to train time / verify time / provenance only.
  - [ ] Explain what the norm kernel fuses.
  - [ ] Explain online softmax with exp2 in the attention kernel.
  - [ ] Write the ID pattern from memory (`kv_cache__decode_attention_mha__b4_s1024`).
  - [ ] Explain why the attention reference itself was broken, and how you'd have caught it earlier.
- **Must answer:** 300 examples without 300 hand-written kernels? Why is the explanation a training target? What happens to a fast but wrong kernel? Why fp16 tolerance ~1e-2?

## Phase D — Shared library (`kernelforge/`)
**Outcome:** explain the train/infer/parse contract. Every function here is on a live request path.

- **Read:** `prompts.py` (output format + repair feedback), `parsing.py`, `dataset.py`, `model.py` (`KernelGenerator`, `stream_chat`, `BatchTextStreamer`), `search.py`.
- **Drills:**
  - [ ] Write the two-section output format from memory.
  - [ ] Explain what breaks if parsing is fragile.
  - [ ] Trace one repair conversation: which turns are kept, and why only the latest attempt.
  - [ ] Explain how one `generate()` call streams 4 sequences.
- **Must answer:** Schema dict → chat messages → tokens? Model choice and why? Loading a LoRA adapter at inference? Why rank wrong-numerics failures above crashes for repair?

## Phase E — Fine-tuning
**Outcome:** explain QLoRA, the hyperparameters, and that the model learns code *and* explanations.

- **Concepts:** LoRA r/α/dropout; target q/k/v/o + MLP; 4-bit base; SFTTrainer + chat template; held-out-template split (why a random shape split leaks); generic-explanation failure mode.
- **Read:** `training/finetune.py`; `training/configs/finetune_default.json`; `training/configs/qwen2.5-coder-1.5b.json`.
- **Drills:**
  - [ ] Recite the config (r=16, α=32, lr 2e-4, 3 epochs, batch 1 × accumulation 8 = effective 8).
  - [ ] Explain why the target is structured.
- **Must answer:** Why QLoRA? Why not RL, and what does the inference-time search give you instead? How would you detect template-wording overfitting? What if quantized_matmul stays near zero after fine-tuning?

## Phase F — Evaluation
**Outcome:** present the category before/after, the search gain, and the rubric.

- **Read:** `evaluation/common.py`, `baseline_eval.py`, `finetuned_eval.py`, `explanation_review.py`.
- **Drills:**
  - [ ] `python evaluation/baseline_eval.py --dry-run --device cpu`, and explain what it exercises.
  - [ ] Draw the category table from memory.
  - [ ] Score 3 explanations with the rubric.
- **Must answer:** Why by category? How do you evaluate explanations without BLEU? "Passed" vs "looks like Triton"? Parse failure vs wrong kernel in the report? Why is pass@1 vs best-of-4 a fair comparison only with the same harness?

## Phase G — Demo + search engine
**Outcome:** diagram the request path and the search loop; say what's optional.

- **Read:** `app/backend/main.py`, `app/backend/catalog.py`, `kernelforge/search.py`, `app/web/app.js`, `app/launch.py`, `tests/test_app.py`, `tests/test_search.py`.
- **Request path:** UI → `POST /generate/stream` → GPU lock → `run_search` → `stream_chat` (N sequences) → parse → verify (cache/dedupe) → select or repair → NDJSON events → UI.
- **Drills:**
  - [ ] List the `GenerateRequest` fields and the stream events.
  - [ ] Explain what happens when the tab closes mid-generation (generation stops before the lock is released).
  - [ ] Explain why verification is a flag.
  - [ ] Explain when the cache is and isn't used, and why timeouts aren't cached.
  - [ ] Explain what the UI shows for unparseable base-model output.
- **Must answer:** "Can I see it?" Why the base model before fine-tuning? What's identical between demo verification and dataset verification? How would you productionize (auth, rate limits, a job queue, multi-GPU)?

## Phase H — End-to-end ownership (interview simulation)
**Whiteboard scripts (out loud):**
1. Problem → prior work → gaps → architecture (5 min).
2. Live demo (2–3 min): run Samples 4 + Repair 1, narrate the candidate timeline, and point at the base-model badge unprompted.
3. Harness deep-dive (5 min): "this is the same code the demo just called."
4. Data → train → eval loop and honest status (5 min).
5. Tradeoffs and limitations (3 min).

**Self-test (all must be yes):**
- [ ] I can name the 5 categories and why each matters for serving.
- [ ] I can explain both contracts and interpreter mode.
- [ ] I can explain why the harness came before trusting the dataset.
- [ ] I can walk the LoRA config and the training format.
- [ ] I can explain the search engine's selection and repair rules.
- [ ] I can cite 4+ prior works and our gaps against them.
- [ ] I can state the current status honestly.
- [ ] I can point to the file for the harness, dataset build, fine-tuning, baseline, search, and API.

**Mock prompt:** "Walk me through KernelForge. Then I'll pick one file and ask about a design decision. Then: what would you do if attention correctness stayed low after fine-tuning?"

## Weekly schedule (build + learn, university first)

| Week | Build | Learn | Done when |
|---|---|---|---|
| **0 (now)** | Kaggle GPU run: demo link + GIF; `build_dataset.py --device cuda` | Pitch + Phase G | Public link works; dataset report exists |
| 1 | Fix any low-pass templates on GPU; widen shapes if needed | Phase A + start B | Softmax fusion story solid |
| 2 | Baseline eval incl. best-of-4 + repair | Finish B + Phase C | Can teach the harness; baseline table filled |
| 3 | Fine-tune; flip the demo badge | Phases D + E | Adapter trained; FT story solid |
| 4 | Fine-tuned eval + rubric; write-up | Phases F + H | Mock interview without notes, live demo included |

**Daily micro-habit (15 min):** one must-answer question out loud, then check it against the file.

## Interview answer bank

- **What does it do?** Op → Triton kernel + explanation, verified for correctness and speed, specialized for LLM-serving ops. Live demo on the base model, fine-tuning in progress.
- **Can I see it?** Yes. [Link, or the GIF.] The harness that gates the training data verifies what you see.
- **What's novel?** Not broad SOTA: data-efficient supervised specialization + explainability on serving kernels under solo compute, with one harness used for dataset, eval, search, and demo.
- **How do you know a kernel is correct?** Subprocess sandbox runs `Model` vs `ModelNew` on identically seeded inputs; `allclose` with dtype-aware tolerances; timeout and crash isolation.
- **How do you get better kernels without RL?** Best-of-N sampling filtered by the harness, then repair turns that show the model the exact mismatch or traceback. Kevin-32B trains that multi-turn loop with RL; we run it at inference time.
- **How was the dataset built?** Hand-written templates × shape grids → harness filter → only verified triples train the model.
- **How did you verify templates without a GPU?** Triton's interpreter in Docker: correctness on small shapes. It found real reference bugs. Speed still needs the GPU run.
- **How did you fine-tune?** Qwen2.5-Coder-1.5B-Instruct, QLoRA r=16 α=32, Unsloth on a T4, structured output. [Say "in progress" or "done" as appropriate.]
- **How do you measure success?** Per category: % correct and average speedup vs baseline, pass@1 vs search, and a manual explanation rubric.
- **What didn't work?** The attention/GQA references and the W8A8 int matmul were broken; single-block attention didn't scale. Found them, fixed them at the template level, and re-verified. Vector-add-style ops don't beat PyTorch; fusion is where the wins are.
- **What next?** GPU-verify the dataset → baseline with search → fine-tune → category eval → rejection-sampling fine-tuning from search outputs → autotune.

## Reading order (one linear path)
1. `README.md`
2. `app/backend/main.py`, `kernelforge/search.py`, `app/web/app.js`, `app/launch.py`
3. `docs/project_writeup.md`
4. This file (Part I)
5. `DOCUMENTATION.md`
6. `examples/kernels/softmax/*`
7. `verification/verify_kernel.py` + `sandbox_runner.py`
8. `tests/test_verify_kernel.py`, `tests/test_search.py`, `tests/test_app.py`
9. `data/SCHEMA.md` + `data/examples/*`
10. `data/templates/common.py` + `norm.py` + `attention.py`
11. `data/build_dataset.py`
12. `kernelforge/prompts.py` → `parsing.py` → `dataset.py` → `model.py`
13. `training/finetune.py` + configs
14. `evaluation/*.py`

## Cheat-sheet

| Topic | One-line rule |
|---|---|
| Fusion | Fewer launches + fewer HBM round-trips → speedup on memory-bound chains |
| Prefill vs decode | Full-sequence attention vs one query against the KV cache |
| Online softmax | Running max + running sum; rescale partial results instead of recomputing |
| QLoRA | Frozen 4-bit base + small trained low-rank adapters |
| Harness first | Same pass/fail definition for dataset, eval, search, and demo |
| Interpreter | Correct on CPU ≠ fast on GPU; never report it as GPU-verified |
| Search | Best-of-N + harness-feedback repair; select the fastest correct kernel |
| Explanation | A trained output field, not a caption added afterwards |
| Scope | Serving ops, not a generic KernelBench sweep |
| Demo honesty | Base model until the adapter is flipped in; say so visibly |

---

# Part III — Immediate next steps

**0. One-step GPU session (recommended):** import `notebooks/kaggle_gpu_session.ipynb` into Kaggle (GPU T4, Internet on) and *Run All*. It runs `scripts/gpu_session.py`:
1. Checks the GPU environment and runs the harness tests on the real GPU.
2. Runs `build_dataset.py --device cuda`.
3. Runs a baseline eval on held-out templates.
4. With `FINETUNE = True`, also runs a QLoRA fine-tune plus its evaluation.

Every artifact (dataset, reports, logs, adapter) ends up in `/kaggle/working/kernelforge_outputs.zip`. Local pre-flight (Oct 1, 2026, Docker interpreter, all 300 grid shapes):
- **261/300 pass and 0 are incorrect.**
- The other 39 were killed for running out of memory (8 parallel jobs in an 8 GB Docker VM). They are the largest shapes: attention/RoPE at batch 8–16 × sequence 2048, and norms at 16k–65k rows.
- On a 16 GB T4 the largest references peak at roughly 9–12 GB, so a few may still run out of memory. The dataset report now lists `oom` separately from incorrect kernels (per category and per template), so a memory limit isn't mistaken for a broken template.

**Evaluation split (changed Oct 1, 2026):** train/val/test is now **split by template**. One whole template per category is held out (100 test entries; 180 train, 20 val). A random shape-level split put the same kernel, with different shape constants, in both train and test, so a fine-tuned model would score by memorisation. `--split random` still exists for comparison, but report the template split.

**1. Ship the live demo (Kaggle, GPU T4, Internet on):**
```bash
git clone https://github.com/Arsalan-05/KernelForge.git && cd KernelForge
pip install -q -r requirements-demo.txt
python app/launch.py --share
```
In the studio: pick an op, set Harness GPU, Samples 4, Repair 1, and generate. Record `docs/assets/demo.gif` and put the link in the README.

**2. GPU-verify the dataset (same session):**
```bash
python data/build_dataset.py --device cuda
```
This fills `data/verified/dataset.jsonl` and prints per-category pass/speedup. If a template's pass rate is low, fix the template (not individual shapes) and rerun.

**3. Local correctness check before any GPU session (Mac, Docker):**
```bash
docker build -f docker/verify.Dockerfile -t kernelforge-verify .
docker run --rm -v "$PWD":/work kernelforge-verify python data/build_dataset.py --interpret --smoke --jobs 8
```

**4. Then:** baseline (with search) → fine-tune → evaluation → write-up, per Part I.

You can interview on architecture before the numbers exist. Just say explicitly that GPU verification and metrics are pending.
