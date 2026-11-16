# Citation verification log

Every entry in `references.bib`, the source it was checked against (2026-11-15), and the
**only** claims the paper attributes to it. If you add a claim about a cited work, re-read
the source first and add the claim here.

| Key | Checked against | What the paper says about it (and nothing more) |
|---|---|---|
| `ouyang2025kernelbench` | arxiv.org/abs/2502.10517; PMLR v267 (ICML 2025), pp. 47356–47415 | 250 PyTorch workloads; `fast_p` metric (fraction correct *and* faster than a threshold p); frontier reasoning models match the PyTorch baseline in under 20% of cases; execution/profiling feedback helps. We reuse its `Model`/`ModelNew` + `get_inputs`/`get_init_inputs` convention. |
| `li2025tritonbench` | arxiv.org/abs/2502.14752 (abstract + HTML body); ACL Anthology 2025.findings-acl.1183 | Two channels: 184 real-world GitHub operators (TritonBench-G) and PyTorch-aligned operators (TritonBench-T). Best execution accuracy 23.91% on G, 53.01% on T; best speedup among correct operators 1.56× on G and 1.91× on T. **Not** "speedup rarely exceeds 1.5×" (see UNVERIFIED.md). |
| `fisches2025kernelllm` | huggingface.co/facebook/KernelLLM (model card, its own `@software` citation) | 8B, based on Llama 3.1 Instruct; SFT on ~25k PyTorch–Triton pairs plus synthetic samples (KernelBook; TheStack + `torch.compile`); 10 epochs, 16 GPUs, ~12 h; KernelBench-Triton Level 1 on H100: 20.2 pass@1, 51.8 pass@10. No natural-language explanation output is described. |
| `baronio2025kevin` | arxiv.org/abs/2507.11948 | "First model trained with multi-turn RL for CUDA kernel generation"; base QwQ-32B; correctness 56%→82%, mean speedup 0.53×→1.10× vs PyTorch eager; serial refinement scales better than parallel sampling. **Generates CUDA, not Triton.** |
| `li2025autotriton` | arxiv.org/abs/2507.05687 | 8B Triton model; SFT on a curated data pipeline then GRPO RL with rule-based + execution-based rewards; evaluated on TritonBench and KernelBench channels; comparable to Claude-4-Sonnet and DeepSeek-R1-0528 (authors' claim). |
| `woo2025tritonrl` | arxiv.org/abs/2510.17891 (v1 Oct 2025, v2 Feb 2026); listed as a poster at an ICML 2026 workshop (cited as arXiv) | 8B Triton model; SFT then RL with verifiable rewards; multi-layered verification and Hierarchical Reward Decomposition to counter reward hacking; KernelBench results reported as state of the art among similar-size Triton models (authors' claim). |
| `liu2026drkernel` | arxiv.org/abs/2602.05885 | Dr. Kernel-14B; KernelGYM environment with reward-hacking checks; multi-turn RL (TRLOO); profiling-based rewards and rejection sampling. |
| `wang2025geak` | arxiv.org/abs/2507.23194 | AMD agentic (training-free) Triton generation framework with generation/evaluation/reflection/optimization agents; targets MI300X/MI250; up to 63% correctness and up to 2.59× speedup on its two benchmarks. |
| `lange2025robustkbench` | arxiv.org/abs/2509.14279; sakana.ai/ai-cuda-engineer | Sakana AI. Agentic PyTorch→CUDA pipeline with evolutionary meta-generation and LLM-based verifiers; argues existing kernel benchmarks have "exploitable loopholes"; introduces robust-kbench. Uses frontier LLMs at test time, no fine-tuning mentioned. |
| `hsu2024liger` | arxiv.org/abs/2410.10989 | Open-source Triton kernels for LLM *training*; fusion and input chunking; ~20% throughput, ~60% memory reduction vs HuggingFace implementations. |
| `chen2023selfdebug` | arxiv.org/abs/2304.05128 | Few-shot prompting teaches an LLM to debug its own code from execution results. |
| `shinn2023reflexion` | arxiv.org/abs/2303.11366; NeurIPS 2023 proceedings page confirmed 2026-11-16: papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html | Agents improve by verbally reflecting on task feedback, without weight updates. NeurIPS 2023 author list: Shinn, Cassano, Gopinath, Narasimhan, Yao (5 authors; Berman present in arXiv v1 but not in proceedings). |
| `hu2022lora` | arxiv.org/abs/2106.09685; ICLR 2022 virtual site | Frozen weights + trainable low-rank matrices per layer. |
| `dettmers2023qlora` | arxiv.org/abs/2305.14314; NeurIPS 2023 proceedings | Backprop through a frozen 4-bit model into LoRA adapters; NF4, double quantization, paged optimizers. |
| `unsloth2023` | github.com/unslothai/unsloth (README citation block) | Software for faster/lower-memory fine-tuning. Cited as software. |
| `hui2024qwen25coder` | arxiv.org/abs/2409.12186 | Qwen2.5-Coder series, 0.5B–32B. We use the 1.5B Instruct checkpoint. 24 authors; bib lists first three + "others". |
| `tillet2019triton` | Harvard PDF + PLDI'19 MAPL page; DOI 10.1145/3315508.3329973 | The Triton language/compiler. |
| `dao2022flashattention` | arxiv.org/abs/2205.14135; NeurIPS 2022 proceedings | IO-aware exact attention via tiling between HBM and SRAM. |
| `dao2024flashattention2` | arxiv.org/abs/2307.08691; ICLR 2024 virtual site | Better work partitioning; ~2× over FlashAttention. |
| `dao2023flashdecoding` | crfm.stanford.edu/2023/10/12/flashdecoding.html; pytorch.org/blog/flash-decoding | Decode attention split along the KV sequence, with a log-sum-exp reduction across splits. **Our decode templates do NOT implement the split-KV pass** (see `data/templates/kv_cache.py`). |
| `milakov2018online` | arxiv.org/abs/1805.02867 | Online (single-pass) softmax normalizer. |
| `ainslie2023gqa` | arxiv.org/abs/2305.13245 (comment: "Accepted at EMNLP 2023") | Grouped-query attention. |
| `su2021roformer` | arxiv.org/abs/2104.09864 | Rotary position embedding. Journal version not verified, so arXiv is cited. |
| `zhang2019rmsnorm` | arxiv.org/abs/1910.07467 (comment: "NeurIPS 2019") | RMSNorm. |
