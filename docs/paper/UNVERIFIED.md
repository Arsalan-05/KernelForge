# Not verified — do not cite (yet)

Things that came up while writing the paper but could not be confirmed from a primary source on
2026-11-15. None of them appear in `paper.tex` or `references.bib`. To promote one, read the
primary source, add it to `references.bib`, and log it in `SOURCES.md`.

## Claims in MASTER_PLAN.md / project_writeup.md that the sources do not support

| Claim in repo docs | What the source actually says | Action |
|---|---|---|
| TritonBench: "speedup rarely exceeds 1.5×" (MASTER_PLAN, Prior work) | The paper reports the **best** speedup among correct operators: 1.56× (TritonBench-G) and 1.91× (TritonBench-T). It makes no "rarely exceeds" claim. | Paper uses the source's numbers. Suggest fixing MASTER_PLAN wording. |
| TritonBench: "frontier models peak below 24% correctness on real-world kernels" | Roughly right: the best execution accuracy on TritonBench-G is 23.91% (GPT-o1, one-shot). On TritonBench-T it is 53.01%. | The paper gives both numbers, not just the lower one. |
| Kevin-32B is a Triton / "kernel self-refinement" model (implied by grouping) | Kevin generates **CUDA** kernels (base QwQ-32B, multi-turn RL). | The paper says CUDA explicitly. |
| Kevin-32B is "by Cognition" | The arXiv page lists authors but no affiliations in the abstract view. A Cognition blog post exists by reputation but I did not read it. | The paper attributes it to Baronio et al. only. |
| "AutoTriton / TritonRL (Li et al., Woo et al.)" are RL-only | Both use **SFT followed by RL**. | The paper describes both stages. |
| "AI CUDA Engineer (Sakana AI, Feb 2025)" | The original Feb 2025 report (title "The AI CUDA Engineer: Agentic CUDA Kernel Discovery, Optimization and Composition", authors Lange et al.) was seen only through search snippets. Its PDF URL returned 404. | Cite the verified successor, arXiv:2509.14279, instead. |
| "None of these output a natural-language rationale" | I checked the abstracts and model cards only. Some of these systems produce reasoning traces (e.g. TritonRL's "think", Kevin's reasoning model base), which are not the same as a deliberate, evaluated explanation field, but I have not read every paper in full. | The paper says "none of the works we reviewed *evaluates* a natural-language explanation as an output," and the owner should re-check before posting. |

## Works not cited because they were not verified

- **Sakana AI reward-hacking incident (Feb 2025):** reported informally that some AI CUDA Engineer kernels exploited the evaluation harness. Not verified from a primary source. Relevant to our "harness-gated" framing; verify before mentioning.
- **KernelBook dataset (GPUMODE):** named on the KernelLLM card. No separate paper read.
- **KernelBench-v2, TritonForge:** mentioned in `data/SCHEMA.md`. Not checked.
- **CUDA-L1** and other CUDA-RL work: not checked.
- **Venues not confirmed:** Self-Debugging (possibly ICLR 2024), Reflexion (possibly NeurIPS 2023), RoFormer (possibly Neurocomputing 2024). Cited as arXiv preprints only.
- **Rejection-sampling fine-tuning / STaR / RFT** for the Future-work section: not checked. Future work describes the idea without a citation.
- **Hardware facts about the T4** (Turing, no native bf16, 16 GB, FP16 tensor-core peak): not cited from an NVIDIA datasheet. The paper only states "16 GB" and "fp16 training as configured". Add an NVIDIA datasheet citation if more hardware detail is needed.
