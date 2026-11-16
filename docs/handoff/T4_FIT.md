# T4 Fit Analysis — Memory Estimates (Estimate Only)

> **Disclaimer:** All figures are *estimates* based on tensor-size arithmetic. T4 fit is untested;
> actual GPU memory will differ due to CUDA context overhead, Triton JIT scratch buffers, driver
> reservations, and allocator fragmentation. These estimates assume a fresh CUDA context with no
> other processes resident on the GPU.

## Summary

- T4 VRAM: **16 GB**
- Largest kernel tensor footprint: **4.50 GB** (`norm/rmsnorm_residual`, rows=65536, hidden=12288)
- All 300 template×shape combinations fit T4 **standalone** (kernel tensors only)
- All 300 fit T4 **with fine-tuning overhead** (~2.5 GB for QLoRA model + LoRA adapters + optimizer + activations)
- All 300 fit T4 **with generation overhead** (~2.0 GB for quantized model + KV cache)
- The 15 interpreter OOMs all hit the **8 GB CPU RAM cap** of the Docker VM, not a GPU limit.
  On a T4 (16 GB VRAM), all 15 OOM entries are estimated to fit comfortably.

## Model Overhead Estimates

| Scenario | Overhead | Basis | Remaining for kernels |
|---|---|---|---|
| **Fine-tuning** (QLoRA, Qwen2.5-Coder-1.5B) | ~2.5 GB | 4-bit weights ≈ 1.1 GB + LoRA adapters ≈ 0.1 GB + optimizer states ≈ 0.2 GB + activations (seq=8192) ≈ 0.7 GB + CUDA overhead ≈ 0.4 GB | ~13.5 GB |
| **Generation** (inference, 4-bit) | ~2.0 GB | 4-bit weights ≈ 1.1 GB + KV cache (28L×2KV×4096 tokens) ≈ 0.1 GB + CUDA overhead ≈ 0.3 GB + margin ≈ 0.5 GB | ~14.0 GB |
| **Verification only** (no model) | ~0.3 GB | CUDA context + Triton JIT | ~15.7 GB |

*Qwen2.5-Coder-1.5B-Instruct: 1.54B params, 28 layers, hidden=1536, GQA 12/2 heads, head_dim=128.*

## Memory Formula (Kernel Tensors Only)

Peak kernel memory counts only the input and output tensors. Flash-attention kernels do not
materialise the full S×S score matrix; the O(S·D) intermediate state lives in SRAM.

| Template | Tensors counted | Formula |
|---|---|---|
| `attention/causal_prefill` | q, k, v, out | 4 × B×H×S×D × fp16 |
| `attention/cross_noncausal` | q, k, v, out | (2×B×H×Sq + 2×B×H×Skv) × D × fp16; Sq = max(64, Skv/4) |
| `attention/gqa_causal_prefill` | q, k, v, out | (2×B×Hq + 2×B×Hkv) × S × D × fp16 |
| `norm/rmsnorm`, `norm/layernorm` | input, output | 2 × rows × hidden × fp16 |
| `norm/rmsnorm_residual` | input, residual, output | 3 × rows × hidden × fp16 |
| `rope/rotate_half`, `rope/interleaved` | q_in, q_out | 2 × B×H×S×D × fp16 |
| `rope/fused_qk` | q_in, k_in, q_out, k_out | 4 × B×H×S×D × fp16 |
| `kv_cache/decode_attention_mha` | q, k_cache, v_cache, out | (2×B×H×S + 2×B×H) × D × fp16 |
| `kv_cache/decode_attention_gqa` | q, k_cache, v_cache, out | (2×B×Hkv×S + 2×B×Hq) × D × fp16 |
| `kv_cache/cache_write` | cache_k, cache_v, new_k, new_v | (2×B×H×S + 2×B×H) × D × fp16 |
| `quantized_matmul/w8a16_*` | activation fp16, weight int8, output fp16 | B×in×fp16 + out×in×int8 + B×out×fp16 |
| `quantized_matmul/w8a8` | activation int8, weight int8, output fp16 | B×in×int8 + out×in×int8 + B×out×fp16 |

Constants: H=32 heads, D=128 head_dim, Hq=32, Hkv=8 (GQA).

## Per-Template × Shape Memory Table

**Fits FT** = fits T4 with fine-tuning model loaded (~2.5 GB overhead).
**Fits Gen** = fits T4 with generation model loaded (~2.0 GB overhead).
**Interp OOM** = crashed exit -9 on 8 GB CPU-RAM Docker VM (not a GPU limit).

### `attention/causal_prefill`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `b1_s256` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b1_s512` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b1_s1024` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b1_s2048` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b2_s256` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b2_s512` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s1024` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b2_s2048` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b4_s256` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b4_s512` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b4_s1024` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b4_s2048` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b8_s256` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b8_s512` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s1024` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b8_s2048` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |
| `b16_s256` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b16_s512` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s1024` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |
| `b16_s2048` | 1024.0 | 1.000 | ✓ | ✓ | ✓ | ⚠ OOM |

*attention/causal_prefill: 20 shapes. Kernel range: 0.008–1.000 GB. Fits FT: 20/20. Interpreter OOMs: 1/20.*

### `attention/cross_noncausal`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `b1_skv256` | 5.0 | 0.005 | ✓ | ✓ | ✓ | — |
| `b1_skv512` | 10.0 | 0.010 | ✓ | ✓ | ✓ | — |
| `b1_skv1024` | 20.0 | 0.020 | ✓ | ✓ | ✓ | — |
| `b1_skv2048` | 40.0 | 0.039 | ✓ | ✓ | ✓ | — |
| `b2_skv256` | 10.0 | 0.010 | ✓ | ✓ | ✓ | — |
| `b2_skv512` | 20.0 | 0.020 | ✓ | ✓ | ✓ | — |
| `b2_skv1024` | 40.0 | 0.039 | ✓ | ✓ | ✓ | — |
| `b2_skv2048` | 80.0 | 0.078 | ✓ | ✓ | ✓ | — |
| `b4_skv256` | 20.0 | 0.020 | ✓ | ✓ | ✓ | — |
| `b4_skv512` | 40.0 | 0.039 | ✓ | ✓ | ✓ | — |
| `b4_skv1024` | 80.0 | 0.078 | ✓ | ✓ | ✓ | — |
| `b4_skv2048` | 160.0 | 0.156 | ✓ | ✓ | ✓ | — |
| `b8_skv256` | 40.0 | 0.039 | ✓ | ✓ | ✓ | — |
| `b8_skv512` | 80.0 | 0.078 | ✓ | ✓ | ✓ | — |
| `b8_skv1024` | 160.0 | 0.156 | ✓ | ✓ | ✓ | — |
| `b8_skv2048` | 320.0 | 0.312 | ✓ | ✓ | ✓ | — |
| `b16_skv256` | 80.0 | 0.078 | ✓ | ✓ | ✓ | — |
| `b16_skv512` | 160.0 | 0.156 | ✓ | ✓ | ✓ | — |
| `b16_skv1024` | 320.0 | 0.312 | ✓ | ✓ | ✓ | — |
| `b16_skv2048` | 640.0 | 0.625 | ✓ | ✓ | ✓ | — |

*attention/cross_noncausal: 20 shapes. Kernel range: 0.005–0.625 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

### `attention/gqa_causal_prefill`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `b1_s256` | 5.0 | 0.005 | ✓ | ✓ | ✓ | — |
| `b1_s512` | 10.0 | 0.010 | ✓ | ✓ | ✓ | — |
| `b1_s1024` | 20.0 | 0.020 | ✓ | ✓ | ✓ | — |
| `b1_s2048` | 40.0 | 0.039 | ✓ | ✓ | ✓ | — |
| `b2_s256` | 10.0 | 0.010 | ✓ | ✓ | ✓ | — |
| `b2_s512` | 20.0 | 0.020 | ✓ | ✓ | ✓ | — |
| `b2_s1024` | 40.0 | 0.039 | ✓ | ✓ | ✓ | — |
| `b2_s2048` | 80.0 | 0.078 | ✓ | ✓ | ✓ | — |
| `b4_s256` | 20.0 | 0.020 | ✓ | ✓ | ✓ | — |
| `b4_s512` | 40.0 | 0.039 | ✓ | ✓ | ✓ | — |
| `b4_s1024` | 80.0 | 0.078 | ✓ | ✓ | ✓ | — |
| `b4_s2048` | 160.0 | 0.156 | ✓ | ✓ | ✓ | — |
| `b8_s256` | 40.0 | 0.039 | ✓ | ✓ | ✓ | — |
| `b8_s512` | 80.0 | 0.078 | ✓ | ✓ | ✓ | — |
| `b8_s1024` | 160.0 | 0.156 | ✓ | ✓ | ✓ | — |
| `b8_s2048` | 320.0 | 0.312 | ✓ | ✓ | ✓ | — |
| `b16_s256` | 80.0 | 0.078 | ✓ | ✓ | ✓ | — |
| `b16_s512` | 160.0 | 0.156 | ✓ | ✓ | ✓ | — |
| `b16_s1024` | 320.0 | 0.312 | ✓ | ✓ | ✓ | — |
| `b16_s2048` | 640.0 | 0.625 | ✓ | ✓ | ✓ | ⚠ OOM |

*attention/gqa_causal_prefill: 20 shapes. Kernel range: 0.005–0.625 GB. Fits FT: 20/20. Interpreter OOMs: 1/20.*

### `norm/rmsnorm`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `r1024_h2048` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `r1024_h4096` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `r1024_h8192` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `r1024_h11008` | 43.0 | 0.042 | ✓ | ✓ | ✓ | — |
| `r1024_h12288` | 48.0 | 0.047 | ✓ | ✓ | ✓ | — |
| `r4096_h2048` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `r4096_h4096` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `r4096_h8192` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `r4096_h11008` | 172.0 | 0.168 | ✓ | ✓ | ✓ | — |
| `r4096_h12288` | 192.0 | 0.188 | ✓ | ✓ | ✓ | — |
| `r16384_h2048` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `r16384_h4096` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `r16384_h8192` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |
| `r16384_h11008` | 688.0 | 0.672 | ✓ | ✓ | ✓ | — |
| `r16384_h12288` | 768.0 | 0.750 | ✓ | ✓ | ✓ | — |
| `r65536_h2048` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |
| `r65536_h4096` | 1024.0 | 1.000 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h8192` | 2048.0 | 2.000 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h11008` | 2752.0 | 2.688 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h12288` | 3072.0 | 3.000 | ✓ | ✓ | ✓ | ⚠ OOM |

*norm/rmsnorm: 20 shapes. Kernel range: 0.008–3.000 GB. Fits FT: 20/20. Interpreter OOMs: 4/20.*

### `norm/layernorm`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `r1024_h2048` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `r1024_h4096` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `r1024_h8192` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `r1024_h11008` | 43.0 | 0.042 | ✓ | ✓ | ✓ | — |
| `r1024_h12288` | 48.0 | 0.047 | ✓ | ✓ | ✓ | — |
| `r4096_h2048` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `r4096_h4096` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `r4096_h8192` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `r4096_h11008` | 172.0 | 0.168 | ✓ | ✓ | ✓ | — |
| `r4096_h12288` | 192.0 | 0.188 | ✓ | ✓ | ✓ | — |
| `r16384_h2048` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `r16384_h4096` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `r16384_h8192` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |
| `r16384_h11008` | 688.0 | 0.672 | ✓ | ✓ | ✓ | — |
| `r16384_h12288` | 768.0 | 0.750 | ✓ | ✓ | ✓ | — |
| `r65536_h2048` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |
| `r65536_h4096` | 1024.0 | 1.000 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h8192` | 2048.0 | 2.000 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h11008` | 2752.0 | 2.688 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h12288` | 3072.0 | 3.000 | ✓ | ✓ | ✓ | ⚠ OOM |

*norm/layernorm: 20 shapes. Kernel range: 0.008–3.000 GB. Fits FT: 20/20. Interpreter OOMs: 4/20.*

### `norm/rmsnorm_residual`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `r1024_h2048` | 12.0 | 0.012 | ✓ | ✓ | ✓ | — |
| `r1024_h4096` | 24.0 | 0.023 | ✓ | ✓ | ✓ | — |
| `r1024_h8192` | 48.0 | 0.047 | ✓ | ✓ | ✓ | — |
| `r1024_h11008` | 64.5 | 0.063 | ✓ | ✓ | ✓ | — |
| `r1024_h12288` | 72.0 | 0.070 | ✓ | ✓ | ✓ | — |
| `r4096_h2048` | 48.0 | 0.047 | ✓ | ✓ | ✓ | — |
| `r4096_h4096` | 96.0 | 0.094 | ✓ | ✓ | ✓ | — |
| `r4096_h8192` | 192.0 | 0.188 | ✓ | ✓ | ✓ | — |
| `r4096_h11008` | 258.0 | 0.252 | ✓ | ✓ | ✓ | — |
| `r4096_h12288` | 288.0 | 0.281 | ✓ | ✓ | ✓ | — |
| `r16384_h2048` | 192.0 | 0.188 | ✓ | ✓ | ✓ | — |
| `r16384_h4096` | 384.0 | 0.375 | ✓ | ✓ | ✓ | — |
| `r16384_h8192` | 768.0 | 0.750 | ✓ | ✓ | ✓ | — |
| `r16384_h11008` | 1032.0 | 1.008 | ✓ | ✓ | ✓ | — |
| `r16384_h12288` | 1152.0 | 1.125 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h2048` | 768.0 | 0.750 | ✓ | ✓ | ✓ | — |
| `r65536_h4096` | 1536.0 | 1.500 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h8192` | 3072.0 | 3.000 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h11008` | 4128.0 | 4.031 | ✓ | ✓ | ✓ | ⚠ OOM |
| `r65536_h12288` | 4608.0 | 4.500 | ✓ | ✓ | ✓ | ⚠ OOM |

*norm/rmsnorm_residual: 20 shapes. Kernel range: 0.012–4.500 GB. Fits FT: 20/20. Interpreter OOMs: 5/20.*

### `rope/rotate_half`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `b1_s256` | 4.0 | 0.004 | ✓ | ✓ | ✓ | — |
| `b1_s512` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b1_s1024` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b1_s2048` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s256` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b2_s512` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b2_s1024` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s2048` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b4_s256` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b4_s512` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b4_s1024` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b4_s2048` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s256` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b8_s512` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b8_s1024` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s2048` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s256` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b16_s512` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b16_s1024` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s2048` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |

*rope/rotate_half: 20 shapes. Kernel range: 0.004–0.500 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

### `rope/interleaved`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `b1_s256` | 4.0 | 0.004 | ✓ | ✓ | ✓ | — |
| `b1_s512` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b1_s1024` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b1_s2048` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s256` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b2_s512` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b2_s1024` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s2048` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b4_s256` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b4_s512` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b4_s1024` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b4_s2048` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s256` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b8_s512` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b8_s1024` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s2048` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s256` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b16_s512` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b16_s1024` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s2048` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |

*rope/interleaved: 20 shapes. Kernel range: 0.004–0.500 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

### `rope/fused_qk`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `b1_s256` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b1_s512` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b1_s1024` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b1_s2048` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b2_s256` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b2_s512` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s1024` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b2_s2048` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b4_s256` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b4_s512` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b4_s1024` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b4_s2048` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b8_s256` | 64.0 | 0.062 | ✓ | ✓ | ✓ | — |
| `b8_s512` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s1024` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b8_s2048` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |
| `b16_s256` | 128.0 | 0.125 | ✓ | ✓ | ✓ | — |
| `b16_s512` | 256.0 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s1024` | 512.0 | 0.500 | ✓ | ✓ | ✓ | — |
| `b16_s2048` | 1024.0 | 1.000 | ✓ | ✓ | ✓ | — |

*rope/fused_qk: 20 shapes. Kernel range: 0.008–1.000 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

### `kv_cache/decode_attention_mha`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `b1_s256` | 4.0 | 0.004 | ✓ | ✓ | ✓ | — |
| `b1_s512` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b1_s1024` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b1_s2048` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s256` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b2_s512` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b2_s1024` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s2048` | 64.0 | 0.063 | ✓ | ✓ | ✓ | — |
| `b4_s256` | 16.1 | 0.016 | ✓ | ✓ | ✓ | — |
| `b4_s512` | 32.1 | 0.031 | ✓ | ✓ | ✓ | — |
| `b4_s1024` | 64.1 | 0.063 | ✓ | ✓ | ✓ | — |
| `b4_s2048` | 128.1 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s256` | 32.1 | 0.031 | ✓ | ✓ | ✓ | — |
| `b8_s512` | 64.1 | 0.063 | ✓ | ✓ | ✓ | — |
| `b8_s1024` | 128.1 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s2048` | 256.1 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s256` | 64.2 | 0.063 | ✓ | ✓ | ✓ | — |
| `b16_s512` | 128.2 | 0.125 | ✓ | ✓ | ✓ | — |
| `b16_s1024` | 256.2 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s2048` | 512.2 | 0.500 | ✓ | ✓ | ✓ | — |

*kv_cache/decode_attention_mha: 20 shapes. Kernel range: 0.004–0.500 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

### `kv_cache/decode_attention_gqa`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `b1_s256` | 1.0 | 0.001 | ✓ | ✓ | ✓ | — |
| `b1_s512` | 2.0 | 0.002 | ✓ | ✓ | ✓ | — |
| `b1_s1024` | 4.0 | 0.004 | ✓ | ✓ | ✓ | — |
| `b1_s2048` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b2_s256` | 2.0 | 0.002 | ✓ | ✓ | ✓ | — |
| `b2_s512` | 4.0 | 0.004 | ✓ | ✓ | ✓ | — |
| `b2_s1024` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b2_s2048` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b4_s256` | 4.1 | 0.004 | ✓ | ✓ | ✓ | — |
| `b4_s512` | 8.1 | 0.008 | ✓ | ✓ | ✓ | — |
| `b4_s1024` | 16.1 | 0.016 | ✓ | ✓ | ✓ | — |
| `b4_s2048` | 32.1 | 0.031 | ✓ | ✓ | ✓ | — |
| `b8_s256` | 8.1 | 0.008 | ✓ | ✓ | ✓ | — |
| `b8_s512` | 16.1 | 0.016 | ✓ | ✓ | ✓ | — |
| `b8_s1024` | 32.1 | 0.031 | ✓ | ✓ | ✓ | — |
| `b8_s2048` | 64.1 | 0.063 | ✓ | ✓ | ✓ | — |
| `b16_s256` | 16.2 | 0.016 | ✓ | ✓ | ✓ | — |
| `b16_s512` | 32.2 | 0.031 | ✓ | ✓ | ✓ | — |
| `b16_s1024` | 64.2 | 0.063 | ✓ | ✓ | ✓ | — |
| `b16_s2048` | 128.2 | 0.125 | ✓ | ✓ | ✓ | — |

*kv_cache/decode_attention_gqa: 20 shapes. Kernel range: 0.001–0.125 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

### `kv_cache/cache_write`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `b1_s256` | 4.0 | 0.004 | ✓ | ✓ | ✓ | — |
| `b1_s512` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b1_s1024` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b1_s2048` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s256` | 8.0 | 0.008 | ✓ | ✓ | ✓ | — |
| `b2_s512` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `b2_s1024` | 32.0 | 0.031 | ✓ | ✓ | ✓ | — |
| `b2_s2048` | 64.0 | 0.063 | ✓ | ✓ | ✓ | — |
| `b4_s256` | 16.1 | 0.016 | ✓ | ✓ | ✓ | — |
| `b4_s512` | 32.1 | 0.031 | ✓ | ✓ | ✓ | — |
| `b4_s1024` | 64.1 | 0.063 | ✓ | ✓ | ✓ | — |
| `b4_s2048` | 128.1 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s256` | 32.1 | 0.031 | ✓ | ✓ | ✓ | — |
| `b8_s512` | 64.1 | 0.063 | ✓ | ✓ | ✓ | — |
| `b8_s1024` | 128.1 | 0.125 | ✓ | ✓ | ✓ | — |
| `b8_s2048` | 256.1 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s256` | 64.2 | 0.063 | ✓ | ✓ | ✓ | — |
| `b16_s512` | 128.2 | 0.125 | ✓ | ✓ | ✓ | — |
| `b16_s1024` | 256.2 | 0.250 | ✓ | ✓ | ✓ | — |
| `b16_s2048` | 512.2 | 0.500 | ✓ | ✓ | ✓ | — |

*kv_cache/cache_write: 20 shapes. Kernel range: 0.004–0.500 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

### `quantized_matmul/w8a16_pertensor`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `attn_proj_b1` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `attn_proj_b8` | 16.1 | 0.016 | ✓ | ✓ | ✓ | — |
| `attn_proj_b32` | 16.5 | 0.016 | ✓ | ✓ | ✓ | — |
| `attn_proj_b128` | 18.0 | 0.018 | ✓ | ✓ | ✓ | — |
| `mlp_up_b1` | 43.0 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_up_b8` | 43.2 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_up_b32` | 43.9 | 0.043 | ✓ | ✓ | ✓ | — |
| `mlp_up_b128` | 46.7 | 0.046 | ✓ | ✓ | ✓ | — |
| `mlp_down_b1` | 43.0 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_down_b8` | 43.2 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_down_b32` | 43.9 | 0.043 | ✓ | ✓ | ✓ | — |
| `mlp_down_b128` | 46.7 | 0.046 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b1` | 48.0 | 0.047 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b8` | 48.2 | 0.047 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b32` | 49.0 | 0.048 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b128` | 52.0 | 0.051 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b1` | 64.0 | 0.063 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b8` | 64.2 | 0.063 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b32` | 65.0 | 0.063 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b128` | 68.0 | 0.066 | ✓ | ✓ | ✓ | — |

*quantized_matmul/w8a16_pertensor: 20 shapes. Kernel range: 0.016–0.066 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

### `quantized_matmul/w8a16_perchannel`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `attn_proj_b1` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `attn_proj_b8` | 16.1 | 0.016 | ✓ | ✓ | ✓ | — |
| `attn_proj_b32` | 16.5 | 0.016 | ✓ | ✓ | ✓ | — |
| `attn_proj_b128` | 18.0 | 0.018 | ✓ | ✓ | ✓ | — |
| `mlp_up_b1` | 43.0 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_up_b8` | 43.2 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_up_b32` | 43.9 | 0.043 | ✓ | ✓ | ✓ | — |
| `mlp_up_b128` | 46.7 | 0.046 | ✓ | ✓ | ✓ | — |
| `mlp_down_b1` | 43.0 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_down_b8` | 43.2 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_down_b32` | 43.9 | 0.043 | ✓ | ✓ | ✓ | — |
| `mlp_down_b128` | 46.7 | 0.046 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b1` | 48.0 | 0.047 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b8` | 48.2 | 0.047 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b32` | 49.0 | 0.048 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b128` | 52.0 | 0.051 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b1` | 64.0 | 0.063 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b8` | 64.2 | 0.063 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b32` | 65.0 | 0.063 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b128` | 68.0 | 0.066 | ✓ | ✓ | ✓ | — |

*quantized_matmul/w8a16_perchannel: 20 shapes. Kernel range: 0.016–0.066 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

### `quantized_matmul/w8a8`

| Shape ID | Kernel MB | Kernel GB | Fits T4 | Fits FT | Fits Gen | Interp OOM |
|---|---:|---:|:---:|:---:|:---:|:---:|
| `attn_proj_b1` | 16.0 | 0.016 | ✓ | ✓ | ✓ | — |
| `attn_proj_b8` | 16.1 | 0.016 | ✓ | ✓ | ✓ | — |
| `attn_proj_b32` | 16.4 | 0.016 | ✓ | ✓ | ✓ | — |
| `attn_proj_b128` | 17.5 | 0.017 | ✓ | ✓ | ✓ | — |
| `mlp_up_b1` | 43.0 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_up_b8` | 43.2 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_up_b32` | 43.8 | 0.043 | ✓ | ✓ | ✓ | — |
| `mlp_up_b128` | 46.2 | 0.045 | ✓ | ✓ | ✓ | — |
| `mlp_down_b1` | 43.0 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_down_b8` | 43.1 | 0.042 | ✓ | ✓ | ✓ | — |
| `mlp_down_b32` | 43.6 | 0.043 | ✓ | ✓ | ✓ | — |
| `mlp_down_b128` | 45.3 | 0.044 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b1` | 48.0 | 0.047 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b8` | 48.2 | 0.047 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b32` | 48.9 | 0.048 | ✓ | ✓ | ✓ | — |
| `qkv_fused_b128` | 51.5 | 0.050 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b1` | 64.0 | 0.063 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b8` | 64.2 | 0.063 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b32` | 64.8 | 0.063 | ✓ | ✓ | ✓ | — |
| `attn_proj_large_b128` | 67.0 | 0.065 | ✓ | ✓ | ✓ | — |

*quantized_matmul/w8a8: 20 shapes. Kernel range: 0.016–0.065 GB. Fits FT: 20/20. Interpreter OOMs: 0/20.*

## Notes on Interpreter OOMs vs T4

The Triton **interpreter** emulates GPU execution in Python/numpy on CPU RAM. It allocates
auxiliary buffers (loop intermediates, masked loads, broadcasting temporaries) that do not
exist in compiled GPU kernels. As a result, interpreter memory consumption is typically
5–15× higher than actual compiled VRAM usage.

All 15 OOM entries hit the 8 GB Docker VM cap in CPU RAM. Their estimated GPU tensor
footprints range from **1.5 GB** (`norm/rmsnorm_residual r65536 h4096`) to **4.5 GB**
(`norm/rmsnorm_residual r65536 h12288`) — well within the 16 GB T4 budget even with the
model loaded for generation (~2 GB overhead → peak budget 14 GB).

The two attention OOM entries (`causal_prefill b16 s2048` and `gqa_causal_prefill b16 s2048`)
have estimated kernel footprints of only **0.536 GB** and **0.268 GB** respectively —
almost certainly runnable on T4. Their interpreter OOM is an artefact of the 8 GB VM cap,
not a real memory constraint.

---

*Generated: 2026-11-15. T4 fit untested — these are estimates only.*