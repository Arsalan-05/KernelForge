from . import attention, kv_cache, norm, quantized_matmul, rope

GENERATORS = {
    "quantized_matmul": quantized_matmul.generate,
    "attention": attention.generate,
    "kv_cache": kv_cache.generate,
    "norm": norm.generate,
    "rope": rope.generate,
}

# One small instance per template (15 total), sized for Triton's CPU interpreter.
SMOKE_GENERATORS = {
    "quantized_matmul": quantized_matmul.generate_smoke,
    "attention": attention.generate_smoke,
    "kv_cache": kv_cache.generate_smoke,
    "norm": norm.generate_smoke,
    "rope": rope.generate_smoke,
}
