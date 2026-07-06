from . import attention, kv_cache, norm, quantized_matmul, rope

GENERATORS = {
    "quantized_matmul": quantized_matmul.generate,
    "attention": attention.generate,
    "kv_cache": kv_cache.generate,
    "norm": norm.generate,
    "rope": rope.generate,
}
