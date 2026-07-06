# vector_add

Elementwise `x + y` over a 1D tensor.

**Why the fused kernel is faster:** it isn't, really — this op is fully
memory-bandwidth-bound (one add per three memory accesses: load x, load y,
store out), so PyTorch eager already gets close to peak bandwidth. This is
the "hello world" kernel: it exists to prove out the launch/grid/mask
mechanics (`program_id`, `arange`, boundary `mask`) that every other kernel
in this repo reuses, not to demonstrate a speedup.
