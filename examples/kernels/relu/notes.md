# relu

Elementwise `max(x, 0)`.

**Why the fused kernel is faster:** same as `vector_add` — memory-bound, not
a speed demo. What it adds over `vector_add` is a second Triton language
primitive (`tl.maximum`) and confirms the same launch pattern generalizes to
any pure elementwise op. Real wins from ReLU only show up when it's *fused*
into a neighboring op (e.g. bias + ReLU, or matmul + ReLU) so the
intermediate never round-trips through global memory — that's the pattern
`bias_relu`-style fused examples in the dataset are meant to demonstrate.
