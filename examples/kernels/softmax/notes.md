# softmax

Row-wise softmax over a 2D tensor (`dim=-1`), one Triton program per row.

**Why the fused kernel is faster:** `torch.softmax` eager still runs as
several kernel launches under the hood (max reduce -> subtract -> exp ->
sum reduce -> divide), each one reading and writing the full row to global
memory. The Triton version loads each row into on-chip memory once, does
max, subtract, exp, sum, and divide entirely there, and writes the result
once — one read + one write per row instead of ~5. This is the first
example in the repo where fusion (not just a hand-rolled launch) is the
actual source of the speedup, and it's memory-traffic-bound so the win
grows as rows get wider (more redundant round-trips saved).
