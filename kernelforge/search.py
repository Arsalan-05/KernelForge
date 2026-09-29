"""Verifier-guided kernel search: best-of-N sampling + harness-feedback repair.

Round 0 samples `num_candidates` completions in one batched generation. Every
parseable candidate goes through the verification harness (identical kernels
share one run). If any candidate passes, the fastest correct one wins.
Otherwise the most promising failure -- wrong numerics beat crashes, crashes
beat unparseable output -- is fed back to the model with the harness's error
as the next user turn, for up to `repair_rounds` more rounds.

Everything model- or harness-specific is injected, so the same loop drives the
demo backend, the mock generator in tests, and (later) pass@k evaluation.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from kernelforge.parsing import parse_model_output
from kernelforge.prompts import build_messages, build_repair_messages

# generate(messages, num_sequences, temperature) -> iterator of (sequence_index, text_chunk)
GenerateFn = Callable[[list, int, Optional[float]], Iterator[tuple]]
# verify(kernel) -> dict with at least ran/status/correct/passed/speedup/error
VerifyFn = Callable[[str], dict]
# skip_reason(kernel or None) -> None when the harness should run
SkipReasonFn = Callable[[Optional[str]], Optional[str]]

SAMPLING_TEMPERATURE = 0.7


@dataclass
class SearchConfig:
    num_candidates: int = 1
    repair_rounds: int = 0
    base_temperature: Optional[float] = None

    def temperature(self, n: int) -> Optional[float]:
        # Best-of-N is pointless if every sample is near-greedy.
        if n > 1:
            return max(self.base_temperature or 0.0, SAMPLING_TEMPERATURE)
        return self.base_temperature


@dataclass
class Candidate:
    id: str
    round: int
    index: int
    raw_output: str = ""
    status: str = "pending"  # ok | unparseable
    triton_kernel: Optional[str] = None
    optimization_explanation: Optional[str] = None
    verification: Optional[dict] = None
    duplicate_of: Optional[str] = None

    @property
    def passed(self) -> bool:
        return bool(self.verification and self.verification.get("passed"))

    def rank(self) -> tuple:
        """Higher is better. Used to pick the winner and the repair target."""
        v = self.verification or {}
        if self.passed:
            return (4, v.get("speedup") or 0.0)
        if v.get("status") == "ok" and v.get("correct") is False:
            return (3, 0.0)
        if self.status == "ok" and not v.get("ran", False):
            return (2, 0.0)  # parsed but not verified (verification skipped)
        if self.status == "ok":
            return (1, 0.0)  # harness error / timeout / crash
        return (0, 0.0)

    def summary(self) -> dict:
        return {
            "candidate": self.id,
            "round": self.round,
            "status": self.status,
            "triton_kernel": self.triton_kernel,
            "optimization_explanation": self.optimization_explanation,
            "raw_output": self.raw_output,
            "verification": self.verification,
            "duplicate_of": self.duplicate_of,
        }


@dataclass
class SearchStats:
    """Counters for one search, merged into server-wide stats by the caller."""

    candidates: int = 0
    tokens: int = 0
    generation_time_s: float = 0.0
    verifications_run: int = 0
    verification_cache_hits: int = 0
    rounds: int = 0
    extra: dict = field(default_factory=dict)


def kernel_key(pytorch_reference: str, kernel: str, *parts) -> str:
    h = hashlib.sha256()
    for piece in (pytorch_reference, kernel, *map(str, parts)):
        h.update(piece.encode())
        h.update(b"\0")
    return h.hexdigest()


def run_search(
    description: str,
    pytorch_reference: str,
    config: SearchConfig,
    generate: GenerateFn,
    verify: VerifyFn,
    skip_reason: SkipReasonFn,
    count_tokens: Callable[[str], int],
    stats: Optional[SearchStats] = None,
) -> Iterator[dict]:
    """Yield NDJSON-ready event dicts. Closing the iterator closes the active generation."""
    stats = stats or SearchStats()
    candidates: list[Candidate] = []
    verified_by_kernel: dict[str, Candidate] = {}
    messages = build_messages(description, pytorch_reference)
    repair_target: Optional[Candidate] = None
    search_start = time.perf_counter()

    for round_idx in range(config.repair_rounds + 1):
        n = config.num_candidates if round_idx == 0 else 1
        if repair_target is not None:
            messages = build_repair_messages(
                description, pytorch_reference, repair_target.raw_output,
                repair_target.status, repair_target.verification,
            )
        round_event = {
            "event": "round",
            "round": round_idx,
            "kind": "sample" if round_idx == 0 else "repair",
            "num_candidates": n,
            "candidates": [f"r{round_idx}c{i}" for i in range(n)],
        }
        if repair_target is not None:
            round_event["repairing"] = repair_target.id
            round_event["feedback"] = messages[-1]["content"]
        yield round_event
        stats.rounds += 1

        round_cands = [Candidate(id=f"r{round_idx}c{i}", round=round_idx, index=i) for i in range(n)]
        candidates.extend(round_cands)
        chunks: list[list[str]] = [[] for _ in range(n)]
        gen_start = time.perf_counter()
        stream = generate(messages, n, config.temperature(n))
        try:
            for index, text in stream:
                chunks[index].append(text)
                yield {"event": "token", "candidate": round_cands[index].id, "text": text}
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        gen_time = time.perf_counter() - gen_start

        num_tokens = 0
        for cand, parts in zip(round_cands, chunks):
            cand.raw_output = "".join(parts).strip()
            num_tokens += count_tokens(cand.raw_output) if cand.raw_output else 0
        stats.candidates += n
        stats.tokens += num_tokens
        stats.generation_time_s += gen_time
        yield {
            "event": "generated",
            "round": round_idx,
            "generation_time_s": round(gen_time, 3),
            "num_tokens": num_tokens,
            "tokens_per_s": round(num_tokens / gen_time, 1) if gen_time > 0 else None,
        }

        for cand in round_cands:
            parsed = parse_model_output(cand.raw_output)
            cand.status = "ok" if parsed.has_kernel else "unparseable"
            cand.triton_kernel = parsed.triton_kernel if parsed.has_kernel else None
            # Without a kernel, the parser's "explanation" is just the raw text.
            cand.optimization_explanation = parsed.optimization_explanation if parsed.has_kernel else None
            yield {
                "event": "parsed",
                "candidate": cand.id,
                "status": cand.status,
                "triton_kernel": cand.triton_kernel,
                "optimization_explanation": cand.optimization_explanation,
                "raw_output": cand.raw_output,
            }

            reason = skip_reason(cand.triton_kernel)
            if reason:
                cand.verification = {"ran": False, "skipped_reason": reason}
            elif cand.triton_kernel in verified_by_kernel:
                original = verified_by_kernel[cand.triton_kernel]
                cand.duplicate_of = original.id
                cand.verification = {**original.verification, "cached": True}
                stats.verification_cache_hits += 1
            else:
                yield {"event": "verifying", "candidate": cand.id}
                cand.verification = verify(cand.triton_kernel)
                if cand.verification.get("cached"):
                    stats.verification_cache_hits += 1
                else:
                    stats.verifications_run += 1
                verified_by_kernel[cand.triton_kernel] = cand
            yield {"event": "verification", "candidate": cand.id, "duplicate_of": cand.duplicate_of,
                   **cand.verification}

        best = max(candidates, key=lambda c: c.rank())
        if best.passed or round_idx == config.repair_rounds:
            break
        repairable = best.rank()[0] in (0, 1, 3)  # unparseable, harness failure, or wrong numerics
        if not repairable:
            break  # parsed but unverifiable: harness feedback is the only repair signal
        repair_target = best

    best = max(candidates, key=lambda c: c.rank())
    passed = [c for c in candidates if c.passed]
    distinct = len({c.triton_kernel for c in passed})
    if best.passed:
        timed = (best.verification or {}).get("speedup") is not None
        if distinct > 1:
            reason = (f"fastest of {distinct} distinct passing kernels" if timed
                      else f"first of {distinct} distinct kernels that passed (no timings to rank by)")
        elif len(passed) > 1:
            reason = f"all {len(passed)} passing candidates produced the same kernel"
        else:
            reason = "only candidate that passed verification"
        if best.round > 0:
            reason += f" (fixed in repair round {best.round})"
    elif best.verification and not best.verification.get("ran"):
        reason = "not verified: " + best.verification.get("skipped_reason", "verification skipped")
    else:
        reason = "no candidate passed; showing the closest attempt"

    yield {
        "event": "selected",
        **best.summary(),
        "reason": reason,
        "num_candidates": len(candidates),
        "num_passed": len(passed),
        "total_time_s": round(time.perf_counter() - search_start, 3),
        "candidates": [
            {"candidate": c.id, "round": c.round, "status": c.status,
             "passed": c.passed, "speedup": (c.verification or {}).get("speedup"),
             "verdict": _verdict(c), "duplicate_of": c.duplicate_of}
            for c in candidates
        ],
    }


def _verdict(c: Candidate) -> str:
    v = c.verification or {}
    if c.status != "ok":
        return "unparseable"
    if not v.get("ran"):
        return "not run"
    if v.get("passed"):
        return "passed"
    if v.get("status") == "ok":
        return "incorrect"
    return v.get("status") or "error"
