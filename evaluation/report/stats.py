"""Small statistics helpers for the results report.

Everything here is plain arithmetic on counts, so it has no dependencies and
is easy to check by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# z for a two-sided 95% interval (the 97.5th percentile of the standard normal).
Z95 = 1.959963984540054

# Below this many samples a percentage is too noisy to compare categories on;
# cells built from fewer samples get a dagger and a footnote.
SMALL_N = 20


def wilson_interval(successes: int, n: int, z: float = Z95) -> Optional[tuple[float, float]]:
    """Wilson score interval for a binomial proportion, as fractions in [0, 1].

    Chosen over the textbook "p +/- z*sqrt(p(1-p)/n)" interval because that one
    collapses to [0, 0] or [1, 1] when every sample fails or passes, which is
    exactly what happens with 5 samples per category. Wilson stays sensible.
    Returns None when n == 0 (no interval exists).
    """
    if n <= 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} must be between 0 and n={n}")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Proportion:
    """k successes out of n, with its Wilson 95% interval."""

    successes: int
    n: int

    @property
    def fraction(self) -> Optional[float]:
        return self.successes / self.n if self.n else None

    @property
    def interval(self) -> Optional[tuple[float, float]]:
        return wilson_interval(self.successes, self.n)

    @property
    def small(self) -> bool:
        return self.n < SMALL_N


def finite(values: Sequence[float]) -> list[float]:
    """Drop None / inf / NaN (the harness reports inf when a kernel times at 0 s)."""
    return [v for v in values if v is not None and math.isfinite(v)]


def mean(values: Sequence[float]) -> Optional[float]:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else None


def geomean(values: Sequence[float]) -> Optional[float]:
    """Geometric mean of positive finite values. The usual average for ratios such as
    speedups: a 2x and a 0.5x kernel average to 1x, not 1.25x."""
    vals = [v for v in finite(values) if v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(finite(values))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


# ---------------------------------------------------------------------------
# pass@k estimator (Chen et al. 2021, "Evaluating Large Language Models
# Trained on Code", arXiv:2107.03374).
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> Optional[float]:
    """Unbiased estimator for pass@k.

    Given n independent samples per problem, c of which are correct, the
    unbiased estimator for the probability that at least one of k samples is
    correct is:

        pass@k = 1 - C(n-c, k) / C(n, k)

    This equals 0.0 when c == 0 and 1.0 when c >= k.

    Args:
        n: total samples per problem (must be >= k)
        c: correct samples (0 <= c <= n)
        k: number of samples to draw

    Returns None when n < k (not enough samples to estimate).
    """
    if n < k:
        return None
    if c < 0 or c > n:
        raise ValueError(f"c={c} must be between 0 and n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        # Fewer than k incorrect samples: impossible to pick k without getting a correct one.
        return 1.0
    # Use log-domain arithmetic to avoid overflow on large n.
    # C(n-c, k) / C(n, k) = prod_{i=0}^{k-1} (n-c-i) / (n-i)
    log_ratio = sum(
        math.log(n - c - i) - math.log(n - i)
        for i in range(k)
    )
    return 1.0 - math.exp(log_ratio)


def pass_at_k_aggregate(counts: Sequence[tuple[int, int]], k: int) -> Optional[float]:
    """Average pass@k over a collection of (n, c) pairs (one per problem).

    Returns the mean pass@k across problems where n >= k. Returns None if no
    problem has enough samples.
    """
    estimates = [pass_at_k(n, c, k) for n, c in counts if pass_at_k(n, c, k) is not None]
    return sum(estimates) / len(estimates) if estimates else None


# ---------------------------------------------------------------------------
# Failure-type breakdown
# ---------------------------------------------------------------------------

# Canonical failure types. These map raw outcome strings from evaluation/common.py
# to a coarser display category for the report.
FAILURE_TYPE_LABELS: dict[str, str] = {
    "no_kernel": "syntax / no kernel",
    "harness_exception": "harness error",
    "not_verified": "not verified",
    "error": "compile / runtime error",
    "crash": "crash (OOM or hard kill)",
    "timeout": "timeout",
    "correct": "correct",          # not a failure, included for completeness
    "incorrect": "wrong output",
}


def failure_breakdown(outcome_counts: dict[str, int]) -> dict[str, int]:
    """Map raw outcome strings to display categories and sum counts.

    Args:
        outcome_counts: dict of {outcome_string: count} as written by
            evaluation.common._count_outcomes().

    Returns a dict of {display_label: total_count} with one entry per
    non-zero failure type. Outcomes not in FAILURE_TYPE_LABELS are grouped
    under "other".
    """
    result: dict[str, int] = {}
    for outcome, count in outcome_counts.items():
        label = FAILURE_TYPE_LABELS.get(outcome, "other")
        result[label] = result.get(label, 0) + count
    return result
