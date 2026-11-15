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
