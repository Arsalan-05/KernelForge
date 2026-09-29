"""The 15 dataset templates as a browsable catalog for the demo.

Each template has two instances of the same op: a realistic serving shape (for
GPU verification, where speedup is meaningful) and a small shape sized for
Triton's CPU interpreter (correctness only). Status labels come from real
artifacts only: data/raw/interpret_report.json for interpreter checks and
data/verified/dataset.jsonl for GPU verification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from data.templates import GENERATORS, SMOKE_GENERATORS
from kernelforge.dataset import load_dataset_jsonl

ROOT = Path(__file__).parent.parent.parent
INTERPRET_REPORT = ROOT / "data" / "raw" / "interpret_report.json"

_REFERENCE_IN_PROMPT = re.compile(r"PyTorch reference:\n```python\n(.*?)\n```", re.S)
_CATEGORY_KEYWORDS = {
    "attention": ("attention", "softmax", "scaled_dot_product", "causal", "attn"),
    "kv_cache": ("cache", "decode", "past_key", "kv"),
    "norm": ("norm", "rms", "variance", "eps"),
    "rope": ("rope", "rotary", "cos", "sin", "rotate"),
    "quantized_matmul": ("int8", "quant", "dequant", "matmul", "linear", "scale"),
}


@dataclass
class Template:
    id: str
    category: str
    op_name: str
    description: str
    gpu: dict
    smoke: dict
    interpreter_checked: bool
    gpu_verified: int
    median_speedup: Optional[float]

    def info(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "op_name": self.op_name,
            "description": self.description,
            "interpreter_checked": self.interpreter_checked,
            "gpu_verified": self.gpu_verified,
            "median_speedup": self.median_speedup,
        }

    def detail(self) -> dict:
        return {
            **self.info(),
            "pytorch_reference": self.gpu["pytorch_reference"],
            "smoke_reference": self.smoke["pytorch_reference"],
            "smoke_description": self.smoke["description"],
            "gpu_shape_id": self.gpu["id"],
            "smoke_shape_id": self.smoke["id"],
            "reference_explanation": self.gpu["optimization_explanation"],
            "tolerance": self.gpu.get("tolerance"),
        }


def _slug(entry_id: str) -> str:
    return "__".join(entry_id.split("__")[:2])


def _interpreter_passes() -> set[str]:
    try:
        report = json.loads(INTERPRET_REPORT.read_text())
    except (OSError, ValueError):
        return set()
    return {r["id"] for results in report.get("templates", {}).values() for r in results if r.get("passed")}


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return round(values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2, 2)


class Catalog:
    def __init__(self) -> None:
        interpreted = _interpreter_passes()
        verified: dict[str, list[float]] = {}
        for e in load_dataset_jsonl(verified_only=True):
            verified.setdefault(_slug(e["id"]), []).append((e.get("verification") or {}).get("speedup") or 0.0)

        self.templates: list[Template] = []
        self._by_reference: dict[str, dict] = {}
        for category, make_smoke in SMOKE_GENERATORS.items():
            full = GENERATORS[category]()
            for smoke in make_smoke():
                slug = _slug(smoke["id"])
                shapes = [e for e in full if _slug(e["id"]) == slug]
                gpu = shapes[len(shapes) // 2]
                speedups = verified.get(slug, [])
                self.templates.append(Template(
                    id=slug,
                    category=category,
                    op_name=smoke["op_name"],
                    description=gpu["description"],
                    gpu=gpu,
                    smoke=smoke,
                    interpreter_checked=smoke["id"] in interpreted,
                    gpu_verified=len(speedups),
                    median_speedup=_median(speedups),
                ))
                for e in (*shapes, smoke):
                    self._by_reference[e["pytorch_reference"].strip()] = e

    def get(self, template_id: str) -> Optional[Template]:
        return next((t for t in self.templates if t.id == template_id), None)

    def match_reference(self, pytorch_reference: str) -> Optional[dict]:
        return self._by_reference.get(pytorch_reference.strip())

    def example_for(self, pytorch_reference: str, description: str = "") -> Optional[dict]:
        """A solved entry to show few-shot: never the requested op itself, preferring an
        interpreter-checked sibling in the same category (the shortest, to keep prompts small)."""
        target = self.match_reference(pytorch_reference)
        target_slug = _slug(target["id"]) if target else None
        pool = [t for t in self.templates if t.id != target_slug]
        checked = [t for t in pool if t.interpreter_checked] or pool
        if not checked:
            return None
        if target is not None:
            category = target["category"]
        else:
            text = f"{description}\n{pytorch_reference}".lower()
            scores = {c: sum(text.count(k) for k in keys) for c, keys in _CATEGORY_KEYWORDS.items()}
            category = max(scores, key=scores.get) if any(scores.values()) else "norm"
        same = [t for t in checked if t.category == category] or checked
        return min(same, key=lambda t: len(t.smoke["triton_kernel"])).smoke

    def match_prompt(self, messages: list[dict]) -> Optional[dict]:
        # Newest first: earlier user turns may hold a few-shot example.
        for m in reversed(messages):
            if m["role"] == "user":
                found = _REFERENCE_IN_PROMPT.search(m["content"])
                if found:
                    return self.match_reference(found.group(1))
        return None
