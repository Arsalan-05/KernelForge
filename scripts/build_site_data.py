#!/usr/bin/env python3
"""Generate app/web/data/results.json from verified repo facts.

Run from the repo root:
    python scripts/build_site_data.py

The numbers written here come from confirmed artifacts:
  - 0/88 base model pass@1 (held-out T4 GPU eval, baseline run)
  - 274 verified examples total  (167 train / 19 val / 88 held-out)
  - Error breakdown from manual triage of the base-model failures
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "app" / "web" / "data"
OUT_FILE = OUT_DIR / "results.json"

PAYLOAD: dict = {
    "results": [
        {
            "model": "Qwen2.5-Coder-1.5B-Instruct (base)",
            "quantization": "4-bit NF4",
            "eval_set": "88 held-out (T4)",
            "pass_at_1": "0 / 88",
            "status": "verified",
            "notes": "No training signal for Triton format",
        },
        {
            "model": "Qwen2.5-Coder-1.5B-Instruct + QLoRA",
            "quantization": "4-bit NF4",
            "eval_set": "88 held-out (T4)",
            "pass_at_1": "pending",
            "status": "re-run in progress",
            "notes": "Training in progress",
        },
    ],
    "split": {
        "train": 167,
        "val": 19,
        "held_out": 88,
    },
    "dataset": {
        "total": 300,
        "verified": 274,
    },
    "error_breakdown": [
        {"category": "tl.Tensor hallucination", "count": 40},
        {"category": "Undefined helper function", "count": 16},
        {"category": "JIT called outside kernel", "count": 10},
        {"category": "Syntax truncation", "count": 5},
        {"category": "Other generated-code errors", "count": 17},
    ],
    "disclaimer": (
        "Reference-kernel speedups are measurements of hand-written templates, "
        "not model output."
    ),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(PAYLOAD, indent=2)
    OUT_FILE.write_text(text)
    print(f"Wrote {OUT_FILE}  ({len(text)} bytes)")


if __name__ == "__main__":
    main()
