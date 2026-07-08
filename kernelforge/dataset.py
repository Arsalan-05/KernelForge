"""Dataset loading, filtering, and train/val/test splitting."""

from __future__ import annotations

import json
import random
from pathlib import Path

CATEGORIES = ("quantized_matmul", "attention", "kv_cache", "norm", "rope")

DEFAULT_DATASET_PATH = Path(__file__).parent.parent / "data" / "verified" / "dataset.jsonl"
EXAMPLES_DIR = Path(__file__).parent.parent / "data" / "examples"


def load_dataset_jsonl(path: Path | str | None = None, *, verified_only: bool = True) -> list[dict]:
    """Load entries from a JSONL file."""
    path = Path(path) if path else DEFAULT_DATASET_PATH
    if not path.exists():
        return []

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if verified_only:
                verification = entry.get("verification", {})
                if not verification.get("verified"):
                    continue
            entries.append(entry)
    return entries


def load_example_entries() -> list[dict]:
    """Load the hand-written schema examples (always available, no GPU verification needed)."""
    entries = []
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        entries.append(json.loads(path.read_text()))
    return entries


def load_training_data(
    path: Path | str | None = None,
    *,
    fallback_to_examples: bool = True,
) -> list[dict]:
    """Load verified dataset entries, optionally falling back to schema examples."""
    entries = load_dataset_jsonl(path, verified_only=True)
    if entries:
        return entries
    if fallback_to_examples:
        return load_example_entries()
    return []


def split_dataset(
    entries: list[dict],
    *,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split entries stratified by category."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for entry in entries:
        by_category.setdefault(entry["category"], []).append(entry)

    train, val, test = [], [], []
    for category_entries in by_category.values():
        if not category_entries:
            continue
        shuffled = category_entries.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio)) if n >= 3 else max(0, n - 2)
        n_val = max(1, int(n * val_ratio)) if n >= 3 else (1 if n - n_train >= 2 else 0)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test
