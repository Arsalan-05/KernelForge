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


def template_of(entry: dict) -> str:
    """Template slug, e.g. "attention__causal_prefill" from "attention__causal_prefill__b2_s512"."""
    return "__".join(entry["id"].split("__")[:2])


def split_by_template(
    entries: list[dict],
    *,
    val_ratio: float = 0.10,
    test_templates_per_category: int = 1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Hold out whole templates for test, so the test set measures unseen ops.

    Entries of one template share the same kernel and differ only in shape
    constants, so a shape-level random split leaks the answer into training.
    Categories with a single template fall back to a shape-level split.
    """
    rng = random.Random(seed)
    by_category: dict[str, dict[str, list[dict]]] = {}
    for entry in entries:
        by_category.setdefault(entry["category"], {}).setdefault(template_of(entry), []).append(entry)

    train, val, test = [], [], []
    for category in sorted(by_category):
        templates = by_category[category]
        names = sorted(templates)
        if len(names) <= test_templates_per_category:
            tr, va, te = split_dataset([e for n in names for e in templates[n]], seed=seed)
            train += tr
            val += va
            test += te
            continue
        held_out = set(rng.sample(names, test_templates_per_category))
        seen = []
        for name in names:
            (test if name in held_out else seen).extend(templates[name])
        rng.shuffle(seen)
        n_val = max(1, round(len(seen) * val_ratio)) if len(seen) > 1 else 0
        val += seen[:n_val]
        train += seen[n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val, test


def make_split(entries: list[dict], strategy: str = "template", seed: int = 42, **ratios):
    if strategy == "template":
        return split_by_template(entries, val_ratio=ratios.get("val_ratio", 0.10), seed=seed)
    if strategy == "random":
        return split_dataset(entries, seed=seed, **ratios)
    raise ValueError(f"unknown split strategy {strategy!r}; use 'template' or 'random'")


def interleave_by_category(entries: list[dict]) -> list[dict]:
    """Round-robin across categories, so `--limit N` samples every category."""
    queues: dict[str, list[dict]] = {}
    for entry in entries:
        queues.setdefault(entry["category"], []).append(entry)
    out = []
    while any(queues.values()):
        for category in sorted(queues):
            if queues[category]:
                out.append(queues[category].pop(0))
    return out
