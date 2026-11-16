"""How many tokens does the model have to generate for each dataset target?

Tokenizes every template entry's target output (kernel + explanation) in the
exact training format (training/finetune.py: system + user + assistant through
the tokenizer's chat template) and reports max / p95 / mean per category and
how many targets exceed a generation cap.

    python scripts/check_token_lengths.py                 # cap 1536
    python scripts/check_token_lengths.py --cap 2048 --json out.json

"Target tokens" = tokens of the full chat text minus tokens of the prompt with
the assistant header (add_generation_prompt=True). That's exactly what the
model must emit, including the end-of-turn token, so it is the number to
compare against max_new_tokens. Needs the tokenizer files (downloaded from the
Hugging Face Hub on first use, ~10 MB); no GPU, no model weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.templates import GENERATORS
from kernelforge.prompts import SYSTEM_PROMPT, format_training_example

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # training/configs/qwen2.5-coder-1.5b.json


def target_token_count(entry: dict, tokenizer) -> tuple[int, int]:
    """(tokens the model must generate, total training-sequence tokens) for one entry."""
    example = format_training_example(entry)
    prompt = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": example["instruction"]}]
    full = prompt + [{"role": "assistant", "content": example["response"]}]
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(full, tokenize=True)
    return len(full_ids) - len(prompt_ids), len(full_ids)


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile (q in 0..100)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def summarize(counts: dict[str, list[int]], cap: int) -> dict:
    """Per-category and overall max / p95 / mean / over-cap from {category: [target tokens]}."""
    def stats(values):
        return {"n": len(values), "max": max(values), "p95": percentile(values, 95),
                "mean": round(sum(values) / len(values), 1), "over_cap": sum(v > cap for v in values)}

    every = [v for values in counts.values() for v in values]
    overall = stats(every)
    overall["max_pct_of_cap"] = round(100 * overall["max"] / cap, 1)
    return {"cap": cap, "per_category": {c: stats(v) for c, v in counts.items()}, "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cap", type=int, default=1536, help="generation cap (max_new_tokens) to compare against")
    parser.add_argument("--json", type=Path, default=None, help="also write the summary here")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    counts: dict[str, list[int]] = {}
    longest = (0, None, 0)
    for category, generate in GENERATORS.items():
        for entry in generate():
            target, total = target_token_count(entry, tokenizer)
            counts.setdefault(category, []).append(target)
            if target > longest[0]:
                longest = (target, entry["id"], total)

    summary = summarize(counts, args.cap)
    summary["model"] = args.model
    summary["longest"] = {"id": longest[1], "target_tokens": longest[0], "full_sequence_tokens": longest[2]}

    print(f"Target tokens (kernel + explanation + end-of-turn), {args.model}, cap {args.cap}")
    print(f"{'category':<18} {'n':>4} {'max':>6} {'p95':>6} {'mean':>8} {'>cap':>5}")
    for category, s in summary["per_category"].items():
        print(f"{category:<18} {s['n']:>4} {s['max']:>6} {s['p95']:>6} {s['mean']:>8} {s['over_cap']:>5}")
    o = summary["overall"]
    print(f"{'ALL':<18} {o['n']:>4} {o['max']:>6} {o['p95']:>6} {o['mean']:>8} {o['over_cap']:>5}")
    print(f"\nLongest target: {longest[1]} = {longest[0]} tokens ({o['max_pct_of_cap']}% of the cap); "
          f"full training sequence {longest[2]} tokens")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
