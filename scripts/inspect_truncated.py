#!/usr/bin/env python3
"""
Diagnostic: capture raw_text for the 4 entries that hit max_new_tokens=1536.
Inspect the last ~400 tokens of each for looping/repetition vs valid-but-long.
Count lines in the first complete top-level function/class definition.

Entries: norm__rmsnorm__smoke, norm__layernorm__smoke,
         rope__rotate_half__b2_s9__smoke, rope__interleaved__b2_s9__smoke

Output: evaluation/results/truncated_raw_texts.json
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import SMOKE_GENERATORS
from kernelforge.prompts import build_messages

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 1536
TRUNCATED_IDS = {
    "norm__rmsnorm__smoke",
    "norm__layernorm__smoke",
    "rope__rotate_half__b2_s9__smoke",
    "rope__interleaved__b2_s9__smoke",
}

OUT_PATH = repo_root / "evaluation" / "results" / "truncated_raw_texts.json"


def _first_kernel_line_count(text: str) -> int:
    """Count lines in the first @triton.jit kernel block (up to next top-level def/class or EOF)."""
    lines = text.split("\n")
    in_kernel = False
    start = None
    for i, line in enumerate(lines):
        if "@triton.jit" in line:
            in_kernel = True
            start = i
        elif in_kernel and start is not None and i > start:
            stripped = line
            if stripped and not stripped[0].isspace() and stripped.startswith(("def ", "class ", "@")):
                return i - start
    if in_kernel and start is not None:
        return len(lines) - start
    return 0


def _is_looping(text: str, window: int = 200, min_rep: int = 3) -> tuple[bool, str]:
    """Check last `window` chars for repeated patterns (≥ min_rep repetitions of ≥20 chars)."""
    tail = text[-window:]
    # Find the longest repeated substring in the tail
    for length in range(60, 15, -1):
        for start in range(len(tail) - length * min_rep + 1):
            snippet = tail[start:start + length]
            if snippet.count(snippet) >= 1:  # always true; count reps manually
                reps = 0
                pos = 0
                while True:
                    idx = tail.find(snippet, pos)
                    if idx == -1:
                        break
                    reps += 1
                    pos = idx + length
                if reps >= min_rep:
                    return True, repr(snippet[:80])
    return False, ""


def main() -> None:
    # Collect entries
    entries = []
    for cat in ["norm", "rope"]:
        for entry in SMOKE_GENERATORS[cat]():
            if entry["id"] in TRUNCATED_IDS:
                entries.append(entry)

    print(f"[inspect_truncated] {len(entries)} entries to inspect")
    print(f"[inspect_truncated] Loading model …")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kernelforge.parsing import parse_model_output

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    model.eval()

    results = []
    for entry in entries:
        print(f"  generating {entry['id']} …", end=" ", flush=True)
        messages = build_messages(entry["description"], entry["pytorch_reference"], notes=True)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=None,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - t0
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        comp_tokens = len(generated_ids)
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        tps = comp_tokens / elapsed if elapsed > 0 else 0.0
        print(f"{comp_tokens} tokens, {tps:.1f} tok/s")

        parsed = parse_model_output(raw_text)
        kernel = parsed.triton_kernel or ""

        # Last 400 chars of raw_text (decoded token view, ~400 characters)
        # Roughly 400 chars ≈ last 100-150 tokens
        tail_chars = raw_text[-500:]

        looping, example = _is_looping(raw_text, window=600, min_rep=3)
        kernel_lines = _first_kernel_line_count(kernel)

        results.append({
            "id": entry["id"],
            "completion_tokens": comp_tokens,
            "truncated": comp_tokens == MAX_NEW_TOKENS,
            "looping": looping,
            "loop_example": example if looping else None,
            "kernel_line_count": kernel_lines,
            "tail_400_chars": tail_chars,
            "raw_text_length_chars": len(raw_text),
            "raw_text": raw_text,  # full text for reference
        })

    del model, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("[inspect_truncated] Model freed.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"[inspect_truncated] Written to {OUT_PATH}")

    print("\n=== TRUNCATION INSPECTION REPORT ===")
    for r in results:
        print(f"\n{r['id']}")
        print(f"  tokens: {r['completion_tokens']} (truncated={r['truncated']})")
        print(f"  raw_text length: {r['raw_text_length_chars']} chars")
        print(f"  kernel lines (first @triton.jit block): {r['kernel_line_count']}")
        print(f"  looping: {r['looping']}")
        if r['looping']:
            print(f"  loop example: {r['loop_example']}")
        print(f"  --- last 300 chars ---")
        print(r['tail_400_chars'][-300:])
        print(f"  ---")


if __name__ == "__main__":
    main()
