#!/usr/bin/env python3
"""
Hosted-model baseline: run all 300 (template × shape) problems through a
hosted LLM via OpenRouter and check the generated kernels with the
interpreter harness.

Strategy (two-tier):
  - LLM generation: all 300 entries.  Cheap (<<$3 for qwen-2.5-coder-32b).
  - Interpreter verification: the 15 smoke entries only (one per template,
    small shape).  Running all 300 through the interpreter on large shapes
    would exceed the 2-hour time budget; the smoke tier is a sound proxy for
    "can the model produce valid Triton kernels at all?"

Usage:
    python scripts/hosted_model_baseline.py --pilot          # 5-problem pilot
    python scripts/hosted_model_baseline.py                  # full run
    python scripts/hosted_model_baseline.py --smoke-only     # smoke entries only

Credentials are read from .env (KERNELFORGE_LLM_API_KEY, KERNELFORGE_LLM_BASE_URL,
KERNELFORGE_LLM_MODEL).  The $3 hard-spend guard uses char-based token estimates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import GENERATORS, SMOKE_GENERATORS
from kernelforge.parsing import parse_model_output
from kernelforge.prompts import build_messages
from kernelforge.remote import RemoteChatGenerator
from verification.verify_kernel import verify_model_kernel

# Hard spend cap; script refuses to continue if estimated cost exceeds this.
SPEND_CAP_USD = 3.0
# Cost estimate (conservative; actual OpenRouter pricing varies by model).
COST_PER_INPUT_TOKEN = 0.5 / 1_000_000   # $0.50 / M input
COST_PER_OUTPUT_TOKEN = 1.5 / 1_000_000  # $1.50 / M output

INTERPRETER_TIMEOUT_S = 90.0
GENERATION_TIMEOUT_S = 60.0

RESULTS_DIR = repo_root / "evaluation" / "results"
LOG_PATH = RESULTS_DIR / "hosted_model_baseline.jsonl"


def _load_env() -> dict:
    env_path = repo_root / ".env"
    out = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    # Let actual env vars override .env
    for k in ("KERNELFORGE_LLM_API_KEY", "KERNELFORGE_LLM_BASE_URL", "KERNELFORGE_LLM_MODEL"):
        if k in os.environ:
            out[k] = os.environ[k]
    return out


def _entries_all() -> list[dict]:
    entries = []
    for gen in GENERATORS.values():
        entries.extend(gen())
    return entries


def _entries_smoke() -> list[dict]:
    entries = []
    for gen in SMOKE_GENERATORS.values():
        entries.extend(gen())
    return entries


def _generate_text(gen: RemoteChatGenerator, entry: dict) -> tuple[str, int, int]:
    """Stream a completion for one entry. Returns (text, input_tokens, output_tokens)."""
    messages = build_messages(entry["description"], entry["pytorch_reference"], notes=True)
    # Estimate input tokens before the call
    input_text = " ".join(m["content"] for m in messages)
    input_tokens = gen.count_tokens(input_text)

    chunks: list[str] = []
    try:
        for _, chunk in gen.stream_chat(messages, num_sequences=1, temperature=0.0):
            chunks.append(chunk)
    except Exception as exc:
        return f"[generation error: {exc}]", input_tokens, 0

    text = "".join(chunks)
    output_tokens = gen.count_tokens(text)
    return text, input_tokens, output_tokens


def _run_interpreter(entry: dict, candidate_code: str) -> dict:
    """Run interpreter verification; return a serialisable result dict."""
    try:
        result = verify_model_kernel(
            entry["pytorch_reference"],
            candidate_code,
            interpret=True,
            timeout_s=INTERPRETER_TIMEOUT_S,
        )
        return {
            "ran": True,
            "status": result.status,
            "correct": result.correct,
            "error": result.error,
            "interpreted": result.interpreted,
        }
    except Exception as exc:
        return {"ran": False, "status": "exception", "error": str(exc)}


def _generate_one(gen: RemoteChatGenerator, entry: dict, verify_ids: set[str],
                  counter: list[int], lock: threading.Lock) -> dict:
    """Generate and optionally verify one entry. Thread-safe for the counter."""
    t0 = time.monotonic()
    text, in_tok, out_tok = _generate_text(gen, entry)
    entry_cost = in_tok * COST_PER_INPUT_TOKEN + out_tok * COST_PER_OUTPUT_TOKEN

    parsed = parse_model_output(text)
    verification = None
    if parsed.has_kernel and entry["id"] in verify_ids:
        verification = _run_interpreter(entry, parsed.triton_kernel)

    elapsed = time.monotonic() - t0
    status = "correct" if (verification and verification.get("correct")) else \
             "incorrect" if (verification and verification.get("ran")) else \
             "not_verified" if parsed.has_kernel else "no_kernel"

    with lock:
        counter[0] += 1
        idx = counter[0]

    v_tag = ""
    if verification:
        v_tag = " PASS" if verification.get("correct") else f" FAIL({verification.get('status')})"
    parse_tag = "parsed" if parsed.has_kernel else "NO_KERNEL"
    print(f"  [{idx:>3}] {entry['id'][:56]:<56} {parse_tag}{v_tag}  {elapsed:.1f}s  ${entry_cost:.5f}",
          flush=True)

    return {
        "id": entry["id"],
        "category": entry["category"],
        "op_name": entry.get("op_name", ""),
        "parsed_kernel": parsed.has_kernel,
        "has_explanation": parsed.has_explanation and parsed.has_kernel,
        "verification": verification,
        "outcome": status,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "cost_usd": entry_cost,
        "elapsed_s": elapsed,
        "generated_text_preview": text[:200],
        "generated_text": text,  # full text for downstream re-verification
    }


def run(entries: list[dict], gen: RemoteChatGenerator, verify_ids: set[str],
        pilot: bool, workers: int = 4) -> list[dict]:
    """Generate and optionally verify entries. Returns list of result rows."""
    n = min(5, len(entries)) if pilot else len(entries)
    entries_to_run = entries[:n]

    print(f"\nRunning {n} entries "
          f"({'pilot' if pilot else 'full'}, {workers} parallel workers, "
          f"{len([e for e in entries_to_run if e['id'] in verify_ids])} interpreter-verified)...")

    lock = threading.Lock()
    counter = [0]
    rows: list[dict] = []
    total_cost = 0.0

    # Pilot is sequential; full run uses parallel workers.
    actual_workers = 1 if pilot else workers
    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = {
            pool.submit(_generate_one, gen, entry, verify_ids, counter, lock): entry
            for entry in entries_to_run
        }
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as exc:
                entry = futures[fut]
                row = {
                    "id": entry["id"], "category": entry["category"],
                    "op_name": entry.get("op_name", ""),
                    "parsed_kernel": False, "has_explanation": False,
                    "verification": None, "outcome": "exception",
                    "in_tokens": 0, "out_tokens": 0,
                    "cost_usd": 0.0, "elapsed_s": 0.0,
                    "generated_text_preview": f"[exception: {exc}]",
                }
            rows.append(row)
            total_cost += row["cost_usd"]

            # Hard spend guard
            if total_cost > SPEND_CAP_USD:
                print(f"\n  SPEND CAP HIT: ${total_cost:.4f} > ${SPEND_CAP_USD:.2f} — cancelling remaining.")
                for f in futures:
                    f.cancel()
                break

    id_order = {e["id"]: i for i, e in enumerate(entries_to_run)}
    rows.sort(key=lambda r: id_order.get(r["id"], 999))

    total_input = sum(r["in_tokens"] for r in rows)
    total_output = sum(r["out_tokens"] for r in rows)
    print(f"\nTotal: {total_input:,} input tokens, {total_output:,} output tokens, "
          f"estimated cost ${total_cost:.4f}")
    return rows


def _summarise(rows: list[dict]) -> dict:
    total = len(rows)
    parsed = sum(1 for r in rows if r["parsed_kernel"])
    verified = [r for r in rows if r["verification"] and r["verification"].get("ran")]
    correct = sum(1 for r in verified if r["verification"].get("correct"))
    incorrect = sum(1 for r in verified if r["verification"] and not r["verification"].get("correct")
                    and r["verification"].get("status") == "ok")
    timeouts = sum(1 for r in verified if r["verification"].get("status") == "timeout")
    errors = sum(1 for r in verified if r["verification"].get("status") in ("error", "crash", "exception"))

    # Per-category
    by_cat: dict[str, dict] = {}
    for r in rows:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "parsed": 0, "correct": 0, "verified": 0}
        by_cat[cat]["total"] += 1
        by_cat[cat]["parsed"] += int(r["parsed_kernel"])
        if r["verification"] and r["verification"].get("ran"):
            by_cat[cat]["verified"] += 1
            by_cat[cat]["correct"] += int(bool(r["verification"].get("correct")))

    return {
        "total": total,
        "parsed_kernel": parsed,
        "verified": len(verified),
        "correct": correct,
        "incorrect": incorrect,
        "timeout": timeouts,
        "error": errors,
        "not_verified": total - len(verified),
        "parse_rate_pct": round(parsed / total * 100, 1) if total else None,
        "correct_rate_of_verified_pct": round(correct / len(verified) * 100, 1) if verified else None,
        "per_category": by_cat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hosted-model baseline via OpenRouter.")
    parser.add_argument("--pilot", action="store_true", help="Run only 5 entries as a pilot")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run only the 15 smoke entries (1 per template)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel LLM generation workers (default 4)")
    args = parser.parse_args()

    env = _load_env()
    api_key = env.get("KERNELFORGE_LLM_API_KEY", "")
    base_url = env.get("KERNELFORGE_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    model = env.get("KERNELFORGE_LLM_MODEL", "qwen/qwen-2.5-coder-32b-instruct")

    if not api_key:
        print("Error: KERNELFORGE_LLM_API_KEY not set in .env or environment", file=sys.stderr)
        return 1

    print("Hosted-model baseline")
    print("=" * 60)
    print(f"Model:    {model}")
    print(f"Base URL: {base_url}")
    print(f"Spend cap: ${SPEND_CAP_USD:.2f}")

    gen = RemoteChatGenerator(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=2048,
        temperature=0.0,
        timeout_s=GENERATION_TIMEOUT_S,
    )

    # Decide which entries to generate and which to interpreter-verify
    if args.smoke_only:
        all_entries = _entries_smoke()
        smoke_ids = {e["id"] for e in all_entries}
        print(f"\nSmoke-only mode: {len(all_entries)} entries, all interpreter-verified")
    else:
        all_entries = _entries_all()
        smoke_entries = _entries_smoke()
        smoke_ids = {e["id"] for e in smoke_entries}
        print(f"\nFull mode: {len(all_entries)} entries, "
              f"{len(smoke_ids)} interpreter-verified (smoke shapes)")

    if args.pilot:
        # Run 5 entries, project full cost
        pilot_rows = run(all_entries, gen, smoke_ids, pilot=True, workers=args.workers)
        pilot_cost = sum(r["cost_usd"] for r in pilot_rows)
        projected = pilot_cost / len(pilot_rows) * len(all_entries)
        print(f"\nPilot result: ${pilot_cost:.4f} for {len(pilot_rows)} entries")
        print(f"Projected full cost: ${projected:.4f} for {len(all_entries)} entries")
        if projected > SPEND_CAP_USD:
            print(f"Projected cost ${projected:.4f} > cap ${SPEND_CAP_USD:.2f} — aborting full run.")
            _write_report(pilot_rows, model, base_url, pilot=True, projected_cost=projected)
            return 0
        print(f"Projected cost ${projected:.4f} < cap ${SPEND_CAP_USD:.2f} — safe to continue.")
        print("Re-run without --pilot to execute the full baseline.")
        _write_report(pilot_rows, model, base_url, pilot=True, projected_cost=projected)
        return 0

    # Full run
    rows = run(all_entries, gen, smoke_ids, pilot=False, workers=args.workers)
    report_path = _write_report(rows, model, base_url, smoke_only=args.smoke_only)

    summary = _summarise(rows)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Entries run:          {summary['total']}")
    print(f"Kernel parsed:        {summary['parsed_kernel']} / {summary['total']}"
          f"  ({summary['parse_rate_pct']}%)")
    print(f"Interpreter-verified: {summary['verified']}")
    if summary['verified']:
        print(f"  Correct:   {summary['correct']} / {summary['verified']}"
              f"  ({summary['correct_rate_of_verified_pct']}%)")
        print(f"  Incorrect: {summary['incorrect']}")
        print(f"  Timeout:   {summary['timeout']}")
        print(f"  Error:     {summary['error']}")
    print(f"\nReport: {report_path}")
    return 0


def _write_report(rows: list[dict], model: str, base_url: str,
                  pilot: bool = False, smoke_only: bool = False,
                  projected_cost: float | None = None) -> Path:
    summary = _summarise(rows)
    total_cost = sum(r["cost_usd"] for r in rows)
    report = {
        "run_name": "hosted_model_baseline",
        "model": model,
        "base_url": base_url,
        "pilot": pilot,
        "smoke_only": smoke_only,
        "projected_full_cost_usd": projected_cost,
        "total_entries": summary["total"],
        "total_estimated_cost_usd": total_cost,
        "summary": summary,
        "entries": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if pilot:
        suffix = "_pilot"
    elif smoke_only:
        suffix = "_smoke"
    else:
        suffix = ""
    path = RESULTS_DIR / f"hosted_model_baseline{suffix}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Results written to {path}")
    return path


if __name__ == "__main__":
    sys.exit(main())
