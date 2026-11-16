#!/usr/bin/env python3
"""
Local base-model evaluation: Qwen/Qwen2.5-Coder-1.5B-Instruct on MPS (float16).

No GPU, no paid API.  Triton CPU interpreter verification inside Docker.

Usage:
    python scripts/local_baseline_eval.py --pilot       # 3-entry pilot
    python scripts/local_baseline_eval.py               # full run (up to 60 entries)
    python scripts/local_baseline_eval.py --resume      # skip already-logged entries

Outputs:
    evaluation/results/local_baseline_eval.jsonl   checkpoint (one JSON per line)
    evaluation/results/local_baseline_report.json  final summary (written at end)

Notes:
    - BASE model, not fine-tuned.  Zero-shot baseline.
    - CPU interpreter correctness only — no GPU speedup measurement.
    - Small N (up to 60); Wilson 95% CIs are wide.
    - Model freed completely before any Docker verification.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import GENERATORS, SMOKE_GENERATORS
from kernelforge.parsing import parse_model_output
from kernelforge.prompts import build_messages

RESULTS_DIR = repo_root / "evaluation" / "results"
LOG_PATH = RESULTS_DIR / "local_baseline_eval.jsonl"
REPORT_PATH = RESULTS_DIR / "local_baseline_report.json"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.0
INTERPRETER_TIMEOUT_S = 90

CATEGORIES = ["quantized_matmul", "attention", "kv_cache", "norm", "rope"]
# Smoke entries first (3 per category), then fill from GENERATORS (9 per category)
ENTRIES_PER_CATEGORY = 12


# ---------------------------------------------------------------------------
# Entry selection: 12 per category, smoke first
# ---------------------------------------------------------------------------

def _build_entry_list(pilot: bool = False) -> list[dict]:
    """Return up to 60 entries (12 per category, smoke first)."""
    selected: list[dict] = []
    for cat in CATEGORIES:
        smoke = list(SMOKE_GENERATORS[cat]())   # 3 entries
        regular = list(GENERATORS[cat]())        # 60 entries

        # Smoke entries come first; they're not in GENERATORS (different IDs)
        cat_entries: list[dict] = smoke[:]
        smoke_ids = {e["id"] for e in smoke}
        for e in regular:
            if e["id"] not in smoke_ids and len(cat_entries) < ENTRIES_PER_CATEGORY:
                cat_entries.append(e)

        selected.extend(cat_entries[:ENTRIES_PER_CATEGORY])

    if pilot:
        # 1 smoke entry from each of 3 categories
        pilot_entries: list[dict] = []
        for cat in ["quantized_matmul", "attention", "norm"]:
            smoke = list(SMOKE_GENERATORS[cat]())
            pilot_entries.append(smoke[0])
        return pilot_entries

    return selected


# ---------------------------------------------------------------------------
# Wilson 95% CI
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval at 95% confidence."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# Generation phase
# ---------------------------------------------------------------------------

def _apply_chat_template(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_one(model, tokenizer, entry: dict, device) -> tuple[str, float, int, int]:
    """
    Generate text for one entry.

    Returns (generated_text, tokens_per_second, prompt_tokens, completion_tokens).
    """
    import torch

    messages = build_messages(entry["description"], entry["pytorch_reference"], notes=True)
    prompt = _apply_chat_template(tokenizer, messages)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=None,          # greedy (do_sample=False)
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.monotonic() - t0

    generated_ids = output[0][prompt_len:]
    completion_tokens = len(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    return text, tps, prompt_len, completion_tokens


# ---------------------------------------------------------------------------
# Docker verification (called AFTER model is freed)
# ---------------------------------------------------------------------------

def _verify_in_docker(entry: dict, generated_text: str) -> dict:
    """
    Parse `generated_text` and verify the kernel in Docker.

    Returns a dict with keys: parse_status, verify_status, correct, error,
    raw_text, extracted_kernel.  Storing raw_text and extracted_kernel in
    the checkpoint allows post-hoc comparison to confirm extraction fidelity.
    """
    parsed = parse_model_output(generated_text)
    if not parsed.has_kernel:
        return {
            "parse_status": "failed",
            "verify_status": None,
            "correct": False,
            "error": "parse_model_output: no kernel: section found or empty kernel",
            "raw_text": generated_text,
            "extracted_kernel": None,
        }

    # Write kernel + reference to temp files inside the repo (mounted at /work in Docker).
    # Using a temp dir that lives inside the repo so Docker can access it.
    tmp_dir = RESULTS_DIR / "_docker_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Use unique filenames to avoid collisions if entries run close together.
    entry_id_safe = entry["id"].replace("/", "_")
    kernel_path = tmp_dir / f"{entry_id_safe}_kernel.py"
    ref_path = tmp_dir / f"{entry_id_safe}_ref.py"

    kernel_path.write_text(parsed.triton_kernel)
    ref_path.write_text(entry["pytorch_reference"])

    # Docker paths (relative to /work mount)
    docker_kernel = f"/work/evaluation/results/_docker_tmp/{entry_id_safe}_kernel.py"
    docker_ref = f"/work/evaluation/results/_docker_tmp/{entry_id_safe}_ref.py"

    verify_script = (
        "import sys, json; sys.path.insert(0, '/work'); "
        "from verification.verify_kernel import verify_model_kernel; "
        f"ref = open({repr(docker_ref)}).read(); "
        f"kernel = open({repr(docker_kernel)}).read(); "
        f"result = verify_model_kernel(ref, kernel, device='cpu', "
        f"    timeout_s={INTERPRETER_TIMEOUT_S}, interpret=True); "
        "print(json.dumps({'status': result.status, "
        "    'correct': result.correct, 'error': result.error}))"
    )

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{repo_root}:/work",
        "kernelforge-verify",
        "python", "-c", verify_script,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=INTERPRETER_TIMEOUT_S + 15,  # outer buffer
        )
        if proc.returncode != 0:
            return {
                "parse_status": "ok",
                "verify_status": "error",
                "correct": False,
                "error": (proc.stderr or proc.stdout or "")[-3000:],
                "raw_text": generated_text,
                "extracted_kernel": parsed.triton_kernel,
            }
        stdout = proc.stdout.strip()
        if not stdout:
            return {
                "parse_status": "ok",
                "verify_status": "error",
                "correct": False,
                "error": f"Docker produced no output. stderr: {proc.stderr[-1000:]}",
                "raw_text": generated_text,
                "extracted_kernel": parsed.triton_kernel,
            }
        last_line = stdout.splitlines()[-1]
        result = json.loads(last_line)
        return {
            "parse_status": "ok",
            "verify_status": result["status"],
            "correct": result["correct"],
            "error": result.get("error"),
            "raw_text": generated_text,
            "extracted_kernel": parsed.triton_kernel,
        }
    except subprocess.TimeoutExpired:
        return {
            "parse_status": "ok",
            "verify_status": "timeout",
            "correct": False,
            "error": f"Docker outer timeout ({INTERPRETER_TIMEOUT_S + 15}s)",
            "raw_text": generated_text,
            "extracted_kernel": parsed.triton_kernel,
        }
    except Exception as exc:
        return {
            "parse_status": "ok",
            "verify_status": "error",
            "correct": False,
            "error": str(exc),
            "raw_text": generated_text,
            "extracted_kernel": parsed.triton_kernel,
        }
    finally:
        # Clean up temp files
        for p in (kernel_path, ref_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict[str, dict]:
    """Return {entry_id: record} for already-completed entries."""
    done: dict[str, dict] = {}
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["id"]] = rec
            except json.JSONDecodeError:
                pass
    return done


def _append_record(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _build_report(records: list[dict]) -> dict:
    total = len(records)
    parse_ok = sum(1 for r in records if r.get("parse_status") == "ok")
    correct = sum(1 for r in records if r.get("correct"))
    verified = sum(1 for r in records if r.get("verify_status") is not None)

    failure_types: dict[str, int] = {}
    for r in records:
        if r.get("correct"):
            continue
        ps = r.get("parse_status", "unknown")
        vs = r.get("verify_status", "unknown")
        err = r.get("error") or ""
        if ps != "ok":
            key = "parse_failure"
        elif vs == "timeout":
            key = "timeout"
        elif vs == "ok":
            key = "wrong_output"
        elif "SyntaxError" in err or "IndentationError" in err:
            key = "syntax_error"
        elif "code policy" in err.lower():
            key = "policy_violation"
        elif "NameError" in err:
            key = "name_error"
        else:
            key = "runtime_error"
        failure_types[key] = failure_types.get(key, 0) + 1

    per_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_recs = [r for r in records if r.get("category") == cat]
        n = len(cat_recs)
        k = sum(1 for r in cat_recs if r.get("correct"))
        lo, hi = _wilson_ci(k, n)
        per_category[cat] = {
            "n": n, "correct": k, "pass_at_1": k / n if n else 0.0,
            "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
        }

    lo_all, hi_all = _wilson_ci(correct, total)
    tps_values = [r["tokens_per_second"] for r in records if r.get("tokens_per_second", 0) > 0]

    return {
        "model": MODEL_NAME,
        "note": "BASE model, not fine-tuned. CPU interpreter only. Small N — CIs are wide.",
        "total_entries": total,
        "parse_ok": parse_ok,
        "parse_rate": round(parse_ok / total, 4) if total else 0.0,
        "verified": verified,
        "correct": correct,
        "pass_at_1": round(correct / total, 4) if total else 0.0,
        "wilson_95_ci": [round(lo_all, 4), round(hi_all, 4)],
        "failure_types": failure_types,
        "per_category": per_category,
        "tokens_per_second_median": round(sorted(tps_values)[len(tps_values) // 2], 1) if tps_values else None,
        "tokens_per_second_mean": round(sum(tps_values) / len(tps_values), 1) if tps_values else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="3-entry pilot only")
    parser.add_argument("--resume", action="store_true", help="skip already-logged entries")
    args = parser.parse_args()

    entries = _build_entry_list(pilot=args.pilot)
    already_done = _load_checkpoint() if args.resume else {}

    todo = [e for e in entries if e["id"] not in already_done]
    print(f"[local_baseline_eval] entries={len(entries)}, todo={len(todo)}, "
          f"model={MODEL_NAME}, max_new_tokens={MAX_NEW_TOKENS}")

    # ------------------------------------------------------------------
    # Phase 1: Generation (model loaded, then freed)
    # ------------------------------------------------------------------
    generations: list[dict] = []  # {id, generated_text, tokens_per_second, ...}

    if todo:
        print("[phase 1] Loading model on MPS …")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[phase 1] device={device}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        total_entries = len(todo)
        for i, entry in enumerate(todo, 1):
            print(f"[gen {i}/{total_entries}] {entry['id']} …", end=" ", flush=True)
            try:
                text, tps, prompt_tok, comp_tok = _generate_one(model, tokenizer, entry, device)
                print(f"{comp_tok} tokens, {tps:.1f} tok/s")
                generations.append({
                    "id": entry["id"],
                    "category": entry["category"],
                    "generated_text": text,
                    "tokens_per_second": tps,
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": comp_tok,
                    "phase": "generated",
                })
                if args.pilot and i == 3:
                    break
            except Exception as exc:
                print(f"GENERATION ERROR: {exc}")
                generations.append({
                    "id": entry["id"],
                    "category": entry["category"],
                    "generated_text": "",
                    "tokens_per_second": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "generation_error": str(exc),
                    "phase": "generated",
                })

        print("[phase 1] Freeing model …")
        del model
        del tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("[phase 1] Model freed.")

    # ------------------------------------------------------------------
    # Phase 2: Docker verification (model is gone)
    # ------------------------------------------------------------------
    print("[phase 2] Starting Docker verification …")

    # Build entry lookup
    entry_by_id = {e["id"]: e for e in entries}

    # Merge with already-done
    all_records: list[dict] = list(already_done.values())

    total_gen = len(generations)
    for i, gen_rec in enumerate(generations, 1):
        entry_id = gen_rec["id"]
        entry = entry_by_id[entry_id]

        if gen_rec.get("generation_error"):
            record = {**gen_rec, "parse_status": "failed",
                      "verify_status": None, "correct": False,
                      "error": gen_rec["generation_error"]}
        else:
            print(f"[verify {i}/{total_gen}] {entry_id} …", end=" ", flush=True)
            t0 = time.monotonic()
            vresult = _verify_in_docker(entry, gen_rec["generated_text"])
            elapsed = time.monotonic() - t0
            status_str = "PASS" if vresult["correct"] else f"FAIL({vresult['verify_status']})"
            print(f"{status_str} ({elapsed:.1f}s)")
            record = {
                "id": entry_id,
                "category": gen_rec["category"],
                "tokens_per_second": gen_rec["tokens_per_second"],
                "prompt_tokens": gen_rec["prompt_tokens"],
                "completion_tokens": gen_rec["completion_tokens"],
                "parse_status": vresult["parse_status"],
                "verify_status": vresult["verify_status"],
                "correct": vresult["correct"],
                "error": vresult.get("error"),
            }

        _append_record(record)
        all_records.append(record)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report = _build_report(all_records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n=== LOCAL BASELINE REPORT ===")
    print(f"Model: {report['model']}")
    print(f"Note: {report['note']}")
    print(f"Entries: {report['total_entries']}")
    print(f"Parse OK: {report['parse_ok']} / {report['total_entries']} "
          f"({report['parse_rate']*100:.1f}%)")
    print(f"Correct (pass@1): {report['correct']} / {report['total_entries']} "
          f"({report['pass_at_1']*100:.1f}%)")
    print(f"Wilson 95% CI: [{report['wilson_95_ci'][0]*100:.1f}%, "
          f"{report['wilson_95_ci'][1]*100:.1f}%]")
    if report["tokens_per_second_median"]:
        print(f"Tokens/s: median={report['tokens_per_second_median']}, "
              f"mean={report['tokens_per_second_mean']}")
    print("\nFailure types:", report["failure_types"])
    print("\nPer-category:")
    for cat, row in report["per_category"].items():
        print(f"  {cat:20s}  {row['correct']:2d}/{row['n']:2d}  "
              f"p={row['pass_at_1']*100:.0f}%  "
              f"CI=[{row['wilson_lo']*100:.0f}%,{row['wilson_hi']*100:.0f}%]")
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
