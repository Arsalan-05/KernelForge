#!/usr/bin/env python3
"""
FP32 base-model baseline: Qwen/Qwen2.5-Coder-1.5B-Instruct on the 60-entry
subset in float32 on MPS, using the TRAINING prompt (notes=False, identical
to finetune.py and the Kaggle eval), max_new_tokens=1536, temperature 0.

Verifies every entry twice:
  no-header  — kernel as extracted
  with-header — 'import torch\\nimport triton\\nimport triton.language as tl\\n\\n'
                prepended (only run if no-header fails with NameError)

Stores raw_text and extracted_kernel per record.
Applies the degenerate_output guard from kernelforge.parsing.

Outputs:
  evaluation/results/fp32_baseline_eval.jsonl   checkpoint (one record per line)
  evaluation/results/fp32_baseline_report.json  final summary

Checkpoint format per line:
  {id, category, phase, ...generation fields..., ...verification fields...}

phase values:
  "generated"           — generation recorded, verification pending
  "verified"            — both verification passes recorded

Resume: --resume skips entries already in the checkpoint with phase="verified".

Usage:
  python scripts/fp32_baseline_eval.py
  python scripts/fp32_baseline_eval.py --resume
  python scripts/fp32_baseline_eval.py --pilot   (3 entries only)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import GENERATORS, SMOKE_GENERATORS
from kernelforge.parsing import is_degenerate_output, parse_model_output
from kernelforge.prompts import build_messages

RESULTS_DIR = repo_root / "evaluation" / "results"
LOG_PATH = RESULTS_DIR / "fp32_baseline_eval.jsonl"
REPORT_PATH = RESULTS_DIR / "fp32_baseline_report.json"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 1536
INTERPRETER_TIMEOUT_S = 90
IMPORT_HEADER = "import torch\nimport triton\nimport triton.language as tl\n\n"

CATEGORIES = ["quantized_matmul", "attention", "kv_cache", "norm", "rope"]
ENTRIES_PER_CATEGORY = 12


# ---------------------------------------------------------------------------
# Entry selection (same as local_baseline_eval.py)
# ---------------------------------------------------------------------------

def _build_entry_list(pilot: bool = False) -> list[dict]:
    selected: list[dict] = []
    for cat in CATEGORIES:
        smoke = list(SMOKE_GENERATORS[cat]())
        regular = list(GENERATORS[cat]())
        cat_entries: list[dict] = smoke[:]
        smoke_ids = {e["id"] for e in smoke}
        for e in regular:
            if e["id"] not in smoke_ids and len(cat_entries) < ENTRIES_PER_CATEGORY:
                cat_entries.append(e)
        selected.extend(cat_entries[:ENTRIES_PER_CATEGORY])
    if pilot:
        return [list(SMOKE_GENERATORS[c]())[0]
                for c in ["quantized_matmul", "attention", "norm"]]
    return selected


# ---------------------------------------------------------------------------
# Wilson 95% CI
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_one(model, tokenizer, entry: dict, device) -> tuple[str, float, int, int, bool]:
    """Returns (text, tok/s, prompt_tokens, completion_tokens, truncated)."""
    import torch

    messages = build_messages(entry["description"], entry["pytorch_reference"])
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

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

    generated_ids = output[0][prompt_len:]
    comp = len(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tps = comp / elapsed if elapsed > 0 else 0.0
    return text, tps, prompt_len, comp, (comp == MAX_NEW_TOKENS)


# ---------------------------------------------------------------------------
# Docker verification (called AFTER model is freed)
# ---------------------------------------------------------------------------

def _docker_verify(entry: dict, kernel_text: str, suffix: str) -> dict:
    """Run one Docker interpreter verification. Returns {correct, error, last_error}."""
    tmp_dir = RESULTS_DIR / "_docker_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    entry_id_safe = entry["id"].replace("/", "_") + suffix
    kernel_path = tmp_dir / f"{entry_id_safe}_kernel.py"
    ref_path    = tmp_dir / f"{entry_id_safe}_ref.py"
    kernel_path.write_text(kernel_text)
    ref_path.write_text(entry["pytorch_reference"])

    dk = f"/work/evaluation/results/_docker_tmp/{entry_id_safe}_kernel.py"
    dr = f"/work/evaluation/results/_docker_tmp/{entry_id_safe}_ref.py"
    script = (
        "import sys, json; sys.path.insert(0, '/work'); "
        "from verification.verify_kernel import verify_model_kernel; "
        f"ref = open({repr(dr)}).read(); kernel = open({repr(dk)}).read(); "
        f"r = verify_model_kernel(ref, kernel, device='cpu', "
        f"    timeout_s={INTERPRETER_TIMEOUT_S}, interpret=True); "
        "print(json.dumps({'status': r.status, 'correct': r.correct, 'error': r.error}))"
    )
    cmd = ["docker", "run", "--rm", "-v", f"{repo_root}:/work",
           "kernelforge-verify", "python", "-c", script]

    def _last(err: str) -> str:
        if not err:
            return "(none)"
        for line in reversed(err.strip().splitlines()):
            if line.strip():
                return line.strip()
        return "(empty)"

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=INTERPRETER_TIMEOUT_S + 15)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-2000:]
            return {"correct": False, "error": err, "last_error": _last(err)}
        last = proc.stdout.strip().splitlines()[-1]
        d = json.loads(last)
        return {"correct": bool(d["correct"]),
                "error": d.get("error"),
                "last_error": _last(d.get("error") or "")}
    except subprocess.TimeoutExpired:
        msg = f"outer timeout ({INTERPRETER_TIMEOUT_S + 15}s)"
        return {"correct": False, "error": msg, "last_error": msg}
    except Exception as exc:
        return {"correct": False, "error": str(exc), "last_error": str(exc)[:200]}
    finally:
        for p in (kernel_path, ref_path):
            try: p.unlink(missing_ok=True)
            except OSError: pass


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict[str, dict]:
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


def _write_record(record: dict) -> None:
    """Overwrite the entry for this id by rewriting the whole checkpoint file."""
    done = _load_checkpoint()
    done[record["id"]] = record
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w") as f:
        for rec in done.values():
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _last_error_line(err: str | None) -> str:
    if not err:
        return "(none)"
    for line in reversed(err.strip().splitlines()):
        if line.strip():
            return line.strip()
    return "(empty)"


def _build_report(records: list[dict]) -> dict:
    total = len(records)
    correct_no_hdr  = sum(1 for r in records if r.get("no_header_correct"))
    correct_with_hdr = sum(1 for r in records if r.get("with_header_correct"))
    degenerate = sum(1 for r in records if r.get("degenerate"))
    truncated = sum(1 for r in records if r.get("truncated"))

    # Error breakdown: last-line of traceback for no-header failures
    no_hdr_errors: dict[str, int] = {}
    with_hdr_errors: dict[str, int] = {}
    for r in records:
        if r.get("degenerate"):
            no_hdr_errors["degenerate_output"] = no_hdr_errors.get("degenerate_output", 0) + 1
            continue
        last = r.get("no_header_last_error") or "(none)"
        no_hdr_errors[last] = no_hdr_errors.get(last, 0) + 1
        if r.get("with_header_last_error"):
            last2 = r.get("with_header_last_error")
            with_hdr_errors[last2] = with_hdr_errors.get(last2, 0) + 1

    # Categorise by error type (not raw message)
    def _categorise(err: str) -> str:
        if "NameError" in err:
            return "name_error"
        if "SyntaxError" in err or "IndentationError" in err:
            return "syntax_error"
        if "AttributeError" in err or "TypeError" in err:
            return "type_error"
        if "degenerate_output" in err:
            return "degenerate_output"
        if err == "(none)":
            return "no_error_reported"
        return "other"

    no_hdr_buckets: dict[str, int] = {}
    for r in records:
        if r.get("degenerate"):
            no_hdr_buckets["degenerate_output"] = no_hdr_buckets.get("degenerate_output", 0) + 1
        else:
            cat = _categorise(r.get("no_header_last_error") or "(none)")
            no_hdr_buckets[cat] = no_hdr_buckets.get(cat, 0) + 1

    with_hdr_buckets: dict[str, int] = {}
    for r in records:
        if r.get("with_header_correct") is not None:
            if r.get("with_header_correct"):
                with_hdr_buckets["correct"] = with_hdr_buckets.get("correct", 0) + 1
            else:
                cat = _categorise(r.get("with_header_last_error") or "(none)")
                with_hdr_buckets[cat] = with_hdr_buckets.get(cat, 0) + 1

    name_error_entries = [r for r in records
                          if not r.get("degenerate")
                          and "NameError" in (r.get("no_header_last_error") or "")]
    correct_with_hdr_ne = sum(1 for r in name_error_entries if r.get("with_header_correct"))

    per_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_recs = [r for r in records if r.get("category") == cat]
        n = len(cat_recs)
        k_no  = sum(1 for r in cat_recs if r.get("no_header_correct"))
        k_hdr = sum(1 for r in cat_recs if r.get("with_header_correct"))
        lo, hi = _wilson_ci(k_no, n)
        per_category[cat] = {
            "n": n,
            "no_header_correct": k_no,
            "with_header_any_correct": k_hdr,
            "pass_at_1_no_header": round(k_no / n, 4) if n else 0.0,
            "wilson_lo": round(lo, 4),
            "wilson_hi": round(hi, 4),
        }

    tps = [r["tokens_per_second"] for r in records if r.get("tokens_per_second", 0) > 0]
    lo_all, hi_all = _wilson_ci(correct_no_hdr, total)

    return {
        "model": MODEL_NAME,
        "dtype": "float32",
        "prompt": "notes=False (training prompt, identical to finetune.py)",
        "max_new_tokens": MAX_NEW_TOKENS,
        "total_entries": total,
        "truncated": truncated,
        "degenerate_output": degenerate,
        "no_header": {
            "correct": correct_no_hdr,
            "pass_at_1": round(correct_no_hdr / total, 4) if total else 0.0,
            "wilson_95_ci": [round(lo_all, 4), round(hi_all, 4)],
            "error_buckets": no_hdr_buckets,
            "top_errors": dict(sorted(no_hdr_errors.items(),
                                      key=lambda x: -x[1])[:10]),
        },
        "with_header": {
            "entries_tested": len(name_error_entries),
            "correct": correct_with_hdr_ne,
            "pass_at_1": round(correct_with_hdr_ne / len(name_error_entries), 4)
                         if name_error_entries else 0.0,
            "error_buckets": with_hdr_buckets,
        },
        "per_category": per_category,
        "tokens_per_second_median": round(sorted(tps)[len(tps) // 2], 1) if tps else None,
        "tokens_per_second_mean": round(sum(tps) / len(tps), 1) if tps else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="3 entries only")
    args = parser.parse_args()

    entries = _build_entry_list(pilot=args.pilot)
    done = _load_checkpoint() if args.resume else {}
    # For resume: skip entries that already have verification complete
    todo_gen = [e for e in entries
                if e["id"] not in done or done[e["id"]].get("phase") == "generated"]
    already_verified = [e for e in entries
                        if e["id"] in done and done[e["id"]].get("phase") == "verified"]

    print(f"[fp32_baseline] entries={len(entries)} "
          f"todo_gen={len(todo_gen)} already_verified={len(already_verified)} "
          f"model={MODEL_NAME} dtype=float32 max_new_tokens={MAX_NEW_TOKENS}")

    # ------------------------------------------------------------------
    # Phase 1: Generation
    # ------------------------------------------------------------------
    if todo_gen:
        print("[phase 1] Loading model (float32 on MPS) …")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[phase 1] device={device}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        for i, entry in enumerate(todo_gen, 1):
            eid = entry["id"]
            if eid in done and done[eid].get("phase") != "generated":
                continue  # already verified on a previous resume
            print(f"[gen {i}/{len(todo_gen)}] {eid} …", end=" ", flush=True)
            try:
                text, tps, ptok, ctok, trunc = _generate_one(model, tokenizer, entry, device)
                degen = is_degenerate_output(text)
                label = ""
                if trunc: label += "  TRUNCATED"
                if degen: label += "  DEGENERATE"
                print(f"{ctok} tok, {tps:.1f} tok/s{label}")
            except Exception as exc:
                print(f"ERROR: {exc}")
                text, tps, ptok, ctok, trunc, degen = "", 0.0, 0, 0, False, False

            rec = {
                "id": eid, "category": entry["category"],
                "phase": "generated",
                "tokens_per_second": tps,
                "prompt_tokens": ptok,
                "completion_tokens": ctok,
                "truncated": trunc,
                "degenerate": degen,
                "raw_text": text,
            }
            _write_record(rec)
            done[eid] = rec

        del model, tokenizer
        gc.collect()
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("[phase 1] Model freed.")

    # ------------------------------------------------------------------
    # Phase 2: Verification
    # ------------------------------------------------------------------
    print("\n[phase 2] Verifying …\n")
    for i, entry in enumerate(entries, 1):
        eid = entry["id"]
        rec = done.get(eid, {})
        if rec.get("phase") == "verified":
            continue  # already done

        text = rec.get("raw_text", "")
        degen = rec.get("degenerate", False) or is_degenerate_output(text)

        if degen or not text:
            rec.update({
                "phase": "verified",
                "extracted_kernel": None,
                "no_header_correct": False,
                "no_header_error": "degenerate_output" if degen else "no text generated",
                "no_header_last_error": "degenerate_output" if degen else "no text",
                "with_header_correct": None,
                "with_header_error": None,
                "with_header_last_error": None,
            })
            _write_record(rec)
            done[eid] = rec
            print(f"  [{i:2d}] {eid}  DEGENERATE — skipping Docker")
            continue

        parsed = parse_model_output(text)
        kernel = parsed.triton_kernel or ""
        rec["extracted_kernel"] = kernel

        # No-header verification
        if kernel:
            print(f"  [{i:2d}] {eid}  verifying no-header …", end=" ", flush=True)
            v = _docker_verify(entry, kernel, "_fp32_nh")
        else:
            v = {"correct": False, "error": "no kernel extracted", "last_error": "no kernel extracted"}

        rec["no_header_correct"] = v["correct"]
        rec["no_header_error"] = v["error"]
        rec["no_header_last_error"] = v.get("last_error", "")
        print(f"{'PASS' if v['correct'] else 'FAIL(' + (v.get('last_error','')[:60]) + ')'}")

        # With-header verification (only if NameError)
        with_hdr = None
        if kernel and "NameError" in (v.get("error") or ""):
            kernel_hdr = IMPORT_HEADER + kernel
            print(f"       {eid}  verifying with-header …", end=" ", flush=True)
            with_hdr = _docker_verify(entry, kernel_hdr, "_fp32_wh")
            print(f"{'PASS' if with_hdr['correct'] else 'FAIL(' + (with_hdr.get('last_error','')[:60]) + ')'}")

        rec["with_header_correct"] = with_hdr["correct"] if with_hdr else None
        rec["with_header_error"] = with_hdr["error"] if with_hdr else None
        rec["with_header_last_error"] = with_hdr.get("last_error") if with_hdr else None
        rec["phase"] = "verified"
        _write_record(rec)
        done[eid] = rec

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    all_records = [done[e["id"]] for e in entries if e["id"] in done]
    report = _build_report(all_records)

    name_err = [r for r in all_records if "NameError" in (r.get("no_header_last_error") or "")]

    print("\n=== FP32 BASELINE REPORT ===")
    print(f"dtype=float32  prompt=notes=False  max_new_tokens={MAX_NEW_TOKENS}")
    print(f"Total: {report['total_entries']}  "
          f"truncated: {report['truncated']}  "
          f"degenerate: {report['degenerate_output']}")
    print(f"No-header: {report['no_header']['correct']}/{report['total_entries']} correct "
          f"pass@1={report['no_header']['pass_at_1']:.3f} "
          f"CI={report['no_header']['wilson_95_ci']}")
    print(f"With-header (NameError entries only): "
          f"{report['with_header']['correct']}/{report['with_header']['entries_tested']} correct")
    print(f"tok/s median={report['tokens_per_second_median']}  mean={report['tokens_per_second_mean']}")
    print("\nNo-header error buckets:", report['no_header']['error_buckets'])
    print("\nPer-category:")
    for cat, m in report['per_category'].items():
        print(f"  {cat:<20} {m['no_header_correct']}/{m['n']} "
              f"CI=[{m['wilson_lo']:.3f},{m['wilson_hi']:.3f}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {REPORT_PATH}")
    print(f"Log:    {LOG_PATH}")


if __name__ == "__main__":
    main()
