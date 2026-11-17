#!/usr/bin/env python3
"""
Evaluate a model on the held-out test split (data/held_out_split.json).

The held-out test set contains entries from 5 templates (1 per category, seed=42)
that were NEVER in the training data. Use this script for any post-fine-tuning
evaluation to avoid train-test leakage.

Device: CUDA → 4-bit NF4 (if bitsandbytes available); MPS/CPU → float32.

Usage:
  # Base model (no adapter)
  python scripts/held_out_eval.py --out evaluation/results/held_out_base.json

  # Fine-tuned adapter
  python scripts/held_out_eval.py \\
      --adapter training/checkpoints/<run>/adapter \\
      --out evaluation/results/held_out_finetuned.json

  # CPU dry-run (no Docker, 4 entries per category = 20 total)
  python scripts/held_out_eval.py --pilot --no-verify

  # Limit entries per category (e.g. 4 for a quick 20-entry run)
  python scripts/held_out_eval.py --per-category 4

Leakage guarantee:
  All entries evaluated here are from held-out templates; none appear in
  train_ids or val_ids of data/held_out_split.json.

Report schema:
  Identical to fp32_baseline_report.json with extra fields:
    "adapter", "quantization", "eval_set": "held_out_test"

Fail conditions:
  - Exit 1 if degenerate_output rate > 10%.
  - Exit 1 if any evaluated entry appears in the train or val split.
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

from data.templates import GENERATORS
from kernelforge.dataset import template_of
from kernelforge.parsing import is_degenerate_output, parse_model_output
from kernelforge.prompts import build_messages

SPLIT_PATH = repo_root / "data" / "held_out_split.json"
RESULTS_DIR = repo_root / "evaluation" / "results"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 1536
INTERPRETER_TIMEOUT_S = 90
IMPORT_HEADER = "import torch\nimport triton\nimport triton.language as tl\n\n"
CATEGORIES = ["quantized_matmul", "attention", "kv_cache", "norm", "rope"]
DEGENERATE_RATE_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Entry selection from held-out test split
# ---------------------------------------------------------------------------

def _load_split() -> dict:
    if not SPLIT_PATH.exists():
        print(f"[ERROR] Split file not found: {SPLIT_PATH}", file=sys.stderr)
        sys.exit(1)
    return json.loads(SPLIT_PATH.read_text())


def _build_held_out_entries(
    model_name: str,
    per_category: int | None = None,
    pilot: bool = False,
) -> list[dict]:
    """Return entries from the held-out test split, optionally capped per category."""
    split = _load_split()
    test_ids = set(split["test_ids"])
    train_ids = set(split["train_ids"])
    val_ids   = set(split["val_ids"])

    # Load all 300 template candidates
    from kernelforge.dataset import load_template_candidates
    all_entries = {e["id"]: e for e in load_template_candidates()}

    # Pilot: 1 entry per category
    if pilot:
        per_category = 1

    selected = []
    for cat in CATEGORIES:
        cat_test = sorted(
            [all_entries[eid] for eid in test_ids if eid in all_entries and eid.startswith(cat + "__")],
            key=lambda e: e["id"],
        )
        if per_category:
            cat_test = cat_test[:per_category]
        selected.extend(cat_test)

    # Leakage guard: abort if any entry is in train/val
    for e in selected:
        eid = e["id"]
        if eid in train_ids:
            print(f"[FATAL] held_out_eval: entry {eid!r} is in the TRAIN split. "
                  "Aborting to prevent leakage.", file=sys.stderr)
            sys.exit(1)
        if eid in val_ids:
            print(f"[FATAL] held_out_eval: entry {eid!r} is in the VAL split. "
                  "Aborting to prevent leakage.", file=sys.stderr)
            sys.exit(1)

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
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_name: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            trust_remote_code=True,
        )
        device, quant = "cuda", "4bit_nf4"
    elif torch.backends.mps.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
            device_map=None, trust_remote_code=True,
        )
        model = model.to("mps")
        device, quant = "mps", "float32"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
            device_map="cpu", trust_remote_code=True,
        )
        device, quant = "cpu", "float32"

    if adapter:
        from peft import PeftModel
        ap = Path(adapter).expanduser().resolve()
        if not ap.exists():
            print(f"[ERROR] Adapter not found: {ap}", file=sys.stderr)
            sys.exit(1)
        model = PeftModel.from_pretrained(model, str(ap))

    model.eval()
    return model, tokenizer, device, quant


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_one(model, tokenizer, entry: dict, device: str):
    import torch
    messages = build_messages(entry["description"], entry["pytorch_reference"])
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device in ("cuda", "mps"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    t0 = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            temperature=None, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.monotonic() - t0
    generated = output[0][prompt_len:]
    comp = len(generated)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    tps = comp / elapsed if elapsed > 0 else 0.0
    return text, tps, prompt_len, comp, (comp == MAX_NEW_TOKENS)


# ---------------------------------------------------------------------------
# Docker verification (identical to fp32_baseline_eval.py)
# ---------------------------------------------------------------------------

def _docker_verify(entry: dict, kernel: str, suffix: str, tmp_dir: Path) -> dict:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe = entry["id"].replace("/", "_") + suffix
    kp = tmp_dir / f"{safe}_kernel.py"
    rp = tmp_dir / f"{safe}_ref.py"
    kp.write_text(kernel)
    rp.write_text(entry["pytorch_reference"])
    dk = f"/work/evaluation/results/_docker_tmp/{safe}_kernel.py"
    dr = f"/work/evaluation/results/_docker_tmp/{safe}_ref.py"
    script = (
        "import sys, json; sys.path.insert(0, '/work'); "
        "from verification.verify_kernel import verify_model_kernel; "
        f"ref = open({repr(dr)}).read(); kernel = open({repr(dk)}).read(); "
        f"r = verify_model_kernel(ref, kernel, device='cpu', "
        f"    timeout_s={INTERPRETER_TIMEOUT_S}, interpret=True); "
        "print(json.dumps({'correct': r.correct, 'error': r.error}))"
    )
    cmd = ["docker", "run", "--rm", "-v", f"{repo_root}:/work",
           "kernelforge-verify", "python", "-c", script]

    def _last(err):
        if not err:
            return "(none)"
        for line in reversed(err.strip().splitlines()):
            if line.strip():
                return line.strip()
        return "(empty)"

    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=INTERPRETER_TIMEOUT_S + 15)
        if p.returncode != 0:
            err = (p.stderr or p.stdout or "")[-2000:]
            return {"correct": False, "error": err, "last_error": _last(err)}
        d = json.loads(p.stdout.strip().splitlines()[-1])
        return {"correct": bool(d["correct"]), "error": d.get("error"),
                "last_error": _last(d.get("error") or "")}
    except subprocess.TimeoutExpired:
        msg = f"outer timeout ({INTERPRETER_TIMEOUT_S + 15}s)"
        return {"correct": False, "error": msg, "last_error": msg}
    except Exception as exc:
        return {"correct": False, "error": str(exc), "last_error": str(exc)[:200]}
    finally:
        for p in (kp, rp):
            try: p.unlink(missing_ok=True)
            except OSError: pass


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(log_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except json.JSONDecodeError:
                    pass
    return done


def _write_record(rec: dict, log_path: Path) -> None:
    done = _load_checkpoint(log_path)
    done[rec["id"]] = rec
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        for r in done.values():
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _categorise(err: str) -> str:
    if "NameError" in err: return "name_error"
    if "SyntaxError" in err or "IndentationError" in err: return "syntax_error"
    if "AttributeError" in err or "TypeError" in err: return "type_error"
    if "degenerate_output" in err: return "degenerate_output"
    if err in ("(none)", "skipped (--no-verify)"): return "not_verified"
    return "other"


def _build_report(records: list[dict], model_name: str, adapter: str | None,
                  quantization: str, per_category: int | None) -> dict:
    total = len(records)
    correct_nh  = sum(1 for r in records if r.get("no_header_correct"))
    degenerate  = sum(1 for r in records if r.get("degenerate"))
    truncated   = sum(1 for r in records if r.get("truncated"))
    parsed      = sum(1 for r in records if r.get("extracted_kernel"))

    # Error buckets
    nh_errors: dict[str, int] = {}
    nh_buckets: dict[str, int] = {}
    wh_buckets: dict[str, int] = {}
    for r in records:
        if r.get("degenerate"):
            nh_errors["degenerate_output"] = nh_errors.get("degenerate_output", 0) + 1
            nh_buckets["degenerate_output"] = nh_buckets.get("degenerate_output", 0) + 1
            continue
        last = r.get("no_header_last_error") or "(none)"
        nh_errors[last] = nh_errors.get(last, 0) + 1
        b = _categorise(last)
        nh_buckets[b] = nh_buckets.get(b, 0) + 1
        if r.get("with_header_last_error"):
            b2 = _categorise(r.get("with_header_last_error", ""))
            wh_buckets[b2] = wh_buckets.get(b2, 0) + 1
        elif r.get("with_header_correct"):
            wh_buckets["correct"] = wh_buckets.get("correct", 0) + 1

    name_error_entries = [
        r for r in records
        if not r.get("degenerate") and "NameError" in (r.get("no_header_last_error") or "")
    ]
    correct_wh_ne = sum(1 for r in name_error_entries if r.get("with_header_correct"))

    per_cat: dict[str, dict] = {}
    for cat in CATEGORIES:
        cr = [r for r in records if r.get("category") == cat]
        n  = len(cr)
        kn = sum(1 for r in cr if r.get("no_header_correct"))
        kw = sum(1 for r in cr if r.get("with_header_correct"))
        lo, hi = _wilson_ci(kn, n)
        per_cat[cat] = {
            "n": n, "no_header_correct": kn, "with_header_any_correct": kw,
            "pass_at_1_no_header": round(kn / n, 4) if n else 0.0,
            "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
        }

    # Held-out template vs all results (always all test entries here)
    tps = [r["tokens_per_second"] for r in records if r.get("tokens_per_second", 0) > 0]
    lo_all, hi_all = _wilson_ci(correct_nh, total)

    return {
        "model": model_name,
        "adapter": str(Path(adapter).resolve()) if adapter else None,
        "quantization": quantization,
        "dtype": "float32" if quantization == "float32" else "float16 (4bit_nf4 compute)",
        "eval_set": "held_out_test",
        "prompt": "notes=False (training prompt, identical to finetune.py)",
        "max_new_tokens": MAX_NEW_TOKENS,
        "total_entries": total,
        "parsed": parsed,
        "parse_rate": round(parsed / total, 4) if total else 0.0,
        "truncated": truncated,
        "degenerate_output": degenerate,
        "no_header": {
            "correct": correct_nh,
            "pass_at_1": round(correct_nh / total, 4) if total else 0.0,
            "wilson_95_ci": [round(lo_all, 4), round(hi_all, 4)],
            "error_buckets": nh_buckets,
            "top_errors": dict(sorted(nh_errors.items(), key=lambda x: -x[1])[:10]),
        },
        "with_header": {
            "entries_tested": len(name_error_entries),
            "correct": correct_wh_ne,
            "pass_at_1": round(correct_wh_ne / len(name_error_entries), 4)
                         if name_error_entries else 0.0,
            "error_buckets": wh_buckets,
        },
        "per_category": per_cat,
        "tokens_per_second_median": round(sorted(tps)[len(tps) // 2], 1) if tps else None,
        "tokens_per_second_mean":   round(sum(tps) / len(tps), 1) if tps else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate on held-out test split (no train/val leakage)."
    )
    parser.add_argument("--model", default=MODEL_NAME,
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter directory (omit for base model).")
    parser.add_argument("--out", default=None,
                        help="Output JSON report path.")
    parser.add_argument("--log", default=None,
                        help="JSONL checkpoint path.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--per-category", type=int, default=None,
                        help="Cap entries per category (e.g. 4 → 20 total).")
    parser.add_argument("--pilot", action="store_true",
                        help="1 entry per category (5 total); for schema validation.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip Docker verification (generation only).")
    args = parser.parse_args()

    tag = "finetuned" if args.adapter else "base"
    if args.out:
        out_path = Path(args.out)
    else:
        suffix = f"_p{args.per_category}" if args.per_category else ""
        out_path = RESULTS_DIR / f"held_out_{tag}{suffix}_report.json"
    log_path = Path(args.log) if args.log else out_path.with_suffix(".jsonl")

    entries = _build_held_out_entries(args.model, per_category=args.per_category,
                                      pilot=args.pilot)
    done = _load_checkpoint(log_path) if args.resume else {}
    todo_gen = [e for e in entries
                if e["id"] not in done or done[e["id"]].get("phase") == "generated"]
    already = [e for e in entries
               if e["id"] in done and done[e["id"]].get("phase") == "verified"]

    per_cat_msg = f" (per_category={args.per_category})" if args.per_category else ""
    print(f"[held_out_eval] tag={tag} entries={len(entries)}{per_cat_msg} "
          f"todo_gen={len(todo_gen)} already_verified={len(already)}")
    print(f"[held_out_eval] out={out_path}")

    model = tokenizer = device = quant = None

    # ------------------------------------------------------------------
    # Phase 1: Generation
    # ------------------------------------------------------------------
    if todo_gen:
        print(f"[phase 1] Loading {args.model} …")
        model, tokenizer, device, quant = _load_model(args.model, args.adapter)
        print(f"[phase 1] device={device} quant={quant}")

        for i, entry in enumerate(todo_gen, 1):
            eid = entry["id"]
            if eid in done and done[eid].get("phase") == "verified":
                continue
            print(f"[gen {i}/{len(todo_gen)}] {eid} …", end=" ", flush=True)
            try:
                text, tps, ptok, ctok, trunc = _generate_one(model, tokenizer, entry, device)
                degen = is_degenerate_output(text)
                label = ("  TRUNC" if trunc else "") + ("  DEGEN" if degen else "")
                print(f"{ctok} tok, {tps:.1f} tok/s{label}")
            except Exception as exc:
                print(f"ERROR: {exc}")
                text, tps, ptok, ctok, trunc, degen = "", 0.0, 0, 0, False, False

            rec = {"id": eid, "category": entry["category"], "phase": "generated",
                   "tokens_per_second": tps, "prompt_tokens": ptok, "completion_tokens": ctok,
                   "truncated": trunc, "degenerate": degen, "raw_text": text}
            _write_record(rec, log_path)
            done[eid] = rec

        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif torch.backends.mps.is_available(): torch.mps.empty_cache()
        except Exception:
            pass
        print("[phase 1] Model freed.")

    # ------------------------------------------------------------------
    # Phase 2: Verification
    # ------------------------------------------------------------------
    tmp_dir = RESULTS_DIR / "_docker_tmp"
    if args.no_verify:
        print("[phase 2] Skipping verification (--no-verify).")
        for entry in entries:
            eid = entry["id"]
            rec = done.get(eid, {})
            if rec.get("phase") != "verified":
                rec.update({"phase": "verified", "extracted_kernel": None,
                             "no_header_correct": False,
                             "no_header_error": "skipped (--no-verify)",
                             "no_header_last_error": "skipped (--no-verify)",
                             "with_header_correct": None, "with_header_error": None,
                             "with_header_last_error": None})
                _write_record(rec, log_path)
                done[eid] = rec
    else:
        print("\n[phase 2] Verifying …\n")
        for i, entry in enumerate(entries, 1):
            eid = entry["id"]
            rec = done.get(eid, {})
            if rec.get("phase") == "verified":
                continue
            text = rec.get("raw_text", "")
            degen = rec.get("degenerate", False) or is_degenerate_output(text)
            if degen or not text:
                rec.update({"phase": "verified", "extracted_kernel": None,
                             "no_header_correct": False,
                             "no_header_error": "degenerate_output" if degen else "no text",
                             "no_header_last_error": "degenerate_output" if degen else "no text",
                             "with_header_correct": None, "with_header_error": None,
                             "with_header_last_error": None})
                _write_record(rec, log_path)
                done[eid] = rec
                print(f"  [{i:2d}] {eid}  DEGENERATE — skip")
                continue

            parsed = parse_model_output(text)
            kernel = parsed.triton_kernel or ""
            rec["extracted_kernel"] = kernel

            if kernel:
                print(f"  [{i:2d}] {eid}  verifying no-header …", end=" ", flush=True)
                v = _docker_verify(entry, kernel, f"_{tag}_nh", tmp_dir)
            else:
                v = {"correct": False, "error": "no kernel extracted",
                     "last_error": "no kernel extracted"}
            rec["no_header_correct"] = v["correct"]
            rec["no_header_error"]   = v["error"]
            rec["no_header_last_error"] = v.get("last_error", "")
            status = "PASS" if v["correct"] else f"FAIL({v.get('last_error','')[:60]})"
            print(status)

            with_hdr = None
            if kernel and "NameError" in (v.get("error") or ""):
                kh = IMPORT_HEADER + kernel
                print(f"       {eid}  verifying with-header …", end=" ", flush=True)
                with_hdr = _docker_verify(entry, kh, f"_{tag}_wh", tmp_dir)
                s2 = "PASS" if with_hdr["correct"] else f"FAIL({with_hdr.get('last_error','')[:60]})"
                print(s2)

            rec["with_header_correct"]    = with_hdr["correct"] if with_hdr else None
            rec["with_header_error"]      = with_hdr["error"] if with_hdr else None
            rec["with_header_last_error"] = with_hdr.get("last_error") if with_hdr else None
            rec["phase"] = "verified"
            _write_record(rec, log_path)
            done[eid] = rec

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    all_records = [done[e["id"]] for e in entries if e["id"] in done]
    if not all_records:
        print("[ERROR] No records.", file=sys.stderr)
        sys.exit(1)

    report = _build_report(all_records, args.model, args.adapter,
                           quant or "float32", args.per_category)

    print("\n=== HELD-OUT EVAL REPORT ===")
    print(f"model={args.model}  tag={tag}  quant={report['quantization']}  "
          f"eval_set={report['eval_set']}")
    print(f"total={report['total_entries']}  parsed={report['parsed']} "
          f"({report['parse_rate']:.1%})  trunc={report['truncated']}  "
          f"degen={report['degenerate_output']}")
    print(f"No-header pass@1={report['no_header']['pass_at_1']:.3f} "
          f"({report['no_header']['correct']}/{report['total_entries']}) "
          f"CI={report['no_header']['wilson_95_ci']}")
    print(f"With-header: {report['with_header']['correct']}/"
          f"{report['with_header']['entries_tested']}")
    print(f"tok/s median={report['tokens_per_second_median']}  "
          f"mean={report['tokens_per_second_mean']}")
    print("No-header error buckets:", report["no_header"]["error_buckets"])
    print("\nPer-category:")
    for cat, m in report["per_category"].items():
        print(f"  {cat:<20} {m['no_header_correct']}/{m['n']} "
              f"CI=[{m['wilson_lo']:.3f},{m['wilson_hi']:.3f}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")
    print(f"Log:    {log_path}")

    # Degenerate rate guard
    total = report["total_entries"]
    deg   = report["degenerate_output"]
    if total > 0 and deg / total > DEGENERATE_RATE_THRESHOLD:
        pct = 100 * deg / total
        print(f"\n[FATAL] Degenerate rate {deg}/{total} = {pct:.1f}% > "
              f"{int(DEGENERATE_RATE_THRESHOLD*100)}% threshold.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
