#!/usr/bin/env python3
"""
Kaggle final eval: compare base model vs fine-tuned adapter on the same 60
entries, same prompt (notes=False), same verifier, same max_new_tokens.

Device selection:
  CUDA available → 4-bit NF4 (BitsAndBytes, same as training)
  MPS available  → float32
  CPU only       → float32

Usage:
  # Base model only (no adapter)
  python scripts/kaggle_eval.py --out evaluation/results/kaggle_base_eval_report.json

  # Fine-tuned adapter
  python scripts/kaggle_eval.py \
      --adapter training/run_*/adapter \
      --out evaluation/results/kaggle_finetuned_eval_report.json

  # CPU dry-run (no Docker, 3 entries only)
  python scripts/kaggle_eval.py --pilot --no-verify

Fail conditions:
  - Exits 1 if degenerate_output rate > 10% (§1f requirement)
  - Exits 1 if total_entries == 0 after generation

Report schema: identical to fp32_baseline_report.json with two extra fields:
  "adapter": path string or null
  "quantization": "4bit_nf4" | "float32"
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

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 1536
INTERPRETER_TIMEOUT_S = 90
IMPORT_HEADER = "import torch\nimport triton\nimport triton.language as tl\n\n"
CATEGORIES = ["quantized_matmul", "attention", "kv_cache", "norm", "rope"]
ENTRIES_PER_CATEGORY = 12
DEGENERATE_RATE_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Entry selection (identical to fp32_baseline_eval.py)
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
# Model loading
# ---------------------------------------------------------------------------

def _load_model(adapter: str | None):
    """
    Returns (model, tokenizer, device_str, quantization_str).

    CUDA:  4-bit NF4 via BitsAndBytesConfig (matches training/finetune.py).
           If adapter given, PeftModel wraps the base model.
    MPS/CPU: float32, no quantization.
             Adapter loading on MPS/CPU requires merge_and_unload() which is
             memory-intensive — warn but proceed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        device = "cuda"
        quant = "4bit_nf4"
    elif torch.backends.mps.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            device_map="mps",
            trust_remote_code=True,
        )
        device = "mps"
        quant = "float32"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        device = "cpu"
        quant = "float32"

    if adapter:
        from peft import PeftModel
        adapter_path = Path(adapter).expanduser().resolve()
        if not adapter_path.exists():
            print(f"[ERROR] Adapter path does not exist: {adapter_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[load] Loading adapter from {adapter_path} …")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
    else:
        model.eval()

    return model, tokenizer, device, quant


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_one(model, tokenizer, entry: dict, device: str) -> tuple[str, float, int, int, bool]:
    """Returns (text, tok/s, prompt_tokens, completion_tokens, truncated)."""
    import torch

    messages = build_messages(entry["description"], entry["pytorch_reference"])
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device != "cpu":
        # For device_map="auto" (CUDA) tensors should go to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
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
# Docker verification
# ---------------------------------------------------------------------------

def _docker_verify(entry: dict, kernel_text: str, suffix: str, tmp_dir: Path) -> dict:
    """Run one Docker interpreter verification. Returns {correct, error, last_error}."""
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
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(log_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["id"]] = rec
            except json.JSONDecodeError:
                pass
    return done


def _write_record(record: dict, log_path: Path) -> None:
    done = _load_checkpoint(log_path)
    done[record["id"]] = record
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        for rec in done.values():
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _categorise_error(err: str) -> str:
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


def _build_report(records: list[dict], model_name: str, adapter: str | None,
                  quantization: str) -> dict:
    total = len(records)
    correct_no_hdr   = sum(1 for r in records if r.get("no_header_correct"))
    correct_with_hdr = sum(1 for r in records if r.get("with_header_correct"))
    degenerate       = sum(1 for r in records if r.get("degenerate"))
    truncated        = sum(1 for r in records if r.get("truncated"))

    no_hdr_errors: dict[str, int] = {}
    no_hdr_buckets: dict[str, int] = {}
    with_hdr_buckets: dict[str, int] = {}

    for r in records:
        if r.get("degenerate"):
            no_hdr_errors["degenerate_output"] = no_hdr_errors.get("degenerate_output", 0) + 1
            no_hdr_buckets["degenerate_output"] = no_hdr_buckets.get("degenerate_output", 0) + 1
            continue
        last = r.get("no_header_last_error") or "(none)"
        no_hdr_errors[last] = no_hdr_errors.get(last, 0) + 1
        bucket = _categorise_error(last)
        no_hdr_buckets[bucket] = no_hdr_buckets.get(bucket, 0) + 1
        if r.get("with_header_last_error"):
            last2 = r.get("with_header_last_error")
            b2 = _categorise_error(last2)
            with_hdr_buckets[b2] = with_hdr_buckets.get(b2, 0) + 1
        elif r.get("with_header_correct"):
            with_hdr_buckets["correct"] = with_hdr_buckets.get("correct", 0) + 1

    name_error_entries = [
        r for r in records
        if not r.get("degenerate") and "NameError" in (r.get("no_header_last_error") or "")
    ]
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
        "model": model_name,
        "adapter": adapter,
        "quantization": quantization,
        "dtype": "float16 (4bit_nf4 compute)" if quantization == "4bit_nf4" else "float32",
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
            "top_errors": dict(sorted(no_hdr_errors.items(), key=lambda x: -x[1])[:10]),
        },
        "with_header": {
            "entries_tested": len(name_error_entries),
            "correct": correct_with_hdr_ne,
            "pass_at_1": (round(correct_with_hdr_ne / len(name_error_entries), 4)
                          if name_error_entries else 0.0),
            "error_buckets": with_hdr_buckets,
        },
        "per_category": per_category,
        "tokens_per_second_median": (round(sorted(tps)[len(tps) // 2], 1) if tps else None),
        "tokens_per_second_mean": (round(sum(tps) / len(tps), 1) if tps else None),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaggle final eval: base vs fine-tuned on the 60-entry subset."
    )
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter directory (omit for base model only).")
    parser.add_argument("--out", default=None,
                        help="Output JSON report path. Defaults to "
                             "evaluation/results/kaggle_{base|finetuned}_eval_report.json")
    parser.add_argument("--log", default=None,
                        help="JSONL checkpoint path. Defaults alongside --out.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing checkpoint.")
    parser.add_argument("--pilot", action="store_true",
                        help="3 entries only (smoke test; skips Docker if --no-verify).")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip Docker verification (CPU dry-run mode).")
    args = parser.parse_args()

    tag = "finetuned" if args.adapter else "base"
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"kaggle_{tag}_eval_report.json"
    log_path = Path(args.log) if args.log else out_path.with_suffix(".jsonl")

    entries = _build_entry_list(pilot=args.pilot)
    done = _load_checkpoint(log_path) if args.resume else {}
    todo_gen = [e for e in entries
                if e["id"] not in done or done[e["id"]].get("phase") == "generated"]
    already_verified = [e for e in entries
                        if e["id"] in done and done[e["id"]].get("phase") == "verified"]

    print(f"[kaggle_eval] tag={tag} entries={len(entries)} "
          f"todo_gen={len(todo_gen)} already_verified={len(already_verified)}")
    print(f"[kaggle_eval] out={out_path}")

    # ------------------------------------------------------------------
    # Phase 1: Generation
    # ------------------------------------------------------------------
    model = tokenizer = device = quant = None

    if todo_gen:
        print("[phase 1] Loading model …")
        model, tokenizer, device, quant = _load_model(args.adapter)
        print(f"[phase 1] device={device} quantization={quant} adapter={args.adapter}")

        for i, entry in enumerate(todo_gen, 1):
            eid = entry["id"]
            if eid in done and done[eid].get("phase") == "verified":
                continue
            print(f"[gen {i}/{len(todo_gen)}] {eid} …", end=" ", flush=True)
            try:
                text, tps, ptok, ctok, trunc = _generate_one(model, tokenizer, entry, device)
                degen = is_degenerate_output(text)
                label = ""
                if trunc:
                    label += "  TRUNCATED"
                if degen:
                    label += "  DEGENERATE"
                print(f"{ctok} tok, {tps:.1f} tok/s{label}")
            except Exception as exc:
                print(f"ERROR: {exc}")
                text, tps, ptok, ctok, trunc, degen = "", 0.0, 0, 0, False, False

            rec: dict = {
                "id": eid,
                "category": entry["category"],
                "phase": "generated",
                "tokens_per_second": tps,
                "prompt_tokens": ptok,
                "completion_tokens": ctok,
                "truncated": trunc,
                "degenerate": degen,
                "raw_text": text,
            }
            _write_record(rec, log_path)
            done[eid] = rec

        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        print("[phase 1] Model freed.")

    # ------------------------------------------------------------------
    # Phase 2: Verification
    # ------------------------------------------------------------------
    if args.no_verify:
        print("\n[phase 2] Skipping verification (--no-verify).")
        # Mark all generated entries as verified with no-verify sentinel
        for entry in entries:
            eid = entry["id"]
            rec = done.get(eid, {})
            if rec.get("phase") != "verified":
                rec.update({
                    "phase": "verified",
                    "extracted_kernel": None,
                    "no_header_correct": False,
                    "no_header_error": "skipped (--no-verify)",
                    "no_header_last_error": "skipped (--no-verify)",
                    "with_header_correct": None,
                    "with_header_error": None,
                    "with_header_last_error": None,
                })
                _write_record(rec, log_path)
                done[eid] = rec
    else:
        tmp_dir = RESULTS_DIR / "_docker_tmp"
        print("\n[phase 2] Verifying …\n")
        for i, entry in enumerate(entries, 1):
            eid = entry["id"]
            rec = done.get(eid, {})
            if rec.get("phase") == "verified":
                continue

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
                _write_record(rec, log_path)
                done[eid] = rec
                print(f"  [{i:2d}] {eid}  DEGENERATE — skipping Docker")
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
            rec["no_header_error"] = v["error"]
            rec["no_header_last_error"] = v.get("last_error", "")
            status = "PASS" if v["correct"] else f"FAIL({(v.get('last_error','')[:60])})"
            print(status)

            with_hdr = None
            if kernel and "NameError" in (v.get("error") or ""):
                kernel_hdr = IMPORT_HEADER + kernel
                print(f"       {eid}  verifying with-header …", end=" ", flush=True)
                with_hdr = _docker_verify(entry, kernel_hdr, f"_{tag}_wh", tmp_dir)
                status2 = "PASS" if with_hdr["correct"] else f"FAIL({(with_hdr.get('last_error','')[:60])})"
                print(status2)

            rec["with_header_correct"] = with_hdr["correct"] if with_hdr else None
            rec["with_header_error"] = with_hdr["error"] if with_hdr else None
            rec["with_header_last_error"] = with_hdr.get("last_error") if with_hdr else None
            rec["phase"] = "verified"
            _write_record(rec, log_path)
            done[eid] = rec

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    all_records = [done[e["id"]] for e in entries if e["id"] in done]

    if not all_records:
        print("[ERROR] No records to report — generation may have failed.", file=sys.stderr)
        sys.exit(1)

    report = _build_report(
        all_records,
        model_name=MODEL_NAME,
        adapter=str(Path(args.adapter).resolve()) if args.adapter else None,
        quantization=quant or ("4bit_nf4" if not args.no_verify else "unknown"),
    )

    print("\n=== KAGGLE EVAL REPORT ===")
    print(f"tag={tag}  quantization={report['quantization']}  "
          f"adapter={report['adapter']}")
    print(f"Total: {report['total_entries']}  "
          f"truncated: {report['truncated']}  "
          f"degenerate: {report['degenerate_output']}")
    print(f"No-header pass@1={report['no_header']['pass_at_1']:.3f} "
          f"({report['no_header']['correct']}/{report['total_entries']}) "
          f"CI={report['no_header']['wilson_95_ci']}")
    print(f"With-header (NameError entries): "
          f"{report['with_header']['correct']}/{report['with_header']['entries_tested']} correct")
    print(f"tok/s median={report['tokens_per_second_median']}  "
          f"mean={report['tokens_per_second_mean']}")
    print("\nNo-header error buckets:", report["no_header"]["error_buckets"])
    print("\nPer-category:")
    for cat, m in report["per_category"].items():
        print(f"  {cat:<20} {m['no_header_correct']}/{m['n']} "
              f"CI=[{m['wilson_lo']:.3f},{m['wilson_hi']:.3f}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")
    print(f"Log:    {log_path}")

    # ------------------------------------------------------------------
    # §1f: Fail loudly if degenerate rate > 10%
    # ------------------------------------------------------------------
    total = report["total_entries"]
    deg_count = report["degenerate_output"]
    if total > 0 and deg_count / total > DEGENERATE_RATE_THRESHOLD:
        pct = 100 * deg_count / total
        print(
            f"\n[FATAL] Degenerate output rate {deg_count}/{total} = {pct:.1f}% "
            f"exceeds threshold {int(DEGENERATE_RATE_THRESHOLD * 100)}%.\n"
            "This suggests fp16 NaN or quantization failure. Check:\n"
            "  1. load_in_4bit=True in training/configs/finetune_default.json\n"
            "  2. No dtype override bypassing quantization in KernelGenerator\n"
            "  3. T4 fp16 support: T4 supports fp16; bf16 is NOT supported on T4\n",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
