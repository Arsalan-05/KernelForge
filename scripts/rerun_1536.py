#!/usr/bin/env python3
"""
Rerun 12 entries with max_new_tokens=1536 (vs 768 in the main baseline run).

Purpose: test whether the 768-token truncation SyntaxErrors are fixed by
         giving the model more room, and whether pass@1 changes.

Selection: 12 smoke entries covering all 5 categories
  quantized_matmul: 3 smoke
  attention:        3 smoke
  kv_cache:         2 smoke
  norm:             2 smoke  (both had '(' was never closed at 768 tokens)
  rope:             2 smoke

Outputs:
  evaluation/results/rerun_1536_eval.jsonl    checkpoint
  evaluation/results/rerun_1536_report.json   summary
"""

from __future__ import annotations

import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import SMOKE_GENERATORS
from kernelforge.parsing import parse_model_output

RESULTS_DIR = repo_root / "evaluation" / "results"
LOG_PATH = RESULTS_DIR / "rerun_1536_eval.jsonl"
REPORT_PATH = RESULTS_DIR / "rerun_1536_report.json"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 1536
TEMPERATURE = 0.0
INTERPRETER_TIMEOUT_S = 90

PLAN = {
    "quantized_matmul": 3,
    "attention":        3,
    "kv_cache":         2,
    "norm":             2,
    "rope":             2,
}


def _build_entry_list() -> list[dict]:
    entries: list[dict] = []
    for cat, n in PLAN.items():
        smoke = list(SMOKE_GENERATORS[cat]())
        entries.extend(smoke[:n])
    return entries


def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _generate_one(model, tokenizer, entry: dict, device):
    import torch
    from kernelforge.prompts import build_messages

    messages = build_messages(entry["description"], entry["pytorch_reference"], notes=True)
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
    completion_tokens = len(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    return text, tps, prompt_len, completion_tokens


def _verify_in_docker(entry: dict, generated_text: str) -> dict:
    parsed = parse_model_output(generated_text)
    if not parsed.has_kernel:
        return {
            "parse_status": "failed",
            "verify_status": None,
            "correct": False,
            "error": "parse_model_output: no kernel section found",
        }

    tmp_dir = RESULTS_DIR / "_docker_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    entry_id_safe = entry["id"].replace("/", "_")
    kernel_path = tmp_dir / f"{entry_id_safe}_k1536.py"
    ref_path = tmp_dir / f"{entry_id_safe}_r1536.py"
    kernel_path.write_text(parsed.triton_kernel)
    ref_path.write_text(entry["pytorch_reference"])

    docker_kernel = f"/work/evaluation/results/_docker_tmp/{entry_id_safe}_k1536.py"
    docker_ref    = f"/work/evaluation/results/_docker_tmp/{entry_id_safe}_r1536.py"

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
            cmd, capture_output=True, text=True,
            timeout=INTERPRETER_TIMEOUT_S + 15,
        )
        if proc.returncode != 0:
            return {
                "parse_status": "ok", "verify_status": "error",
                "correct": False,
                "error": (proc.stderr or proc.stdout or "")[-3000:],
            }
        last_line = proc.stdout.strip().splitlines()[-1]
        result = json.loads(last_line)
        return {
            "parse_status": "ok",
            "verify_status": result["status"],
            "correct": result["correct"],
            "error": result.get("error"),
        }
    except subprocess.TimeoutExpired:
        return {"parse_status": "ok", "verify_status": "timeout",
                "correct": False, "error": "timeout"}
    except Exception as exc:
        return {"parse_status": "ok", "verify_status": "error",
                "correct": False, "error": str(exc)}
    finally:
        for p in (kernel_path, ref_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


def _last_error_line(err: str) -> str:
    if not err:
        return "(none)"
    for line in reversed(err.strip().split("\n")):
        if line.strip():
            return line.strip()
    return "(empty)"


def main() -> None:
    entries = _build_entry_list()
    print(f"[rerun_1536] {len(entries)} entries, max_new_tokens={MAX_NEW_TOKENS}, model={MODEL_NAME}")

    # Phase 1: generation
    print("[phase 1] Loading model on MPS …")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[phase 1] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    model.eval()

    generations: list[dict] = []
    for i, entry in enumerate(entries, 1):
        print(f"[gen {i}/{len(entries)}] {entry['id']} …", end=" ", flush=True)
        try:
            text, tps, prompt_tok, comp_tok = _generate_one(model, tokenizer, entry, device)
            truncated = comp_tok == MAX_NEW_TOKENS
            print(f"{comp_tok} tokens, {tps:.1f} tok/s{'  TRUNCATED' if truncated else ''}")
            generations.append({
                "id": entry["id"], "category": entry["category"],
                "generated_text": text, "tokens_per_second": tps,
                "prompt_tokens": prompt_tok, "completion_tokens": comp_tok,
                "truncated": truncated,
            })
        except Exception as exc:
            print(f"ERROR: {exc}")
            generations.append({
                "id": entry["id"], "category": entry["category"],
                "generated_text": "", "tokens_per_second": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "truncated": False, "generation_error": str(exc),
            })

    print("[phase 1] Freeing model …")
    del model, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("[phase 1] Model freed.")

    # Phase 2: Docker verification
    print("[phase 2] Starting Docker verification …")
    entry_by_id = {e["id"]: e for e in entries}
    records: list[dict] = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, gen in enumerate(generations, 1):
        entry = entry_by_id[gen["id"]]
        print(f"[verify {i}/{len(generations)}] {gen['id']} …", end=" ", flush=True)
        t0 = time.monotonic()
        vresult = _verify_in_docker(entry, gen.get("generated_text", ""))
        elapsed = time.monotonic() - t0
        status_str = "PASS" if vresult["correct"] else f"FAIL({vresult['verify_status']})"
        print(f"{status_str} ({elapsed:.1f}s)")

        record = {
            "id": gen["id"], "category": gen["category"],
            "tokens_per_second": gen["tokens_per_second"],
            "prompt_tokens": gen["prompt_tokens"],
            "completion_tokens": gen["completion_tokens"],
            "truncated": gen.get("truncated", False),
            "parse_status": vresult["parse_status"],
            "verify_status": vresult["verify_status"],
            "correct": vresult["correct"],
            "last_error_line": _last_error_line(vresult.get("error") or ""),
            "error": vresult.get("error"),
        }
        records.append(record)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # Report
    n = len(records)
    correct = sum(1 for r in records if r["correct"])
    lo, hi = _wilson_ci(correct, n)
    truncated_count = sum(1 for r in records if r.get("truncated"))

    import collections
    error_dist = collections.Counter(r["last_error_line"] for r in records if not r["correct"])

    per_cat: dict[str, dict] = {}
    for cat in PLAN:
        cat_recs = [r for r in records if r["category"] == cat]
        k = sum(1 for r in cat_recs if r["correct"])
        clo, chi = _wilson_ci(k, len(cat_recs))
        per_cat[cat] = {"n": len(cat_recs), "correct": k,
                        "wilson_lo": round(clo, 4), "wilson_hi": round(chi, 4)}

    report = {
        "model": MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "note": "BASE model, not fine-tuned. CPU interpreter only. N=12.",
        "total_entries": n,
        "correct": correct,
        "pass_at_1": round(correct / n, 4) if n else 0.0,
        "wilson_95_ci": [round(lo, 4), round(hi, 4)],
        "truncated": truncated_count,
        "error_distribution": dict(error_dist.most_common(10)),
        "per_category": per_cat,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n=== RERUN 1536 REPORT ===")
    print(f"max_new_tokens=1536  (baseline was 768)")
    print(f"Correct (pass@1): {correct}/{n} ({correct/n*100:.1f}%)")
    print(f"Wilson 95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"Truncated at 1536: {truncated_count}/{n}")
    print("\nError distribution (last error line):")
    for msg, cnt in error_dist.most_common():
        print(f"  {cnt:3d}  {msg[:100]}")
    print("\nPer-category:")
    for cat, row in per_cat.items():
        print(f"  {cat:20s}  {row['correct']}/{row['n']}")
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
