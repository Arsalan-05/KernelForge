#!/usr/bin/env python3
"""
Rerun the 4 entries that produced "!!!!" output under fp16 MPS with float32.

Hypothesis: the "!!!!" pattern is the fp16 NaN signature on MPS — NaN logits
cause the argmax to return token 0, which decodes as "!". Switching to float32
should eliminate NaN propagation and produce valid output.

Entries: norm__rmsnorm__smoke, norm__layernorm__smoke,
         rope__rotate_half__b2_s9__smoke, rope__interleaved__b2_s9__smoke

For each entry reports:
  - Whether "!" output disappears
  - Token count and truncation flag
  - Last 300 chars of output
  - Verify result as-generated (no header)
  - Verify result with "import triton\nimport triton.language as tl\n\n" prepended
    (if no-header fails with NameError)

Output: evaluation/results/fp32_rerun_report.json
"""

from __future__ import annotations

import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import SMOKE_GENERATORS
from kernelforge.parsing import parse_model_output
from kernelforge.prompts import build_messages

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 1536
INTERPRETER_TIMEOUT_S = 90

RESULTS_DIR = repo_root / "evaluation" / "results"
OUT_PATH = RESULTS_DIR / "fp32_rerun_report.json"
TMP_DIR = RESULTS_DIR / "_docker_tmp"

TRUNCATED_IDS = {
    "norm__rmsnorm__smoke",
    "norm__layernorm__smoke",
    "rope__rotate_half__b2_s9__smoke",
    "rope__interleaved__b2_s9__smoke",
}


def _last_error_line(err: str) -> str:
    if not err:
        return "(none)"
    for line in reversed(err.strip().split("\n")):
        if line.strip():
            return line.strip()
    return "(empty)"


def _has_bang_tail(text: str, window: int = 300) -> bool:
    """Return True if the last `window` chars consist mostly of '!' characters."""
    tail = text[-window:]
    if not tail:
        return False
    bang_ratio = tail.count("!") / len(tail)
    return bang_ratio > 0.9


def _docker_verify(entry: dict, kernel_text: str, suffix: str = "") -> dict:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    entry_id_safe = entry["id"].replace("/", "_") + suffix
    kernel_path = TMP_DIR / f"{entry_id_safe}_kernel.py"
    ref_path    = TMP_DIR / f"{entry_id_safe}_ref.py"
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
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=INTERPRETER_TIMEOUT_S + 15)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-2000:]
            return {"correct": False, "error": err, "last_error": _last_error_line(err)}
        last = proc.stdout.strip().splitlines()[-1]
        d = json.loads(last)
        return {"correct": d["correct"], "error": d.get("error"),
                "last_error": _last_error_line(d.get("error") or "")}
    except Exception as exc:
        return {"correct": False, "error": str(exc), "last_error": str(exc)}
    finally:
        for p in (kernel_path, ref_path):
            try: p.unlink(missing_ok=True)
            except OSError: pass


def main() -> None:
    # Collect the 4 entries
    entries = []
    for cat in ["norm", "rope"]:
        for entry in SMOKE_GENERATORS[cat]():
            if entry["id"] in TRUNCATED_IDS:
                entries.append(entry)
    print(f"[fp32_rerun] {len(entries)} entries")

    print("[fp32_rerun] Loading model in float32 …")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,       # float32 instead of float16
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    generations = []
    for i, entry in enumerate(entries, 1):
        print(f"[gen {i}/{len(entries)}] {entry['id']} …", end=" ", flush=True)
        messages = build_messages(entry["description"], entry["pytorch_reference"], notes=True)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.monotonic()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                temperature=None, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - t0
        ids = out[0][inputs["input_ids"].shape[1]:]
        comp = len(ids)
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        tps = comp / elapsed if elapsed > 0 else 0
        truncated = comp == MAX_NEW_TOKENS
        bang_tail = _has_bang_tail(raw)
        print(f"{comp} tok, {tps:.1f} tok/s{'  TRUNCATED' if truncated else ''}{'  BANG-TAIL' if bang_tail else ''}")
        generations.append({
            "entry": entry,
            "raw_text": raw,
            "comp": comp,
            "truncated": truncated,
            "bang_tail": bang_tail,
        })

    del model, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("[fp32_rerun] Model freed.")

    print("\n[fp32_rerun] Verifying …\n")
    header = "import triton\nimport triton.language as tl\n\n"
    results = []

    for g in generations:
        entry = g["entry"]
        raw = g["raw_text"]
        parsed = parse_model_output(raw)
        kernel = parsed.triton_kernel or ""

        tail_300 = raw[-300:]

        v_no_hdr = _docker_verify(entry, kernel, "_fp32_nohdr") if kernel else {
            "correct": False, "error": "no kernel extracted", "last_error": "no kernel extracted"
        }

        v_with_hdr = None
        if "NameError" in (v_no_hdr.get("error") or ""):
            kernel_with_hdr = header + kernel
            v_with_hdr = _docker_verify(entry, kernel_with_hdr, "_fp32_hdr")

        status_no  = "PASS" if v_no_hdr["correct"] else f"FAIL({v_no_hdr['last_error'][:60]})"
        status_hdr = ("PASS" if v_with_hdr["correct"] else
                      f"FAIL({v_with_hdr['last_error'][:60]})") if v_with_hdr else "skipped"

        print(f"  {entry['id']}")
        print(f"    tokens={g['comp']} truncated={g['truncated']} bang_tail={g['bang_tail']}")
        print(f"    no-header: {status_no}")
        if v_with_hdr is not None:
            print(f"    with-header: {status_hdr}")
        print(f"    tail-300: {repr(tail_300[-100:])}")
        print()

        results.append({
            "id": entry["id"],
            "category": entry["category"],
            "completion_tokens": g["comp"],
            "truncated": g["truncated"],
            "bang_tail_fp32": g["bang_tail"],
            "tail_300_chars": tail_300,
            "no_header_correct": v_no_hdr["correct"],
            "no_header_last_error": v_no_hdr["last_error"],
            "with_header_correct": v_with_hdr["correct"] if v_with_hdr else None,
            "with_header_last_error": v_with_hdr["last_error"] if v_with_hdr else None,
        })

    bang_gone = [r for r in results if not r["bang_tail_fp32"]]
    name_err  = [r for r in results if "NameError" in r["no_header_last_error"]]
    name_err_pass = [r for r in name_err if r.get("with_header_correct")]

    print("=== FP32 RERUN REPORT ===")
    print(f"Bang-tail gone (fp32): {len(bang_gone)}/{len(results)}")
    print(f"Truncated: {sum(1 for r in results if r['truncated'])}/{len(results)}")
    print(f"No-header correct: {sum(1 for r in results if r.get('no_header_correct'))}/{len(results)}")
    print(f"With-header correct (NameError entries): {len(name_err_pass)}/{len(name_err)}")
    for r in results:
        print(f"  {r['id']}: bang_tail={r['bang_tail_fp32']} tokens={r['completion_tokens']} "
              f"no_hdr={'PASS' if r['no_header_correct'] else r['no_header_last_error'][:50]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nReport written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
