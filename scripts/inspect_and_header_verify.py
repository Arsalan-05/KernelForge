#!/usr/bin/env python3
"""
Two diagnostics in one generation pass over 12 rerun entries:

A) Truncation inspection (4 entries that hit 1536 tokens):
   - Last ~400 chars of raw_text
   - Check for repeating/looping patterns
   - Line count of first complete @triton.jit kernel block
   - Quoted example if looping

B) With-header verification (6 NameError entries from the rerun):
   - Prepend "import triton\nimport triton.language as tl\n" to extracted kernel
   - Docker verify; compare pass rate vs without headers
   - Reports both numbers: N/6 pass with-header vs 0/6 without

Output: evaluation/results/inspect_report.json
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
OUT_PATH = RESULTS_DIR / "inspect_report.json"

# 12 entries from rerun_1536
PLAN = {
    "quantized_matmul": 3,
    "attention":        3,
    "kv_cache":         2,
    "norm":             2,
    "rope":             2,
}
TRUNCATED_IDS = {
    "norm__rmsnorm__smoke",
    "norm__layernorm__smoke",
    "rope__rotate_half__b2_s9__smoke",
    "rope__interleaved__b2_s9__smoke",
}


def _build_entries() -> list[dict]:
    entries = []
    for cat, n in PLAN.items():
        smoke = list(SMOKE_GENERATORS[cat]())
        entries.extend(smoke[:n])
    return entries


def _last_error_line(err: str) -> str:
    if not err:
        return "(none)"
    for line in reversed(err.strip().split("\n")):
        if line.strip():
            return line.strip()
    return "(empty)"


# ── looping detection ──────────────────────────────────────────────────────────

def _detect_loop(text: str, min_len: int = 20, min_reps: int = 3) -> tuple[bool, str]:
    """Find the longest pattern that repeats >= min_reps times consecutively in `text`."""
    # Search in the last 1000 chars where looping is most visible
    tail = text[-1000:]
    best_len = 0
    best_snippet = ""
    for length in range(80, min_len - 1, -1):
        for start in range(len(tail) - length * min_reps + 1):
            snippet = tail[start:start + length]
            # Count consecutive repetitions from `start`
            reps = 0
            pos = start
            while tail[pos:pos + length] == snippet:
                reps += 1
                pos += length
            if reps >= min_reps and length > best_len:
                best_len = length
                best_snippet = snippet
    if best_len > 0:
        return True, repr(best_snippet[:80])
    return False, ""


# ── kernel line counter ────────────────────────────────────────────────────────

def _first_kernel_line_count(kernel_text: str) -> int:
    """Lines in the first @triton.jit decorated function (up to next top-level def/class or EOF)."""
    lines = kernel_text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if "@triton.jit" in line:
            start = i
        if start is not None and i > start:
            stripped = line
            if stripped and not stripped[0].isspace() and re.match(r"(def |class |@)", stripped):
                return i - start
    if start is not None:
        return len(lines) - start
    return 0


# ── Docker verification helpers ───────────────────────────────────────────────

TMP_DIR = RESULTS_DIR / "_docker_tmp"


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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    entries = _build_entries()
    print(f"[inspect_and_header_verify] {len(entries)} entries")

    # Phase 1: generate all 12
    print("[phase 1] Loading model …")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map=device, trust_remote_code=True,
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
        print(f"{comp} tok, {tps:.1f} tok/s{'  TRUNCATED' if truncated else ''}")
        generations.append({"entry": entry, "raw_text": raw, "comp": comp, "truncated": truncated})

    del model, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("[phase 1] Model freed.")

    # Phase 2: analyze + verify
    print("\n[phase 2] Analyzing and verifying …\n")
    results = []
    header = "import triton\nimport triton.language as tl\n\n"

    for g in generations:
        entry = g["entry"]
        raw = g["raw_text"]
        truncated = g["truncated"]
        parsed = parse_model_output(raw)
        kernel = parsed.triton_kernel or ""

        # --- truncation analysis ---
        tail_300 = raw[-300:]
        is_loop, loop_ex = _detect_loop(raw) if truncated else (False, "")
        klines = _first_kernel_line_count(kernel)

        # --- without-header verification (baseline) ---
        v_no_hdr = _docker_verify(entry, kernel, "_nohdr") if kernel else {
            "correct": False, "error": "no kernel extracted", "last_error": "no kernel extracted"
        }

        # --- with-header verification ---
        v_with_hdr = None
        if "NameError" in (v_no_hdr.get("error") or ""):
            kernel_with_hdr = header + kernel
            v_with_hdr = _docker_verify(entry, kernel_with_hdr, "_hdr")

        status_no  = "PASS" if v_no_hdr["correct"] else f"FAIL({v_no_hdr['last_error'][:60]})"
        status_hdr = ("PASS" if v_with_hdr["correct"] else
                      f"FAIL({v_with_hdr['last_error'][:60]})") if v_with_hdr else "skipped"
        print(f"  {entry['id']}")
        print(f"    tokens={g['comp']} truncated={truncated} looping={is_loop} kernel_lines={klines}")
        print(f"    no-header: {status_no}")
        if v_with_hdr is not None:
            print(f"    with-header: {status_hdr}")
        if truncated:
            print(f"    tail-300:\n      {repr(tail_300[-200:])}")
        print()

        results.append({
            "id": entry["id"],
            "category": entry["category"],
            "completion_tokens": g["comp"],
            "truncated": truncated,
            "is_loop": is_loop,
            "loop_example": loop_ex if is_loop else None,
            "kernel_line_count": klines,
            "tail_300_chars": tail_300,
            "no_header_correct": v_no_hdr["correct"],
            "no_header_last_error": v_no_hdr["last_error"],
            "with_header_correct": v_with_hdr["correct"] if v_with_hdr else None,
            "with_header_last_error": v_with_hdr["last_error"] if v_with_hdr else None,
        })

    # Summary
    name_err = [r for r in results if "NameError" in r["no_header_last_error"]]
    name_err_with_hdr = [r for r in name_err if r.get("with_header_correct")]
    truncated = [r for r in results if r["truncated"]]
    looping = [r for r in truncated if r["is_loop"]]

    print("=== INSPECT + HEADER VERIFY REPORT ===")
    print(f"Without-header: 0/{len(results)} correct")
    print(f"NameError entries: {len(name_err)}")
    print(f"With-header (NameError entries only): {len(name_err_with_hdr)}/{len(name_err)} correct")
    print()
    print(f"Truncated (hit {MAX_NEW_TOKENS}): {len(truncated)}/{len(results)}")
    print(f"Looping among truncated: {len(looping)}/{len(truncated)}")
    for r in truncated:
        print(f"  {r['id']}: looping={r['is_loop']} kernel_lines={r['kernel_line_count']}")
        if r["is_loop"] and r["loop_example"]:
            print(f"    example: {r['loop_example']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nReport written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
