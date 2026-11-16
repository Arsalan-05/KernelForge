#!/usr/bin/env python3
"""
Task 2 investigation: streaming vs non-streaming newline comparison.

Sends the SAME prompt once in streaming mode (SSE) and once in non-streaming
mode (plain POST), captures both raw responses, and compares whether the
model-generated newline at the SyntaxError location appears in both.

Purpose: confirm that missing newlines are a MODEL artifact (not a streaming
or JSON-parsing bug in our pipeline).

Usage:
    python scripts/newline_comparison.py

Credentials from .env (KERNELFORGE_LLM_API_KEY, KERNELFORGE_LLM_BASE_URL,
KERNELFORGE_LLM_MODEL).  Max 10 API calls, $0.50 hard cap.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import SMOKE_GENERATORS
from kernelforge.prompts import build_messages
from kernelforge.remote import RemoteChatGenerator

SPEND_CAP_USD = 0.50
MAX_CALLS = 10
MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _load_env() -> dict:
    env_path = repo_root / ".env"
    out = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    import os
    for k in ("KERNELFORGE_LLM_API_KEY", "KERNELFORGE_LLM_BASE_URL", "KERNELFORGE_LLM_MODEL"):
        if k in os.environ:
            out[k] = os.environ[k]
    return out


def _parse_retry_after(error_text: str) -> int:
    m = re.search(r'"Retry-After"\s*:\s*"?(\d+)"?', error_text)
    return int(m.group(1)) + 10 if m else 130


# ---------------------------------------------------------------------------
# Non-streaming call
# ---------------------------------------------------------------------------

def _call_nonstreaming(client: httpx.Client, url: str, api_key: str, model: str,
                        messages: list[dict]) -> tuple[str, dict]:
    """POST without stream. Returns (content_text, usage_dict)."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = client.post(f"{url}/chat/completions", json=body, headers=headers, timeout=120.0)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage


# ---------------------------------------------------------------------------
# Streaming call (using existing RemoteChatGenerator)
# ---------------------------------------------------------------------------

def _call_streaming(gen: RemoteChatGenerator, messages: list[dict]) -> tuple[str, object]:
    """Stream via RemoteChatGenerator. Returns (joined_text, usage)."""
    chunks: list[str] = []
    for _, chunk in gen.stream_chat(messages, num_sequences=1, temperature=0.0):
        chunks.append(chunk)
    text = "".join(chunks)
    usage = gen.last_usage.get(0)
    return text, usage


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _has_syntax_error(text: str) -> tuple[bool, str]:
    """Try to exec/compile the text; return (has_error, detail)."""
    try:
        compile(text, "<generated>", "exec")
        return False, ""
    except SyntaxError as e:
        return True, str(e)


def _locate_missing_newlines(text: str, window: int = 60) -> list[str]:
    """Find positions where two statement-like tokens appear on the same line
    without a newline separator (the specific SyntaxError pattern seen in
    the first baseline run).
    Returns list of snippets showing the problem region.
    """
    hits = []
    # Heuristic: look for lines that contain a statement-ending token followed
    # immediately (without \n) by another statement-starting token.
    # Pattern: digit/identifier followed by whitespace and then a statement keyword/identifier
    # without an intervening newline.
    for m in re.finditer(r'[0-9a-zA-Z_><=!)\]]\s{2,}[a-zA-Z_]', text):
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 20)
        snippet = repr(text[start:end])
        hits.append(snippet)
    return hits[:5]  # at most 5 examples


def _char_level_inspect(text: str, search: str, context: int = 15) -> str:
    """Find `search` in text and show characters around the match with ordinals."""
    idx = text.find(search)
    if idx < 0:
        return f"  '{search}' not found"
    start = max(0, idx - context)
    end = min(len(text), idx + len(search) + context)
    snippet = text[start:end]
    chars = " | ".join(f"'{c}' ({ord(c)})" for c in snippet)
    return f"  position {idx}: {chars}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    env = _load_env()
    api_key = env.get("KERNELFORGE_LLM_API_KEY", "")
    base_url = env.get("KERNELFORGE_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    model = env.get("KERNELFORGE_LLM_MODEL", "qwen/qwen-2.5-coder-32b-instruct")

    if not api_key:
        print("ERROR: KERNELFORGE_LLM_API_KEY not set")
        return 1

    # Pick the entry that had SyntaxErrors in the original run
    entries = list(SMOKE_GENERATORS["quantized_matmul"]())
    entry = entries[0]  # quantized_matmul__w8a16_pertensor__smoke_b5
    print(f"Entry: {entry['id']}")

    messages = build_messages(entry["description"], entry["pytorch_reference"], notes=True)
    total_prompt_chars = sum(len(m["content"]) for m in messages)
    print(f"Prompt characters: {total_prompt_chars:,} (~{total_prompt_chars//4:,} tokens)")
    print()

    client = httpx.Client(timeout=httpx.Timeout(130.0, connect=10.0))
    gen = RemoteChatGenerator(base_url, api_key, model, max_tokens=MAX_TOKENS, temperature=0.0)

    total_cost = 0.0
    call_count = 0

    # --- Call 1: STREAMING ---
    print("=" * 60)
    print("CALL 1: STREAMING (SSE)")
    print("=" * 60)
    streaming_text = None
    for attempt in range(MAX_CALLS):
        if call_count >= MAX_CALLS:
            print("MAX_CALLS reached — aborting")
            return 1
        try:
            call_count += 1
            t0 = time.monotonic()
            streaming_text, s_usage = _call_streaming(gen, messages)
            elapsed = time.monotonic() - t0
            cost = getattr(s_usage, "cost_usd", None) or 0.0
            total_cost += cost
            print(f"  Got {len(streaming_text)} chars in {elapsed:.1f}s")
            print(f"  Usage: {s_usage}")
            print(f"  Cost: ${cost:.5f}  Running total: ${total_cost:.5f}")
            break
        except Exception as exc:
            err = str(exc)
            if "402" in err and "in_flight" in err:
                wait = _parse_retry_after(err)
                print(f"  [402] waiting {wait}s (attempt {attempt+1})...")
                time.sleep(wait)
                continue
            print(f"  ERROR: {exc}")
            return 1

    if streaming_text is None:
        print("Could not get streaming response")
        return 1

    if total_cost > SPEND_CAP_USD:
        print(f"SPEND CAP exceeded: ${total_cost:.5f} > ${SPEND_CAP_USD}")
        return 1

    print()
    # Wait before next call to clear in-flight budget
    wait_between = 10
    print(f"Waiting {wait_between}s before non-streaming call...")
    time.sleep(wait_between)

    # --- Call 2: NON-STREAMING ---
    print("=" * 60)
    print("CALL 2: NON-STREAMING (plain POST)")
    print("=" * 60)
    nonstream_text = None
    for attempt in range(MAX_CALLS - call_count):
        if call_count >= MAX_CALLS:
            print("MAX_CALLS reached — aborting")
            return 1
        try:
            call_count += 1
            t0 = time.monotonic()
            nonstream_text, ns_usage = _call_nonstreaming(client, base_url, api_key, model, messages)
            elapsed = time.monotonic() - t0
            ns_cost = ns_usage.get("cost", 0.0) or 0.0
            total_cost += ns_cost
            print(f"  Got {len(nonstream_text)} chars in {elapsed:.1f}s")
            print(f"  Usage: {ns_usage}")
            print(f"  Cost: ${ns_cost:.5f}  Running total: ${total_cost:.5f}")
            break
        except RuntimeError as exc:
            err = str(exc)
            if "402" in err and "in_flight" in err:
                wait = _parse_retry_after(err)
                print(f"  [402] waiting {wait}s (attempt {attempt+1})...")
                time.sleep(wait)
                continue
            print(f"  ERROR: {exc}")
            return 1

    if nonstream_text is None:
        print("Could not get non-streaming response")
        return 1

    # --- Analysis ---
    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print()

    # 1. Lengths
    print(f"Streaming length:     {len(streaming_text):,} chars")
    print(f"Non-streaming length: {len(nonstream_text):,} chars")
    print(f"Identical:            {streaming_text == nonstream_text}")
    print()

    # 2. Syntax errors
    s_has_err, s_detail = _has_syntax_error(streaming_text)
    n_has_err, n_detail = _has_syntax_error(nonstream_text)
    print(f"Streaming SyntaxError:     {s_has_err}  {s_detail[:80] if s_detail else ''}")
    print(f"Non-streaming SyntaxError: {n_has_err}  {n_detail[:80] if n_detail else ''}")
    print()

    # 3. Locate missing newlines
    print("Missing-newline candidates in STREAMING output:")
    hits = _locate_missing_newlines(streaming_text)
    if hits:
        for h in hits:
            print(f"  {h}")
    else:
        print("  None found (all statements appear on separate lines)")
    print()

    print("Missing-newline candidates in NON-STREAMING output:")
    hits2 = _locate_missing_newlines(nonstream_text)
    if hits2:
        for h in hits2:
            print(f"  {h}")
    else:
        print("  None found")
    print()

    # 4. Character-level inspection at known problem location
    # The specific pattern seen in the original run:
    probe = "/ 127"
    print(f"Char inspection around '{probe}':")
    print(f"  Streaming:     {_char_level_inspect(streaming_text, probe)}")
    print(f"  Non-streaming: {_char_level_inspect(nonstream_text, probe)}")
    print()

    # 5. Show first 400 chars of each for visual comparison
    print("First 300 chars of streaming output:")
    print(repr(streaming_text[:300]))
    print()
    print("First 300 chars of non-streaming output:")
    print(repr(nonstream_text[:300]))
    print()

    # 6. Conclusion
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    both_missing = s_has_err and n_has_err
    neither_missing = not s_has_err and not n_has_err
    only_streaming = s_has_err and not n_has_err
    only_nonstream = not s_has_err and n_has_err

    if both_missing:
        print("BOTH streaming and non-streaming have SyntaxErrors.")
        print("→ Missing newlines are a MODEL artifact, not a streaming/parsing bug.")
        print("  The model itself does not generate the newline token at that position.")
        conclusion = "MODEL_ARTIFACT"
    elif neither_missing:
        print("NEITHER response has SyntaxErrors.")
        print("→ This run did not reproduce the original SyntaxError artifact.")
        print("  The model generated valid code on this attempt. Prior SyntaxErrors")
        print("  are intermittent model-quality issues (temperature=0.0 but non-deterministic")
        print("  at the token level for certain positions).")
        conclusion = "CLEAN_RUN_NO_REPRO"
    elif only_streaming:
        print("ONLY streaming has a SyntaxError — non-streaming is clean.")
        print("→ This would indicate a streaming/parsing bug. INVESTIGATE FURTHER.")
        conclusion = "STREAMING_ARTIFACT"
    else:
        print("ONLY non-streaming has a SyntaxError — streaming is clean.")
        print("→ Unexpected. INVESTIGATE FURTHER.")
        conclusion = "NONSTREAMING_ARTIFACT"

    print()
    print(f"Total API calls: {call_count}")
    print(f"Total cost: ${total_cost:.5f}")
    print(f"Conclusion code: {conclusion}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
