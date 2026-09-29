"""Kernel search engine, batched streamer, and harness mismatch reporting (no GPU, no model)."""

import pytest

from kernelforge.model import BatchTextStreamer
from kernelforge.search import SearchConfig, run_search


def _reply(tag: str) -> str:
    return f"kernel:\nimport torch\n# {tag}\nclass ModelNew: pass\n\nexplanation:\nfused {tag}"


def _fake_generate(outputs_per_round):
    """outputs_per_round[i] is the list of completions returned for round i."""
    rounds = iter(outputs_per_round)
    seen = []

    def generate(messages, n, temperature):
        seen.append((messages, n, temperature))
        texts = next(rounds)
        assert len(texts) == n
        for i, text in enumerate(texts):
            half = len(text) // 2
            yield i, text[:half]
            yield i, text[half:]

    return generate, seen


def _run(outputs, verdicts, config, skip=lambda kernel: None if kernel else "no kernel"):
    generate, seen = _fake_generate(outputs)
    verified = []

    def verify(kernel):
        verified.append(kernel)
        for tag, result in verdicts.items():
            if f"# {tag}\n" in kernel:
                return {"ran": True, **result}
        raise AssertionError("unexpected kernel")

    events = list(run_search("desc", "ref", config, generate, verify, skip, lambda t: len(t.split())))
    return events, seen, verified


PASS = {"status": "ok", "correct": True, "passed": True}


def test_best_of_n_picks_fastest_passing_candidate():
    events, seen, _ = _run(
        [[_reply("slow"), _reply("broken"), _reply("fast")]],
        {"slow": {**PASS, "speedup": 1.1}, "broken": {"status": "error", "error": "boom"},
         "fast": {**PASS, "speedup": 2.4}},
        SearchConfig(num_candidates=3, base_temperature=0.2),
    )
    selected = events[-1]
    assert selected["event"] == "selected"
    assert selected["candidate"] == "r0c2" and selected["num_passed"] == 2
    assert "fastest of 2 distinct" in selected["reason"]
    assert seen[0][2] == 0.7  # best-of-N raises near-greedy temperature


def test_single_sample_keeps_configured_temperature():
    _, seen, _ = _run([[_reply("a")]], {"a": PASS}, SearchConfig(base_temperature=0.2))
    assert seen[0][2] == 0.2


def test_repair_uses_best_failure_and_stops_on_pass():
    events, seen, _ = _run(
        [[_reply("crash"), _reply("wrong")], [_reply("fixed")]],
        {"crash": {"status": "error", "error": "Traceback: boom"},
         "wrong": {"status": "ok", "correct": False, "error": "output mismatch: max abs error 3.2"},
         "fixed": {**PASS, "speedup": 1.3}},
        SearchConfig(num_candidates=2, repair_rounds=2),
    )
    rounds = [e for e in events if e["event"] == "round"]
    assert [r["kind"] for r in rounds] == ["sample", "repair"]  # stopped after the fix
    # Wrong numerics outrank a crash as the repair target.
    assert rounds[1]["repairing"] == "r0c1"
    repair_messages = seen[1][0]
    assert "max abs error 3.2" in repair_messages[-1]["content"]
    assert repair_messages[-2] == {"role": "assistant", "content": _reply("wrong")}
    assert events[-1]["candidate"] == "r1c0" and "repair round 1" in events[-1]["reason"]


def test_unparseable_output_is_repaired_with_format_feedback():
    events, seen, verified = _run(
        [["I think you should use shared memory."], [_reply("ok")]],
        {"ok": PASS},
        SearchConfig(repair_rounds=1),
    )
    assert "could not be used" in seen[1][0][-1]["content"]
    assert len(verified) == 1
    assert events[-1]["verification"]["passed"]


def test_identical_kernels_are_verified_once():
    events, _, verified = _run(
        [[_reply("same"), _reply("same")]], {"same": PASS}, SearchConfig(num_candidates=2),
    )
    assert len(verified) == 1
    second = [e for e in events if e["event"] == "verification"][1]
    assert second["duplicate_of"] == "r0c0" and second["cached"] is True


def test_skipped_verification_never_triggers_repair():
    events, seen, verified = _run(
        [[_reply("a")]], {}, SearchConfig(repair_rounds=2), skip=lambda kernel: "Verification disabled.",
    )
    assert len(seen) == 1 and not verified
    assert events[-1]["reason"] == "not verified: Verification disabled."


def test_all_failures_report_closest_attempt():
    events, _, _ = _run(
        [[_reply("bad")], [_reply("worse")]],
        {"bad": {"status": "ok", "correct": False, "error": "mismatch"},
         "worse": {"status": "error", "error": "crash"}},
        SearchConfig(repair_rounds=1),
    )
    selected = events[-1]
    assert selected["candidate"] == "r0c0"  # wrong numerics beats the crash from the repair
    assert selected["reason"].startswith("no candidate passed")
    assert [c["verdict"] for c in selected["candidates"]] == ["incorrect", "error"]


def test_tokens_are_tagged_and_reassemble_exactly():
    events, _, _ = _run([[_reply("x"), _reply("y")]], {"x": PASS, "y": PASS}, SearchConfig(num_candidates=2))
    for cid, tag in (("r0c0", "x"), ("r0c1", "y")):
        text = "".join(e["text"] for e in events if e["event"] == "token" and e["candidate"] == cid)
        assert text == _reply(tag)


class _CharTokenizer:
    """One token per character; id 0 is EOS, ids are ord(ch)."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if int(i) != 0)


def test_batch_streamer_splits_rows_and_stops_at_eos():
    torch = pytest.importorskip("torch")
    streamer = BatchTextStreamer(_CharTokenizer(), num_sequences=2, eos_token_id=0)
    streamer.put(torch.tensor([[1, 2, 3]]))  # the prompt is skipped
    for a, b in zip("h\ni", "ok?"):
        streamer.put(torch.tensor([ord(a), ord(b)]))
    streamer.put(torch.tensor([0, ord("!")]))
    streamer.put(torch.tensor([ord("z"), 0]))  # row 0 already finished: ignored
    streamer.end()
    out = {0: "", 1: ""}
    for i, chunk in streamer:
        out[i] += chunk
    assert out == {0: "h\ni", 1: "ok?!"}


def test_harness_mismatch_message_locates_the_error():
    torch = pytest.importorskip("torch")
    from verification.sandbox_runner import _compare

    ref = torch.zeros(4, 4)
    cand = ref.clone()
    cand[2, 3] = 5.0
    ok, msg = _compare(cand, ref, 1e-2, 1e-2)
    assert not ok and "1/16 elements" in msg and "(2, 3)" in msg

    assert "shape (4,)" in _compare(torch.zeros(4), ref, 1e-2, 1e-2)[1]
    nan = ref.clone()
    nan[0, 0] = float("nan")
    assert "NaN (1 elements)" in _compare(nan, ref, 1e-2, 1e-2)[1]
    assert _compare((ref, ref), ref, 1e-2, 1e-2)[1].startswith("candidate returned 2 output(s)")
    assert _compare(ref.clone(), ref, 1e-2, 1e-2) == (True, None)
