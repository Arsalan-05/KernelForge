"""CPU-only tests for data/build_dataset.py's run loop: OOM classification,
the incremental progress log, --resume, the solo OOM retry and the time budget.

Nothing here touches a GPU or Triton. `verify` is replaced by a fake that
returns canned VerifyResults, and every entry is a synthetic stand-in
(ids like "synthetic__tpl__N"), not a real dataset entry or result.
"""

import json
from argparse import Namespace

import pytest

from data import build_dataset as bd
from verification.verify_kernel import VerifyResult

PASS = VerifyResult(status="ok", correct=True, speedup=1.5)
WRONG = VerifyResult(status="ok", correct=False, error="max abs error 3.2")
CUDA_OOM = VerifyResult(status="error", error="torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
KILLED = VerifyResult(status="crash", error="no output from sandbox (exit code -9); stderr:\n")


def synthetic_entries(n, category="synthetic"):
    return [{"id": f"{category}__tpl__{i}", "category": category} for i in range(n)]


class FakeVerify:
    """Returns a scripted result per entry id (a list = one result per call) and records calls."""

    def __init__(self, script, default=PASS):
        self.script = {k: list(v) if isinstance(v, list) else [v] for k, v in script.items()}
        self.default = default
        self.calls = []

    def __call__(self, entry):
        self.calls.append(entry["id"])
        queue = self.script.get(entry["id"])
        if queue:
            return queue.pop(0) if len(queue) > 1 else queue[0]
        return self.default


# ---------------------------------------------------------------- OOM classification

@pytest.mark.parametrize("result", [CUDA_OOM, KILLED, VerifyResult(status="error", error="Traceback...\nMemoryError")])
def test_oom_results_are_classified_as_oom(result):
    assert bd.is_oom(result)
    assert bd._rejection_reason(result) == "oom"


def test_incorrect_is_not_oom():
    assert not bd.is_oom(WRONG)
    assert bd._rejection_reason(WRONG) == "incorrect"


def test_triton_shared_memory_limit_is_not_oom():
    # "out of resource" is a kernel config problem (too much shared memory), not a memory limit.
    result = VerifyResult(status="error", error="triton.runtime.errors.OutOfResources: out of resource: shared memory")
    assert not bd.is_oom(result)
    assert bd._rejection_reason(result) == "error"


def test_passing_result_is_never_oom():
    assert not bd.is_oom(VerifyResult(status="ok", correct=True, error="note: out of memory warning earlier"))


# ---------------------------------------------------------------- progress log

def test_progress_round_trip_and_last_record_wins(tmp_path):
    path = tmp_path / "progress.jsonl"
    with open(path, "a") as f:
        bd.append_progress(f, "a", CUDA_OOM, retried=False)
        bd.append_progress(f, "a", PASS, retried=True)
        bd.append_progress(f, "b", WRONG, retried=False)
    records = bd.load_progress(path)
    assert set(records) == {"a", "b"}
    assert records["a"]["retried"] is True
    assert bd.result_from_record(records["a"]) == PASS
    assert bd.result_from_record(records["b"]) == WRONG


def test_truncated_last_line_is_ignored(tmp_path):
    path = tmp_path / "progress.jsonl"
    with open(path, "a") as f:
        bd.append_progress(f, "a", PASS, retried=False)
        f.write('{"id": "b", "result": {"sta')  # session died mid-write
    assert set(bd.load_progress(path)) == {"a"}


def test_missing_progress_file_is_empty(tmp_path):
    assert bd.load_progress(tmp_path / "nope.jsonl") == {}


def test_progress_file_is_per_mode():
    assert bd.progress_path("cuda", interpret=False) != bd.progress_path("cuda", interpret=True)


# ---------------------------------------------------------------- verify_all / resume

def test_fresh_run_verifies_everything_and_logs_each_result(tmp_path):
    entries = synthetic_entries(4)
    fake = FakeVerify({})
    progress = tmp_path / "progress.jsonl"
    results, stats = bd.verify_all(entries, fake, progress)
    assert [e["id"] for e, _ in results] == [e["id"] for e in entries]  # entry order kept
    assert len(progress.read_text().splitlines()) == 4
    assert stats == {"resumed": 0, "oom_retried": 0, "oom_recovered_on_retry": 0, "not_run": []}


def test_resume_skips_completed_ids(tmp_path):
    entries = synthetic_entries(5)
    progress = tmp_path / "progress.jsonl"
    with open(progress, "a") as f:  # an earlier session finished the first two
        bd.append_progress(f, entries[0]["id"], PASS, retried=False)
        bd.append_progress(f, entries[1]["id"], WRONG, retried=False)
    fake = FakeVerify({})
    results, stats = bd.verify_all(entries, fake, progress, resume=True)
    assert fake.calls == [e["id"] for e in entries[2:]]
    assert stats["resumed"] == 2
    assert len(results) == 5
    assert dict((e["id"], r) for e, r in results)[entries[1]["id"]] == WRONG  # old result kept, not rerun


def test_without_resume_old_progress_is_discarded(tmp_path):
    entries = synthetic_entries(2)
    progress = tmp_path / "progress.jsonl"
    with open(progress, "a") as f:
        bd.append_progress(f, entries[0]["id"], WRONG, retried=False)
    fake = FakeVerify({})
    bd.verify_all(entries, fake, progress, resume=False)
    assert fake.calls == [e["id"] for e in entries]


def test_resume_ignores_ids_not_in_this_run(tmp_path):
    progress = tmp_path / "progress.jsonl"
    with open(progress, "a") as f:
        bd.append_progress(f, "other__tpl__9", PASS, retried=False)
    results, stats = bd.verify_all(synthetic_entries(1), FakeVerify({}), progress, resume=True)
    assert stats["resumed"] == 0 and len(results) == 1


# ---------------------------------------------------------------- OOM retry

def test_oom_is_retried_once_alone_and_can_recover(tmp_path):
    entries = synthetic_entries(3)
    oom_id = entries[1]["id"]
    fake = FakeVerify({oom_id: [CUDA_OOM, PASS]})
    results, stats = bd.verify_all(entries, fake, tmp_path / "p.jsonl", jobs=3)
    assert fake.calls.count(oom_id) == 2
    assert fake.calls[-1] == oom_id  # the retry runs after the main pass
    assert stats["oom_retried"] == 1 and stats["oom_recovered_on_retry"] == 1
    assert dict((e["id"], r) for e, r in results)[oom_id] == PASS


def test_persistent_oom_is_retried_only_once_and_stays_oom(tmp_path):
    entries = synthetic_entries(2)
    oom_id = entries[0]["id"]
    fake = FakeVerify({oom_id: KILLED})
    results, stats = bd.verify_all(entries, fake, tmp_path / "p.jsonl")
    assert fake.calls.count(oom_id) == 2
    assert stats["oom_recovered_on_retry"] == 0
    assert bd._rejection_reason(dict((e["id"], r) for e, r in results)[oom_id]) == "oom"


def test_incorrect_results_are_not_retried(tmp_path):
    entries = synthetic_entries(2)
    fake = FakeVerify({entries[0]["id"]: WRONG})
    bd.verify_all(entries, fake, tmp_path / "p.jsonl")
    assert fake.calls.count(entries[0]["id"]) == 1


def test_resume_does_a_pending_oom_retry_but_never_a_second(tmp_path):
    entries = synthetic_entries(2)
    progress = tmp_path / "p.jsonl"
    with open(progress, "a") as f:
        bd.append_progress(f, entries[0]["id"], CUDA_OOM, retried=False)  # died before its retry
        bd.append_progress(f, entries[1]["id"], CUDA_OOM, retried=True)   # already retried
    fake = FakeVerify({}, default=PASS)
    bd.verify_all(entries, fake, progress, resume=True)
    assert fake.calls == [entries[0]["id"]]


# ---------------------------------------------------------------- time budget

def test_deadline_in_the_past_runs_nothing_and_reports_not_run(tmp_path):
    entries = synthetic_entries(3)
    fake = FakeVerify({})
    results, stats = bd.verify_all(entries, fake, tmp_path / "p.jsonl", deadline=0.0)
    assert fake.calls == [] and results == []
    assert stats["not_run"] == [e["id"] for e in entries]


def test_budget_then_resume_finishes_the_rest(tmp_path, monkeypatch):
    entries = synthetic_entries(4)
    progress = tmp_path / "p.jsonl"
    clock = {"t": 0.0}
    monkeypatch.setattr(bd.time, "monotonic", lambda: clock["t"])

    def slow_verify(entry):  # each job "takes" 10 s of the fake clock
        clock["t"] += 10
        return PASS

    # Budget of 15 s: jobs start at t=0 and t=10, the third would start at t=20.
    _, stats = bd.verify_all(entries, slow_verify, progress, deadline=15.0)
    assert len(stats["not_run"]) == 2

    fake = FakeVerify({})
    results, stats = bd.verify_all(entries, fake, progress, resume=True)
    assert fake.calls == [e["id"] for e in entries[2:]]
    assert len(results) == 4 and stats["not_run"] == []


# ---------------------------------------------------------------- report writing

def test_dataset_report_marks_partial_runs_and_counts_oom(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "VERIFIED_PATH", tmp_path / "verified.jsonl")
    monkeypatch.setattr(bd, "REJECTED_PATH", tmp_path / "rejected.jsonl")
    monkeypatch.setattr(bd, "REPORT_PATH", tmp_path / "report.json")
    entries = synthetic_entries(3)
    results = [(entries[0], PASS), (entries[1], CUDA_OOM)]
    stats = {"resumed": 0, "oom_retried": 1, "oom_recovered_on_retry": 0, "not_run": [entries[2]["id"]]}

    bd._write_dataset(results, Namespace(device="cuda", smoke=False), stats)

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["partial"] is True
    assert report["total_generated"] == 3
    assert report["per_category"]["synthetic"]["rejection_reasons"] == {"oom": 1}
    assert len((tmp_path / "verified.jsonl").read_text().splitlines()) == 1
