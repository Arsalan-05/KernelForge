#!/usr/bin/env python3
"""
Large AST-based mutation campaign for the Triton kernel verification harness.

Classification rules (declared BEFORE running):
  - harmless_change : mutant produces IDENTICAL outputs to reference on ALL 5
                      differential-screening seeds (seeds 0-4).  Classification
                      is objective and computed before the official harness run.
  - expected_caught : mutant produces DIFFERENT output on ANY of the 5 seeds.

Scoring:
  Official score seed = 42 (not in the screening set {0..4}).
  Mutation score = caught_by_seed42 / expected_caught_count.
  Wilson 95% CI is reported alongside.

Usage:
    python scripts/ast_mutation_campaign.py --dry-run   # list mutations, no run
    python scripts/ast_mutation_campaign.py             # full 90-min run

Outputs:
    evaluation/results/ast_mutation_results.jsonl   per-mutation checkpoint
    evaluation/results/ast_mutation_report.json     final summary

Constraints:
  - Docker run limit: 2 containers in parallel.
  - 90-minute hard time-box from first Docker call.
  - Checkpoint after every entry so the run can be resumed.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from data.templates import SMOKE_GENERATORS

RESULTS_DIR = repo_root / "evaluation" / "results"
LOG_PATH = RESULTS_DIR / "ast_mutation_results.jsonl"
REPORT_PATH = RESULTS_DIR / "ast_mutation_report.json"
TMP_DIR = RESULTS_DIR / "_docker_tmp"

DOCKER_IMAGE = "kernelforge-verify"
INTERPRETER_TIMEOUT_S = 60   # per Docker call; smaller than main test to stay in budget
DOCKER_PARALLEL = 2          # max concurrent Docker containers
CAMPAIGN_TIME_BOX_S = 90 * 60   # 90 minutes

# Screening seeds (5) and official seed (not in this set)
SCREEN_SEEDS = [0, 1, 2, 3, 4]
OFFICIAL_SEED = 42


# ---------------------------------------------------------------------------
# AST-based mutation generation
# ---------------------------------------------------------------------------

class _NthNodeTransformer(ast.NodeTransformer):
    """Replace the Nth occurrence (0-indexed) of a matched node."""
    def __init__(self, nth: int):
        self._target_nth = nth
        self._count = 0

    def _matches(self, node: ast.AST) -> bool:
        raise NotImplementedError

    def _replace(self, node: ast.AST) -> ast.AST:
        raise NotImplementedError

    def _visit_candidate(self, node: ast.AST) -> ast.AST:
        if self._matches(node):
            if self._count == self._target_nth:
                self._count += 1
                new_node = self._replace(copy.deepcopy(node))
                ast.copy_location(new_node, node)
                ast.fix_missing_locations(new_node)
                return new_node
            self._count += 1
        return self.generic_visit(node)


class FlipBinOp(_NthNodeTransformer):
    """Swap one binary operator: e.g. Add↔Sub, Mult↔Div."""
    _PAIRS: dict[type, type] = {
        ast.Add: ast.Sub, ast.Sub: ast.Add,
        ast.Mult: ast.Div, ast.Div: ast.Mult,
        ast.LShift: ast.RShift, ast.RShift: ast.LShift,
    }

    def __init__(self, op_type: type, nth: int):
        super().__init__(nth)
        self._op_type = op_type
        self._op_flip = self._PAIRS[op_type]

    def _matches(self, node: ast.AST) -> bool:
        return isinstance(node, ast.BinOp) and isinstance(node.op, self._op_type)

    def _replace(self, node: ast.BinOp) -> ast.BinOp:
        node.op = self._op_flip()
        return node

    def visit_BinOp(self, node):
        return self._visit_candidate(node)


class FlipCmpOp(_NthNodeTransformer):
    """Flip one comparison operator: < ↔ >, <= ↔ >=."""
    _PAIRS: dict[type, type] = {
        ast.Lt: ast.Gt, ast.Gt: ast.Lt,
        ast.LtE: ast.GtE, ast.GtE: ast.LtE,
    }

    def __init__(self, op_type: type, nth: int):
        super().__init__(nth)
        self._op_type = op_type
        self._op_flip = self._PAIRS[op_type]

    def _matches(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Compare) and any(
            isinstance(op, self._op_type) for op in node.ops
        )

    def _replace(self, node: ast.Compare) -> ast.Compare:
        node.ops = [
            self._op_flip() if isinstance(op, self._op_type) else op
            for op in node.ops
        ]
        return node

    def visit_Compare(self, node):
        return self._visit_candidate(node)


class FlipKeywordAxis(_NthNodeTransformer):
    """Flip `axis=0` ↔ `axis=1` (or `axis=N` → `axis=1-N` for N in {0,1})."""
    def __init__(self, from_val: int, nth: int):
        super().__init__(nth)
        self._from_val = from_val

    def _matches(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.keyword)
            and node.arg == "axis"
            and isinstance(node.value, ast.Constant)
            and node.value.value == self._from_val
        )

    def _replace(self, node: ast.keyword) -> ast.keyword:
        node.value = ast.Constant(value=1 - self._from_val)
        return node

    def visit_keyword(self, node):
        return self._visit_candidate(node)


class SwapBuiltinName(_NthNodeTransformer):
    """Replace `tl.foo(...)` with `tl.bar(...)` for builtin names."""
    def __init__(self, from_name: str, to_name: str, nth: int):
        super().__init__(nth)
        self._from = from_name
        self._to = to_name

    def _matches(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "tl"
            and node.attr == self._from
        )

    def _replace(self, node: ast.Attribute) -> ast.Attribute:
        node.attr = self._to
        return node

    def visit_Attribute(self, node):
        return self._visit_candidate(node)


class RemoveBinOpRight(_NthNodeTransformer):
    """Replace `a * b` or `a / b` with just `a` (drop the right operand)."""
    def __init__(self, op_type: type, nth: int):
        super().__init__(nth)
        self._op_type = op_type

    def _matches(self, node: ast.AST) -> bool:
        return isinstance(node, ast.BinOp) and isinstance(node.op, self._op_type)

    def _replace(self, node: ast.BinOp) -> ast.AST:
        return self.generic_visit(node.left)   # drop right side

    def visit_BinOp(self, node):
        return self._visit_candidate(node)


class ReplaceConstant(_NthNodeTransformer):
    """Replace one specific numeric constant with another."""
    def __init__(self, from_val, to_val, nth: int):
        super().__init__(nth)
        self._from = from_val
        self._to = to_val

    def _matches(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value == self._from

    def _replace(self, node: ast.Constant) -> ast.Constant:
        node.value = self._to
        return node

    def visit_Constant(self, node):
        return self._visit_candidate(node)


# ---------------------------------------------------------------------------
# Mutation site counting helpers
# ---------------------------------------------------------------------------

def _count_binop(src: str, op_type: type) -> int:
    tree = ast.parse(src)
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, op_type))


def _count_cmpop(src: str, op_type: type) -> int:
    tree = ast.parse(src)
    return sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Compare) and any(isinstance(op, op_type) for op in n.ops)
    )


def _count_keyword_axis(src: str, val: int) -> int:
    tree = ast.parse(src)
    return sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.keyword) and n.arg == "axis"
        and isinstance(n.value, ast.Constant) and n.value.value == val
    )


def _count_builtin(src: str, name: str) -> int:
    tree = ast.parse(src)
    return sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        and n.value.id == "tl" and n.attr == name
    )


def _count_const(src: str, val) -> int:
    tree = ast.parse(src)
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == val)


def _apply_transformer(src: str, transformer: ast.NodeTransformer) -> str | None:
    """Return mutated source, or None if transformer made no change."""
    tree = ast.parse(src)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    new_src = ast.unparse(new_tree)
    return new_src if new_src != ast.unparse(ast.parse(src)) else None


# ---------------------------------------------------------------------------
# Mutation definition
# ---------------------------------------------------------------------------

def generate_mutations_for_kernel(
    entry_id: str, category: str, ref_code: str, kernel_src: str
) -> list[dict]:
    """
    Return list of mutation dicts:
      {id, category, entry_id, mut_type, description, mutated_kernel}
    """
    mutations: list[dict] = []

    def _add(mut_type: str, desc: str, transformer: ast.NodeTransformer, nth: int) -> bool:
        new_src = _apply_transformer(kernel_src, transformer)
        if new_src is None:
            return False
        mut_id = f"{entry_id}__{mut_type}_{nth}"
        mutations.append({
            "id": mut_id,
            "category": category,
            "entry_id": entry_id,
            "mut_type": mut_type,
            "description": desc,
            "ref_code": ref_code,
            "mutated_kernel": new_src,
        })
        return True

    # Arithmetic flips: Add↔Sub
    n_add = _count_binop(kernel_src, ast.Add)
    for i in range(min(n_add, 4)):   # up to 4 per template
        _add("flip_add_to_sub", f"BinOp[{i}]: + → -",
             FlipBinOp(ast.Add, i), i)

    n_sub = _count_binop(kernel_src, ast.Sub)
    for i in range(min(n_sub, 4)):
        _add("flip_sub_to_add", f"BinOp[{i}]: - → +",
             FlipBinOp(ast.Sub, i), i)

    # Arithmetic flips: Mult↔Div
    n_mul = _count_binop(kernel_src, ast.Mult)
    for i in range(min(n_mul, 4)):
        _add("flip_mul_to_div", f"BinOp[{i}]: * → /",
             FlipBinOp(ast.Mult, i), i)

    n_div = _count_binop(kernel_src, ast.Div)
    for i in range(min(n_div, 4)):
        _add("flip_div_to_mul", f"BinOp[{i}]: / → *",
             FlipBinOp(ast.Div, i), i)

    # Comparison flips
    for cmp_type, label in [(ast.Lt, "<→>"), (ast.Gt, ">→<"),
                             (ast.LtE, "<=→>="), (ast.GtE, ">=→<=")]:
        n = _count_cmpop(kernel_src, cmp_type)
        for i in range(min(n, 3)):
            _add(f"flip_cmp_{cmp_type.__name__.lower()}", f"Compare[{i}]: {label}",
                 FlipCmpOp(cmp_type, i), i)

    # Axis flips
    for axis_val in [0, 1]:
        n = _count_keyword_axis(kernel_src, axis_val)
        for i in range(min(n, 3)):
            _add(f"flip_axis_{axis_val}_to_{1-axis_val}",
                 f"axis={axis_val}→{1-axis_val} (occurrence {i})",
                 FlipKeywordAxis(axis_val, i), i)

    # Builtin swaps
    for from_fn, to_fn in [("sqrt", "rsqrt"), ("rsqrt", "sqrt"), ("exp", "log"), ("log", "exp")]:
        n = _count_builtin(kernel_src, from_fn)
        for i in range(min(n, 3)):
            _add(f"swap_{from_fn}_to_{to_fn}", f"tl.{from_fn}→tl.{to_fn} (occurrence {i})",
                 SwapBuiltinName(from_fn, to_fn, i), i)

    # Remove right operand (missing scale / weight / bias)
    for op_type, label in [(ast.Mult, "*drop_right"), (ast.Div, "/drop_right")]:
        n = _count_binop(kernel_src, op_type)
        for i in range(min(n, 3)):
            _add(f"remove_binop_{op_type.__name__.lower()}_right_{i}",
                 f"Drop right of {label}[{i}]",
                 RemoveBinOpRight(op_type, i), i)

    # Constant mutations
    for const_val, replacement in [(127, 126), (127, 128), (0.0, 1.0), (1.0, 0.0)]:
        n = _count_const(kernel_src, const_val)
        for i in range(min(n, 2)):
            _add(f"replace_const_{const_val}_to_{replacement}_{i}",
                 f"const[{i}]: {const_val} → {replacement}",
                 ReplaceConstant(const_val, replacement, i), i)

    return mutations


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def _write_tmp(name: str, content: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    p = TMP_DIR / name
    p.write_text(content)
    return p


def _run_harness_in_docker(ref_path: Path, kernel_path: Path, seed: int) -> dict:
    """Run verify_model_kernel in Docker with given seed. Returns result dict."""
    docker_ref = f"/work/evaluation/results/_docker_tmp/{ref_path.name}"
    docker_kernel = f"/work/evaluation/results/_docker_tmp/{kernel_path.name}"

    script = (
        "import sys, json; sys.path.insert(0, '/work'); "
        "from verification.verify_kernel import verify_model_kernel; "
        f"ref = open({repr(docker_ref)}).read(); "
        f"kernel = open({repr(docker_kernel)}).read(); "
        f"result = verify_model_kernel(ref, kernel, device='cpu', "
        f"timeout_s={INTERPRETER_TIMEOUT_S}, interpret=True, seed={seed}); "
        "print(json.dumps({'status': result.status, "
        "    'correct': result.correct, 'error': result.error}))"
    )
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{repo_root}:/work",
        DOCKER_IMAGE,
        "python", "-c", script,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=INTERPRETER_TIMEOUT_S + 15
        )
        if proc.returncode != 0:
            return {"status": "error", "correct": False, "error": proc.stderr[-1000:]}
        stdout = proc.stdout.strip()
        if not stdout:
            return {"status": "error", "correct": False, "error": "no output"}
        return json.loads(stdout.splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "correct": False, "error": "outer timeout"}
    except Exception as exc:
        return {"status": "error", "correct": False, "error": str(exc)}


def _run_screen_and_official_in_docker(
    ref_path: Path, kernel_path: Path,
    screen_seeds: list[int], official_seed: int,
) -> tuple[bool, dict]:
    """
    Run all 5 screening seeds + the official seed in ONE Docker container to
    amortise startup cost.  Script written to a temp file to avoid shell-quoting
    and one-liner limitations.

    Returns (any_wrong_in_screening, official_result_dict).
    """
    docker_ref = f"/work/evaluation/results/_docker_tmp/{ref_path.name}"
    docker_kernel = f"/work/evaluation/results/_docker_tmp/{kernel_path.name}"

    script_body = f"""\
import sys, json
sys.path.insert(0, '/work')
from verification.verify_kernel import verify_model_kernel

ref = open({repr(docker_ref)}).read()
kernel = open({repr(docker_kernel)}).read()

screen_seeds = {screen_seeds!r}
any_wrong = False
screen_err = None
for s in screen_seeds:
    r = verify_model_kernel(ref, kernel, device='cpu',
                            timeout_s={INTERPRETER_TIMEOUT_S}, interpret=True, seed=s)
    if not r.correct:
        any_wrong = True
        screen_err = r.error
        break

official = None
if any_wrong:
    ro = verify_model_kernel(ref, kernel, device='cpu',
                             timeout_s={INTERPRETER_TIMEOUT_S}, interpret=True,
                             seed={official_seed!r})
    official = {{'status': ro.status, 'correct': ro.correct, 'error': ro.error}}

print(json.dumps({{'any_wrong': any_wrong, 'screen_err': screen_err,
                   'official': official}}))
"""
    # Write runner script into the shared temp dir so Docker can access it
    runner_path = TMP_DIR / f"{ref_path.stem}_runner.py"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    runner_path.write_text(script_body)
    docker_runner = f"/work/evaluation/results/_docker_tmp/{runner_path.name}"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{repo_root}:/work",
        DOCKER_IMAGE,
        "python", docker_runner,
    ]
    timeout = (len(screen_seeds) + 1) * (INTERPRETER_TIMEOUT_S + 5) + 15
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            official = {"status": "error", "correct": False, "error": proc.stderr[-800:]}
            return True, official
        stdout = proc.stdout.strip()
        if not stdout:
            official = {"status": "error", "correct": False,
                        "error": f"no docker output; stderr={proc.stderr[-400:]}"}
            return True, official
        result = json.loads(stdout.splitlines()[-1])
        any_wrong = result["any_wrong"]
        official_raw = result.get("official")
        if official_raw is None:
            official = {"status": "skipped", "correct": True, "error": None}
        else:
            official = official_raw
        return any_wrong, official
    except subprocess.TimeoutExpired:
        return True, {"status": "timeout", "correct": False, "error": "outer timeout"}
    except Exception as exc:
        return True, {"status": "error", "correct": False, "error": str(exc)}
    finally:
        try:
            runner_path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _load_done() -> dict[str, dict]:
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


def _append(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List mutations without running")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-checkpointed mutations")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Collect all mutations from all 15 smoke entries
    # ------------------------------------------------------------------
    all_mutations: list[dict] = []
    print("=== AST Mutation Campaign ===")
    print("Classification rules:")
    print("  harmless_change : identical output on ALL seeds {0,1,2,3,4}")
    print("  expected_caught : different output on ANY of seeds {0,1,2,3,4}")
    print("  Official score  : caught by seed=42 / expected_caught_count")
    print()
    print("Collecting mutations from 15 smoke templates …")

    for cat, smoke_gen in SMOKE_GENERATORS.items():
        entries = list(smoke_gen())
        for entry in entries:
            muts = generate_mutations_for_kernel(
                entry["id"], cat, entry["pytorch_reference"], entry["triton_kernel"]
            )
            all_mutations.extend(muts)
            print(f"  {entry['id']:50s}  {len(muts):3d} mutations")

    print(f"\nTotal mutations generated: {len(all_mutations)}")

    if args.dry_run:
        for m in all_mutations:
            print(f"  [{m['mut_type']:40s}] {m['id']}")
        return

    if len(all_mutations) < 200:
        print(f"WARNING: only {len(all_mutations)} mutations; target is ≥200.")

    done = _load_done() if args.resume else {}
    todo = [m for m in all_mutations if m["id"] not in done]
    print(f"Mutations to run: {len(todo)} (skipping {len(done)} already done)")

    campaign_start = time.monotonic()

    # ------------------------------------------------------------------
    # Step 2: For each mutation, run screening + official test in Docker
    # ------------------------------------------------------------------
    all_records: list[dict] = list(done.values())

    for i, mut in enumerate(todo, 1):
        elapsed = time.monotonic() - campaign_start
        if elapsed > CAMPAIGN_TIME_BOX_S:
            print(f"\n[time-box] 90 minutes reached after {i-1} mutations. Stopping.")
            break

        entry_id_safe = mut["id"].replace("/", "_")
        ref_path = _write_tmp(f"{entry_id_safe}_ref.py", mut["ref_code"])
        kernel_path = _write_tmp(f"{entry_id_safe}_kernel.py", mut["mutated_kernel"])

        print(f"[{i}/{len(todo)}] {mut['id'][:60]} …", end=" ", flush=True)
        t0 = time.monotonic()

        # Single Docker call: screen on seeds 0-4, then run official (seed=42)
        # if any screening seed fails. Amortises Docker startup cost.
        any_wrong, official = _run_screen_and_official_in_docker(
            ref_path, kernel_path, SCREEN_SEEDS, OFFICIAL_SEED
        )

        classification = "expected_caught" if any_wrong else "harmless_change"
        caught = (classification == "expected_caught" and
                  not official.get("correct", True) and
                  official.get("status") != "skipped")

        elapsed_this = time.monotonic() - t0
        print(
            f"{classification[:7]} | "
            f"{'CAUGHT' if caught else ('skip' if classification == 'harmless_change' else 'SURVIVED')} "
            f"({elapsed_this:.1f}s)"
        )

        record = {
            "id": mut["id"],
            "category": mut["category"],
            "entry_id": mut["entry_id"],
            "mut_type": mut["mut_type"],
            "description": mut["description"],
            "classification": classification,
            "caught": caught,
            "screen_seeds": SCREEN_SEEDS,
            "screen_any_wrong": any_wrong,
            "official_seed": OFFICIAL_SEED,
            "official_correct": official.get("correct"),
            "official_status": official.get("status"),
            "official_error": (official.get("error") or "")[:300],
        }
        _append(record)
        all_records.append(record)

        # Clean up temp files
        for p in (ref_path, kernel_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Step 3: Report
    # ------------------------------------------------------------------
    expected_caught = [r for r in all_records if r["classification"] == "expected_caught"]
    harmless = [r for r in all_records if r["classification"] == "harmless_change"]
    caught = [r for r in expected_caught if r["caught"]]
    survived = [r for r in expected_caught if not r["caught"]]

    n_ec = len(expected_caught)
    n_caught = len(caught)
    lo, hi = _wilson_ci(n_caught, n_ec)

    # Per-type breakdown
    by_type: dict[str, dict] = {}
    for r in expected_caught:
        mt = r["mut_type"]
        if mt not in by_type:
            by_type[mt] = {"expected_caught": 0, "caught": 0}
        by_type[mt]["expected_caught"] += 1
        if r["caught"]:
            by_type[mt]["caught"] += 1

    # Per-template breakdown
    by_entry: dict[str, dict] = {}
    for r in all_records:
        eid = r["entry_id"]
        if eid not in by_entry:
            by_entry[eid] = {"total": 0, "expected_caught": 0, "caught": 0, "harmless": 0}
        by_entry[eid]["total"] += 1
        if r["classification"] == "expected_caught":
            by_entry[eid]["expected_caught"] += 1
            if r["caught"]:
                by_entry[eid]["caught"] += 1
        else:
            by_entry[eid]["harmless"] += 1

    report = {
        "classification_rules": {
            "harmless_change": "identical output on ALL seeds {0,1,2,3,4}",
            "expected_caught": "different output on ANY seed in {0,1,2,3,4}",
            "official_seed": OFFICIAL_SEED,
        },
        "total_mutations_generated": len(all_mutations),
        "total_mutations_run": len(all_records),
        "harmless_change": len(harmless),
        "expected_caught": n_ec,
        "caught": n_caught,
        "survived": len(survived),
        "mutation_score": round(n_caught / n_ec, 4) if n_ec else 0.0,
        "wilson_95_ci": [round(lo, 4), round(hi, 4)],
        "by_type": by_type,
        "by_entry": by_entry,
        "survivors": [{"id": r["id"], "mut_type": r["mut_type"],
                        "description": r["description"],
                        "error": r.get("official_error", "")[:200]}
                       for r in survived],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n=== AST MUTATION CAMPAIGN REPORT ===")
    print(f"Classification: harmless_change if correct on ALL seeds {{0..4}}; "
          f"expected_caught if wrong on ANY seed")
    print(f"Official seed : {OFFICIAL_SEED}")
    print(f"Total generated : {len(all_mutations)}")
    print(f"Run            : {len(all_records)}")
    print(f"harmless_change: {len(harmless)}")
    print(f"expected_caught: {n_ec}")
    print(f"Caught         : {n_caught}")
    print(f"Survived       : {len(survived)}")
    if n_ec > 0:
        print(f"Mutation score : {n_caught}/{n_ec} = {n_caught/n_ec*100:.1f}%")
        print(f"Wilson 95% CI  : [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"\nReport: {REPORT_PATH}")
    if survived:
        print("\nSurvivors:")
        for s in report["survivors"]:
            print(f"  {s['id']}")
            print(f"    type: {s['mut_type']}")
            print(f"    desc: {s['description']}")


if __name__ == "__main__":
    main()
