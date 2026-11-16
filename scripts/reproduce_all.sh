#!/usr/bin/env bash
# reproduce_all.sh
#
# CPU-only reproducibility script. No GPU, no paid API calls, no internet.
# Runs all checks that can be done locally:
#
#   Step 1  Python test suite                   ~35s
#   Step 2  Mutation test (Docker, interpreter)  ~25s
#   Step 3  Report generation (from any results) <5s
#
# Usage (from repo root):
#   bash scripts/reproduce_all.sh
#
# Requirements:
#   - Python virtualenv active (.venv or the conda env the user configured)
#   - Docker available with kernelforge-verify:latest image
#   - 5 GB free disk
#
# Exit codes:
#   0  all steps passed
#   1  one or more steps failed

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO}/evaluation/results"

# Colour helpers — degrade gracefully when stdout is not a terminal.
_bold=""  _reset=""  _green=""  _red=""  _yellow=""
if [ -t 1 ] && command -v tput &>/dev/null; then
    _bold="$(tput bold 2>/dev/null || true)"
    _reset="$(tput sgr0 2>/dev/null || true)"
    _green="$(tput setaf 2 2>/dev/null || true)"
    _red="$(tput setaf 1 2>/dev/null || true)"
    _yellow="$(tput setaf 3 2>/dev/null || true)"
fi

_header() { printf "\n%s=== %s ===%s\n" "${_bold}" "$*" "${_reset}"; }
_ok()     { printf "%s[  OK  ]%s %s\n" "${_green}"  "${_reset}" "$*"; }
_fail()   { printf "%s[ FAIL ]%s %s\n" "${_red}"    "${_reset}" "$*"; }
_info()   { printf "%s[ INFO ]%s %s\n" "${_yellow}"  "${_reset}" "$*"; }

# Simple timing: bash-3-compatible.
_now() { date +%s; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
_header "Preflight"

if ! python --version &>/dev/null; then
    _fail "python not found — activate the virtualenv first"; exit 1
fi
_info "Python: $(python --version 2>&1)"

if ! docker info &>/dev/null; then
    _fail "Docker daemon not running — start Docker Desktop and retry"; exit 1
fi

if ! docker image inspect kernelforge-verify:latest &>/dev/null; then
    _fail "Docker image kernelforge-verify:latest not found"
    _info "Build it with: docker build -t kernelforge-verify -f docker/Dockerfile.verify ."
    exit 1
fi
_info "Docker image: kernelforge-verify:latest  OK"

DISK_FREE_GB=$(df -g "${REPO}" 2>/dev/null | awk 'NR==2{print $4}' \
              || df -BG "${REPO}" 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' \
              || echo 999)
_info "Disk free: ~${DISK_FREE_GB} GB"

cd "${REPO}"

# ---------------------------------------------------------------------------
# Step 1: Test suite
# ---------------------------------------------------------------------------
_header "Step 1: Python test suite"
T1_START=$(_now)
if python -m pytest --tb=short -q 2>&1 | tee /tmp/kf_pytest.log; then
    T1_END=$(_now)
    T1_ELAPSED=$((T1_END - T1_START))
    PYTEST_SUMMARY=$(tail -1 /tmp/kf_pytest.log || echo "?")
    _ok "pytest passed in ${T1_ELAPSED}s — ${PYTEST_SUMMARY}"
    STEP1_STATUS=PASS
else
    T1_END=$(_now)
    T1_ELAPSED=$((T1_END - T1_START))
    _fail "pytest FAILED after ${T1_ELAPSED}s — see /tmp/kf_pytest.log"
    STEP1_STATUS=FAIL
fi

# ---------------------------------------------------------------------------
# Step 2: Mutation test (Docker, Triton interpreter)
# ---------------------------------------------------------------------------
_header "Step 2: Mutation test (Docker, interpreter mode)"
T2_START=$(_now)
if docker run --rm \
       -v "${REPO}":/work \
       kernelforge-verify \
       python scripts/mutation_test.py 2>&1 | tee /tmp/kf_mutation.log; then
    T2_END=$(_now)
    T2_ELAPSED=$((T2_END - T2_START))
    MUTATION_SCORE=$(grep "^Mutation score" /tmp/kf_mutation.log || echo "?")
    _ok "mutation_test passed in ${T2_ELAPSED}s — ${MUTATION_SCORE}"
    STEP2_STATUS=PASS
else
    T2_END=$(_now)
    T2_ELAPSED=$((T2_END - T2_START))
    _fail "mutation_test FAILED after ${T2_ELAPSED}s — see /tmp/kf_mutation.log"
    STEP2_STATUS=FAIL
fi

# ---------------------------------------------------------------------------
# Step 3: Report generation
# ---------------------------------------------------------------------------
_header "Step 3: Report generation"
REPORT_OUT="${REPO}/docs/results"
mkdir -p "${REPORT_OUT}"

# Use evaluation/results/ if it has JSON files; otherwise an empty dir.
if ls "${RESULTS_DIR}"/*.json &>/dev/null 2>&1; then
    REPORT_SRC="${RESULTS_DIR}"
    _info "Report source: evaluation/results/"
else
    REPORT_SRC="$(mktemp -d)"
    _info "Report source: empty dir (no GPU results — stages will show absent)"
fi

T3_START=$(_now)
if python -m evaluation.report "${REPORT_SRC}" \
       --out "${REPORT_OUT}" --no-figures 2>&1 | tee /tmp/kf_report.log; then
    T3_END=$(_now)
    T3_ELAPSED=$((T3_END - T3_START))
    _ok "report generated in ${T3_ELAPSED}s — written to docs/results/"
    STEP3_STATUS=PASS
else
    T3_END=$(_now)
    T3_ELAPSED=$((T3_END - T3_START))
    _fail "report FAILED after ${T3_ELAPSED}s — see /tmp/kf_report.log"
    STEP3_STATUS=FAIL
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
_header "Summary"
printf "  %-20s  %s  (%s)\n" "pytest"       "${STEP1_STATUS}" "${T1_ELAPSED:-?}s"
printf "  %-20s  %s  (%s)\n" "mutation_test" "${STEP2_STATUS}" "${T2_ELAPSED:-?}s"
printf "  %-20s  %s  (%s)\n" "report"       "${STEP3_STATUS}" "${T3_ELAPSED:-?}s"
echo ""

if [ "${STEP1_STATUS}" = PASS ] && [ "${STEP2_STATUS}" = PASS ] && [ "${STEP3_STATUS}" = PASS ]; then
    _ok "All steps PASSED"
    exit 0
else
    _fail "One or more steps FAILED"
    exit 1
fi
