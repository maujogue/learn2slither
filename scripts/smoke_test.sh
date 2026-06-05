#!/usr/bin/env bash
# smoke_test.sh — end-to-end sanity check for learn2slither
# Usage: bash scripts/smoke_test.sh
# All tests run headlessly; no GUI is launched.

set -euo pipefail

PASS=0
FAIL=0
SKIP=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pass() { echo -e "  ${GREEN}✓${RESET} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}✗${RESET} $1"; FAIL=$((FAIL + 1)); }
section() { echo -e "\n${BOLD}$1${RESET}"; }

# Run a command, assert it exits with the expected code, and optionally
# assert that stdout/stderr contains (or does not contain) a pattern.
#
#   check <label> <expected_exit> [--contains <pattern>] [--not-contains <pattern>] -- <cmd...>
check() {
    local label="$1"
    local expected_exit="$2"
    shift 2

    local contains_pat=""
    local not_contains_pat=""

    while [[ $# -gt 0 && "$1" != "--" ]]; do
        case "$1" in
            --contains)      contains_pat="$2";     shift 2 ;;
            --not-contains)  not_contains_pat="$2"; shift 2 ;;
            *) break ;;
        esac
    done
    [[ "$1" == "--" ]] && shift

    local output
    local actual_exit=0
    output=$("$@" 2>&1) || actual_exit=$?

    local ok=true

    if [[ "$actual_exit" != "$expected_exit" ]]; then
        ok=false
        fail "$label  (exit $actual_exit, want $expected_exit)"
        echo "      cmd : $*"
        echo "      output: $(echo "$output" | head -5)"
        return
    fi

    if [[ -n "$contains_pat" && ! "$output" =~ $contains_pat ]]; then
        ok=false
        fail "$label  (pattern not found: '$contains_pat')"
        echo "      output: $(echo "$output" | head -5)"
        return
    fi

    if [[ -n "$not_contains_pat" && "$output" =~ $not_contains_pat ]]; then
        ok=false
        fail "$label  (unexpected pattern found: '$not_contains_pat')"
        echo "      output: $(echo "$output" | head -5)"
        return
    fi

    $ok && pass "$label"
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TMP_Q="/tmp/smoke_q_$$.json"
TMP_NN="/tmp/smoke_nn_$$.json"
trap 'rm -f "$TMP_Q" "${TMP_Q%.*}_validation.svg" "$TMP_NN" "${TMP_NN%.*}_validation.svg"' EXIT

Q_MODEL="src/models/q_table_70000.json"
NN_MODEL="src/models/dqn_2002.json"

echo -e "${BOLD}learn2slither smoke test${RESET}"
echo "repo: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Bad arguments
# ---------------------------------------------------------------------------
section "1. Bad arguments"

check "no subcommand → exit 2" 2 \
    -- uv run l2s

check "unknown subcommand → exit 2" 2 \
    -- uv run l2s foobar

check "train missing --sessions → exit 2" 2 \
    -- uv run l2s train --headless

check "train --engine invalid → exit 2" 2 \
    -- uv run l2s train --headless --sessions 5 --engine bad

check "test --headless no path → exit 1" 1 \
    --contains "requires a model path" \
    -- uv run l2s test --headless

check "test --runs 0 → exit 1" 1 \
    --contains "must be at least 1" \
    -- uv run l2s test --headless --runs 0 "$Q_MODEL"

check "test --runs -3 → exit 1" 1 \
    --contains "must be at least 1" \
    -- uv run l2s test --headless --runs -3 "$Q_MODEL"

check "train --sessions 0 → exit 1" 1 \
    --contains "must be at least 1" \
    -- uv run l2s train --headless --sessions 0

check "train --sessions -1 → exit 1" 1 \
    --contains "must be at least 1" \
    -- uv run l2s train --headless --sessions -1

check "test non-existent model → exit 0 with warning" 0 \
    --contains "does not exist" \
    -- uv run l2s test --headless --runs 1 /nonexistent/path.json

# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------
section "2. Training"

check "train q-table 10 sessions" 0 \
    --contains "Training completed" \
    -- uv run l2s train --headless --engine q --sessions 10 --path "$TMP_Q"

check "train DQN 10 sessions" 0 \
    --contains "Training completed" \
    -- uv run l2s train --headless --engine nn --sessions 10 --path "$TMP_NN"

check "board clamp: --width 0 clamped to 5" 0 \
    --contains "5x5" \
    -- uv run l2s train --headless --engine q --sessions 2 --width 0 --height 0 --path /tmp/smoke_clamp.json
rm -f /tmp/smoke_clamp.json /tmp/smoke_clamp_validation.svg

check "board clamp: --width 999 clamped to 25" 0 \
    --contains "25x25" \
    -- uv run l2s train --headless --engine q --sessions 2 --width 999 --height 999 --path /tmp/smoke_clamp.json
rm -f /tmp/smoke_clamp.json /tmp/smoke_clamp_validation.svg

# ---------------------------------------------------------------------------
# 3. Testing — Q-table
# ---------------------------------------------------------------------------
section "3. Testing — Q-table"

check "test q-table 5 runs" 0 \
    --contains "Mean Score" \
    -- uv run l2s test --headless --engine q --runs 5 "$Q_MODEL"

check "test q-table verbose prints agent choices" 0 \
    --contains "Agent choice:" \
    -- uv run l2s test --headless --verbose --engine q --runs 1 "$Q_MODEL"

check "test q-table reaches length ≥ 10 in majority of runs" 0 \
    -- bash -c "
        output=\$(uv run l2s test --headless --engine q --runs 20 '$Q_MODEL' 2>&1)
        scores=\$(echo \"\$output\" | grep '^Scores:' | sed 's/Scores: //')
        above=0; total=0
        IFS=', []' read -ra arr <<< \"\$scores\"
        for s in \"\${arr[@]}\"; do
            [[ \"\$s\" =~ ^[0-9]+\$ ]] || continue
            total=\$((total + 1))
            (( s >= 10 )) && above=\$((above + 1)) || true
        done
        echo \"Scores ≥10: \$above / \$total\"
        (( above * 2 > total ))
    "

# ---------------------------------------------------------------------------
# 4. Testing — DQN (with auto-detection)
# ---------------------------------------------------------------------------
section "4. Testing — DQN"

check "test DQN 5 runs (explicit --engine nn)" 0 \
    --contains "Mean Score" \
    -- uv run l2s test --headless --engine nn --runs 5 "$NN_MODEL"

check "test DQN auto-detects engine from filename" 0 \
    --contains "Auto-detected engine 'nn'" \
    -- uv run l2s test --headless --runs 5 "$NN_MODEL"

check "test DQN verbose prints agent choices" 0 \
    --contains "Agent choice:" \
    -- uv run l2s test --headless --verbose --engine nn --runs 1 "$NN_MODEL"

# ---------------------------------------------------------------------------
# 5. Resilience — corrupt / empty / wrong-engine models
# ---------------------------------------------------------------------------
section "5. Resilience"

echo '{}' > /tmp/smoke_empty.json
check "empty model file → warning + untrained agent runs" 0 \
    --contains "Could not load" \
    -- uv run l2s test --headless --engine q --runs 2 /tmp/smoke_empty.json
rm -f /tmp/smoke_empty.json

echo 'not{valid[json' > /tmp/smoke_corrupt.json
check "corrupt JSON → warning + untrained agent runs" 0 \
    --contains "Could not load" \
    -- uv run l2s test --headless --engine q --runs 2 /tmp/smoke_corrupt.json
rm -f /tmp/smoke_corrupt.json

check "wrong engine for model → auto-corrected via detection" 0 \
    --contains "Auto-detected engine 'q'" \
    -- uv run l2s test --headless --engine nn --runs 2 "$Q_MODEL"

# ---------------------------------------------------------------------------
# 6. Benchmark
# ---------------------------------------------------------------------------
section "6. Benchmark"

check "benchmark q engine" 0 \
    --contains "Best Models" \
    -- uv run l2s benchmark --headless --engine q --runs 3 --top 2

check "benchmark nn engine" 0 \
    --contains "Best Models" \
    -- uv run l2s benchmark --headless --engine nn --runs 3 --top 2

check "benchmark all engines" 0 \
    --contains "Best Models" \
    -- uv run l2s benchmark --headless --engine all --runs 3 --top 3

check "benchmark --runs clamped: -1 treated as 1" 0 \
    --contains "Best Models" \
    -- uv run l2s benchmark --headless --engine q --runs -1 --top 1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL))
echo ""
echo -e "${BOLD}Results: ${GREEN}${PASS}/${TOTAL} passed${RESET}$([ $FAIL -gt 0 ] && echo -e ", ${RED}${FAIL} failed${RESET}" || true)"

[[ $FAIL -eq 0 ]]
