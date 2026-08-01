#!/bin/bash
set -euo pipefail

# Benchmarks decode speed and asserts the generated text is deterministic:
# the model's stdout must be byte-identical to the committed baseline in
# $GOLDEN, otherwise the run fails instead of recording a metric. Re-baseline
# with `BENCH_REGEN=1` (or by deleting $GOLDEN) after a legitimately expected
# change in model output (e.g. a correctness fix).

MODEL="${1:-models/OLMoE-1B-7B-0924-Instruct/}"
CSV="${2:-metrics.csv}"
GOLDEN="${3:-benchmark.expected.txt}"
TIMEOUT_SEC=60
HEADER="commit,tk/s"
PROMPT="The number of planets in the Solar System is: "

COMMIT=$(git rev-parse --short HEAD)

out=$(mktemp)
err=$(mktemp)
trap 'rm -f "$out" "$err"' EXIT

# stdout carries the generated text (asserted deterministic below), stderr
# the load debug lines and the final speed report.
status=0
echo "$PROMPT" | timeout --signal INT --kill-after=5s "$TIMEOUT_SEC" ./build/main "$MODEL" > "$out" 2> "$err" || status=$?

if [ "$status" -ne 0 ]; then
    echo "ERROR: benchmark run failed (exit $status), stderr:" >&2
    cat "$err" >&2
    exit 1
fi

if [ ! -f "$GOLDEN" ]; then
    cp "$out" "$GOLDEN"
    echo "[warn] recorded baseline output in $GOLDEN; delete it or set BENCH_REGEN=1 to re-baseline" >&2
elif [ -n "${BENCH_REGEN:-}" ]; then
    cp "$out" "$GOLDEN"
    echo "[warn] overwrote baseline output in $GOLDEN (BENCH_REGEN=1)" >&2
elif ! cmp -s "$out" "$GOLDEN"; then
    echo "ERROR: benchmark output differs from $GOLDEN, generation is not deterministic:" >&2
    diff -u "$GOLDEN" "$out" | head -40 >&2 || true
    exit 1
fi

tk_s=$(grep -oP 'speed: \K[\d.]+' "$err" | head -1)
if [ -z "$tk_s" ]; then
    echo "ERROR: no 'speed:' line in model stderr" >&2
    exit 1
fi

if [ ! -s "$CSV" ]; then
    echo "$HEADER" > "$CSV"
fi

echo "$COMMIT,$tk_s" >> "$CSV"
echo "Wrote to $CSV: $COMMIT,$tk_s"
