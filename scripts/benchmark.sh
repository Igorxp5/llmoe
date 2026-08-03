#!/bin/bash
set -euo pipefail

# Benchmarks decode speed and asserts the generated text is deterministic:
# the model's stdout must be byte-identical to the committed baseline in
# $GOLDEN, otherwise the run fails instead of recording a metric. Re-baseline
# with `BENCH_REGEN=1` (or by deleting $GOLDEN) after a legitimately expected
# change in model output (e.g. a correctness fix).
#
# Each row in $CSV records the commit, tokens/s and the host's static hardware
# fingerprint (backend, CPU name, cores/threads, RAM). All values except
# RAM speed and slot count are read from /proc without privileges; pass
# `--sudo-hw` to also read those two via `sudo dmidecode` (root), otherwise
# they are left empty.

TIMEOUT_SEC=60
PROMPT="The number of planets in the Solar System is: "
BACKEND="CPU"

# Parse --sudo-hw first; the remaining positional args are MODEL CSV GOLDEN.
SUDO_HW=0
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --sudo-hw) SUDO_HW=1; shift ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

MODEL="${POSITIONAL[0]:-models/OLMoE-1B-7B-0924-Instruct/}"
CSV="${POSITIONAL[1]:-benchmarks/metrics.csv}"
GOLDEN="${POSITIONAL[2]:-benchmarks/benchmark.expected.txt}"

# ── Static hardware fingerprint (same machine for every historical run) ─────
detect_cpu_name() {
    awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo
}

detect_cores_threads() {
    # e.g. "12c/16t": physical cores and online logical threads from
    # /proc/cpuinfo (which lists only CPUs currently online; the `siblings`
    # field counts offlined threads too).
    local cores threads
    cores=$(awk '/^cpu cores/{print $4; exit}' /proc/cpuinfo)
    threads=$(grep -c '^processor' /proc/cpuinfo)
    echo "${cores}c/${threads}t"
}

detect_ram_mb() {
    # MemTotal is in KiB; report in MB (rounded) like `free -m`.
    local kb
    kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
    echo "$(( (kb + 512) / 1024 ))"
}

fetch_hw() {
    # RAM speed and slot count live in SMBIOS and need root; best-effort only.
    local dmi
    if [ "$SUDO_HW" -ne 1 ]; then
        return
    fi
    if ! dmi=$(sudo dmidecode --type memory 2>/dev/null); then
        echo "[warn] sudo dmidecode failed; ram_speed/ram_slots left empty" >&2
        return
    fi
    ram_speed=$(echo "$dmi" | grep -oE '[[:space:]]Speed: [0-9]+ MT/s' | head -1 |
        sed -E 's/.*Speed: ([0-9]+) MT\/s/\1/')
    ram_slots=$(echo "$dmi" | grep -c 'Form Factor:')
}

CPU_NAME=$(detect_cpu_name)
CPU_CORES_THREADS=$(detect_cores_threads)
RAM_MB=$(detect_ram_mb)
RAM_SPEED=""
RAM_SLOTS=""
fetch_hw

COMMIT=$(git rev-parse --short HEAD)
HEADER="commit,tk_s,backend,cpu_name,cpu_cores_threads,ram_mb,ram_speed_mts,ram_slots"

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

row="$COMMIT,$tk_s,$BACKEND,$CPU_NAME,$CPU_CORES_THREADS,$RAM_MB,$RAM_SPEED,$RAM_SLOTS"
echo "$row" >> "$CSV"
echo "Wrote to $CSV: $row"