#!/bin/bash
set -euo pipefail

MODEL="${1:-models/OLMoE-1B-7B-0924-Instruct/}"
CSV="${2:-metrics.csv}"
TIMEOUT_SEC=60
HEADER="commit,tk/s"

COMMIT=$(git rev-parse --short HEAD)

stderr=$(echo "The number of planets in the Solar System is: " | timeout --signal INT --kill-after=5s "$TIMEOUT_SEC" ./build/main "$MODEL" 2>&1 >/dev/null) || true

tk_s=$(echo "$stderr" | grep -oP 'speed: \K[\d.]+' | head -1)

if [ ! -f "$CSV" ]; then
    echo "$HEADER" > "$CSV"
fi

echo "$COMMIT,$tk_s" >> "$CSV"
echo "Wrote to $CSV: $COMMIT,$tk_s"
