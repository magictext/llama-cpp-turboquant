#!/bin/bash
# TurboQuant Decode Method Comparison
# Tests constant-LUT vs bit-arithmetic dequant on the current hardware.
# The build auto-detects: M1/M2/M3/M4 → bit-arithmetic, M5+ → constant LUT.
#
# Usage: bash scripts/turbo-decode-comparison.sh [model.gguf]
#
# Outputs a comparison table of decode speed at multiple context depths.

set -uo pipefail

MODEL="${1:-}"
BENCH="./build/bin/llama-bench"

# Auto-find model
if [ -z "$MODEL" ]; then
    MODEL=$(find ./models ../models ~/local_llms/models -name "*.gguf" -type f 2>/dev/null | head -1)
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "ERROR: No model found. Pass path: bash $0 /path/to/model.gguf"
    exit 1
fi

if [ ! -f "$BENCH" ]; then
    echo "ERROR: llama-bench not found. Build first:"
    echo "  cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build -j"
    exit 1
fi

echo "======================================================"
echo "  TurboQuant Decode Method Comparison"
echo "======================================================"
echo "Model: $(basename "$MODEL")"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Capture which method was selected
echo "=== GPU Info ==="
$BENCH -m "$MODEL" -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 -p 64 -n 0 -r 1 2>&1 | grep -E "GPU name|GPU family|has tensor|bit-arithmetic|turbo3" | head -10
echo ""

echo "=== q8_0 Baseline ==="
echo "| Depth | tok/s |"
echo "|-------|-------|"
for DEPTH in 0 4096 8192 16384 32768; do
    LABEL="short"
    ARGS="-p 0 -n 128"
    if [ "$DEPTH" -gt 0 ]; then
        LABEL="${DEPTH}"
        ARGS="-p 0 -n 128 -d $DEPTH"
    fi
    RESULT=$($BENCH -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 $ARGS -r 3 2>&1 | grep "tg128" | awk -F'|' '{print $NF}' | sed 's/[^0-9.]//g' | head -1)
    echo "| $LABEL | $RESULT |"
done
echo ""

echo "=== turbo3 (auto-selected method) ==="
echo "| Depth | tok/s |"
echo "|-------|-------|"
for DEPTH in 0 4096 8192 16384 32768; do
    LABEL="short"
    ARGS="-p 0 -n 128"
    if [ "$DEPTH" -gt 0 ]; then
        LABEL="${DEPTH}"
        ARGS="-p 0 -n 128 -d $DEPTH"
    fi
    RESULT=$($BENCH -m "$MODEL" -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 $ARGS -r 3 2>&1 | grep "tg128" | awk -F'|' '{print $NF}' | sed 's/[^0-9.]//g' | head -1)
    echo "| $LABEL | $RESULT |"
done
echo ""

echo "=== Prefill ==="
echo "| Depth | q8_0 | turbo3 |"
echo "|-------|------|--------|"
for DEPTH in 2048 4096 8192 16384; do
    Q8=$($BENCH -m "$MODEL" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p $DEPTH -n 0 -r 3 2>&1 | grep "pp${DEPTH}" | awk -F'|' '{print $NF}' | sed 's/[^0-9.]//g' | head -1)
    T3=$($BENCH -m "$MODEL" -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 -p $DEPTH -n 0 -r 3 2>&1 | grep "pp${DEPTH}" | awk -F'|' '{print $NF}' | sed 's/[^0-9.]//g' | head -1)
    echo "| $DEPTH | $Q8 | $T3 |"
done
echo ""

echo "=== PPL ==="
WIKI="./wikitext-2-raw/wiki.test.raw"
if [ -f "$WIKI" ]; then
    PPL_Q8=$(./build/bin/llama-perplexity -m "$MODEL" -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -f "$WIKI" --chunks 8 2>&1 | grep "Final" | awk '{print $4}')
    PPL_T3=$(./build/bin/llama-perplexity -m "$MODEL" -ngl 99 -fa on --cache-type-k turbo3 --cache-type-v turbo3 -f "$WIKI" --chunks 8 2>&1 | grep "Final" | awk '{print $4}')
    echo "q8_0:   $PPL_Q8"
    echo "turbo3: $PPL_T3"
else
    echo "SKIPPED (wikitext-2-raw not found)"
fi

echo ""
echo "======================================================"
echo "  DONE"
echo "======================================================"
