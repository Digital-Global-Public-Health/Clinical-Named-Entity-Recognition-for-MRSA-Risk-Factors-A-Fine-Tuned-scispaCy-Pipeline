#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

usage() {
    echo "Usage: $0 {2000|5000|10000|full} [GPU_ID]" >&2
    echo "GPU_ID defaults to -1 (CPU); use 0, 1, ... to select a GPU." >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

SIZE="$1"
GPU_ID="${2:--1}"

case "$SIZE" in
    2000|5000|10000)
        TRAIN_DATA="splits/train_${SIZE}.spacy"
        ;;
    full)
        TRAIN_DATA="splits/train.spacy"
        ;;
    *)
        echo "Error: size must be one of 2000, 5000, 10000, or full (got '$SIZE')." >&2
        usage
        exit 2
        ;;
esac

if [[ ! "$GPU_ID" =~ ^-1$|^[0-9]+$ ]]; then
    echo "Error: GPU_ID must be -1 for CPU or a non-negative integer (got '$GPU_ID')." >&2
    exit 2
fi

CONFIG="configs/ner_sci.cfg"
CUSTOM_CODE="configs/custom_code.py"
DEV_DATA="${DEV_DATA:-splits/dev_500.spacy}"
OUTPUT="models/ner_${SIZE}/"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Error: requested training DocBin is missing: $TRAIN_DATA" >&2
    exit 1
fi

if [[ ! -f "$DEV_DATA" ]]; then
    echo "Error: development DocBin is missing: $DEV_DATA" >&2
    exit 1
fi

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

run spacy debug config "$CONFIG" \
    --code "$CUSTOM_CODE" \
    --paths.train "$TRAIN_DATA" \
    --paths.dev "$DEV_DATA"

run spacy debug data "$CONFIG" \
    --code "$CUSTOM_CODE" \
    --paths.train "$TRAIN_DATA" \
    --paths.dev "$DEV_DATA"

run spacy train "$CONFIG" \
    --code "$CUSTOM_CODE" \
    --paths.train "$TRAIN_DATA" \
    --paths.dev "$DEV_DATA" \
    --output "$OUTPUT" \
    --gpu-id "$GPU_ID"
