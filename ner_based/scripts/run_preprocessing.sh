#!/usr/bin/env bash
# scripts/run_preprocessing.sh
#
# Pipeline Step 2: Clean and normalise raw clinical note chunks for NER.
#
# Prerequisites:
#   1. conda env mrsa-nlp-ner is activated
#   2. Cohort builder has been run (notes present in data/interim/airms/notes/)
#
# Usage:
#   bash scripts/run_preprocessing.sh [--debug]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV="mrsa-nlp-ner"
LOG_LEVEL="INFO"
RAW_NOTES_DIR="data/interim/airms/notes"
OUT_DIR="data/interim/airms/notes_preprocessed"
MAX_TOKENS=0         # 0 = no BERT windowing; set 512 for HF track
DEBUG=false
DEBUG_N_NOTES=200

for arg in "$@"; do
    case $arg in
        --debug) DEBUG=true ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"

if [[ -f ".env" ]]; then
    set -a; source ".env"; set +a
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "[run] Activated conda env: $CONDA_ENV"
fi

# ---------------------------------------------------------------------------
# Run pipeline step
# ---------------------------------------------------------------------------
echo "[run] Starting NER preprocessing (debug=${DEBUG}, max_tokens=${MAX_TOKENS})"

DEBUG_FLAGS="--no-debug"
if [[ "$DEBUG" == "true" ]]; then
    DEBUG_FLAGS="--debug --debug-n-notes $DEBUG_N_NOTES"
fi

python -m src.cli \
    --log-level "$LOG_LEVEL" \
    preprocess \
    --raw-notes-dir "$RAW_NOTES_DIR" \
    --out-dir "$OUT_DIR" \
    --max-tokens "$MAX_TOKENS" \
    --expand-abbrev \
    $DEBUG_FLAGS

echo "[run] Preprocessing complete."
