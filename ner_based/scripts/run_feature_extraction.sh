#!/usr/bin/env bash
# scripts/run_feature_extraction.sh
#
# Pipeline Steps 5 + 6: NER inference on notes + feature aggregation.
#
# Prerequisites:
#   1. conda env mrsa-nlp-ner is activated
#   2. Preprocessing has been run
#   3. A trained model is present at MODEL_PATH
#
# Usage:
#   bash scripts/run_feature_extraction.sh [--track spacy|hf] [--debug]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV="mrsa-nlp-ner"
LOG_LEVEL="INFO"
PREPROCESSED_DIR="data/interim/airms/notes_preprocessed"
EXTRACTIONS_DIR="data/interim/airms/ner_extractions"
COHORT_PATH="data/interim/airms/mrsa_cohort_person_list.parquet"
MODEL_PATH="models/airms_ner_v1.0"
TRACK="spacy"
BATCH_SIZE=32
NEGATION_WINDOW=5
LEVEL="visit"
DEBUG=false
DEBUG_N_NOTES=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --track) TRACK="$2"; shift 2 ;;
        --debug) DEBUG=true; shift ;;
        *)       shift ;;
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

DEBUG_FLAGS="--no-debug"
if [[ "$DEBUG" == "true" ]]; then
    DEBUG_FLAGS="--debug --debug-n-notes $DEBUG_N_NOTES"
fi

# ---------------------------------------------------------------------------
# Step 5 — NER extraction
# ---------------------------------------------------------------------------
echo "[run] Step 5/6: NER extraction (track=${TRACK}, debug=${DEBUG})"

python -m src.cli \
    --log-level "$LOG_LEVEL" \
    extract \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --out-dir "$EXTRACTIONS_DIR" \
    --model-path "$MODEL_PATH" \
    --model-track "$TRACK" \
    --batch-size "$BATCH_SIZE" \
    --negation-window "$NEGATION_WINDOW" \
    $DEBUG_FLAGS

echo "[run] NER extraction complete."

# ---------------------------------------------------------------------------
# Step 6 — Feature aggregation
# ---------------------------------------------------------------------------
echo "[run] Step 6/6: Feature aggregation (level=${LEVEL})"

python -m src.cli \
    --log-level "$LOG_LEVEL" \
    aggregate-features \
    --extractions-dir "$EXTRACTIONS_DIR" \
    --cohort-path "$COHORT_PATH" \
    --level "$LEVEL" \
    --include-negated \
    --include-counts \
    $DEBUG_FLAGS

echo "[run] Feature aggregation complete."
echo "[run] Output: outputs/ner_feature_aggregation_<timestamp>/"
