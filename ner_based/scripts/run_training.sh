#!/usr/bin/env bash
# scripts/run_training.sh
#
# Pipeline Step 4: Train / fine-tune a NER model on annotated clinical notes.
#
# Track A (spaCy): runs on CPU, ~30 min for a small annotated set.
# Track B (HF):    requires GPU — submit as a Minerva HPC job or run on a
#                  GPU node with the mrsa-nlp-ner env.
#
# Prerequisites:
#   1. conda env mrsa-nlp-ner is activated
#   2. Annotated training/validation data present in annotations/
#      (train.spacy, val.spacy, test.spacy  OR  train.csv, val.csv, test.csv)
#   3. For Track B: GPU available
#
# Usage:
#   bash scripts/run_training.sh --track spacy [--debug]
#   bash scripts/run_training.sh --track hf    [--debug]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV="mrsa-nlp-ner"
LOG_LEVEL="INFO"
TRACK="spacy"
BASE_MODEL="en_core_sci_sm"    # spaCy; change to emilyalsentzer/Bio_ClinicalBERT for HF
TRAIN_DATA="annotations/train.spacy"
VAL_DATA="annotations/val.spacy"
TEST_DATA="annotations/test.spacy"
MODEL_OUT="models/airms_ner_v1.0"
N_EPOCHS=30
BATCH_SIZE=16
DROPOUT=0.3
LEARNING_RATE=0.001
DEVICE="cpu"
SEED=42
DEBUG=false
DEBUG_N_EXAMPLES=50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --track)   TRACK="$2"; shift 2 ;;
        --debug)   DEBUG=true; shift ;;
        *)         shift ;;
    esac
done

# HF track defaults
if [[ "$TRACK" == "hf" ]]; then
    BASE_MODEL="emilyalsentzer/Bio_ClinicalBERT"
    DEVICE="cuda:0"
    BATCH_SIZE=8
    LEARNING_RATE=0.00005
    echo "[run] HF track selected — ensure GPU is available."
fi

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
# Run training
# ---------------------------------------------------------------------------
echo "[run] Starting NER training"
echo "      track:      $TRACK"
echo "      base model: $BASE_MODEL"
echo "      device:     $DEVICE"
echo "      debug:      $DEBUG"

DEBUG_FLAGS="--no-debug"
if [[ "$DEBUG" == "true" ]]; then
    DEBUG_FLAGS="--debug --debug-n-examples $DEBUG_N_EXAMPLES"
fi

python -m src.cli \
    --log-level "$LOG_LEVEL" \
    train \
    --track "$TRACK" \
    --base-model "$BASE_MODEL" \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --test-data "$TEST_DATA" \
    --model-out-dir "$MODEL_OUT" \
    --n-epochs "$N_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --dropout "$DROPOUT" \
    --learning-rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $DEBUG_FLAGS

echo "[run] Training complete.  Model saved to: $MODEL_OUT"
echo "[run] Training curves:  outputs/train_${TRACK}_<timestamp>/training_curves.png"
