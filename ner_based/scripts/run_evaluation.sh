#!/usr/bin/env bash
# scripts/run_evaluation.sh
#
# Pipeline Step 7: Evaluate NER extraction quality and generate reports.
#
# Prerequisites:
#   1. conda env mrsa-nlp-ner is activated
#   2. Feature aggregation has been run — locate the output CSV path
#
# Usage:
#   bash scripts/run_evaluation.sh <path/to/ner_features.csv> \
#       [--test-annotations <path/to/test.spacy>] \
#       [--rule-features   <path/to/rule_features.csv>] \
#       [--debug]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV="mrsa-nlp-ner"
LOG_LEVEL="INFO"
TARGET_F1=0.70
FEATURES_PATH=""
TEST_ANNOTATIONS=""
RULE_FEATURES=""
DEBUG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test-annotations) TEST_ANNOTATIONS="$2"; shift 2 ;;
        --rule-features)    RULE_FEATURES="$2";    shift 2 ;;
        --debug)            DEBUG=true;             shift ;;
        *)                  FEATURES_PATH="$1";     shift ;;
    esac
done

if [[ -z "$FEATURES_PATH" ]]; then
    echo "Usage: $0 <path/to/ner_features.csv> [--test-annotations <path>] [--rule-features <path>] [--debug]"
    exit 1
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
# Run evaluation
# ---------------------------------------------------------------------------
echo "[run] Evaluating NER features: $FEATURES_PATH"

TEST_FLAG=""
[[ -n "$TEST_ANNOTATIONS" ]] && TEST_FLAG="--test-annotations $TEST_ANNOTATIONS"

RULE_FLAG=""
[[ -n "$RULE_FEATURES" ]] && RULE_FLAG="--rule-features-path $RULE_FEATURES"

DEBUG_FLAG="--no-debug"
[[ "$DEBUG" == "true" ]] && DEBUG_FLAG="--debug"

python -m src.cli \
    --log-level "$LOG_LEVEL" \
    evaluate \
    "$FEATURES_PATH" \
    $TEST_FLAG \
    $RULE_FLAG \
    --target-f1 "$TARGET_F1" \
    $DEBUG_FLAG

echo "[run] Evaluation complete.  See outputs/ner_evaluation_<timestamp>/"
