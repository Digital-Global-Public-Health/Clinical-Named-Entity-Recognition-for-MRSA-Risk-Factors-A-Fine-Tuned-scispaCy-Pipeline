#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Configuration
# -------------------------
LOGIN_NODE="li04e02"
REMOTE_HOST="db.airms.mssm.edu"
REMOTE_PORT=30041

AIRMS_DB="AIRMS"
AIRMS_USER="ahmeda47"

SSL_ENCRYPT="TRUE"
SSL_VALIDATE="FALSE"
SSL_HOSTNAME_IN_CERT="hana-pa2.mssm.edu"
SSL_TRUSTSTORE="None"
CONNECT_TIMEOUT="0"

PORT_START=50000
PORT_END=51000

SCHEMA="CDMPHI"
CHUNK_SIZE=500
MIN_NOTE_DATE="2014-07-14"
DEBUG=false
DEBUG_N_PERSONS=20
SEED=7

LOG_LEVEL="INFO"

# -------------------------
# Parse optional args
# -------------------------
# Usage:
#   bash scripts/run_cohort_builder.sh [--debug]
while [ $# -gt 0 ]; do
  case "$1" in
    --debug)     DEBUG=true; shift ;;
    --seed)      shift; SEED="$1"; shift ;;
    *)           shift ;;
  esac
done

# -------------------------
# Helpers
# -------------------------
find_open_port() {
  local s=$1
  local e=$2
  for ((p=s; p<=e; p++)); do
    if ! nc -z 127.0.0.1 "$p" 2>/dev/null && ! lsof -ti :"$p" >/dev/null 2>&1; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

cleanup_tunnel() {
  if [[ -n "${TUNNEL_PID:-}" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
  fi
}

trap cleanup_tunnel EXIT

# -------------------------
# Main
# -------------------------
main() {
  echo "=== MRSA NLP (NER-Based) — Cohort Builder ==="

  read -s -p "Enter AIRMS HANA password for user '${AIRMS_USER}': " AIRMS_PASSWORD
  echo

  local LPORT
  LPORT=$(find_open_port "$PORT_START" "$PORT_END") || {
    echo "ERROR: no free local port between $PORT_START and $PORT_END" >&2
    exit 1
  }
  echo "[run] Using local port $LPORT"

  if ! ssh -f -N \
       -L "${LPORT}:${REMOTE_HOST}:${REMOTE_PORT}" \
       -o ExitOnForwardFailure=yes \
       -o ServerAliveInterval=60 \
       -o ServerAliveCountMax=3 \
       "${LOGIN_NODE}"; then
    echo "ERROR: Failed to establish SSH tunnel through ${LOGIN_NODE}" >&2
    exit 2
  fi

  echo "[run] ✓ Tunnel established: localhost:${LPORT} → ${REMOTE_HOST}:${REMOTE_PORT}"

  # Export environment variables for connection
  export AIRMS_HOST="127.0.0.1"
  export AIRMS_PORT="${LPORT}"
  export AIRMS_DATABASE="${AIRMS_DB}"
  export AIRMS_USER="${AIRMS_USER}"
  export AIRMS_PASSWORD="${AIRMS_PASSWORD}"
  export AIRMS_ENCRYPT="${SSL_ENCRYPT}"
  export AIRMS_SSL_VALIDATE_CERTIFICATE="${SSL_VALIDATE}"
  export AIRMS_SSL_HOSTNAME_IN_CERT="${SSL_HOSTNAME_IN_CERT}"
  export AIRMS_SSL_TRUSTSTORE="${SSL_TRUSTSTORE}"
  export AIRMS_CONNECT_TIMEOUT="${CONNECT_TIMEOUT}"

  # Thread management on HPC
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

  # Activate conda env if not already active
  CONDA_ENV="mrsa-nlp-ner"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "[run] Activated conda env: $CONDA_ENV"
  fi

  echo "[run] Starting cohort builder (debug=${DEBUG}, seed=${SEED})"

  # Run pipeline
  if [[ "$DEBUG" == "true" ]]; then
    python -m src.cli \
        --log-level "$LOG_LEVEL" \
        --seed "$SEED" \
        build-cohort \
        --schema "$SCHEMA" \
        --chunk-size "$CHUNK_SIZE" \
        --min-note-date "$MIN_NOTE_DATE" \
        --debug \
        --debug-n-persons "$DEBUG_N_PERSONS"
  else
    python -m src.cli \
        --log-level "$LOG_LEVEL" \
        --seed "$SEED" \
        build-cohort \
        --schema "$SCHEMA" \
        --chunk-size "$CHUNK_SIZE" \
        --min-note-date "$MIN_NOTE_DATE" \
        --no-debug
  fi

  echo "[run] Cohort builder complete."
}

main "$@"
