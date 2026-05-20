#!/usr/bin/env bash
# scripts/start_airms_tunnel.sh
#
# Opens an SSH tunnel to the AIR.MS HANA database server.
# Tunnel maps localhost:${LOCAL_PORT} → db.airms.mssm.edu:30041
#
# Usage:
#   bash scripts/start_airms_tunnel.sh          # foreground (blocks terminal)
#   bash scripts/start_airms_tunnel.sh &         # background
#   bash scripts/start_airms_tunnel.sh --check   # exit 0 if tunnel running, 1 if not
#
# Environment variables (set in .env or export before calling):
#   AIRMS_SSH_USER   — your SSH username (default: $USER)
#   AIRMS_LOCAL_PORT — local port to forward (default: 30041)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env if present
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

SSH_USER="${AIRMS_SSH_USER:-$USER}"
LOCAL_PORT="${AIRMS_LOCAL_PORT:-30041}"
REMOTE_HOST="db.airms.mssm.edu"
REMOTE_PORT=30041
PID_FILE="$PROJECT_ROOT/.tunnel.pid"

# Function: check if tunnel is running
check_tunnel() {
    # Test if local port is listening
    timeout 2 bash -c "echo > /dev/tcp/localhost/${LOCAL_PORT}" 2>/dev/null || return 1
}

# Function: start tunnel in background
start_tunnel_background() {
    echo "[tunnel] Starting SSH tunnel in background..."
    echo "[tunnel] localhost:${LOCAL_PORT} → ${REMOTE_HOST}:${REMOTE_PORT}"

    ssh -N -L "${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" \
        "${SSH_USER}@${REMOTE_HOST}" &

    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > "$PID_FILE"

    # Wait for tunnel to establish
    sleep 2

    if check_tunnel; then
        echo "[tunnel] ✓ Tunnel established (PID: $TUNNEL_PID)"
        return 0
    else
        echo "[tunnel] ✗ Tunnel failed to establish. Check SSH credentials."
        rm -f "$PID_FILE"
        return 1
    fi
}

# Function: stop tunnel
stop_tunnel() {
    if [[ -f "$PID_FILE" ]]; then
        TUNNEL_PID=$(cat "$PID_FILE")
        if kill "$TUNNEL_PID" 2>/dev/null; then
            echo "[tunnel] ✓ Tunnel closed (PID: $TUNNEL_PID)"
        fi
        rm -f "$PID_FILE"
    fi
}

# Handle --check flag
if [[ "${1:-}" == "--check" ]]; then
    if check_tunnel; then
        echo "[tunnel] ✓ Tunnel is running on localhost:${LOCAL_PORT}"
        exit 0
    else
        echo "[tunnel] ✗ Tunnel is NOT running on localhost:${LOCAL_PORT}"
        exit 1
    fi
fi

# Default: start tunnel in foreground
echo "[tunnel] Opening SSH tunnel:"
echo "         localhost:${LOCAL_PORT} → ${REMOTE_HOST}:${REMOTE_PORT}"
echo "         SSH user: ${SSH_USER}"
echo "         Press Ctrl+C to close tunnel."
echo ""

ssh -N -L "${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" \
    "${SSH_USER}@${REMOTE_HOST}"
