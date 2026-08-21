#!/usr/bin/env bash
# Start (or reuse) a Cloudflare Tunnel to the Streamlit frontend and print
# the public URL. Run this once before a demo session.
set -euo pipefail

PORT="${FRONTEND_PORT:-8502}"
LOGFILE="$(mktemp -t cloudflared-tunnel-log)"

if pgrep -f "cloudflared tunnel --url http://localhost:${PORT}" >/dev/null 2>&1; then
    echo "A tunnel to localhost:${PORT} is already running." >&2
    echo "Kill it first if you want a fresh URL: pkill -f 'cloudflared tunnel'" >&2
    exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared not found. Install it with: brew install cloudflared" >&2
    exit 1
fi

echo "Starting tunnel to http://localhost:${PORT} ..."
nohup cloudflared tunnel --url "http://localhost:${PORT}" > "$LOGFILE" 2>&1 &
TUNNEL_PID=$!
disown

for _ in $(seq 1 30); do
    URL=$(grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' "$LOGFILE" | head -1 || true)
    if [ -n "${URL:-}" ]; then
        echo ""
        echo "Public URL: ${URL}"
        echo "(tunnel pid ${TUNNEL_PID}, logs at ${LOGFILE})"
        exit 0
    fi
    sleep 1
done

echo "Timed out waiting for tunnel URL. Check logs at ${LOGFILE}" >&2
exit 1
