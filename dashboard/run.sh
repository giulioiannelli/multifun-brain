#!/usr/bin/env bash
# One-command launcher: build the frontend (if needed) and serve everything
# from a single FastAPI port. Collaborators just run this and open the browser.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PORT="${MFB_DASHBOARD_PORT:-8000}"
URL="http://localhost:${PORT}"

cd "$HERE/frontend"
if [ ! -d node_modules ]; then
  echo "[dashboard] installing frontend dependencies (first run only)…"
  npm install
fi
echo "[dashboard] building frontend…"
npm run build

cd "$ROOT"
echo "[dashboard] serving at ${URL}  (Ctrl-C to stop)"
# Open the browser shortly after the server starts (best-effort, non-fatal).
(
  sleep 2
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  elif command -v open >/dev/null 2>&1; then open "$URL"
  fi
) >/dev/null 2>&1 &

exec python -m uvicorn dashboard.backend.app:app --host 127.0.0.1 --port "$PORT"
