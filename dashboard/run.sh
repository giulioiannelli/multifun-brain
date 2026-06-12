#!/usr/bin/env bash
# One-command launcher: serve the dashboard from a single FastAPI port.
#
#   ./dashboard/run.sh                       # build once if needed, then serve
#   MFB_DASHBOARD_REBUILD=1 ./dashboard/run.sh   # force a frontend rebuild
#   MFB_DASHBOARD_PORT=8001 ./dashboard/run.sh   # use a different port
#
# The build runs ONLY when there's no existing build (or you force it) — so
# repeat launches start instantly instead of rebuilding every time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PORT="${MFB_DASHBOARD_PORT:-8000}"
URL="http://localhost:${PORT}"

# 1. Refuse to start (with a clear message) if the port is already taken —
#    otherwise a stale/suspended server silently makes the page "never load".
PORT_STATE="$(python - "$PORT" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1]))); print("free")
except OSError:
    print("busy")
finally:
    s.close()
PY
)"
if [ "$PORT_STATE" = "busy" ]; then
  echo "[dashboard] port ${PORT} is already in use — a previous server may still be running."
  echo "  inspect:  lsof -iTCP:${PORT} -sTCP:LISTEN"
  echo "  free it:  kill \$(lsof -tiTCP:${PORT} -sTCP:LISTEN)"
  echo "  or pick another port:  MFB_DASHBOARD_PORT=8001 $0"
  exit 1
fi

# 2. Install frontend deps on first run only.
cd "$HERE/frontend"
if [ ! -d node_modules ]; then
  echo "[dashboard] installing frontend dependencies (first run only)…"
  npm install
fi

# 3. Build only when needed (missing build, or explicitly forced).
if [ "${MFB_DASHBOARD_REBUILD:-0}" = "1" ] || [ ! -f dist/index.html ]; then
  echo "[dashboard] building frontend…"
  npm run build
else
  echo "[dashboard] using existing build (MFB_DASHBOARD_REBUILD=1 to rebuild)."
fi

# 4. Serve everything from one port; open the browser best-effort.
cd "$ROOT"
echo "[dashboard] serving at ${URL}  (Ctrl-C to stop)"
(
  sleep 2
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  elif command -v open >/dev/null 2>&1; then open "$URL"
  fi
) >/dev/null 2>&1 &

exec python -m uvicorn dashboard.backend.app:app --host 127.0.0.1 --port "$PORT"
