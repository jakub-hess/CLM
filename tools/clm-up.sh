#!/usr/bin/env bash
# Install CLM into a venv and run clm-serve in the background against an external /v1/embeddings endpoint (e.g. an LLM gateway).
#
#   ./clm-up.sh          install/update, check the embedder, (re)start
#   ./clm-up.sh stop     stop the server
#
# Everything lives in $CLM_HOME (default: this script's directory): .venv, .env, clm.log, clm.pid.
set -euo pipefail

CLM_HOME="${CLM_HOME:-$(cd "$(dirname "$0")" && pwd)}"
CLM_REPO="${CLM_REPO:-https://github.com/jakub-hess/CLM.git}"
CLM_REF="${CLM_REF:-feat/emb-api-key}"
PORT="${CLM_PORT:-8700}"
PY="${PYTHON:-python3}"

mkdir -p "$CLM_HOME"
cd "$CLM_HOME"
PIDFILE="$CLM_HOME/clm.pid"

stop() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "stopping clm-serve (pid $(cat "$PIDFILE"))"
        kill "$(cat "$PIDFILE")"
        for _ in $(seq 20); do kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break; sleep 0.5; done
    fi
    rm -f "$PIDFILE"
}

if [ "${1:-}" = "stop" ]; then stop; exit 0; fi

# ── .env ──
if [ ! -f .env ]; then
    cat > .env <<'EOF'
CLM_EMB_URL="https://<your-gateway>/v1/embeddings"
CLM_EMB_MODEL="<model name of a Qwen3-8B last-token pooling endpoint>"
CLM_EMB_API_KEY=
# Optional: require "Authorization: Bearer <key>" from CLM clients
# CLM_API_KEY=
CLM_DEVICE=cpu
EOF
    chmod 600 .env
    echo "wrote $CLM_HOME/.env -- set CLM_EMB_URL, CLM_EMB_MODEL and CLM_EMB_API_KEY, then rerun"
    exit 1
fi
set -a; . ./.env; set +a
# The heads are tiny; keep them off a GPU that other services already fill
export CLM_DEVICE="${CLM_DEVICE:-cpu}"
: "${CLM_EMB_API_KEY:?CLM_EMB_API_KEY is empty in $CLM_HOME/.env}"
case "$CLM_EMB_URL$CLM_EMB_MODEL" in *"<"*) echo "fill in CLM_EMB_URL and CLM_EMB_MODEL in $CLM_HOME/.env"; exit 1;; esac

# ── install ──
if [ ! -x .venv/bin/pip ]; then
    rm -rf .venv
    if ! "$PY" -m venv .venv; then
        rm -rf .venv
        echo "venv creation failed; on Debian/Ubuntu: sudo apt install python3-venv"; exit 1
    fi
fi
echo "installing (first run downloads ~200 MB of CPU torch)..."
.venv/bin/pip install --upgrade pip
# CPU-only torch: the heads are tiny, and this skips ~2 GB of CUDA wheels
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
# Serving deps only: the package also lists vllm (local embedder) and pyarrow (training),
# which the gateway setup doesn't need and which would pull CUDA torch back in
.venv/bin/pip install numpy requests fastapi uvicorn
.venv/bin/pip install --upgrade --force-reinstall --no-deps "git+$CLM_REPO@$CLM_REF"

# ── embedder smoke test: reachable, authenticated, 4096-d ──
.venv/bin/python - <<'EOF'
import os, sys
from clm.embedder import Embedder
e = Embedder(os.environ["CLM_EMB_URL"], os.environ["CLM_EMB_MODEL"],
             int(os.environ.get("CLM_EMB_MAX_TOKENS", 2048)), api_key=os.environ["CLM_EMB_API_KEY"])
v, tokens = e.embed(["hello world"])
print(f"embedder ok: dim {v.shape[1]}, {tokens} tokens")
if v.shape[1] != 4096:
    sys.exit("expected 4096-d Qwen3-8B embeddings; the head will not work with this model")
EOF

# ── (re)start ──
stop
nohup .venv/bin/clm-serve --port "$PORT" >> clm.log 2>&1 &
echo $! > "$PIDFILE"

for _ in $(seq 60); do
    if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null; then
        echo "clm-serve up on :$PORT (pid $(cat "$PIDFILE")), log: $CLM_HOME/clm.log"
        curl -s "http://127.0.0.1:$PORT/health"; echo
        exit 0
    fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "clm-serve died, last log lines:"; tail -20 clm.log; exit 1; }
    sleep 1
done
echo "clm-serve not healthy after 60s, see $CLM_HOME/clm.log"; exit 1
