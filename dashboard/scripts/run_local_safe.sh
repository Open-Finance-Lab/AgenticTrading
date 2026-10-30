#!/usr/bin/env bash
# Start Agentic Trading Lab locally in a no-money-at-risk configuration.
#   - runs against a COPY of the seed DB (the committed backtest.db is never touched)
#   - blanks every broker/LLM key, so nothing can trade or bill
#   - blanks the three *_DATABASE_URL, mail, Stripe and iFinD keys, so a
#     dashboard/.env configured like prod cannot write prod Postgres, send real
#     mail or reach a paid gateway (app.py's load_dotenv never overrides a set
#     variable, and an empty URL selects SQLite)
#   - Robinhood live execution forced off
#   - offline simulated market data (vn.py) enabled
# Usage: dashboard/scripts/run_local_safe.sh [port]   (default 8765)
# See docs/local-run.md for the full guide.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

PORT="${1:-8765}"
SCRATCH="${ATL_LOCAL_DIR:-$HOME/.cache/atl-local}"
mkdir -p "$SCRATCH/bar_cache"
[ -f "$SCRATCH/backtest.db" ] || cp dashboard/storage/data/backtest.db "$SCRATCH/backtest.db"

# A venv keeps its interpreter in Scripts/ on Windows (Git Bash) and bin/
# everywhere else. Picked by OS, not by which file exists: a checkout shared
# with Windows can hold a Windows venv that Linux cannot run.
case "${OSTYPE:-}" in
  msys*|cygwin*) PY=.venv/Scripts/python.exe ;;
  *)             PY=.venv/bin/python ;;
esac

if ! "$PY" -c "import uvicorn" 2>/dev/null; then
  echo "Dependencies missing. Install with:" >&2
  echo "  UV_HTTP_TIMEOUT=180 uv pip install --python $PY -r requirements.txt -r requirements-vnpy.txt pytest pytest-timeout" >&2
  exit 1
fi

echo "DB copy: $SCRATCH/backtest.db   (delete it to reset to the seed data)"
echo "Open:    http://127.0.0.1:$PORT/app"
exec env \
  DATABASE_PATH="$SCRATCH/backtest.db" \
  ATL_BAR_CACHE_DIR="$SCRATCH/bar_cache" \
  ROBINHOOD_EXECUTE=false \
  ATL_BAR_CACHE_WARM=0 \
  LEADERBOARD_DAILY_AUTO_DEPLOY=0 \
  ENABLE_VNPY_SIMULATION=true \
  ALPACA_API_KEY= ALPACA_SECRET_KEY= \
  ANTHROPIC_API_KEY= OPENAI_API_KEY= DEEPSEEK_API_KEY= \
  COMMONSTACK_API_KEY= OPENROUTER_API_KEY= \
  USERS_DATABASE_URL= CONTENT_DATABASE_URL= AGENT_RUNS_DATABASE_URL= \
  BREVO_API_KEY= \
  ATL_STRIPE_TEST_BILLING_ENABLED= STRIPE_SECRET_KEY= STRIPE_WEBHOOK_SECRET= \
  IFIND_REFRESH_TOKEN= IFIND_ACCESS_TOKEN= \
  "$PY" -m uvicorn dashboard.backend.app:app --host 127.0.0.1 --port "$PORT"
