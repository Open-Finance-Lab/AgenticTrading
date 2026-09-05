# Credentials (local only)

This folder is not tracked in git. For local development:

```bash
cp credentials/alpaca.json.example credentials/alpaca.json
# Edit credentials/alpaca.json with your Alpaca paper-trading keys
```

Alternatively, set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env` (see `.env.example` at the repo root).

For **live** (real-money) Alpaca trading, use a separate key pair via `credentials/alpaca_live.json.example` → `credentials/alpaca_live.json`, or `ALPACA_LIVE_API_KEY` / `ALPACA_LIVE_SECRET_KEY` in `.env` — never reuse the paper credentials above.
