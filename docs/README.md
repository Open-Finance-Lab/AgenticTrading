# Agentic Trading Documentation

Sphinx sources for Read the Docs. From the **repository root**:

Install dependencies (matches Read the Docs):

```bash
pip install -r requirements-sphinx.txt
```

Live preview with auto-reload:

```bash
cd docs/source
sphinx-autobuild . ../build --open-browser --port 8000
```

Open http://127.0.0.1:8000. One-off HTML build:

```bash
cd docs
make html
# open docs/build/html/index.html
```

Edit files under `docs/source/`:

- **Lab** — `docs/source/lab/`
- **Orchestration framework** — `docs/source/orchestration/`

Integration guides outside the Sphinx site:

- **vn.py simulated market data (Chinese)** — `docs/integrations/vnpy-simulation.md`
- **Stripe Credits Test Mode** — `docs/integrations/stripe-credits-test-mode.md`

Then commit and push.

Developer reference (GitHub only; the repo `CLAUDE.md` holds the rules and points here):

- **Configuration (per-variable reference; families documented only in `.env.example` are not repeated)** — `docs/architecture/configuration.md`
- **Backend internals (subsystem detail)** — `docs/architecture/backend-internals.md`
- **Local setup and safe local run** — `docs/local-run.md`
