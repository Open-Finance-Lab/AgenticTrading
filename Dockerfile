FROM python:3.13-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt requirements-ai-hedge-fund.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Keep virattt/ai-hedge-fund and its LangChain graph isolated from ATL's main
# dependency set. The adapter invokes this interpreter over a JSON subprocess
# boundary; marketplace users never clone or install the upstream project.
#
# Upstream pins numpy^1.24 (<2). numpy 1.26.x has no cp313 wheels, and
# python:*-slim has no C compiler to build from source — so the isolated
# venv must run on 3.12 even though ATL itself stays on 3.13. uv installs
# a managed 3.12 under /opt/uv-python; the venv keeps an absolute reference
# to it, so uninstalling the uv build tool afterwards is safe.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN pip install --no-cache-dir uv \
    && uv python install 3.12 \
    && uv venv --python 3.12 /opt/ai-hedge-fund-venv \
    && uv pip install --python /opt/ai-hedge-fund-venv \
       -r requirements-ai-hedge-fund.txt \
    && /opt/ai-hedge-fund-venv/bin/python -c "from src.main import run_hedge_fund" \
    && pip uninstall -y uv

# Copy dashboard application
COPY dashboard ./dashboard

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV AI_HEDGE_FUND_PYTHON=/opt/ai-hedge-fund-venv/bin/python

# Health check (PORT-aware so it stays valid when PORT is overridden)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os, requests; requests.get('http://localhost:' + os.environ.get('PORT', '8000') + '/health')" || exit 1

# Run backend via the canonical ASGI package target.
# WORKDIR is /app (repo-root-compatible), so `dashboard.backend.app` imports
# with no extra import-path configuration. sh -c is used so ${PORT} is expanded
# at runtime; defaults to 8000 when PORT is absent.
CMD ["sh", "-c", "uvicorn dashboard.backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
