FROM python:3.12-slim AS python-base

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────────
# Copy only the dependency manifest first so this layer is cached independently
# of source-code changes.  The editable install (-e) happens after COPY . .
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    "numpy>=2.0" "pydantic>=2.7" "pyyaml>=6.0" \
    "fastapi>=0.111" "uvicorn[standard]>=0.30" "websockets>=12.0" \
    "pytest>=8.2" "ruff>=0.6"

# CPU-only PyTorch for the deep methods (MAPPO, QMIX, TransfQMix, Ensemble, MoE).
# The CPU wheel keeps the image small enough to train on a normal machine; no
# GPU is required.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ── Frontend build ──────────────────────────────────────────────────────
COPY src/rescue_sim/visualization/frontend/package.json \
     src/rescue_sim/visualization/frontend/package-lock.json* \
     /app/src/rescue_sim/visualization/frontend/
RUN cd /app/src/rescue_sim/visualization/frontend && npm ci --ignore-scripts 2>/dev/null || npm install

COPY src/rescue_sim/visualization/frontend/ /app/src/rescue_sim/visualization/frontend/
RUN cd /app/src/rescue_sim/visualization/frontend && npm run build && \
    mkdir -p /app/frontend_dist && \
    cp -r dist/* /app/frontend_dist/

# ── Copy all project files ──────────────────────────────────────────────
COPY . .

# Re-install the project in editable mode now that all source is available
RUN pip install --no-cache-dir -e ".[dev]"

ENV FRONTEND_DIST_DIR=/app/frontend_dist

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default: run the visualization backend (serves frontend from dist/)
CMD ["uvicorn", "src.rescue_sim.visualization.api:app", "--host", "0.0.0.0", "--port", "8000"]
