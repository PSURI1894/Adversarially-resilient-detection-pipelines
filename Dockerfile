# ==============================================================================
# MULTI-STAGE DOCKERFILE — ADVERSARIALLY RESILIENT IDS PIPELINE
# ==============================================================================
# Stage 1: Build dependencies
# Stage 2: Runtime (slim, non-root)
# ==============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ci.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-ci.txt \
    && find /install -name "*.pyc" -delete \
    && find /install -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r idsuser && useradd -r -g idsuser -d /app -s /sbin/nologin idsuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY main_pipeline.py .
COPY simulation_engine.py .

# Create necessary directories
RUN mkdir -p data/raw data/processed reports/figures reports/audit_logs \
    reports/experiments reports/dashboards models/registry \
    && chown -R idsuser:idsuser /app

# Switch to non-root user
USER idsuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check endpoint (FastAPI on port 8000)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/status')" || exit 1

# Expose ports
# 8000 = FastAPI API
# 9090 = Prometheus metrics
EXPOSE 8000 9090

# Default entrypoint
CMD ["python", "main_pipeline.py"]
