FROM python:3.12-slim

# build-essential: some deps (e.g. chromadb's hnswlib) may need to compile if no
# prebuilt wheel matches the platform. curl: used by docker-entrypoint.sh to wait
# for the backend to become healthy before starting the frontend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x docker-entrypoint.sh \
    && mkdir -p .data/chromadb .data/uploads .data/evaluation .data/logs \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# API_BASE_URL points the Streamlit frontend at the FastAPI backend running in
# the same container. Deliberately no OPENROUTER_API_KEY here — leave it unset
# so every request must supply its own key from the frontend (see README).
ENV API_BASE_URL=http://localhost:8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/api/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
