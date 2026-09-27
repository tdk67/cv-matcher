#!/bin/bash
# Starts the FastAPI backend and Streamlit frontend in one container.
# Backend is internal-only (localhost); only the frontend port needs to be
# public on platforms that expose a single port (see README Deployment section).
set -e

uvicorn src.main:app --host 0.0.0.0 --port 8000 --proxy-headers &
BACKEND_PID=$!

echo "Waiting for backend to become healthy..."
until curl -sf http://localhost:8000/api/health >/dev/null 2>&1; do
    sleep 1
done
echo "Backend is up."

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
FRONTEND_PID=$!

_term() {
    echo "Shutting down..."
    kill -TERM "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
}
trap _term TERM INT

# Exit (and let the container stop) as soon as either process dies.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
EXIT_CODE=$?
_term
exit "$EXIT_CODE"
