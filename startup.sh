#!/bin/bash
set -e

# This script starts the appropriate application based on the STARTUP_COMMAND env var.
echo "=== Startup Script Debug Info ==="
echo "STARTUP_COMMAND: ${STARTUP_COMMAND:-'not set'}"
echo "PORT: ${PORT:-'not set'}"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Gunicorn version: $(python -m gunicorn --version)"
echo "Available files: $(ls -la)"
echo "Models directory: $(ls -la models/ 2>/dev/null || echo 'models directory not found')"
echo "=================================="

# Set default port if not provided
if [ -z "$PORT" ]; then
  export PORT=10000
  echo "PORT not set, using default: $PORT"
fi

if [ "$STARTUP_COMMAND" = "api" ]; then
  echo "Starting Fraud Detection API on port $PORT..."
  echo "Testing Python import..."
  python -c "import serve_model; print('serve_model imported successfully')"
  echo "Starting gunicorn..."
  exec python -m gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --log-level debug serve_model:app
elif [ "$STARTUP_COMMAND" = "dashboard" ]; then
  echo "Starting Fraud Detection Dashboard on port $PORT..."
  echo "API_BASE_URL: ${API_BASE_URL:-'not set'}"
  echo "Testing Python import..."
  python -c "import dashboard_app; print('dashboard_app imported successfully')"
  echo "Starting gunicorn..."
  exec python -m gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --log-level debug dashboard_app:server
else
  echo "Error: STARTUP_COMMAND environment variable not set or invalid."
  echo "Available values: 'api' or 'dashboard'"
  echo "Current value: '${STARTUP_COMMAND:-'not set'}'"
  echo "Environment variables:"
  env | grep -E "(STARTUP|PORT)" || echo "No relevant environment variables found"
  exit 1
fi
