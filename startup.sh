#!/bin/bash
set -e

# This script starts the appropriate application based on the STARTUP_COMMAND env var.

if [ "$STARTUP_COMMAND" = "api" ]; then
  echo "Starting Fraud Detection API..."
  exec gunicorn --bind 0.0.0.0:$PORT serve_model:app
elif [ "$STARTUP_COMMAND" = "dashboard" ]; then
  echo "Starting Fraud Detection Dashboard..."
  exec gunicorn --bind 0.0.0.0:$PORT dashboard_app:server
else
  echo "Error: STARTUP_COMMAND environment variable not set or invalid."
  echo "Set it to 'api' or 'dashboard'."
  exit 1
fi
