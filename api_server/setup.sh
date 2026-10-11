#!/usr/bin/env bash
set -euo pipefail
SERVER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SERVER_PYTHON="${API_SERVER_PYTHON:-${PIPELINE_PYTHON:-}}"
if [[ -z "$SERVER_PYTHON" && -r "$SERVER_DIR/.python-path" ]]; then
  IFS= read -r SERVER_PYTHON < "$SERVER_DIR/.python-path" || true
fi
SERVER_PYTHON="${SERVER_PYTHON:-$SERVER_DIR/.venv/bin/python}"
if [[ ! -x "$SERVER_PYTHON" ]]; then
  echo "API Python interpreter is unavailable: $SERVER_PYTHON" >&2
  echo "Create a local virtual environment, then set API_SERVER_PYTHON to its bin/python and run setup.sh again." >&2
  exit 1
fi
exec "$SERVER_PYTHON" -m pip install -r "$SERVER_DIR/requirements.txt"
