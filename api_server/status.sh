#!/usr/bin/env bash
set -euo pipefail
SERVER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SERVER_DIR/start.sh" --status "$@"
