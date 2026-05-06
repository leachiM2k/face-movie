#!/usr/bin/env bash
set -euo pipefail

# Mode dispatch:
#   face-movie [args...]      → CLI (default)
#   face-movie web [args...]  → Web UI on 0.0.0.0:8080

if [[ "${1:-}" == "web" ]]; then
    shift
    echo "++++++++++++++ Face-Movie: web UI on http://0.0.0.0:8080"
    exec uvicorn webapp.server:app --host 0.0.0.0 --port 8080 "$@"
fi

echo "++++++++++++++ Face-Movie: align + morph"
python main.py "$@"
echo "++++++++++++++ done"
