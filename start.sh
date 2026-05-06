#!/usr/bin/env bash
set -euo pipefail

echo "++++++++++++++ Face-Movie: align + morph"
python main.py "$@"
echo "++++++++++++++ done"
