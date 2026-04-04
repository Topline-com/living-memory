#!/bin/bash
# Dreamcatcher Nightly Consolidation
# Add to crontab: 0 3 * * * /path/to/dreamcatcher/scripts/train_nightly.sh >> /path/to/dreamcatcher/logs/nightly.log 2>&1

set -euo pipefail
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${PROJ_DIR}/logs"

echo "=================================================="
echo "Dreamcatcher Nightly — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "=================================================="

[ -d "${PROJ_DIR}/.venv" ] && source "${PROJ_DIR}/.venv/bin/activate"
cd "${PROJ_DIR}"
python -m dreamcatcher nightly
python -m dreamcatcher cleanup --keep 5

echo "Complete at $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "=================================================="
