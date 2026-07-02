#!/usr/bin/env bash
# Run every model / simulation type, on all compute options
# (OpenMC CPU + YAMC CPU + YAMC GPU), for stable isotopes/elements only,
# then audit YAMC vs OpenMC agreement across the populated cache.
set -euo pipefail
cd "$(dirname "$0")"

# Prefer the repo virtualenv's python; fall back to whatever is on PATH.
PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python"

YAMC_VERIFY_NUMBER_OF_SIMULATIONS=-1 MC_BACKENDS=openmc-cpu,yamc-cpu,yamc-gpu ./run_models.sh "$@"

echo
echo "=== YAMC vs OpenMC agreement audit ==="
"$PYTHON" -m yamc_verify.audit
