#!/usr/bin/env bash
#
# Run the YAMC-verification model notebooks with a live, in-place progress
# dashboard (see yamc_verify/orchestrator.py) instead of scrolling output.
# Each notebook is executed by jupyter nbconvert with its native output
# captured, so the terminal shows only the indented task > code > counts view.
#
# Usage:
#   ./run_models.sh                                  # all endf-b8.1 notebooks
#   YAMC_VERIFY_NUMBER_OF_SIMULATIONS=10 ./run_models.sh
#   MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh
#   ./run_models.sh tasks/neutron/sphere/endf-b8.1.ipynb   # specific notebook(s)
#
# Env vars (read by the notebooks / orchestrator):
#   YAMC_VERIFY_NUMBER_OF_SIMULATIONS  cap on nuclides/elements per notebook (default 3)
#   MC_BACKENDS                        comma list of openmc-cpu,yamc-cpu,yamc-gpu (default openmc-cpu)
#   YAMC_VERIFY_NUCLIDES / YAMC_VERIFY_ELEMENTS  explicit item list (overrides the cap)
#
# Notebooks run in place (outputs written back). Run `git restore tasks` (or the
# repo's clear-output pre-commit hook) afterwards if you only want the cache.

set -uo pipefail
cd "$(dirname "$0")"

export YAMC_VERIFY_NUMBER_OF_SIMULATIONS="${YAMC_VERIFY_NUMBER_OF_SIMULATIONS:-3}"
export MC_BACKENDS="${MC_BACKENDS:-openmc-cpu}"

# Prefer the repo virtualenv's python; fall back to whatever is on PATH.
PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python"

exec "$PYTHON" -m yamc_verify.orchestrator "$@"
