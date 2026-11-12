#!/usr/bin/env bash
#
# Run the YAMC-verification tasks with a live, in-place progress dashboard
# (see yamc_verify/orchestrator.py) instead of scrolling output. Each task is
# driven directly by `python -m yamc_verify.run_task` with its output captured,
# so the terminal shows only the indented task > code > counts view.
#
# Usage (arguments are task keys; the old notebook paths still work too):
#   MC_BACKENDS=openmc-cpu ./run_models.sh                       # all endf-b8.1 tasks
#   YAMC_VERIFY_NUMBER_OF_SIMULATIONS=10 MC_BACKENDS=openmc-cpu ./run_models.sh
#   MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh
#   MC_BACKENDS=openmc-cpu ./run_models.sh neutron/sphere/endf-b8.1   # specific task(s)
#
# Env vars (read by the orchestrator / run_task):
#   YAMC_VERIFY_NUMBER_OF_SIMULATIONS  cap on nuclides/elements per task (default 3;
#                                      0 = every isotope, -1 = every stable isotope)
#   MC_BACKENDS                        comma list of openmc-cpu,yamc-cpu,yamc-gpu (required, no default)
#   YAMC_VERIFY_NUCLIDES / YAMC_VERIFY_ELEMENTS  explicit item list (overrides the cap)
#
# Results are written to the on-disk cache and viewed in the benchmark database
# page; nothing is written back to any notebook.

set -uo pipefail
cd "$(dirname "$0")"

export YAMC_VERIFY_NUMBER_OF_SIMULATIONS="${YAMC_VERIFY_NUMBER_OF_SIMULATIONS:-3}"
if [ -z "${MC_BACKENDS:-}" ]; then
    echo "error: MC_BACKENDS is not set. Name the backends to run, e.g." >&2
    echo "  MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh" >&2
    echo "Valid backends: openmc-cpu, yamc-cpu, yamc-gpu" >&2
    exit 2
fi
export MC_BACKENDS

# Prefer the repo virtualenv's python; fall back to whatever is on PATH.
PYTHON=".venv/bin/python"
if [ -x "$PYTHON" ]; then
    # Put the venv's bin on PATH so subprocesses find its console scripts even
    # when the venv is not activated. In particular OpenMC's model.run() spawns
    # the `openmc` executable and locates it via PATH; without this, uncached
    # OpenMC tasks fail with "No such file or directory: 'openmc'".
    export PATH="$PWD/.venv/bin:$PATH"
else
    PYTHON="python"
fi

exec "$PYTHON" -m yamc_verify.orchestrator "$@"
