#!/usr/bin/env bash
#
# Run the YAMC-verification model notebooks on a few nuclides/elements to
# populate the on-disk results cache (e.g. the slow OpenMC reference results).
#
# Usage:
#   ./run_models.sh                                  # 3 items each, OpenMC only (build cache)
#   YAMC_VERIFY_NUMBER_OF_SIMULATIONS=10 ./run_models.sh
#   MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh
#   ./run_models.sh tasks/neutron/sphere/endf-b8.1.ipynb   # run specific notebook(s)
#
# Env vars (read by the notebooks themselves):
#   YAMC_VERIFY_NUMBER_OF_SIMULATIONS  cap on nuclides/elements per notebook (default 3)
#   MC_BACKENDS                        comma list of openmc-cpu,yamc-cpu,yamc-gpu (default openmc-cpu)
#
# Notebooks run in place (outputs written back); the cache lands in
# tasks/<physics>/<geometry>/cache/. Run `git restore tasks` afterwards if you
# only want the cache and not the notebook-output changes.

set -uo pipefail
cd "$(dirname "$0")"

export YAMC_VERIFY_NUMBER_OF_SIMULATIONS="${YAMC_VERIFY_NUMBER_OF_SIMULATIONS:-3}"
export MC_BACKENDS="${MC_BACKENDS:-openmc-cpu}"

# Prefer the repo virtualenv's jupyter; fall back to whatever is on PATH.
JUPYTER=".venv/bin/jupyter"
[ -x "$JUPYTER" ] || JUPYTER="jupyter"

# Default to every endf-b8.1 model notebook (the FENDL variants have no data).
if [ "$#" -gt 0 ]; then
  notebooks=("$@")
else
  notebooks=(tasks/*/*/endf-b8.1.ipynb)
fi

echo "backends=$MC_BACKENDS  items/notebook=$YAMC_VERIFY_NUMBER_OF_SIMULATIONS  notebooks=${#notebooks[@]}"
echo

passed=()
failed=()
for nb in "${notebooks[@]}"; do
  echo "=== $nb ==="
  # timeout=-1 disables the per-cell timeout so full-statistics transport cells finish.
  if "$JUPYTER" nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=-1 "$nb"; then
    passed+=("$nb")
  else
    echo "!! FAILED: $nb"
    failed+=("$nb")
  fi
  echo
done

echo "================ summary ================"
echo "passed: ${#passed[@]} / ${#notebooks[@]}"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "failed:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
