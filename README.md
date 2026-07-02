# yamc-verification

Verification of YAMC by comparing its simulation results against the gold-standard
Monte Carlo code, OpenMC.

## Setup

```bash
pip install -r requirements.txt
```

YAMC is used as an editable install from a sibling `yamc` checkout (`../yamc`).

## Running the verification models

`run_models.sh` executes the model notebooks in place and populates the on-disk
results cache. It drives them through the `yamc_verify` orchestrator
(`python -m yamc_verify.orchestrator`), which runs each notebook silently
(`jupyter nbconvert` with native YAMC/OpenMC output captured) and renders a
single **live, in-place dashboard** instead of a large scrolling log:

```text
running with YAMC 0.4.0, built 3 mins ago
120/300 cached, 80/180 completed ETA 30 mins
neutron sphere - completed 15 mins
    yamc-cpu - 50/50 completed 10 mins
    yamc-gpu - 20/50 cached 30/50 completed 5 mins
    openmc - 50/50 cached
photon sphere
    yamc-cpu - 30/50 ETA 9 mins  <--- in progress
    yamc-gpu - 50/50 todo
    openmc - 50/50 cached
```

The full plan (every task, code, and cached/to-do split) is computed up front
by inspecting the cache, and the counts + ETA update live as each item
finishes. Environment variables control it:

- `MC_BACKENDS` - comma-separated list of `openmc-cpu`, `yamc-cpu`, `yamc-gpu` (default: `openmc-cpu`).
- `YAMC_VERIFY_NUCLIDES` - comma-separated list naming the exact isotope(s) to run
  on the nuclide-based models (neutron, secondary-photon, decay-photon, transmutation),
  e.g. `Fe56`. Overrides the count cap.
- `YAMC_VERIFY_ELEMENTS` - comma-separated list naming the exact element(s) to run on
  the photon models, e.g. `Fe`. Overrides the count cap.
- `YAMC_VERIFY_NUMBER_OF_SIMULATIONS` - cap on nuclides/elements per model when no
  explicit list is given (default: `3`). Special values:
  - `0` - no cap: run **every** discovered nuclide/element (includes radioactive ones).
  - `-1` - run every **stable** (non-radioactive) nuclide, and for the photon
    (element) models every element that has a stable isotope. Stability comes from
    the depletion chain. The already physics-restricted models (decay-photon D1S
    parents, transmutation stable seeds) are unaffected.

Examples:

```bash
# OpenMC only, 3 nuclides/elements per model (builds the reference cache)
./run_models.sh

# yamc + OpenMC, 3 each (this is what fills the comparison plots)
MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh

# yamc + OpenMC, 10 nuclides/elements per model
YAMC_VERIFY_NUMBER_OF_SIMULATIONS=10 MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh

# EVERYTHING: every stable isotope/element, every model, on all three codes
# (OpenMC CPU + YAMC CPU + YAMC GPU). `=-1` keeps only stable, non-radioactive
# nuclides (and elements with a stable isotope); no notebook argument means all
# endf-b8.1 models. This is the full verification sweep - expect it to take a
# long time and rely heavily on the cache.
YAMC_VERIFY_NUMBER_OF_SIMULATIONS=-1 \
  MC_BACKENDS=openmc-cpu,yamc-cpu,yamc-gpu ./run_models.sh

# Same, but every isotope including radioactive ones (`=0` removes the cap).
YAMC_VERIFY_NUMBER_OF_SIMULATIONS=0 \
  MC_BACKENDS=openmc-cpu,yamc-cpu,yamc-gpu ./run_models.sh

# one specific isotope/element through every model on all three codes
YAMC_VERIFY_NUCLIDES=Fe56 YAMC_VERIFY_ELEMENTS=Fe \
  MC_BACKENDS=openmc-cpu,yamc-cpu,yamc-gpu ./run_models.sh

# a single model
MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh tasks/transmutation/material-reaction-rate/endf-b8.1.ipynb
```

Run `git restore tasks` afterwards if you only want the regenerated cache and not
the executed notebook outputs.

You can also run the orchestrator directly (this is what `run_models.sh` calls):

```bash
python -m yamc_verify.orchestrator                                  # all endf-b8.1 notebooks
python -m yamc_verify.orchestrator tasks/neutron/sphere/endf-b8.1.ipynb
```

## Shared code (`yamc_verify/`)

The per-notebook logic that used to be copy-pasted across every model notebook
now lives in the `yamc_verify/` package, so the notebooks are thin (a task
lookup plus a few calls):

- `tasks.py` — the task registry (one `Task` per notebook: library, geometry,
  particle statistics, which codes run, whether YAMC is cached). Single source
  of truth shared by the notebooks and the orchestrator.
- `families/` — per-physics-family simulation runners, plotting, metrics, and
  the run/display glue (`neutron`, `photon`, `secondary_photon`, `decay_photon`,
  `transmutation`). YAMC runs with `verbose=[]` and OpenMC with `verbosity=1` +
  `output=False`, so a notebook is silent whether run here or directly in Jupyter.
- `geometry.py`, `discovery.py`, `plotting.py` — shared geometry builders, item
  discovery, and generic spectrum-comparison plotting/metrics.
- `progress.py`, `plan.py`, `dashboard.py`, `orchestrator.py` — the live
  dashboard: a progress side-channel, the up-front plan, rendering, and the
  notebook runner.

A model notebook is now roughly:

```python
from yamc_verify import backends, get_task
from yamc_verify.families import neutron
TASK = get_task("neutron/sphere/endf-b8.1")
all_results, first_yamc_model = neutron.run_all(TASK, backends.resolve())
neutron.display_results(TASK, all_results)
```

## Auditing YAMC vs OpenMC agreement

To find which nuclides / simulation modes disagree with OpenMC, scan the cache:

```bash
python -m yamc_verify.audit                 # rank all mismatches
python -m yamc_verify.audit --worst 20       # only the 20 worst
python -m yamc_verify.audit --current-only    # ignore stale-build YAMC caches
```

It compares every `(task, nuclide, backend)` for which both an OpenMC and a
YAMC result are cached — reduced-χ² and flux-weighted relative difference of the
flux spectrum plus the worst reaction-rate difference (neutron), or the worst
nuclide-density difference across timesteps (transmutation) — and ranks the
disagreements. Only modes that cache **both** codes can be audited offline;
modes that cache OpenMC only are reported as "not auditable offline (run them to
compare)".

## Using the latest yamc

The notebooks import yamc as an editable install, but its compiled Rust core is
only rebuilt on demand. After pulling new yamc changes, rebuild the core so the
notebooks pick them up:

```bash
cd ../yamc && maturin develop --release
```

Run this from your `yamc` checkout in an environment that has `maturin`; it updates
the editable extension the notebooks import.

## Building the book

```bash
jupyter-book start   # serves at http://localhost:3000
```
