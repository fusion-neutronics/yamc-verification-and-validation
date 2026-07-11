# yamc-verification

Verification of YAMC by comparing its simulation results against the gold-standard
Monte Carlo code, OpenMC.

## Setup

```bash
pip install -r requirements.txt
```

YAMC is used as an editable install from a sibling `yamc` checkout (`../yamc`).

## Running the verification models

`run_models.sh` runs the verification tasks and populates the on-disk results
cache. It drives them through the `yamc_verify` orchestrator
(`python -m yamc_verify.orchestrator`), which runs each task silently
(`python -m yamc_verify.run_task` as a subprocess, output captured) and renders
a single **live, in-place dashboard** instead of a large scrolling log:

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
# nuclides (and elements with a stable isotope); no task argument means all
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

# a single model (task key)
MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh transmutation/material-reaction-rate/endf-b8.1
```

The runs only write to the on-disk cache (the database the benchmark page reads);
there are no notebooks to clean up afterwards.

You can also run the orchestrator directly (this is what `run_models.sh` calls),
or a single task without the dashboard:

```bash
python -m yamc_verify.orchestrator                          # all endf-b8.1 tasks
python -m yamc_verify.orchestrator neutron/sphere/endf-b8.1
python -m yamc_verify.run_task neutron/sphere/endf-b8.1     # one task, no dashboard
```

## Shared code (`yamc_verify/`)

All the model logic lives in the `yamc_verify/` package. Runs are driven directly
from it (there are no notebooks):

- `tasks.py` — the task registry (one `Task` per model/geometry/library: library,
  geometry, particle statistics, which codes run, whether YAMC is cached). Single
  source of truth shared by `run_task` and the orchestrator.
- `families/` — per-physics-family simulation runners, plotting, and metrics
  (`neutron`, `photon`, `secondary_photon`, `decay_photon`, `transmutation`),
  each exposing `run_all(task, backends, cache_dir=...)`. YAMC runs with
  `verbose=[]` and OpenMC with `verbosity=1` + `output=False`, so runs are silent.
- `run_task.py` — drives one task: dispatches `task.family` to its family module's
  `run_all` and caches the results. This is what the orchestrator runs per task.
- `geometry.py`, `discovery.py`, `plotting.py` — shared geometry builders, item
  discovery, and generic spectrum-comparison plotting/metrics.
- `progress.py`, `plan.py`, `dashboard.py`, `orchestrator.py` — the live
  dashboard: a progress side-channel, the up-front plan, rendering, and the runner.
- `index_builder.py`, `reduced_chains.py` — build the `benchmarks-index.json` the
  web page reads, and precompute OpenMC reduced-chain membership for it.

Running one task in code is roughly:

```python
from yamc_verify import backends, get_task
from yamc_verify.run_task import run_task
run_task(get_task("neutron/sphere/endf-b8.1"), backends.resolve())
```

## Adding a nuclear data library

1. Add the library path to `NUCLEAR_DATA_DIR` in `yamc_verify/tasks.py`.
2. Register a `Task` for each model/geometry using the new library (copy an
   existing entry and change `library`/`particles`).
3. Run it (`./run_models.sh <task-key>`); the index builder and the benchmark
   database page pick up the new library automatically (library is a facet).

## Adding a model

1. Add a new family module under `yamc_verify/families/` (or reuse one), exposing
   a `run_all(task, backends, cache_dir=...)` like the others.
2. Register `Task`s for it in `yamc_verify/tasks.py`, and map the new
   `task.family` to the module in `yamc_verify/run_task.py`.
3. Run it with `./run_models.sh <task-key>`; results land in the task's cache and
   appear in the benchmark database page.

## Results cache

Simulation results are cached as JSON under
`tasks/<model>/<...>/cache/<library>/<code>/<nuclide>.json` (via `cache_utils.py`),
so re-running a task does not repeat simulations that are already cached. Two
invalidation policies:

- **OpenMC** results are keyed by the run parameters only (library, particles,
  batches, source energy). OpenMC is fixed, so these stay valid until a parameter
  changes.
- **YAMC** results are additionally keyed by the **yamc version and git commit**
  (`cache_utils.yamc_fingerprint()`). Rebuilding or updating yamc auto-invalidates
  the cache and the affected simulations re-run on the next run. On a wheel
  install with no git checkout the commit is `null` and invalidation falls back to
  the version.

To force everything to re-run, delete the cache directories:

```bash
rm -rf tasks/*/*/cache
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

YAMC is used as an editable install, but its compiled Rust core is only rebuilt
on demand. After pulling new yamc changes, rebuild the core so the runs pick them
up:

```bash
cd ../yamc && maturin develop --release
```

Run this from your `yamc` checkout in an environment that has `maturin`; it updates
the editable extension the runs import.

## The benchmark database page (the site)

The deployed site is a single standalone page, `web/benchmarks.html`, served as
the root `index.html`. It reads a generated `benchmarks-index.json` and lets you
filter every cached benchmark by simulation type / geometry / library / element /
nuclide, with headline agreement stats. Clicking a row expands a detail view with
the interactive model geometry, the flux-spectrum comparison plot, and the
reaction-rate table, pulled live from the on-disk cache.

The page is a pure viewer: it holds no results data, inlines its own CSS, and
uses only relative paths. It never needs editing when results change; only its
data does. When the cache changes, regenerate the index (and, on the published
site, the copied cache) - the page itself is untouched:

```bash
python -m yamc_verify.index_builder --out benchmarks-index.json
```

CI (`.github/workflows/pages.yml`) is correspondingly minimal: it installs
`numpy pandas plotly`, generates the index, copies `benchmarks.html` (as the root
`index.html`), `logo.png`, and `web/models/`, drops in the results cache, and
publishes to GitHub Pages. Every push to `main` regenerates the index, so the
published site always reflects the current cache. There is no MyST, Node, or
`jupyter-book` involved.

### Serving it locally

`fetch` does not work from `file://`, so serve the page over HTTP with
`benchmarks-index.json`, `models/`, and the `tasks/` cache reachable at the paths
it requests. Build the index into `web/`, symlink the cache in beside it, and
serve that directory (the generated files are git-ignored):

```bash
python -m yamc_verify.index_builder --out web/benchmarks-index.json
ln -sfn ../tasks web/tasks
cp web/benchmarks.html web/index.html    # so "/" serves the page, mirroring deploy
python -m http.server 8321 --directory web
# open http://localhost:8321/
```

Rebuild the index whenever the cache changes to pick up new results.

## Troubleshooting

### A task run fails

Check the per-task log the orchestrator prints on failure. Common issues:

- Missing nuclear data files (correct the data path used).
- OpenMC or YAMC not installed in the virtual environment.
