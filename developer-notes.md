# Developer Notes

This page documents how to build and maintain the YAMC Verification Jupyter Book.

## Prerequisites

- Python 3.10+
- OpenMC and YAMC installed
- Nuclear data libraries (ENDF/B-VIII.0, FENDL-3.2c, etc.)

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Building the Book

To build the book:

```bash
jupyter-book build .
```

The site output will be in `_build/site/`.

To start a local dev server:

```bash
jupyter-book start
```

### Build Options

Force rebuild of all pages:

```bash
jupyter-book build . --all
```

Clean build artifacts:

```bash
jupyter-book clean .
```

## Project Structure

```
yamc-verification/
├── myst.yml             # Jupyter Book / MyST configuration & TOC
├── intro.md             # Landing page
├── developer-notes.md   # This file
├── requirements.txt     # Python dependencies
├── run_models.sh        # -> python -m yamc_verify.orchestrator (live dashboard)
├── cache_utils.py       # JSON results cache + progress-emitting cached_or_run/select_requested
├── yamc_verify/         # Shared package (thin notebooks import from here)
│   ├── tasks.py         # Task registry: one Task per notebook (config source of truth)
│   ├── backends.py      # MC_BACKENDS parsing
│   ├── geometry.py      # sphere/broomstick region builders (yamc + openmc)
│   ├── discovery.py     # nuclide/element discovery per family
│   ├── plotting.py      # generic spectrum-comparison plotting + metrics
│   ├── progress.py      # progress side-channel (YAMC_VERIFY_PROGRESS)
│   ├── plan.py          # up-front cached/to-do plan from the cache
│   ├── dashboard.py     # live in-place terminal dashboard
│   ├── orchestrator.py  # runs notebooks silently + tails progress
│   └── families/        # per-physics runners/plots/run+display glue
│       ├── neutron.py
│       ├── photon.py
│       ├── secondary_photon.py
│       ├── decay_photon.py
│       └── transmutation.py
└── tasks/
    └── <physics>/<geometry>/
        ├── endf-b8.1.ipynb   # thin: task lookup + run_all + display_results
        ├── fendl-3.2c.ipynb
        └── cache/            # on-disk JSON results cache
```

## Adding a New Nuclear Data Library

1. Add the library path to `NUCLEAR_DATA_DIR` in `yamc_verify/tasks.py`
2. Register a `Task` for each model/geometry using the new library (copy an
   existing entry and change `library`/`particles`)
3. Copy an existing thin notebook and point its `get_task("…")` at the new key
4. Update the markdown title and add the notebook to `toc` in `myst.yml`

## Adding a New Model

1. Add a new family module under `yamc_verify/families/` (or reuse one) and
   register `Task`s for it in `yamc_verify/tasks.py`
2. Create thin notebooks under `tasks/` that call the family's `run_all` /
   `display_results`
3. Add a new section to the `toc` in `myst.yml`:

```yaml
toc:
  - file: intro.md
  - title: Sphere Model
    children:
      - file: tasks/sphere/endf-b8.0.ipynb
      - file: tasks/sphere/fendl-3.2c.ipynb
  - title: Cylinder Model
    children:
      - file: tasks/cylinder/endf-b8.0.ipynb
      - file: tasks/cylinder/fendl-3.2c.ipynb
```

## Configuration

### Simulation Parameters

Per-notebook run parameters now live in the task registry
(`yamc_verify/tasks.py`), one `Task` per notebook, rather than in each
notebook's setup cell:

| Field | Description |
|-------|-------------|
| `particles` | Particles per batch |
| `batches` | Number of batches |
| `source_energy` | Source energy (eV) |
| `dims` | Geometry dimensions (sphere radius / cylinder radius+half-height) |
| `codes` | Which codes this task can run |
| `caches_yamc` | Whether YAMC results are cached on disk |
| `extras` | Family-specific config (group structures, reduce levels, timesteps, …) |

`particles`/`batches` are intentionally not exposed via env vars (lowering them
produces statistical noise that masquerades as YAMC/OpenMC disagreement).

### Notebook Execution

Notebooks are not executed during the build. Run them in-place before building.
The recommended path is the orchestrator, which runs them quietly with a live
progress dashboard (see the README):

```bash
./run_models.sh                 # all endf-b8.1 notebooks
./run_models.sh tasks/neutron/sphere/endf-b8.1.ipynb
```

You can still drive `nbconvert` directly for a single notebook if needed:

```bash
jupyter nbconvert --to notebook --execute --inplace tasks/*/*/*.ipynb
```

To clear all notebook outputs (e.g. before committing):

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --inplace tasks/*/*/*.ipynb
```

#### Git hooks

A pre-commit hook that clears notebook outputs automatically (keeping diffs
source-only) is tracked in `.githooks/`. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

It strips outputs from `tasks/*/*/*.ipynb` and re-stages them. If `jupyter` is
not on `PATH` it falls back to `python -m nbconvert`, and if neither is
available it prints a warning rather than blocking the commit.

#### Test vs full mode

Each notebook iterates over many nuclides or elements. For a quick smoke test
without re-running everything, set `YAMC_VERIFY_NUMBER_OF_SIMULATIONS` to clip
the iteration list to the first N items:

```bash
# Smoke test: first 5 nuclides only
YAMC_VERIFY_NUMBER_OF_SIMULATIONS=5 jupyter nbconvert --to notebook --execute --inplace tasks/neutron/sphere/endf-b8.1.ipynb

# All notebooks, first 5 each
YAMC_VERIFY_NUMBER_OF_SIMULATIONS=5 jupyter nbconvert --to notebook --execute --inplace tasks/*/*/*.ipynb

# Full run (unset or 0)
jupyter nbconvert --to notebook --execute --inplace tasks/neutron/sphere/endf-b8.1.ipynb
```

`PARTICLES` and `BATCHES` are intentionally not exposed via env vars. Lowering
them produces statistical noise that masquerades as YAMC/OpenMC disagreement,
which would defeat the purpose of verification.

#### Selecting which codes/backends to run

By default each notebook runs OpenMC plus YAMC on both CPU and GPU. To pick a
subset, set the `MC_BACKENDS` env var to a comma-separated list of tokens.
Valid tokens: `openmc-cpu`, `yamc-cpu`, `yamc-gpu`. Any one, two, or all three
may be selected.

```bash
# Compare yamc cpu and yamc gpu only (skip OpenMC)
MC_BACKENDS=yamc-cpu,yamc-gpu jupyter nbconvert --to notebook --execute --inplace tasks/neutron/sphere/endf-b8.1.ipynb

# OpenMC vs yamc cpu only (no GPU; useful on CI runners without an f64 Vulkan adapter)
MC_BACKENDS=openmc-cpu,yamc-cpu jupyter nbconvert --to notebook --execute --inplace tasks/neutron/sphere/endf-b8.1.ipynb

# Default: all three
jupyter nbconvert --to notebook --execute --inplace tasks/neutron/sphere/endf-b8.1.ipynb
```

When `openmc-cpu` is excluded, the ratio panel and rel-diff columns use the
first YAMC backend in `MC_BACKENDS` as the reference instead.

### Results cache

Simulation results are cached as JSON under
`tasks/<model>/<...>/cache/<library>/<code>/<nuclide>.json` (via `cache_utils.py`),
so re-running a notebook to iterate on plots does not re-run the simulations.
Two invalidation policies:

- **OpenMC** results are keyed by the run parameters only (library, particles,
  batches, source energy). OpenMC is fixed, so these stay valid until a
  parameter changes.
- **YAMC** results are additionally keyed by the **yamc version and git
  commit** (`cache_utils.yamc_fingerprint()`). When you rebuild or update yamc
  (a new release, or a new commit for an editable/source install) the cache
  auto-invalidates and the affected simulations re-run on the next notebook
  execution. The version string does not bump per development build, so the git
  commit is what catches day-to-day changes; on a wheel install with no git
  checkout the commit is `null` and invalidation falls back to the version.

To force everything to re-run, delete the cache directories:

```bash
rm -rf tasks/*/*/cache
```

## Troubleshooting

### Notebook execution fails

Check the error in the build output. Common issues:
- Missing nuclear data files, perhaps correct the data path used
- OpenMC/YAMC not installed in the virtual environment
- Timeout exceeded (increase in `myst.yml`)


### Cache issues

Clear the build cache if pages aren't rebuilding properly:

```bash
jupyter-book clean .
jupyter-book build .
```
