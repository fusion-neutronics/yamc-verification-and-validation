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
└── tasks/
    └── sphere/          # Sphere model
        ├── endf-b8.0.ipynb
        └── fendl-3.2c.ipynb
```

## Adding a New Nuclear Data Library

1. Copy an existing notebook (e.g., `endf-b8.0.ipynb`) to a new file
2. Update `NUCLEAR_DATA` variable to the new library name
3. Add the library path to `NUCLEAR_DATA_DIR` dictionary
4. Update the markdown title and description
5. Add the new notebook to the `toc` section in `myst.yml`

## Adding a New Model

1. Create a new directory under `tasks/` (e.g., `tasks/cylinder/`)
2. Create notebooks for each nuclear data library
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

Each notebook has configurable parameters in the setup cell:

| Parameter | Description |
|-----------|-------------|
| `PARTICLES` | Particles per batch |
| `BATCHES` | Number of batches |
| `SOURCE_ENERGY` | Neutron source energy (eV) |
| `FLUX_DECADES` | Dynamic range for flux plots |

### Notebook Execution

Notebooks are not executed during the build. Run them in-place before building:

```bash
jupyter nbconvert --to notebook --execute --inplace tasks/*/*.ipynb
```

To clear all notebook outputs (e.g. before committing):

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --inplace tasks/*/*.ipynb
```

#### Test vs full mode

Each notebook iterates over many nuclides or elements. For a quick smoke test
without re-running everything, set `YAMC_VERIFY_NUMBER_OF_SIMULATIONS` to clip
the iteration list to the first N items:

```bash
# Smoke test: first 5 nuclides only
YAMC_VERIFY_NUMBER_OF_SIMULATIONS=5 jupyter nbconvert --to notebook --execute --inplace tasks/sphere/endf-b8.1.ipynb

# All notebooks, first 5 each
YAMC_VERIFY_NUMBER_OF_SIMULATIONS=5 jupyter nbconvert --to notebook --execute --inplace tasks/*/*.ipynb

# Full run (unset or 0)
jupyter nbconvert --to notebook --execute --inplace tasks/sphere/endf-b8.1.ipynb
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
