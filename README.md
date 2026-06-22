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
results cache. Two environment variables control it:

- `MC_BACKENDS` - comma-separated list of `openmc-cpu`, `yamc-cpu`, `yamc-gpu` (default: `openmc-cpu`).
- `YAMC_VERIFY_NUMBER_OF_SIMULATIONS` - cap on nuclides/elements per model (default: `3`).

Examples:

```bash
# OpenMC only, 3 nuclides/elements per model (builds the reference cache)
./run_models.sh

# yamc + OpenMC, 3 each (this is what fills the comparison plots)
MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh

# yamc + OpenMC, 10 nuclides/elements per model
YAMC_VERIFY_NUMBER_OF_SIMULATIONS=10 MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh

# a single model
MC_BACKENDS=openmc-cpu,yamc-cpu ./run_models.sh tasks/transmutation/material-reaction-rate/endf-b8.1.ipynb
```

Run `git restore tasks` afterwards if you only want the regenerated cache and not
the executed notebook outputs.

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
