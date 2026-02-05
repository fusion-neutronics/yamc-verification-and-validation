# Developer Notes

This page documents how to build and maintain the YAMC Verification Jupyter Book.

## Prerequisites

- Python 3.9+
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
