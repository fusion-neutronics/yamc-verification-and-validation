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

To build the book with notebook execution:

```bash
jupyter-book build .
```

The HTML output will be in `_build/html/`. Open `_build/html/index.html` in a browser to view.

### Build Options

Force re-execution of all notebooks:

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
├── _config.yml          # Jupyter Book configuration
├── _toc.yml             # Table of contents / navigation
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
5. Add the new notebook to `_toc.yml`

## Adding a New Model

1. Create a new directory under `tasks/` (e.g., `tasks/cylinder/`)
2. Create notebooks for each nuclear data library
3. Add a new `parts` section in `_toc.yml`:

```yaml
parts:
  - caption: Sphere Model
    chapters:
      - file: tasks/sphere/endf-b8.0
      - file: tasks/sphere/fendl-3.2c
  - caption: Cylinder Model
    chapters:
      - file: tasks/cylinder/endf-b8.0
      - file: tasks/cylinder/fendl-3.2c
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

Configured in `_config.yml`:

- `execute_notebooks: auto` - Only re-run if source changed
- `timeout: 600` - 10 minute timeout per notebook

## Troubleshooting

### Notebook execution fails

Check the error in the build output. Common issues:
- Missing nuclear data files
- OpenMC/YAMC not installed in the virtual environment
- Timeout exceeded (increase in `_config.yml`)

### Plots not displaying

Ensure `plotly` and `kaleido` are installed. The notebooks use HTML output for Plotly compatibility with static sites.

### Cache issues

Clear the Jupyter cache if notebooks aren't re-executing properly:

```bash
rm -rf _build/.jupyter_cache
jupyter-book build .
```
