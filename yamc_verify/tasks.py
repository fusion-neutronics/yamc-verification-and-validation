"""Registry of verification tasks (one per model notebook).

This is the single source of truth for per-notebook configuration: which
physics family and geometry, which nuclear-data library, particle statistics,
which codes it can run, and whether it caches YAMC results. The thin notebooks
import their :class:`Task` from here so the notebook and the orchestrator's
planner never disagree about what will run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .geometry import GeometryDims

# User-specific nuclear-data HDF5 roots (mirrors the notebooks' NUCLEAR_DATA_DIR).
NUCLEAR_DATA_DIR = {
    "endf-b8.1": "/home/jon/nuclear_data/endf-b8.1-hdf5/endfb-viii.1-hdf5/",
    "endf-b8.0": "/home/jon/nuclear_data/endf-b8.0-hdf5/",
    "tendl-2021": "/home/jon/nuclear_data/tendl-2021-hdf5/tendl-2021-hdf5/",
    "fendl-3.2c": "/home/jon/nuclear_data/fendl-3.2c-hdf5/",
}


@dataclass(frozen=True)
class Task:
    """Everything needed to plan and run one verification notebook."""

    key: str                      # e.g. "neutron/sphere/endf-b8.1"
    notebook: str                 # repo-relative .ipynb path
    family: str                   # neutron | photon | secondary-photon | decay-photon | transmutation
    geometry: str                 # sphere | broomstick | material
    library: str                  # endf-b8.1 | fendl-3.2c
    label: str                    # dashboard label, e.g. "neutron sphere"
    item_kind: str                # nuclide | element
    codes: tuple[str, ...]        # superset of code labels this task can run
    caches_yamc: bool             # whether YAMC results go through the on-disk cache
    particles: int | None = None
    batches: int | None = None
    source_energy: float | None = None
    dims: GeometryDims = field(default_factory=GeometryDims)
    extras: dict = field(default_factory=dict)

    @property
    def cache_dir(self) -> str:
        """Repo-relative results cache directory for this notebook."""
        return os.path.join(os.path.dirname(self.notebook), "cache")

    @property
    def nuclear_data_dir(self) -> str:
        return NUCLEAR_DATA_DIR[self.library]


def _mc_codes():
    return ("openmc", "yamc-cpu", "yamc-gpu")


_SPHERE = GeometryDims(sphere_radius=35.0)
_BROOM = GeometryDims(cylinder_radius=1.0, cylinder_half_height=50.0)

_TASKS: list[Task] = []


def _reg(**kw):
    _TASKS.append(Task(**kw))


# ---- neutron (nuclide; caches YAMC) ---------------------------------------
for _lib, _parts in (("endf-b8.1", 5_000_000), ("fendl-3.2c", 500_000)):
    _reg(key=f"neutron/sphere/{_lib}", notebook=f"tasks/neutron/sphere/{_lib}.ipynb",
         family="neutron", geometry="sphere", library=_lib, label="neutron sphere",
         item_kind="nuclide", codes=_mc_codes(), caches_yamc=True,
         particles=_parts, batches=10, source_energy=14.06e6, dims=_SPHERE)
    _reg(key=f"neutron/broomstick/{_lib}", notebook=f"tasks/neutron/broomstick/{_lib}.ipynb",
         family="neutron", geometry="broomstick", library=_lib, label="neutron broomstick",
         item_kind="nuclide", codes=_mc_codes(), caches_yamc=True,
         particles=_parts, batches=10, source_energy=14.06e6, dims=_BROOM)

# ---- neutron tokamak (fixed multi-material geometry; per-cell flux spectra) --
_reg(key="neutron/tokamak/endf-b8.1", notebook="tasks/neutron/tokamak/endf-b8.1.ipynb",
     family="neutron", geometry="tokamak", library="endf-b8.1", label="neutron tokamak",
     item_kind="cell", codes=("openmc", "yamc-cpu"), caches_yamc=True,
     particles=2_000_000, batches=5, source_energy=14.06e6, dims=_SPHERE)

# ---- neutron multimaterial slab (fixed layered geometry; per-layer flux spectra) --
# Runs on all three backends: the GPU multi-material XS-extraction bug (yamc#208)
# is fixed, and the geometry is fully bounded (vacuum implicit complement).
_reg(key="neutron/multimaterial-slab/endf-b8.1",
     notebook="tasks/neutron/multimaterial-slab/endf-b8.1.ipynb",
     family="neutron", geometry="multimaterial-slab", library="endf-b8.1",
     label="neutron multimaterial slab", item_kind="cell",
     codes=("openmc", "yamc-cpu", "yamc-gpu"), caches_yamc=True,
     particles=2_000_000, batches=5, source_energy=14.06e6, dims=_SPHERE)

# ---- photon (element; caches YAMC) ----------------------------------------
for _lib, _parts in (("endf-b8.1", 5_000_000), ("fendl-3.2c", 500_000)):
    _reg(key=f"photon/sphere/{_lib}", notebook=f"tasks/photon/sphere/{_lib}.ipynb",
         family="photon", geometry="sphere", library=_lib, label="photon sphere",
         item_kind="element", codes=_mc_codes(), caches_yamc=True,
         particles=_parts, batches=10, source_energy=1.0e6, dims=_SPHERE,
         extras={"photon_group_structure": "CCFE-24-PHOTON"})
    _reg(key=f"photon/broomstick/{_lib}", notebook=f"tasks/photon/broomstick/{_lib}.ipynb",
         family="photon", geometry="broomstick", library=_lib, label="photon broomstick",
         item_kind="element", codes=_mc_codes(), caches_yamc=True,
         particles=_parts, batches=10, source_energy=1.0e6, dims=_BROOM,
         extras={"photon_group_structure": "CCFE-24-PHOTON"})

# ---- secondary-photon (coupled n+photon; nuclide; caches YAMC) ------------
for _geom, _dims, _nb in (("sphere", _SPHERE, "sphere"), ("broomstick", _BROOM, "broomstick")):
    _reg(key=f"secondary-photon/{_geom}/endf-b8.1",
         notebook=f"tasks/secondary-photon/{_nb}/endf-b8.1.ipynb",
         family="secondary-photon", geometry=_geom, library="endf-b8.1",
         label=f"secondary-photon {_geom}", item_kind="nuclide", codes=_mc_codes(),
         caches_yamc=True, particles=5_000_000, batches=10, source_energy=14.06e6, dims=_dims,
         extras={"neutron_group_structure": "VITAMIN-J-175",
                 "photon_group_structure": "VITAMIN-J-42",
                 # Matched electron treatments: yamc and OpenMC are each run under
                 # both LED and TTB so they compare like-for-like (a mismatch
                 # makes the low-E photon spectrum diverge spuriously; yamc #179).
                 "electron_treatments": ["led", "ttb"]})

# ---- decay-photon (D1S; nuclide; caches YAMC) -----------------------------
for _geom, _dims, _nb in (("sphere", _SPHERE, "sphere"), ("broomstick", _BROOM, "broomstick")):
    _reg(key=f"decay-photon/{_geom}/endf-b8.1",
         notebook=f"tasks/decay-photon/{_nb}/endf-b8.1.ipynb",
         family="decay-photon", geometry=_geom, library="endf-b8.1",
         label=f"decay-photon {_geom}", item_kind="nuclide", codes=_mc_codes(),
         caches_yamc=True, particles=5_000_000, batches=10, source_energy=14.06e6, dims=_dims,
         extras={"photon_group_structure": "VITAMIN-J-42",
                 "timesteps": [3600.0, 3600.0, 3600.0, 15 * 86400.0, 15 * 86400.0],
                 "source_rates": [1.0, 1.0, 1.0, 0.0, 0.0],
                 "phases": ["irrad", "irrad", "irrad", "cool", "cool"]})

# ---- transmutation (two distinct notebooks) -------------------------------
_reg(key="transmutation/material-reaction-rate/endf-b8.1",
     notebook="tasks/transmutation/material-reaction-rate/endf-b8.1.ipynb",
     family="transmutation-reaction-rate", geometry="material", library="endf-b8.1",
     label="transmutation material-reaction-rate", item_kind="nuclide",
     codes=("openmc", "yamc-cpu"), caches_yamc=True,
     particles=1_000_000, source_energy=14.06e6, dims=GeometryDims(sphere_radius=5.0),
     extras={"timesteps": [3600.0, 3600.0, 3600.0, 15 * 86400.0, 15 * 86400.0],
             "source_rates": [1e18, 1e18, 1e18, 0.0, 0.0],
             "phases": ["irrad", "irrad", "irrad", "cool", "cool"]})
_reg(key="transmutation/material-flux-spectrum/endf-b8.1",
     notebook="tasks/transmutation/material-flux-spectrum/endf-b8.1.ipynb",
     family="transmutation-flux-spectrum", geometry="material", library="endf-b8.1",
     label="transmutation material-flux-spectrum", item_kind="nuclide",
     codes=("openmc", "yamc"), caches_yamc=True,
     extras={"energy_groups": [1e-5, 0.625, 1e5, 2e7],
             "multigroup_flux": [1e12, 5e12, 1e14],
             "timesteps": [3600.0, 3600.0, 3600.0, 15 * 86400.0, 15 * 86400.0],
             "yamc_source_rates": [1.0, 1.0, 1.0, 0.0, 0.0],
             "openmc_source_rates": [1.06e14, 1.06e14, 1.06e14, 0.0, 0.0],
             "phases": ["irrad", "irrad", "irrad", "cool", "cool"]})


_BY_KEY = {t.key: t for t in _TASKS}


def all_tasks() -> list[Task]:
    """All registered tasks, in registry (display) order."""
    return list(_TASKS)


def get_task(key: str) -> Task:
    """Look up a task by its key (e.g. ``"neutron/sphere/endf-b8.1"``)."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown task {key!r}; known: {sorted(_BY_KEY)}") from None


def task_for_notebook(notebook: str) -> Task | None:
    """Return the task whose notebook path matches ``notebook`` (repo-relative)."""
    norm = notebook.replace("\\", "/")
    for t in _TASKS:
        if norm.endswith(t.notebook):
            return t
    return None
