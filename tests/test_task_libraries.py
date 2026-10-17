"""Every task names a library both codes can read.

yamc reads a task's library by keyword, so the keyword must be one the yamc
data origin publishes (each has a ``<keyword>/latest.json`` on
yamc-data.xsplot.com). OpenMC reads the directory ``NUCLEAR_DATA_DIR`` names
for it. The FENDL tasks once named ``fendl-3.2c``, which yamc does not
publish, so their yamc side could not run at all.
"""
import sys
import types

import pytest

from yamc_verify import tasks

# The library keywords the yamc data origin publishes. A task on a new library
# adds it here once the library is published.
YAMC_LIBRARIES = {
    "endf-b8.1", "fendl-3.2d", "irdff-ii", "jeff-4.0", "jendl-5.0", "tendl-2017", "tendl-2025",
}


def test_every_task_names_a_library_yamc_publishes():
    unpublished = sorted({t.library for t in tasks.all_tasks()} - YAMC_LIBRARIES)
    assert not unpublished, f"yamc publishes no library called {unpublished}"


def test_every_task_library_has_an_openmc_data_directory():
    missing = sorted({t.library for t in tasks.all_tasks()} - set(tasks.NUCLEAR_DATA_DIR))
    assert not missing, f"no OpenMC data directory for {missing}"


def test_the_fendl_tasks_run_fendl_3_2d():
    keys = {t.key for t in tasks.all_tasks() if t.library.startswith("fendl")}
    assert keys == {f"{family}/{geometry}/fendl-3.2d"
                    for family in ("neutron", "photon") for geometry in ("sphere", "broomstick")}


class _Part:
    """Any geometry or source object the runner builds and only passes on."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    below = above = property(lambda self: self)

    def __and__(self, other):
        return self


@pytest.mark.parametrize("key", [t.key for t in tasks.all_tasks() if t.family == "photon"])
def test_yamc_photon_runs_read_photon_data_from_the_task_library(key, monkeypatch):
    """yamc takes photon data from endf-b8.1 whatever the neutron library unless
    told otherwise; OpenMC reads the task library's own photon files. A FENDL
    photon task must give yamc FENDL's photon data too."""
    from yamc_verify.families import photon

    loads = []

    class Material(_Part):
        def read_nuclear_data(self, source, photon_data=None):
            loads.append((source, photon_data))

    class Tally(_Part):
        energy_bins = [1.0e3, 1.0e6]
        name = property(lambda self: self.kwargs["name"])

    class Results:
        particles_per_second = 1.0

        def __getitem__(self, tally):
            return types.SimpleNamespace(mean=[1.0], standard_deviation=[0.1])

    class Model(_Part):
        def simulate_transport(self, **kwargs):
            return Results()

    fake = types.SimpleNamespace(Material=Material, Tally=Tally, Model=Model,
                                 sources=types.SimpleNamespace(Isotropic=_Part, Discrete=_Part))
    for name in ("Sphere", "Cylinder", "Plane", "Cell", "Geometry", "PhotonSource"):
        setattr(fake, name, _Part)
    monkeypatch.setitem(sys.modules, "yamc", fake)

    task = tasks.get_task(key)
    photon.run_yamc_simulation("Ag", task)
    assert loads == [(task.library, {"Ag": task.library})]
