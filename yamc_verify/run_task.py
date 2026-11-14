"""Run one verification task's simulations (drive + cache), without a notebook.

The runners (``vv run`` through :mod:`yamc_verify.project`, and the
in-place :mod:`yamc_verify.orchestrator`) launch
``python -m yamc_verify.run_task <task-key>`` as a subprocess (with
``VV_PROGRESS`` set), which resolves the task and backends and calls the
family's ``run_all`` directly: it discovers the items, runs each enabled
backend, caches the results on disk, and emits the progress events the live
display tails. Honors the selection env vars (``MC_BACKENDS``,
``YAMC_VERIFY_NUCLIDES``, ``YAMC_VERIFY_ELEMENTS``,
``YAMC_VERIFY_NUMBER_OF_SIMULATIONS``).

``--discover-with-openmc`` takes the items from the OpenMC library even when
OpenMC does not run in this process, so a yamc process that is part of a
verification with OpenMC runs the same items as the OpenMC one.

Usage::

    python -m yamc_verify.run_task neutron/sphere/endf-b8.1
    MC_BACKENDS=yamc-cpu python -m yamc_verify.run_task neutron/sphere/endf-b8.1 --discover-with-openmc
"""
from __future__ import annotations

import argparse
import dataclasses
import os

from . import backends as _backends
from . import tasks as _tasks
from .families import decay_photon, neutron, photon, secondary_photon, transmutation

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# task.family -> the family module whose run_all drives it
_FAMILY_MODULE = {
    "neutron": neutron,
    "photon": photon,
    "secondary-photon": secondary_photon,
    "decay-photon": decay_photon,
    "transmutation-reaction-rate": transmutation,
    "transmutation-flux-spectrum": transmutation,
}


def family_module(task):
    """Return the family module whose ``run_all`` drives ``task``."""
    try:
        return _FAMILY_MODULE[task.family]
    except KeyError:
        raise KeyError(f"no run_all module for task family {task.family!r}") from None


def run_task(task, backends=None):
    """Run every enabled backend for every discovered item of ``task``; cache results.

    Returns whatever the family ``run_all`` returns (typically
    ``(all_results, first_yamc_model)``). ``cache_dir`` is made absolute so the
    results land in the task's own cache dir regardless of the caller's cwd.
    """
    mod = family_module(task)
    b = backends if backends is not None else _backends.resolve()
    cache_dir = os.path.join(_REPO_ROOT, task.cache_dir)
    return mod.run_all(task, b, cache_dir=cache_dir)


def main(argv=None):
    """Run one task from the command line; return the exit status."""
    ap = argparse.ArgumentParser(prog="python -m yamc_verify.run_task", description=__doc__.split("\n\n")[0])
    ap.add_argument("task", help="task key, e.g. neutron/sphere/endf-b8.1")
    ap.add_argument("--discover-with-openmc", action="store_true",
                    help="take the items from the OpenMC library even if OpenMC does not run here")
    args = ap.parse_args(argv)
    task = _tasks.get_task(args.task)
    b = _backends.resolve()
    if args.discover_with_openmc:
        b = dataclasses.replace(b, discover_with_openmc=True)
    run_task(task, b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
