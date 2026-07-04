"""Run one verification task's simulations (drive + cache), without a notebook.

This replaces the per-task driver notebooks. The orchestrator launches
``python -m yamc_verify.run_task <task-key>`` as a subprocess (with
``YAMC_VERIFY_PROGRESS`` set), which resolves the task and backends and calls the
family's ``run_all`` directly: it discovers the items, runs each enabled backend,
caches the results on disk, and emits the same progress events the live dashboard
tails. Honors the same selection env vars as before (``MC_BACKENDS``,
``YAMC_VERIFY_NUCLIDES``, ``YAMC_VERIFY_ELEMENTS``,
``YAMC_VERIFY_NUMBER_OF_SIMULATIONS``).

Usage::

    python -m yamc_verify.run_task neutron/sphere/endf-b8.1
"""
from __future__ import annotations

import os
import sys

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
    argv = list(argv if argv is not None else sys.argv[1:])
    if len(argv) != 1:
        print("usage: python -m yamc_verify.run_task <task-key>", file=sys.stderr)
        return 2
    task = _tasks.get_task(argv[0])
    run_task(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
