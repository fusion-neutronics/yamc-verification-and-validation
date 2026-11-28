"""What each verification task would run, and which results already have a record.

For a task the planner discovers the item list (the same discovery the family
run loops use, so no drift) and, for each code label, counts the items that
already have a record the run would accept. Wall times stored in the records
(from any build) seed the time estimate. Both runners use it:
:mod:`yamc_verify.project` (``vv plan`` / ``vv run``) and the in-place
:mod:`yamc_verify.orchestrator`. The plan is an *estimate*: the live
``cached``/``computed`` events emitted while a task runs are authoritative.
"""
from __future__ import annotations

import glob
import os
import time
from typing import Callable, Sequence

from vv import Row
from vv.records import find_record, latest_record

from . import discovery
from .backends import Backends
from .tasks import OPENMC_BUILD_KEYED_FAMILIES, Task

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def yamc_build_age() -> float | None:
    """Seconds since the installed compiled yamc core (`_core*.so`) was built, or None."""
    try:
        import yamc

        pkg = os.path.dirname(yamc.__file__)
        sos = glob.glob(os.path.join(pkg, "**", "*.so"), recursive=True)
        if not sos:
            return None
        newest = max(os.path.getmtime(s) for s in sos)
        return max(0.0, time.time() - newest)
    except Exception:
        return None


def code_enabled(code: str, backends: Backends) -> bool:
    """Whether a task's declared ``code`` runs under the resolved backends."""
    if code == "openmc":
        return backends.run_openmc
    if code == "yamc-cpu" or code == "yamc":
        return "cpu" in backends.computes
    if code == "yamc-gpu":
        return "gpu" in backends.computes
    return False


def discover_items(task: Task, discover_with_openmc: bool) -> list[str]:
    """Return the item-name list for ``task`` (empty if discovery is unavailable).

    ``discover_with_openmc`` takes the items from the OpenMC library, as the run
    does whenever OpenMC takes part, rather than from yamc's data cache. The
    fixed-geometry neutron models (tokamak, multimaterial slab) have their cells
    or layers as items.
    """
    fam = task.family
    if fam == "oktavian":
        from .families import oktavian

        return oktavian.discover(task, discover_with_openmc)
    ndir = task.nuclear_data_dir
    if fam == "neutron" and task.geometry == "tokamak":
        from .families import tokamak

        return tokamak.discover(task, discover_with_openmc)
    if fam == "neutron" and task.geometry == "multimaterial-slab":
        from .families import multimaterial_slab

        return multimaterial_slab.discover(task, discover_with_openmc)
    if fam == "neutron":
        return discovery.discover_neutron_nuclides(ndir, task.library, discover_with_openmc)
    if fam == "photon":
        return discovery.discover_photon_elements(ndir, task.library, discover_with_openmc)
    if fam == "secondary-photon":
        return [t[0] for t in discovery.discover_coupled_nuclides(ndir, task.library, discover_with_openmc)]
    if fam == "decay-photon":
        chain = discovery.resolve_openmc_chain(task.library)
        tdata = discovery.transmutation_data_path(task.library)
        return [t[0] for t in discovery.discover_d1s_nuclides(ndir, task.library, chain, tdata)]
    if fam in ("transmutation-reaction-rate", "transmutation-flux-spectrum"):
        chain = discovery.resolve_openmc_chain(task.library)
        tdata = discovery.transmutation_data_path(task.library)
        items = discovery.discover_stable_nuclides(ndir, task.library, chain, tdata, discover_with_openmc)
        return [i if isinstance(i, str) else i[0] for i in items]
    return []


def real_codes(base_codes: list[str], task: Task) -> list[str]:
    """Real cache/progress code labels for ``task``, in display order.

    Secondary-photon runs each backend once per electron treatment and stores
    and reports the result under a treatment-suffixed code (``openmc-led``,
    ``yamc-cpu-ttb``, ...), exactly as :func:`secondary_photon.run_all` builds
    them. Every other family uses the backend code unchanged. Keeping this in
    step with the family is what lets the live progress events land on the
    matching display row instead of being dropped as unknown codes.
    """
    treatments = task.extras.get("electron_treatments")
    if not treatments:
        return list(base_codes)
    return [f"{base}-{t}" for base in base_codes for t in treatments]


def _build_keyed(code: str, task: Task, backends: Backends) -> bool:
    """Whether ``code``'s records must come from the installed build.

    yamc always; OpenMC for the photon-bearing families while OpenMC runs, the
    same rule their run loops pass to the cache.
    """
    if code.startswith("yamc"):
        return True
    return task.family in OPENMC_BUILD_KEYED_FAMILIES and backends.run_openmc


def task_rows(task: Task, labels: Sequence[str], items: Sequence[str],
              find: Callable[[str], dict]) -> list[Row]:
    """One :class:`vv.Row` per code label: the items, how many have a record, earlier times.

    ``find(label)`` gives the keyword arguments of :func:`vv.records.find_record`
    that say which record a run of that label accepts (a ``build_id``, an
    ``accept`` predicate, or nothing for any build). The time per item is the
    average wall time of the newest record of each item from any build, so a new
    build still starts with an estimate from the last one.
    """
    cache_dir = os.path.join(REPO_ROOT, task.cache_dir)
    settings = {"library": task.library}
    rows = []
    for label in labels:
        how = find(label)
        done, times = 0, []
        for key in items:
            if find_record(cache_dir, task.library, label, key, settings=settings, **how) is not None:
                done += 1
            newest = latest_record(cache_dir, task.library, label, key)
            elapsed = newest[1].get("run", {}).get("elapsed_seconds") if newest else None
            if isinstance(elapsed, (int, float)):
                times.append(float(elapsed))
        rows.append(Row(label=label, items=list(items), done=done,
                        seconds_per_item=(sum(times) / len(times)) if times else None))
    return rows
