"""Build the up-front execution plan for the orchestrator.

For every task the planner discovers the item list (same discovery the thin
notebooks use, so no drift) and classifies each ``(item, code)`` as already
cached or still to-do by inspecting the on-disk cache. Cached-item wall times
recorded in the cache metadata (including stale entries from a previous build)
seed the ETA estimate. The plan is an *estimate*: the live ``cached``/``computed``
events emitted while a notebook runs are authoritative and correct it.
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import yamc

from cache_utils import _cache_path, openmc_fingerprint, yamc_fingerprint

from . import discovery
from .backends import Backends
from .tasks import OPENMC_BUILD_KEYED_FAMILIES, Task


def yamc_build_age() -> float | None:
    """Seconds since the compiled yamc core (`_core*.so`) was built, or None."""
    try:
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


def discover_items(task: Task, run_openmc: bool) -> list[str]:
    """Return the item-name list for ``task`` (empty if discovery is unavailable)."""
    fam = task.family
    ndir = task.nuclear_data_dir
    if fam == "neutron":
        return discovery.discover_neutron_nuclides(ndir, task.library, run_openmc)
    if fam == "photon":
        return discovery.discover_photon_elements(ndir, task.library, run_openmc)
    if fam == "secondary-photon":
        return [t[0] for t in discovery.discover_coupled_nuclides(ndir, task.library, run_openmc)]
    if fam == "decay-photon":
        chain = discovery.resolve_openmc_chain(task.library)
        tdata = discovery.transmutation_data_path(task.library)
        return [t[0] for t in discovery.discover_d1s_nuclides(ndir, task.library, chain, tdata)]
    if fam in ("transmutation-reaction-rate", "transmutation-flux-spectrum"):
        chain = discovery.resolve_openmc_chain(task.library)
        tdata = discovery.transmutation_data_path(task.library)
        items = discovery.discover_stable_nuclides(ndir, task.library, chain, tdata, run_openmc)
        return [i if isinstance(i, str) else i[0] for i in items]
    return []


def _cache_info(cache_dir, library, code, key, meta):
    """Return (cached_valid: bool, elapsed_seconds: float | None) for one entry."""
    p = _cache_path(cache_dir, library, code, key)
    if not Path(p).exists():
        return False, None
    try:
        with open(p) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, None
    m = data.get("_meta", {})
    elapsed = m.get("elapsed_seconds")
    valid = all(m.get(k) == v for k, v in meta.items())
    return valid, (float(elapsed) if isinstance(elapsed, (int, float)) else None)


def real_codes(base_codes: list[str], task: Task) -> list[str]:
    """Real cache/progress code labels for ``task``, in display order.

    Secondary-photon runs each backend once per electron treatment and stores
    and reports the result under a treatment-suffixed code (``openmc-led``,
    ``yamc-cpu-ttb``, ...), exactly as :func:`secondary_photon.run_all` builds
    them. Every other family uses the backend code unchanged. Keeping this in
    step with the family is what lets the live progress events land on the
    matching dashboard row instead of being dropped as unknown codes.
    """
    treatments = task.extras.get("electron_treatments")
    if not treatments:
        return list(base_codes)
    return [f"{base}-{t}" for base in base_codes for t in treatments]


def _code_meta(code: str, task: Task, fingerprint: dict) -> dict:
    """Minimal cache-validity meta for the up-front estimate.

    A matching library is enough for most OpenMC references. Those of the
    photon-bearing families must also match the OpenMC build, and YAMC must
    match the current yamc build fingerprint (a new build invalidates its cache).
    """
    meta = {"library": task.library}
    if code.startswith("yamc"):
        meta = {**meta, **fingerprint}
    elif task.family in OPENMC_BUILD_KEYED_FAMILIES:
        meta = {**meta, **openmc_fingerprint()}
    return meta


def build_state(tasks: list[Task], backends: Backends) -> dict:
    """Build the initial dashboard state (see :mod:`yamc_verify.dashboard`)."""
    fingerprint = yamc_fingerprint()
    state = {
        "yamc_version": fingerprint.get("yamc_version", "?"),
        "yamc_built_ago": yamc_build_age(),
        "tasks": [],
    }
    for task in tasks:
        codes = real_codes([c for c in task.codes if code_enabled(c, backends)], task)
        try:
            items = discover_items(task, backends.run_openmc)
        except Exception:
            items = []
        code_entries = {}
        for code in codes:
            meta = _code_meta(code, task, fingerprint)
            cached = 0
            seed_times = []
            for key in items:
                valid, elapsed = _cache_info(task.cache_dir, task.library, code, key, meta)
                if valid:
                    cached += 1
                if elapsed is not None:
                    seed_times.append(elapsed)
            seed_avg = (sum(seed_times) / len(seed_times)) if seed_times else None
            code_entries[code] = {
                "total": len(items),
                "cached": cached,
                "computed": 0,
                "running_key": None,
                "status": "queued",
                "t_start": None,
                "t_end": None,
                "obs_times": [],
                "seed_avg": seed_avg,
            }
        state["tasks"].append({
            "key": task.key,
            "label": task.label,
            "notebook": task.notebook,
            "status": "queued",
            "t_start": None,
            "t_end": None,
            "codes": code_entries,
        })
    return state
