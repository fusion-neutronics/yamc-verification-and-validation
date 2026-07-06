"""JSON cache for verification-notebook simulation results.

Each cache entry is a single JSON file at
``<cache_dir>/<library>/<code>/<key>.json`` with the shape::

    {"_meta": {...run parameters that produced this result...},
     "results": {...arbitrary results dict, numpy arrays JSON-encoded...}}

Numpy arrays are encoded as ``{"_np_array": [...], "_dtype": "..."}`` and
restored on load. The cache is invalidated automatically when the metadata
in the cached file does not match the metadata passed by the caller (e.g.
``PARTICLES`` or ``BATCHES`` changed), so the simulation is re-run and the
cache file overwritten.

The module is imported by the notebooks under ``tasks/`` to avoid re-running
simulations every notebook session. OpenMC reference results are keyed by run
parameters only (OpenMC is fixed); YAMC results additionally fold
:func:`yamc_fingerprint` into their metadata, so the cache auto-invalidates and
the simulation re-runs whenever the installed yamc version or compiled core
(``_core*.so``) changes. Set ``YAMC_VERIFY_ACCEPT_OLD_CACHE=1`` to accept
previously cached YAMC results regardless of which build produced them.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

ACCEPT_OLD_CACHE_ENV = "YAMC_VERIFY_ACCEPT_OLD_CACHE"

# Explicit-item-list env var per selection kind (see :func:`select_requested`).
_ITEM_ENV = {"nuclide": "YAMC_VERIFY_NUCLIDES", "element": "YAMC_VERIFY_ELEMENTS"}


def _progress():
    """Return the :mod:`yamc_verify.progress` module, or ``None`` if unavailable.

    Kept lazy and defensive so ``cache_utils`` has no hard dependency on the
    package and progress reporting can never break a simulation run.
    """
    try:
        from yamc_verify import progress
    except Exception:
        return None
    return progress


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"_np_array": obj.tolist(), "_dtype": str(obj.dtype)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "_np_array" in obj and "_dtype" in obj:
            return np.array(obj["_np_array"], dtype=obj["_dtype"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def _cache_path(cache_dir: Path | str, library: str, code: str, key: str) -> Path:
    safe_key = str(key).replace("/", "_")
    return Path(cache_dir) / str(library) / str(code) / f"{safe_key}.json"


def _meta_matches(cached: Mapping, expected: Mapping) -> bool:
    for k, v in expected.items():
        if cached.get(k) != v:
            return False
    return True


def load_cached(
    cache_dir: Path | str,
    library: str,
    code: str,
    key: str,
    *,
    expected_meta: Optional[Mapping] = None,
) -> Optional[dict]:
    """Return the cached results dict, or ``None`` if no usable cache exists."""
    p = _cache_path(cache_dir, library, code, key)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if expected_meta is not None and not _meta_matches(data.get("_meta", {}), expected_meta):
        return None
    return _decode(data.get("results"))


def save_cached(
    cache_dir: Path | str,
    library: str,
    code: str,
    key: str,
    results: Mapping,
    *,
    meta: Optional[Mapping] = None,
) -> Path:
    """Write ``results`` to the cache file and return its path."""
    p = _cache_path(cache_dir, library, code, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_meta": dict(meta or {}), "results": _encode(dict(results))}
    with p.open("w") as f:
        json.dump(payload, f, indent=2)
    return p


def cached_or_run(
    cache_dir: Path | str,
    library: str,
    code: str,
    key: str,
    run_fn: Callable[[], Mapping],
    *,
    meta: Optional[Mapping] = None,
    verbose: bool = True,
) -> dict:
    """Return cached results if available with matching ``meta``, else run and cache.

    ``run_fn`` is a zero-argument callable that returns a results dict.

    Emits progress events (see :mod:`yamc_verify.progress`) so the verification
    orchestrator can render a live dashboard: a ``cached`` event on a cache hit,
    or an ``item_start`` / ``computed`` pair (with wall-clock seconds) on a run.
    The wall time is also stored under ``elapsed_seconds`` in the cache metadata
    so a later orchestrator run can seed its ETA estimate. Reporting is a no-op
    outside an orchestrated run and never affects the returned results.
    """
    import time

    _p = _progress()
    cached = load_cached(cache_dir, library, code, key, expected_meta=meta)
    p = _cache_path(cache_dir, library, code, key)
    if cached is not None:
        if _p:
            _p.report_item(code, key, status="cached")
        if verbose:
            print(f"  {code}: loaded from cache ({p})")
        return cached
    if _p:
        _p.report_item_start(code, key)
    t0 = time.perf_counter()
    results = run_fn()
    dt = time.perf_counter() - t0
    save_cached(
        cache_dir, library, code, key, results,
        meta={**dict(meta or {}), "elapsed_seconds": dt},
    )
    if _p:
        _p.report_item(code, key, status="computed", seconds=dt)
    if verbose:
        print(f"  {code}: cached to {p}")
    return dict(results)


def select_requested(items, *, kind, key=None, stable_getter=None):
    """Filter a discovered nuclide/element list using the verification env vars.

    Selection priority (highest first):

    1. **Explicit list** - ``YAMC_VERIFY_NUCLIDES`` (when ``kind="nuclide"``) or
       ``YAMC_VERIFY_ELEMENTS`` (when ``kind="element"``): a comma-separated
       list naming the exact items to run. The returned order follows the
       request, so ``YAMC_VERIFY_NUCLIDES=Fe56`` runs only Fe56.
    2. **Count cap** - ``YAMC_VERIFY_NUMBER_OF_SIMULATIONS``:

       * ``> 0`` keeps the first N discovered items (the smoke-test behaviour).
       * ``0`` (or unset) keeps **every** discovered item.
       * ``< 0`` (e.g. ``-1``) keeps every discovered item that is (or, for
         elements, maps to) a **stable, non-radioactive** nuclide - see
         ``stable_getter``.

    Parameters
    ----------
    items : list
        Discovered candidates: bare strings (e.g. ``"Fe56"``, ``"Fe"``), or
        tuples/objects whose comparison key is extracted by ``key``.
    kind : {"nuclide", "element"}
        Selects which explicit-list env var applies. Element-based notebooks
        (photon) pass ``"element"``; everything else passes ``"nuclide"``.
    key : callable, optional
        Maps an item to its string key (default: identity). Pass
        ``lambda t: t[0]`` for tuple lists such as ``(name, path, ...)``.
    stable_getter : callable, optional
        Zero-argument callable returning the set of *keys* considered stable
        (stable nuclide names, or - for ``kind="element"`` - element symbols
        that have a stable isotope). Called **only** when the cap is negative,
        so the (chain-loading) stability lookup stays lazy. When the cap is
        negative and no ``stable_getter`` is given, the full discovered list is
        returned unchanged (the caller cannot distinguish stability, e.g. an
        already physics-restricted list).

    Raises
    ------
    ValueError
        If an explicitly requested item is not present in the discovered list
        (catches typos and unavailable data rather than silently running
        nothing).
    """
    import os

    if key is None:
        def key(x):
            return x

    env_name = _ITEM_ENV[kind]

    raw = os.environ.get(env_name, "").strip()
    if raw:
        requested = [t.strip() for t in raw.split(",") if t.strip()]
        by_key: dict = {}
        for it in items:
            by_key.setdefault(key(it), it)
        missing = [r for r in requested if r not in by_key]
        if missing:
            available = sorted(by_key)
            raise ValueError(
                f"{env_name}={raw!r} requested {missing}, which is not among "
                f"the {len(available)} discovered {kind}(s). "
                f"Available (first 40): {available[:40]}"
            )
        result = [by_key[r] for r in requested]
    else:
        raw_cap = os.environ.get("YAMC_VERIFY_NUMBER_OF_SIMULATIONS", "0").strip()
        try:
            cap = int(raw_cap) if raw_cap else 0
        except ValueError:
            cap = 0
        if cap < 0:
            if stable_getter is not None:
                stable_keys = set(stable_getter())
                result = [it for it in items if key(it) in stable_keys]
            else:
                result = list(items)
        elif cap > 0:
            result = items[:cap]
        else:
            result = list(items)

    _p = _progress()
    if _p:
        _p.report_plan([key(it) for it in result], kind=kind)
    return result


def requested_items(kind: str) -> list[str]:
    """Explicitly requested item names for ``kind`` in request order, or ``[]``.

    Reads the same ``YAMC_VERIFY_NUCLIDES`` / ``YAMC_VERIFY_ELEMENTS`` list that
    :func:`select_requested` honours. Discovery uses it to add back an
    explicitly named item that a family's heuristic filter would otherwise drop
    (e.g. a decay-photon nuclide with no gamma emission, or an unstable
    transmutation target): naming an item means "run it".
    """
    raw = os.environ.get(_ITEM_ENV[kind], "").strip()
    return [t.strip() for t in raw.split(",") if t.strip()] if raw else []


@functools.lru_cache(maxsize=1)
def _yamc_core_hash() -> Optional[str]:
    """Short content hash of the compiled yamc core (``_core*.so``), or ``None``.

    Identifies the actual binary in use: it changes when the extension is
    rebuilt (``maturin develop``) but NOT when a git checkout, merge, or pull
    moves the source repo's HEAD without touching the binary. That is the whole
    point of keying on the artifact rather than the commit, so a no-op merge no
    longer invalidates every cached YAMC result. ``None`` when no compiled
    extension can be found (e.g. a pure-Python stand-in).
    """
    try:
        import glob
        import hashlib

        import yamc

        pkg = os.path.dirname(yamc.__file__)
        sos = sorted(glob.glob(os.path.join(pkg, "**", "*.so"), recursive=True))
        if not sos:
            return None
        h = hashlib.sha256()
        for so in sos:
            h.update(os.path.relpath(so, pkg).encode())  # name, so an added/renamed ext shows up
            with open(so, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return None


def _accept_old_cache() -> bool:
    """Whether ``YAMC_VERIFY_ACCEPT_OLD_CACHE`` opts out of build-based invalidation."""
    return os.environ.get(ACCEPT_OLD_CACHE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def yamc_fingerprint() -> dict:
    """Identity of the installed yamc, for cache invalidation.

    Returns ``{"yamc_version": yamc.__version__, "yamc_core_sha": ...}``. Fold
    this into the ``meta`` of a YAMC cache entry so the cache invalidates
    automatically (and the simulation re-runs) whenever the *compiled* yamc core
    changes. Keying on a content hash of ``_core*.so`` (rather than the git
    commit) means a checkout / merge / pull that leaves the binary untouched
    keeps the cache valid, while a real ``maturin develop`` rebuild invalidates
    it. ``yamc_core_sha`` is ``None`` when no compiled extension is found,
    leaving the version as the invalidation key.

    When ``YAMC_VERIFY_ACCEPT_OLD_CACHE`` is set this returns ``{}``, so YAMC
    results are keyed by run parameters only and any previously cached result is
    accepted regardless of which build produced it. Use it when you know the
    current binary matches the cached results and want to skip a recompute.
    """
    if _accept_old_cache():
        return {}
    import yamc

    return {"yamc_version": yamc.__version__, "yamc_core_sha": _yamc_core_hash()}
