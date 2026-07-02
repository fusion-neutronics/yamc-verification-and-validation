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
the simulation re-runs whenever the installed yamc version or git commit
changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np


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


def select_requested(items, *, kind, key=None):
    """Filter a discovered nuclide/element list using the verification env vars.

    Selection priority (highest first):

    1. **Explicit list** - ``YAMC_VERIFY_NUCLIDES`` (when ``kind="nuclide"``) or
       ``YAMC_VERIFY_ELEMENTS`` (when ``kind="element"``): a comma-separated
       list naming the exact items to run. The returned order follows the
       request, so ``YAMC_VERIFY_NUCLIDES=Fe56`` runs only Fe56.
    2. **Count cap** - ``YAMC_VERIFY_NUMBER_OF_SIMULATIONS``: keep the first N
       discovered items (the existing smoke-test behaviour).
    3. **Full list** - neither env var set.

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

    env_name = {
        "nuclide": "YAMC_VERIFY_NUCLIDES",
        "element": "YAMC_VERIFY_ELEMENTS",
    }[kind]

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
        cap = int(os.environ.get("YAMC_VERIFY_NUMBER_OF_SIMULATIONS", "0")) or None
        result = items[:cap] if cap else items

    _p = _progress()
    if _p:
        _p.report_plan([key(it) for it in result], kind=kind)
    return result


def _yamc_git_commit() -> Optional[str]:
    """Short git commit of the installed yamc source, or ``None``.

    Only meaningful for an editable / source install where ``yamc.__file__``
    lives inside a git checkout (the usual yamc development setup); returns
    ``None`` for a packaged wheel install with no surrounding repository.
    """
    try:
        import subprocess

        import yamc

        src = Path(yamc.__file__).resolve()
        for parent in src.parents:
            if (parent / ".git").exists():
                out = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return out.stdout.strip() if out.returncode == 0 else None
        return None
    except Exception:
        return None


def yamc_fingerprint() -> dict:
    """Identity of the installed yamc, for cache invalidation.

    Returns ``{"yamc_version": yamc.__version__, "yamc_commit": ...}``. Fold
    this into the ``meta`` of a YAMC cache entry so the cache invalidates
    automatically (and the simulation re-runs) whenever the installed yamc
    changes: a new released version, or a new git commit for an editable /
    source install. The version string does not bump per development build, so
    the commit is what catches day-to-day changes. The commit is ``None`` for a
    packaged wheel install with no surrounding git checkout, leaving the version
    as the invalidation key.
    """
    import yamc

    return {"yamc_version": yamc.__version__, "yamc_commit": _yamc_git_commit()}
