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
    """
    cached = load_cached(cache_dir, library, code, key, expected_meta=meta)
    p = _cache_path(cache_dir, library, code, key)
    if cached is not None:
        if verbose:
            print(f"  {code}: loaded from cache ({p})")
        return cached
    results = run_fn()
    save_cached(cache_dir, library, code, key, results, meta=meta)
    if verbose:
        print(f"  {code}: cached to {p}")
    return dict(results)


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
