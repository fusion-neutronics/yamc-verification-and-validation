"""JSON cache for verification-notebook simulation results.

Each cache entry is a single JSON file at
``<cache_dir>/<library>/<code>/<key>.json`` with the shape::

    {"_meta": {...run parameters that produced this result...},
     "results": {...arbitrary results dict, numpy arrays JSON-encoded...}}

Numpy arrays are encoded as ``{"_np_array": [...], "_dtype": "..."}`` and
restored on load. The cache is invalidated automatically when the metadata
in the cached file does not match the metadata passed by the caller (e.g.
``PARTICLES`` or ``BATCHES`` changed) — the simulation is re-run and the
cache file overwritten.

The module is imported by the notebooks under ``tasks/`` to avoid running
the (slow, deterministic) reference OpenMC simulations every notebook
session. The same helpers are intended to cache YAMC results in the future.
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
