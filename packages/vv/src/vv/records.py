"""Result records: one self-describing JSON file per result and build.

Every result is a record at::

    <cache_dir>/<library>/<code>/<item>/<build_id>.json

with the shape::

    {"schema": 1,
     "library": "endf-b8.1", "code": "code-a", "item": "Fe56",
     "code_identity": {"name": "code-a", "version": "1.2.0",
                       "commit": "86ef200e...", "dirty": false,
                       "binary_sha": null, "backend": "cpu",
                       "build_id": "1.2.0+g86ef200e6a0e"},
     "settings": {...run parameters...},
     "settings_hash": "...",
     "run": {"date": "2026-11-12T20:41:07+00:00", "host": "...",
             "elapsed_seconds": 812.4},
     "results": {...numpy arrays JSON-encoded...}}

``code`` is the name the caller files results under (one code can have several,
for example one per backend). ``code_identity`` is supplied by the caller and
must hold a ``build_id``, which names the file: a record is never overwritten by
a different build, so results from two builds can be compared. Re-running the
same build replaces its own record. Numpy arrays are encoded as
``{"_np_array": [...], "_dtype": "..."}`` and restored on load.

Which record satisfies a run is the caller's policy: :func:`find_record` takes
the build to require, or a predicate, or neither (the newest record with
matching settings).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import socket
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

import numpy as np

SCHEMA = 1


def encode(obj: Any) -> Any:
    """Return ``obj`` with numpy arrays and scalars made JSON-serialisable."""
    if isinstance(obj, np.ndarray):
        return {"_np_array": obj.tolist(), "_dtype": str(obj.dtype)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def decode(obj: Any) -> Any:
    """Inverse of :func:`encode`: restore the numpy arrays in a loaded record."""
    if isinstance(obj, dict):
        if "_np_array" in obj and "_dtype" in obj:
            return np.array(obj["_np_array"], dtype=obj["_dtype"])
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


def record_dir(cache_dir: Path | str, library: str, code: str, item: str) -> Path:
    """The folder holding every build's record of one item."""
    return Path(cache_dir) / str(library) / str(code) / str(item).replace("/", "_")


def settings_hash(settings: Mapping) -> str:
    """Stable hash of a settings dict, identifying like-for-like runs across builds."""
    canonical = json.dumps(encode(dict(settings)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def settings_match(stored: Mapping, expected: Mapping) -> bool:
    """Whether every key of ``expected`` has the same value in ``stored``."""
    return all(stored.get(k) == v for k, v in expected.items())


def build_id(version: Optional[str], commit: Optional[str], dirty: Optional[bool],
             binary_sha: Optional[str]) -> str:
    """File-name-safe id of one build: ``<version>+g<commit12>[.dirty][.so<sha>]``.

    The binary hash is included only when the commit does not pin the binary
    (unknown or dirty). With neither, the build is ``<version>+unknown``, and
    with no version either, ``unknown``.
    """
    out = str(version or "unknown")
    if commit:
        out += f"+g{commit[:12]}"
        if dirty:
            out += ".dirty"
    if binary_sha and (not commit or dirty):
        out += f"{'.' if commit else '+'}so{binary_sha}"
    if version and not commit and not binary_sha:
        out += "+unknown"
    return out


def read_record(path: Path | str) -> Optional[dict]:
    """The record at ``path``, or ``None`` if it is unreadable or another schema."""
    try:
        with Path(path).open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == SCHEMA else None


def iter_records(cache_dir: Path | str, library: str, code: str, item: str) -> Iterator[tuple[Path, dict]]:
    """Yield ``(path, record)`` for every readable record of one item, newest first."""
    d = record_dir(cache_dir, library, code, item)
    if not d.is_dir():
        return
    found = []
    for path in d.glob("*.json"):
        record = read_record(path)
        if record is not None:
            # Records converted from the old layout have no run date: they sort last.
            found.append(((record.get("run") or {}).get("date") or "", path, record))
    for _, path, record in sorted(found, key=lambda t: t[0], reverse=True):
        yield path, record


def cached_items(cache_dir: Path | str, library: str, code: str) -> list[str]:
    """Item keys that have at least one record for ``code``."""
    d = Path(cache_dir) / str(library) / str(code)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and any(p.glob("*.json")))


def latest_record(cache_dir: Path | str, library: str, code: str, item: str) -> Optional[tuple[Path, dict]]:
    """The most recent record of one item from any build, or ``None``."""
    return next(iter_records(cache_dir, library, code, item), None)


def find_record(
    cache_dir: Path | str,
    library: str,
    code: str,
    item: str,
    *,
    settings: Optional[Mapping] = None,
    build_id: Optional[str] = None,
    accept: Optional[Callable[[dict], bool]] = None,
) -> Optional[tuple[Path, dict]]:
    """The record of one item whose settings include ``settings``, or ``None``.

    With ``build_id`` only that build's record counts. Otherwise the newest
    record that ``accept`` (a predicate on the whole record) allows is used, or
    the newest of any build when ``accept`` is not given.
    """
    expected = dict(settings or {})
    if build_id is not None:
        path = record_dir(cache_dir, library, code, item) / f"{build_id}.json"
        record = read_record(path) if path.exists() else None
        if record is not None and settings_match(record.get("settings", {}), expected):
            return path, record
        return None
    for path, record in iter_records(cache_dir, library, code, item):
        if settings_match(record.get("settings", {}), expected) and (accept is None or accept(record)):
            return path, record
    return None


def save_record(
    cache_dir: Path | str,
    library: str,
    code: str,
    item: str,
    results: Mapping,
    *,
    identity: Mapping,
    settings: Optional[Mapping] = None,
    elapsed_seconds: Optional[float] = None,
) -> Path:
    """Write the record of one result from the build ``identity`` names; return its path.

    ``identity`` describes the build (see the module docstring) and must hold a
    ``build_id``.
    """
    identity = dict(identity)
    if not identity.get("build_id"):
        raise ValueError(f"the code identity {identity!r} has no build_id, which names the record")
    settings = dict(settings or {})
    path = record_dir(cache_dir, library, code, item) / f"{identity['build_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": SCHEMA,
        "library": library,
        "code": code,
        "item": item,
        "code_identity": identity,
        "settings": encode(settings),
        "settings_hash": settings_hash(settings),
        "run": {
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "elapsed_seconds": elapsed_seconds,
        },
        "results": encode(dict(results)),
    }
    with path.open("w") as f:
        json.dump(record, f, indent=2)
    return path
