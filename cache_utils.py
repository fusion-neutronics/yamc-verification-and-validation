"""Result records for the verification tasks.

Every simulation result is a self-describing JSON record at::

    <cache_dir>/<library>/<code>/<item>/<build_id>.json

with the shape::

    {"schema": 1,
     "library": "endf-b8.1", "code": "yamc-cpu", "item": "Fe56",
     "code_identity": {"name": "yamc", "version": "0.21.0",
                       "commit": "86ef200e...", "dirty": false,
                       "binary_sha": null, "backend": "cpu",
                       "build_id": "0.21.0+g86ef200e6a0e"},
     "settings": {...run parameters...},
     "settings_hash": "...",
     "run": {"date": "2026-11-12T20:41:07+00:00", "host": "...",
             "elapsed_seconds": 812.4},
     "results": {...numpy arrays JSON-encoded...}}

A record is never overwritten by a different build: a new yamc commit or a new
OpenMC release writes a record next to the old one, so results from two builds
can be compared. Re-running the same build with the same settings replaces its
own record. Numpy arrays are encoded as ``{"_np_array": [...], "_dtype": "..."}``
and restored on load.

Which record satisfies a run:

* **yamc** codes always need a record from the installed build.
* **OpenMC** codes need one from the installed OpenMC only when the caller asks
  (``build_keyed=True``), which the photon-bearing families do while running
  OpenMC: their references changed with the OpenMC build (openmc#4036). Other
  OpenMC references are keyed by settings alone, and the most recent matching
  record is used.

Set ``YAMC_VERIFY_ACCEPT_OLD_CACHE=1`` to accept the most recent record with
matching settings from any build.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
import socket
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


RECORD_SCHEMA = 1


def _record_dir(cache_dir: Path | str, library: str, code: str, key: str) -> Path:
    safe_key = str(key).replace("/", "_")
    return Path(cache_dir) / str(library) / str(code) / safe_key


def settings_hash(settings: Mapping) -> str:
    """Stable hash of a settings dict, identifying like-for-like runs across builds."""
    canonical = json.dumps(_encode(dict(settings)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _settings_match(stored: Mapping, expected: Mapping) -> bool:
    return all(stored.get(k) == v for k, v in expected.items())


def _read(path: Path) -> Optional[dict]:
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("schema") == RECORD_SCHEMA else None


def iter_records(cache_dir: Path | str, library: str, code: str, key: str):
    """Yield ``(path, record)`` for every readable record of one item, newest first."""
    d = _record_dir(cache_dir, library, code, key)
    if not d.is_dir():
        return
    found = []
    for path in d.glob("*.json"):
        record = _read(path)
        if record is not None:
            found.append((record.get("run", {}).get("date", ""), path, record))
    for _, path, record in sorted(found, key=lambda t: t[0], reverse=True):
        yield path, record


def cached_items(cache_dir: Path | str, library: str, code: str) -> list[str]:
    """Item keys that have at least one record for ``code``."""
    d = Path(cache_dir) / str(library) / str(code)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and any(p.glob("*.json")))


def latest_record(cache_dir, library, code, key) -> Optional[tuple[Path, dict]]:
    """The most recent record of one item from any build, or ``None``."""
    return next(iter_records(cache_dir, library, code, key), None)


def _uses_build(code: str, build_keyed: Optional[bool]) -> bool:
    if _accept_old_cache():
        return False
    if build_keyed is not None:
        return build_keyed
    return code_name(code) == "yamc"


def find_record(
    cache_dir: Path | str,
    library: str,
    code: str,
    key: str,
    *,
    settings: Optional[Mapping] = None,
    build_keyed: Optional[bool] = None,
) -> Optional[tuple[Path, dict]]:
    """The record that satisfies a run of ``code`` on ``key``, or ``None``.

    ``settings`` must be a subset of the record's settings. ``build_keyed``
    defaults to ``True`` for yamc codes and ``False`` for OpenMC (see the module
    docstring); either way ``YAMC_VERIFY_ACCEPT_OLD_CACHE`` turns it off.
    """
    expected = dict(settings or {})
    if _uses_build(code, build_keyed):
        path = _record_dir(cache_dir, library, code, key) / f"{code_identity(code)['build_id']}.json"
        record = _read(path) if path.exists() else None
        if record is not None and _settings_match(record.get("settings", {}), expected):
            return path, record
        return None
    for path, record in iter_records(cache_dir, library, code, key):
        if _settings_match(record.get("settings", {}), expected):
            return path, record
    return None


def load_cached(cache_dir, library, code, key, *, settings=None, build_keyed=None) -> Optional[dict]:
    """Return the decoded results of the record :func:`find_record` picks, or ``None``."""
    found = find_record(cache_dir, library, code, key, settings=settings, build_keyed=build_keyed)
    return None if found is None else _decode(found[1].get("results"))


def save_cached(
    cache_dir: Path | str,
    library: str,
    code: str,
    key: str,
    results: Mapping,
    *,
    settings: Optional[Mapping] = None,
    elapsed_seconds: Optional[float] = None,
    identity: Optional[Mapping] = None,
) -> Path:
    """Write a record for the installed build of ``code`` and return its path."""
    identity = dict(identity or code_identity(code))
    settings = dict(settings or {})
    path = _record_dir(cache_dir, library, code, key) / f"{identity['build_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": RECORD_SCHEMA,
        "library": library,
        "code": code,
        "item": key,
        "code_identity": identity,
        "settings": _encode(settings),
        "settings_hash": settings_hash(settings),
        "run": {
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "elapsed_seconds": elapsed_seconds,
        },
        "results": _encode(dict(results)),
    }
    with path.open("w") as f:
        json.dump(record, f, indent=2)
    return path


def cached_or_run(
    cache_dir: Path | str,
    library: str,
    code: str,
    key: str,
    run_fn: Callable[[], Mapping],
    *,
    settings: Optional[Mapping] = None,
    build_keyed: Optional[bool] = None,
    verbose: bool = True,
) -> dict:
    """Return the matching record's results, else run ``run_fn`` and record them.

    ``run_fn`` is a zero-argument callable that returns a results dict.

    Emits progress events (see :mod:`yamc_verify.progress`) so the verification
    orchestrator can render a live dashboard: a ``cached`` event on a hit, or an
    ``item_start`` / ``computed`` pair (with wall-clock seconds) on a run. The
    wall time is stored in the record so a later run can seed its ETA estimate.
    """
    import time

    _p = _progress()
    found = find_record(cache_dir, library, code, key, settings=settings, build_keyed=build_keyed)
    if found is not None:
        if _p:
            _p.report_item(code, key, status="cached")
        if verbose:
            print(f"  {code}: loaded from cache ({found[0]})")
        return _decode(found[1].get("results"))
    if _p:
        _p.report_item_start(code, key)
    t0 = time.perf_counter()
    results = run_fn()
    dt = time.perf_counter() - t0
    path = save_cached(cache_dir, library, code, key, results, settings=settings, elapsed_seconds=dt)
    if _p:
        _p.report_item(code, key, status="computed", seconds=dt)
    if verbose:
        print(f"  {code}: cached to {path}")
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
def _yamc_binary_sha() -> Optional[str]:
    """Short content hash of the compiled yamc core (``_core*.so``), or ``None``.

    Part of the build id only when the commit cannot identify the binary on its
    own: a wheel that predates ``yamc.__commit__``, or a build from a checkout
    with uncommitted changes, where each rebuild must still get its own record.
    """
    try:
        import glob

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


def code_name(code: str) -> str:
    """``"openmc"`` or ``"yamc"`` for a cache code label such as ``yamc-cpu-ttb``."""
    name = str(code).split("-", 1)[0]
    if name not in ("openmc", "yamc"):
        raise ValueError(f"cache code {code!r} names neither openmc nor yamc")
    return name


def _backend(code: str) -> Optional[str]:
    parts = str(code).split("-")
    return parts[1] if code_name(code) == "yamc" and len(parts) > 1 else None


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


@functools.lru_cache(maxsize=1)
def openmc_identity() -> dict:
    """Name, version and commit of the OpenMC on ``PATH`` (the one ``Model.run`` starts)."""
    import re
    import shutil
    import subprocess

    exe = shutil.which("openmc")
    if exe is None:
        raise FileNotFoundError(
            "No openmc executable on PATH: the OpenMC references need one, and each "
            "record stores its version and commit."
        )
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True).stdout
    version = re.search(r"OpenMC version (\S+)", out)
    commit = re.search(r"Commit hash: (\S+)", out)
    if version is None or commit is None:
        raise RuntimeError(f"Could not read the version and commit from `{exe} --version`:\n{out}")
    v, c = version.group(1), commit.group(1)
    return {"name": "openmc", "version": v, "commit": c, "dirty": None,
            "binary_sha": None, "build_id": build_id(v, c, None, None)}


@functools.lru_cache(maxsize=1)
def _yamc_identity() -> dict:
    import yamc

    commit = getattr(yamc, "__commit__", None)
    dirty = getattr(yamc, "__commit_dirty__", None)
    sha = _yamc_binary_sha() if (not commit or dirty) else None
    return {"name": "yamc", "version": yamc.__version__, "commit": commit, "dirty": dirty,
            "binary_sha": sha, "build_id": build_id(yamc.__version__, commit, dirty, sha)}


def code_identity(code: str) -> dict:
    """The installed build behind a cache code label, as stored in each record."""
    if code_name(code) == "openmc":
        return {**openmc_identity(), "backend": None}
    return {**_yamc_identity(), "backend": _backend(code)}


def _accept_old_cache() -> bool:
    """Whether ``YAMC_VERIFY_ACCEPT_OLD_CACHE`` accepts records from any build."""
    return os.environ.get(ACCEPT_OLD_CACHE_ENV, "").strip().lower() in ("1", "true", "yes", "on")
