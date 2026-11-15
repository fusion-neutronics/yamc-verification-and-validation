"""Which result records the yamc and OpenMC runs accept, and the code identities they store.

The record format itself is :mod:`vv.records` (shared, code-agnostic). This
module is the yamc-verification policy on top of it:

* the code identity each record stores: OpenMC from ``openmc --version``, yamc
  from its version and ``yamc.__commit__``, plus a hash of the compiled
  extension when the commit is unknown or dirty;
* which record satisfies a run:

  * **yamc** codes always need a record from the installed build.
  * **OpenMC** codes need one from the installed OpenMC only when the caller asks
    (``build_keyed=True``), which the photon-bearing families do while running
    OpenMC: their references changed with the OpenMC build (openmc#4036). Other
    OpenMC references are keyed by settings alone, and the most recent matching
    record is used.

  Set ``YAMC_VERIFY_ACCEPT_OLD_CACHE=1`` to accept the most recent record with
  matching settings from any build;
* which items a run selects (``YAMC_VERIFY_NUCLIDES``, ``YAMC_VERIFY_ELEMENTS``,
  ``YAMC_VERIFY_NUMBER_OF_SIMULATIONS``).

Code labels are ``openmc`` or ``yamc[-<backend>]``, optionally with an electron
treatment suffix (``openmc-led``, ``yamc-cpu-ttb``).
"""

from __future__ import annotations

import functools
import hashlib
import os
import time
from pathlib import Path
from typing import Callable, Mapping, Optional

from vv import progress
from vv.records import build_id, decode, find_record, save_record

ACCEPT_OLD_CACHE_ENV = "YAMC_VERIFY_ACCEPT_OLD_CACHE"

# Explicit-item-list env var per selection kind (see :func:`select_requested`).
_ITEM_ENV = {"nuclide": "YAMC_VERIFY_NUCLIDES", "element": "YAMC_VERIFY_ELEMENTS"}


def _uses_build(code: str, build_keyed: Optional[bool]) -> bool:
    if accept_old_cache():
        return False
    if build_keyed is not None:
        return build_keyed
    return code_name(code) == "yamc"


def find_cached(
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
    build = code_identity(code)["build_id"] if _uses_build(code, build_keyed) else None
    return find_record(cache_dir, library, code, key, settings=settings, build_id=build)


def load_cached(cache_dir, library, code, key, *, settings=None, build_keyed=None) -> Optional[dict]:
    """Return the decoded results of the record :func:`find_cached` picks, or ``None``."""
    found = find_cached(cache_dir, library, code, key, settings=settings, build_keyed=build_keyed)
    return None if found is None else decode(found[1].get("results"))


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
    """Write a record for the installed build of ``code`` (or ``identity``) and return its path."""
    return save_record(cache_dir, library, code, key, results,
                       identity=identity or code_identity(code),
                       settings=settings, elapsed_seconds=elapsed_seconds)


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

    Emits progress events (see :mod:`vv.progress`) so the live display can follow
    the run: a ``cached`` event on a hit, or an ``item_start`` / ``computed``
    pair (with wall-clock seconds) on a run. The wall time is stored in the
    record so a later run can seed its ETA estimate.
    """
    found = find_cached(cache_dir, library, code, key, settings=settings, build_keyed=build_keyed)
    if found is not None:
        progress.report_item(code, key, status="cached")
        if verbose:
            print(f"  {code}: loaded from cache ({found[0]})")
        return decode(found[1].get("results"))
    progress.report_item_start(code, key)
    t0 = time.perf_counter()
    results = run_fn()
    dt = time.perf_counter() - t0
    path = save_cached(cache_dir, library, code, key, results, settings=settings, elapsed_seconds=dt)
    progress.report_item(code, key, status="computed", seconds=dt)
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

    progress.report_plan([key(it) for it in result], kind=kind)
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


def accept_old_cache() -> bool:
    """Whether ``YAMC_VERIFY_ACCEPT_OLD_CACHE`` accepts records from any build."""
    return os.environ.get(ACCEPT_OLD_CACHE_ENV, "").strip().lower() in ("1", "true", "yes", "on")
