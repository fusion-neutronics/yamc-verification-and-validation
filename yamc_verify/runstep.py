"""Uniform run-one-item wrapper used by every family's run loop.

Some families cache their YAMC results (neutron, transmutation-reaction-rate)
and some do not (photon, decay-photon, secondary-photon, transmutation-flux).
OpenMC is always cached. ``run_step`` unifies both: it routes cached runs
through :func:`cache_utils.cached_or_run` (which reports progress itself) and
times + reports uncached runs the same way, so the orchestrator dashboard sees
every ``(item, code)`` regardless of caching policy.
"""
from __future__ import annotations

import time
from typing import Callable, Mapping, Optional

from . import progress


def run_step(
    cache_dir,
    library: str,
    code: str,
    key: str,
    run_fn: Callable[[], Mapping],
    *,
    meta: Optional[Mapping] = None,
    cache: bool = True,
    verbose: bool = False,
) -> dict:
    """Run (or load) one ``(key, code)`` result, emitting progress either way.

    Parameters mirror :func:`cache_utils.cached_or_run`. When ``cache`` is
    ``False`` the result is always computed fresh (no cache read or write) but
    still timed and reported as ``computed``.
    """
    if cache:
        from cache_utils import cached_or_run

        return cached_or_run(
            cache_dir, library, code, key, run_fn, meta=meta, verbose=verbose
        )

    progress.report_item_start(code, key)
    t0 = time.perf_counter()
    result = run_fn()
    dt = time.perf_counter() - t0
    progress.report_item(code, key, status="computed", seconds=dt)
    return dict(result) if isinstance(result, Mapping) else result
