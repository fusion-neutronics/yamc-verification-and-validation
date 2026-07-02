"""Structured progress side-channel for the verification orchestrator.

A running notebook (or any code using :func:`report`) appends JSON-line events
to the file named by the ``YAMC_VERIFY_PROGRESS`` environment variable. The
orchestrator (:mod:`yamc_verify.orchestrator`) sets that variable to a unique
path per notebook launch and tails the file to drive the live dashboard.

When ``YAMC_VERIFY_PROGRESS`` is unset -- the normal interactive-notebook case
-- every function here is a cheap no-op, so the instrumentation is invisible
outside an orchestrated run.

Event shapes (one JSON object per line)::

    {"t": "plan",       "kind": "nuclide", "items": ["Fe56", ...]}
    {"t": "item_start", "code": "yamc-cpu", "key": "Fe56"}
    {"t": "item",       "code": "yamc-cpu", "key": "Fe56",
                        "status": "cached" | "computed", "seconds": 1.23}
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

ENV_VAR = "YAMC_VERIFY_PROGRESS"


def enabled() -> bool:
    """True when a progress side-channel file is configured."""
    return bool(os.environ.get(ENV_VAR))


def report(event: Mapping[str, Any]) -> None:
    """Append one event as a JSON line to the progress file, if configured.

    Never raises: progress reporting must not be able to break a simulation.
    Uses a single ``write`` of one line, which is atomic enough for a reader
    that parses complete lines and ignores a trailing partial line.
    """
    path = os.environ.get(ENV_VAR)
    if not path:
        return
    try:
        line = json.dumps(dict(event), separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def report_plan(items: Sequence[Any], *, kind: str) -> None:
    """Emit the per-notebook plan: the ordered list of items to be processed."""
    report({"t": "plan", "kind": str(kind), "items": [str(i) for i in items]})


def report_item_start(code: str, key: Any) -> None:
    """Emit that a (non-cached) simulation for ``code``/``key`` is starting."""
    report({"t": "item_start", "code": str(code), "key": str(key)})


def report_item(code: str, key: Any, *, status: str, seconds: float | None = None) -> None:
    """Emit that ``code``/``key`` finished, either from ``cache`` or ``computed``."""
    ev: dict[str, Any] = {"t": "item", "code": str(code), "key": str(key), "status": str(status)}
    if seconds is not None:
        ev["seconds"] = float(seconds)
    report(ev)
