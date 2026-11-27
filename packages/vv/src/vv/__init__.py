"""Code-agnostic verification and validation runs.

* :mod:`vv.records`: the result record format, one self-describing JSON file
  per result and build.
* :mod:`vv.runner`: checking a request, planning it and running it, each code in
  its own environment, through a :class:`Project` the repository supplies.
* :mod:`vv.progress` and :mod:`vv.dashboard`: the progress events a run emits
  and the live display that follows them.
* :mod:`vv.publish`: comparing the records with the published run, publishing
  them, and the release archive of each published run.
* :mod:`vv.cli`: the ``vv plan``, ``vv run``, ``vv compare``, ``vv publish``
  and ``vv release`` commands.

Nothing here names a particular code: a code is a name plus the recipe for its
environment (:class:`Code`), supplied by the project.
"""
from __future__ import annotations

from . import dashboard, progress, publish, records
from .runner import (
    ALL,
    Code,
    Command,
    Job,
    Plan,
    Project,
    Row,
    RunResult,
    SelectionError,
    plan,
    run,
    run_jobs,
    select,
)

__all__ = [
    "ALL",
    "Code",
    "Command",
    "Job",
    "Plan",
    "Project",
    "Row",
    "RunResult",
    "SelectionError",
    "dashboard",
    "plan",
    "progress",
    "publish",
    "records",
    "run",
    "run_jobs",
    "select",
]
