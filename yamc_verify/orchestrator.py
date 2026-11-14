"""Run the verification tasks with the codes installed in this environment.

This is the in-place runner behind ``run_models.sh``: it uses whatever OpenMC
and yamc are installed here (a local core build, say), selected by
``MC_BACKENDS``. To run named code versions, each in its own environment, use
``vv run`` (:mod:`yamc_verify.project`) instead.

Each task is run by ``python -m yamc_verify.run_task`` as a subprocess with its
output captured, and the shared live display (:func:`vv.run_jobs`) follows its
progress events.

Usage (arguments are task keys)::

    python -m yamc_verify.orchestrator                       # all endf-b8.1 tasks
    python -m yamc_verify.orchestrator neutron/sphere/endf-b8.1 ...

Honors ``MC_BACKENDS`` (required), ``YAMC_VERIFY_NUCLIDES``,
``YAMC_VERIFY_ELEMENTS`` and ``YAMC_VERIFY_NUMBER_OF_SIMULATIONS``.
"""
from __future__ import annotations

import sys

from vv import Command, Job, run_jobs
from vv.dashboard import humanize

from cache_utils import accept_old_cache, code_identity

from . import backends as _backends
from . import plan, tasks
from .plan import REPO_ROOT


def _select_tasks(argv):
    """Resolve task-key arguments to registry tasks (default: all endf-b8.1)."""
    if argv:
        return [tasks.get_task(arg) for arg in argv]
    return [t for t in tasks.all_tasks() if t.library == "endf-b8.1"]


def _jobs(selected, b) -> list[Job]:
    """One job per task, covering every enabled backend, with its records counted."""
    jobs = []
    for task in selected:
        labels = plan.real_codes([c for c in task.codes if plan.code_enabled(c, b)], task)
        try:
            items = plan.discover_items(task, b.discover_with_openmc)
        except Exception:
            items = []

        def find(label, task=task):
            if accept_old_cache() or not plan._build_keyed(label, task, b):
                return {}
            return {"build_id": code_identity(label)["build_id"]}

        model, library = task.key.rsplit("/", 1)
        jobs.append(Job(model, library, None, plan.task_rows(task, labels, items, find)))
    return jobs


def run(argv=None) -> int:
    """Run the selected tasks with the live display; return the exit status."""
    argv = list(argv if argv is not None else sys.argv[1:])
    b = _backends.resolve()
    selected = _select_tasks(argv)
    if not selected:
        print("no tasks to run", file=sys.stderr)
        return 1
    jobs = _jobs(selected, b)
    commands = [Command([sys.executable, "-m", "yamc_verify.run_task", f"{job.model}/{job.library}"],
                        cwd=REPO_ROOT) for job in jobs]
    header = []
    if any(row.label.startswith("yamc") for job in jobs for row in job.rows):
        age = plan.yamc_build_age()
        version = code_identity("yamc")["version"]
        header.append(f"running with the installed YAMC {version}"
                      + (f", built {humanize(age)} ago" if age is not None else ""))
    failures = run_jobs(jobs, commands, header=header)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
