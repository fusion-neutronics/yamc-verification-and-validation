"""The yamc verification models, and ``verification()`` to run them.

The task registry (:mod:`yamc_verify.tasks`), geometry builders, item
discovery, per-family simulation runners with plotting and metrics, one Python
environment per code version (:mod:`yamc_verify.environments`), and
:func:`verification`, which runs named code versions through the shared ``vv``
package (also available as ``vv plan`` / ``vv run``).
"""
from __future__ import annotations

from . import backends, discovery, geometry, tasks
from .project import verification, verification_plan
from .tasks import Task, all_tasks, get_task, task_for_notebook

__all__ = [
    "backends",
    "discovery",
    "geometry",
    "tasks",
    "Task",
    "all_tasks",
    "get_task",
    "task_for_notebook",
    "verification",
    "verification_plan",
]
