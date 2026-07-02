"""Shared code for the YAMC verification model notebooks.

Extracted from the per-notebook cells to remove duplication: the task registry
(:mod:`yamc_verify.tasks`), geometry builders, item discovery, progress
side-channel, per-family simulation runners + plotting/metrics, and the live
dashboard orchestrator that runs the notebooks with quiet, in-place progress.
"""
from __future__ import annotations

from . import backends, discovery, geometry, progress, tasks
from .tasks import Task, all_tasks, get_task, task_for_notebook

__all__ = [
    "backends",
    "discovery",
    "geometry",
    "progress",
    "tasks",
    "Task",
    "all_tasks",
    "get_task",
    "task_for_notebook",
]
