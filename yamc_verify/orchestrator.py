"""Run the verification tasks with a live, in-place dashboard.

Replaces the scrolling ``run_models.sh`` output with the indented layout from
issue #29. Each task is run by ``python -m yamc_verify.run_task`` as a subprocess
with its stdout/stderr captured (so nothing scrolls) and ``YAMC_VERIFY_PROGRESS``
pointing at a per-task side-channel file. The orchestrator tails that file and
re-renders the dashboard as items complete.

Usage (mirrors run_models.sh); arguments are task keys or (for backwards
compatibility) the old notebook paths::

    python -m yamc_verify.orchestrator                       # all endf-b8.1 tasks
    python -m yamc_verify.orchestrator neutron/sphere/endf-b8.1 ...

Honors the same env vars as before: ``MC_BACKENDS``, ``YAMC_VERIFY_NUCLIDES``,
``YAMC_VERIFY_ELEMENTS``, ``YAMC_VERIFY_NUMBER_OF_SIMULATIONS``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

from . import backends as _backends
from . import dashboard, plan, tasks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _select_tasks(argv, run_openmc):
    """Resolve args to registry tasks (default: all endf-b8.1).

    Each arg may be a task key (``neutron/sphere/endf-b8.1``) or, for backwards
    compatibility, the old notebook path (``tasks/neutron/sphere/endf-b8.1.ipynb``).
    """
    if argv:
        out = []
        for arg in argv:
            try:
                out.append(tasks.get_task(arg))
                continue
            except KeyError:
                pass
            t = tasks.task_for_notebook(arg)
            if t is None:
                print(f"warning: no registered task for {arg}; skipping", file=sys.stderr)
                continue
            out.append(t)
        return out
    return [t for t in tasks.all_tasks() if t.library == "endf-b8.1"]


def _apply_event(task_state: dict, ev: dict):
    """Fold one progress event into a running task's state."""
    t = ev.get("t")
    codes = task_state["codes"]
    if t == "plan":
        # authoritative item count for this notebook; correct any drift
        n = len(ev.get("items", []))
        for c in codes.values():
            if c["total"] != n:
                c["total"] = n
        return
    code = ev.get("code")
    c = codes.get(code)
    if c is None:
        return
    now = time.monotonic()
    if c["status"] == "queued":
        c["status"] = "running"
        c["t_start"] = now
    if t == "item_start":
        c["running_key"] = ev.get("key")
    elif t == "item":
        if ev.get("status") == "cached":
            c["cached"] += 1
        else:
            c["computed"] += 1
            secs = ev.get("seconds")
            if secs is not None:
                c["obs_times"].append(float(secs))
        if c.get("running_key") == ev.get("key"):
            c["running_key"] = None
        if c["cached"] + c["computed"] >= c["total"] and c["total"] > 0:
            c["status"] = "done"
            c["t_end"] = now


def _finalize_task(task_state: dict):
    now = time.monotonic()
    for c in task_state["codes"].values():
        if c["status"] != "done":
            c["status"] = "done"
            c["t_end"] = now
            c["running_key"] = None
    task_state["status"] = "done"
    task_state["t_end"] = now


def run(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    b = _backends.resolve()
    selected = _select_tasks(argv, b.run_openmc)
    if not selected:
        print("no notebooks to run", file=sys.stderr)
        return 1

    state = plan.build_state(selected, b)
    by_key = {ts["key"]: ts for ts in state["tasks"]}
    display = dashboard.LiveDisplay()
    tty = display._tty

    display.update(state, time.monotonic())

    tmpdir = tempfile.mkdtemp(prefix="yamc_verify_progress_")
    failures = []

    for task in selected:
        ts = by_key[task.key]
        ts["status"] = "running"
        ts["t_start"] = time.monotonic()
        # The up-front plan seeded cached/computed as an estimate; once this
        # notebook runs its live cached/computed events are authoritative, so
        # zero the live counters and let the events rebuild them exactly.
        for c in ts["codes"].values():
            c["cached"] = 0
            c["computed"] = 0
            c["obs_times"] = []
        progress_file = os.path.join(tmpdir, task.key.replace("/", "_") + ".jsonl")
        open(progress_file, "w").close()

        env = dict(os.environ)
        env["YAMC_VERIFY_PROGRESS"] = progress_file
        log_path = os.path.join(tmpdir, task.key.replace("/", "_") + ".log")
        log = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "yamc_verify.run_task", task.key],
            cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )

        seen = 0
        while True:
            seen = _drain(progress_file, seen, ts)
            display.update(state, time.monotonic())
            if proc.poll() is not None:
                break
            time.sleep(0.3 if tty else 1.0)
        # final drain of anything written between last poll and exit
        _drain(progress_file, seen, ts)
        log.close()
        _finalize_task(ts)
        if proc.returncode != 0:
            failures.append((task.key, log_path))
        display.update(state, time.monotonic())

    # Leave the completed dashboard on screen, then a compact summary.
    display.update(state, time.monotonic())
    print()
    if failures:
        print(f"{len(failures)} task(s) FAILED:")
        for key, log_path in failures:
            print(f"  {key}   (log: {log_path})")
        return 1
    print(f"all {len(selected)} task(s) completed")
    return 0


def _drain(progress_file: str, seen: int, task_state: dict) -> int:
    """Apply progress lines beyond the first ``seen`` complete lines; return new count.

    Only newline-terminated lines are processed, so a line mid-write is picked
    up on the next tick. Files are tiny, so re-reading each tick is cheap.
    """
    try:
        with open(progress_file) as f:
            lines = f.readlines()
    except OSError:
        return seen
    complete = [ln for ln in lines if ln.endswith("\n")]
    for ln in complete[seen:]:
        ln = ln.strip()
        if ln:
            try:
                _apply_event(task_state, json.loads(ln))
            except json.JSONDecodeError:
                pass
    return len(complete)


if __name__ == "__main__":
    raise SystemExit(run())
