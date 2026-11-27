"""Plan and run a verification: models x libraries x codes, each code in its own environment.

vv knows nothing about any particular code. A repository describes its models,
libraries and codes with a :class:`Project`, and a code is just a name plus the
recipe for its environment (:class:`Code`). vv then:

1. checks the request up front (:func:`select`): every argument is required,
   and anything unknown or missing stops it with a list of what is available;
2. builds the plan (:func:`plan`): one :class:`Job` per model, library and code,
   with the items each will run, how many already have a record, and a rough
   time from the times earlier records took;
3. runs it (:func:`run`): every code's environment is made ready first, then each
   job runs as a separate process inside its code's environment while the live
   display (:mod:`vv.dashboard`) follows its progress events (:mod:`vv.progress`).
"""
from __future__ import annotations

import abc
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, TextIO, Union

from . import dashboard, progress

ALL = "all"
"""Pass ``models="all"`` to run every model defined on the chosen libraries."""


class SelectionError(ValueError):
    """The requested models, libraries or codes cannot be run; the message lists what can."""


@dataclass(frozen=True)
class Code:
    """One code to verify: a name plus the recipe for its environment.

    Attributes
    ----------
    spec : str
        The code as requested, for example ``"code-a==1.2.0"``. It names the
        code in the plan and on the display.
    name : str
        The name its results are filed under in the records.
    environment : callable
        Builds the code's environment if needed and returns its Python
        interpreter. Called only by :func:`run`, never by :func:`plan`.
    is_ready : callable
        Whether the environment is already built.
    check : callable, optional
        Raises ``ValueError``, listing what is available, when the code cannot
        be installed (an unknown version, say). Called before anything runs,
        and only while the environment is not built.
    build_seconds : float, optional
        Rough time to build the environment, for the estimate.
    environment_key : str, optional
        Codes with the same key share one environment (two backends of one
        build, say), which is then built and counted once. Defaults to ``spec``.
    """

    spec: str
    name: str
    environment: Callable[[], Union[Path, str]]
    is_ready: Callable[[], bool]
    check: Optional[Callable[[], None]] = None
    build_seconds: Optional[float] = None
    environment_key: Optional[str] = None

    @property
    def key(self) -> str:
        """The environment this code runs in (see ``environment_key``)."""
        return self.environment_key or self.spec


@dataclass
class Row:
    """One line of the plan and the display: the items one code files under one name.

    ``label`` is the name the results are filed under, which the progress events
    carry. ``done`` counts the items that already have a record this code would
    accept, and ``seconds_per_item`` is the average time earlier records took.
    ``name`` is filled in by vv from the code unless the project sets it.
    """

    label: str
    items: list[str]
    done: int = 0
    seconds_per_item: Optional[float] = None
    name: str = ""


@dataclass
class Job:
    """The runs of one code on one model and library: one process in the code's environment.

    ``code`` is ``None`` for a job that runs whatever is installed in the
    current environment (see :func:`run_jobs`).
    """

    model: str
    library: str
    code: Optional[Code]
    rows: list[Row]

    @property
    def group(self) -> str:
        """The heading this job's rows appear under."""
        return f"{self.model} {self.library}"


@dataclass(frozen=True)
class Command:
    """How to start one job: its arguments, extra environment variables and directory."""

    argv: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: Optional[Union[Path, str]] = None


class Project(abc.ABC):
    """What a repository supplies so vv can plan and run its verification.

    Subclass it and implement the abstract methods; :attr:`name` heads the
    display. Nothing here knows which codes the project compares.
    """

    name: str = "verification"
    root: Optional[Path] = None
    """The repository the records live in, which comparing and publishing need.
    ``vv`` sets it to the folder holding ``vv.toml`` when the project leaves it unset."""

    @abc.abstractmethod
    def models(self) -> list[str]:
        """Every model, in display order."""

    @abc.abstractmethod
    def libraries(self) -> dict[str, Optional[str]]:
        """Every library, mapped to ``None`` if it can be used or to why it cannot."""

    @abc.abstractmethod
    def defined(self, model: str, library: str) -> bool:
        """Whether ``model`` is defined on ``library``."""

    @abc.abstractmethod
    def parse_code(self, spec: str) -> Code:
        """The :class:`Code` for ``spec``; raise ``ValueError`` naming the accepted forms."""

    def runs(self, model: str, library: str, code: Code) -> bool:
        """Whether ``model`` on ``library`` runs ``code`` at all (default: always)."""
        return True

    @abc.abstractmethod
    def rows(self, model: str, library: str, code: Code, codes: Sequence[Code]) -> list[Row]:
        """The rows of one job: its items, what is already done, and earlier timings.

        ``codes`` is every code in the verification, for projects whose item
        lists depend on which codes take part.
        """

    @abc.abstractmethod
    def command(self, job: Job, codes: Sequence[Code], python: Path) -> Command:
        """How to run ``job`` with ``python``, the interpreter of its code's environment."""

    def fixed_records(self, model: str, library: str) -> list[Path]:
        """Records that come from no code, a measurement say, and belong to every run of ``model`` on ``library``.

        Comparing and publishing include them beside the codes' records,
        filed under their own name (each record's ``code``), whenever a code
        in the run runs the model. Default: none.
        """
        return []

    def records(self, model: str, library: str, code: Code) -> list[Path]:
        """The record of each item that ``code`` has on ``model`` and ``library``, from the stored records alone.

        One path per item and label: the record a run of ``code`` would use.
        Needed by :mod:`vv.publish` (comparing and publishing), not by planning
        or running.
        """
        raise NotImplementedError(f"{type(self).__name__} does not say which records each code has")


@dataclass
class Selection:
    """A checked request: the models, libraries and codes, and the pairs defined."""

    models: list[str]
    libraries: list[str]
    codes: list[Code]
    pairs: list[tuple[str, str]]
    skipped: list[str]


@dataclass
class Plan:
    """What a verification would run, what already has records, and a rough time."""

    jobs: list[Job]
    codes: list[Code]
    skipped: list[str]

    def environments_to_build(self) -> list[Code]:
        """One code per environment a run would build first."""
        used = {job.code.spec for job in self.jobs if job.code is not None}
        out: dict[str, Code] = {}
        for c in self.codes:
            if c.spec in used and c.key not in out and not c.is_ready():
                out[c.key] = c
        return list(out.values())

    def state(self, header: Sequence[str] = ()) -> dict:
        """The display state before anything runs (see :mod:`vv.dashboard`)."""
        return build_state(self.jobs, header)

    def estimate(self) -> tuple[float, int]:
        """Rough seconds to run it, and how many rows have no earlier timing to go on.

        Includes building the environments that are not ready.
        """
        seconds, unknown = dashboard.remaining_seconds(self.state())
        for code in self.environments_to_build():
            if code.build_seconds is None:
                unknown += 1
            else:
                seconds += code.build_seconds
        return seconds, unknown

    def format(self) -> str:
        """The plan as text: environments, rows (done and to do) and the estimate."""
        lines = ["codes:"]
        to_build = {c.key for c in self.environments_to_build()}
        width = max((len(c.spec) for c in self.codes), default=0)
        for code in self.codes:
            if code.key in to_build:
                note = "environment to build"
                if code.build_seconds is not None:
                    note += f" (about {dashboard.humanize(code.build_seconds)})"
            else:
                note = "environment ready"
            if code.key != code.spec:
                note += f", shared as {code.key}"
            lines.append(f"    {code.spec:{width}s}  {note}")
        lines.append("")
        lines += dashboard.render_lines(self.state(), time.monotonic())
        if self.skipped:
            lines += ["", "not run:"] + [f"    {s}" for s in self.skipped]
        seconds, unknown = self.estimate()
        if seconds == 0 and not unknown:
            return "\n".join(lines + ["", "estimate: nothing left to run"])
        estimate = f"estimate: about {dashboard.humanize(seconds)}"
        if to_build:
            estimate += f", including building {len(to_build)} environment(s)"
        if unknown:
            estimate += f"; {unknown} row(s) have no earlier timing, so it is a lower bound"
        lines += ["", estimate]
        return "\n".join(lines)


@dataclass
class RunResult:
    """The outcome of :func:`run`: its plan and the jobs that failed, with their logs."""

    plan: Plan
    failures: list[tuple[Job, Path]]

    @property
    def ok(self) -> bool:
        """True when every job finished without an error."""
        return not self.failures


def _names(kind: str, value, available: Sequence[str], problems: list[str]) -> list[str]:
    """Check one required list argument against what is available."""
    if value is None or isinstance(value, str) or not list(value):
        given = "a single string" if isinstance(value, str) else "nothing"
        problems.append(f"{kind}: got {given}; name them as a list. Available: {list(available)}")
        return []
    out: list[str] = []
    for v in value:
        if v not in out:
            out.append(v)
    unknown = [v for v in out if v not in available]
    if unknown:
        problems.append(f"unknown {kind} {unknown}. Available: {list(available)}")
    return [v for v in out if v in available]


def select(project: Project, *, models, libraries, codes, records_only: bool = False) -> Selection:
    """Check a request before anything runs; raise :class:`SelectionError` listing every problem.

    ``models`` is a list of model names or ``"all"``; ``libraries`` and ``codes``
    are lists. None has a default. Unknown models, libraries or codes, libraries
    whose data is missing, and codes that cannot be installed all stop it, with
    a list of what is available. ``records_only`` is for work on the stored
    records alone (comparing and publishing them): the libraries' data and the
    codes' installability are then not needed, so not checked.
    """
    problems: list[str] = []
    known_models = list(project.models())
    if isinstance(models, str) and models == ALL:
        chosen_models = known_models
    else:
        chosen_models = _names("models", models, known_models, problems)

    known_libraries = project.libraries()
    usable = [lib for lib, why in known_libraries.items() if why is None]
    chosen_libraries = _names("libraries", libraries, list(known_libraries), problems)
    for lib in list(chosen_libraries):
        why = known_libraries[lib]
        if why is not None and not records_only:
            problems.append(f"library {lib} is missing: {why}. Available: {usable}")
            chosen_libraries.remove(lib)

    chosen_codes: list[Code] = []
    if codes is None or isinstance(codes, str) or not list(codes):
        given = "a single string" if isinstance(codes, str) else "nothing"
        problems.append(f"codes: got {given}; name each code and its version as a list")
    else:
        for spec in codes:
            try:
                code = project.parse_code(spec)
            except ValueError as e:
                problems.append(str(e))
                continue
            if any(c.spec == code.spec for c in chosen_codes):
                problems.append(f"code {code.spec} is named twice")
                continue
            chosen_codes.append(code)

    pairs, skipped = [], []
    for model in chosen_models:
        for lib in chosen_libraries:
            if project.defined(model, lib):
                pairs.append((model, lib))
            else:
                skipped.append(f"{model} is not defined on {lib}")
    if chosen_libraries:
        for model in chosen_models:
            if not any(m == model for m, _ in pairs):
                on = [lib for lib in known_libraries if project.defined(model, lib)]
                problems.append(f"model {model} is not defined on {chosen_libraries}; it is defined on {on}")
    if chosen_models:
        for lib in chosen_libraries:
            if not any(lb == lib for _, lb in pairs):
                problems.append(f"library {lib} has none of the models {chosen_models}")

    for code in chosen_codes:
        if code.check is not None and not records_only and not code.is_ready():
            try:
                code.check()
            except ValueError as e:
                problems.append(str(e))

    if problems:
        raise SelectionError("cannot run this verification:\n" + "\n".join(f"  - {p}" for p in problems))
    return Selection(chosen_models, chosen_libraries, chosen_codes, pairs, skipped)


def make_plan(project: Project, selection: Selection) -> Plan:
    """The plan for a checked selection (see :func:`plan`)."""
    jobs: list[Job] = []
    skipped = list(selection.skipped)
    for model, lib in selection.pairs:
        for code in selection.codes:
            if not project.runs(model, lib, code):
                skipped.append(f"{model} on {lib} does not run {code.name}")
                continue
            try:
                rows = project.rows(model, lib, code, selection.codes)
            except Exception as e:
                raise SelectionError(f"cannot plan {model} on {lib} for {code.spec}: {e}") from e
            for row in rows:
                if not row.name:
                    row.name = code.spec if row.label == code.name else f"{code.spec} ({row.label})"
            jobs.append(Job(model, lib, code, rows))
    if not jobs:
        raise SelectionError("nothing to run:\n" + "\n".join(f"  - {s}" for s in skipped))
    return Plan(jobs, list(selection.codes), skipped)


def plan(project: Project, *, models, libraries, codes) -> Plan:
    """What :func:`run` would run, what already has records, and a rough time; builds nothing."""
    return make_plan(project, select(project, models=models, libraries=libraries, codes=codes))


def run(project: Project, *, models, libraries, codes, stream: Optional[TextIO] = None) -> RunResult:
    """Run a verification with the live display and return what failed.

    Every argument is required (see :func:`select`). Each code's environment is
    made ready before any simulation starts, so a code that cannot be built
    stops the run at the start. Each job then runs as a separate process with its
    code's interpreter.
    """
    out = stream or sys.stdout
    the_plan = plan(project, models=models, libraries=libraries, codes=codes)
    used = {job.code.spec for job in the_plan.jobs}
    pythons: dict[str, Path] = {}
    for code in the_plan.codes:
        if code.spec not in used or code.key in pythons:
            continue
        if not code.is_ready():
            print(f"building the environment for {code.spec}", file=out, flush=True)
        pythons[code.key] = Path(code.environment())
    commands = [project.command(job, the_plan.codes, pythons[job.code.key]) for job in the_plan.jobs]
    header = [f"{project.name}: " + ", ".join(c.spec for c in the_plan.codes if c.spec in used)]
    failures = run_jobs(the_plan.jobs, commands, header=header, stream=out)
    return RunResult(the_plan, failures)


def build_state(jobs: Sequence[Job], header: Sequence[str] = ()) -> dict:
    """The display state for ``jobs``: one heading per model and library, one line per row."""
    tasks: dict[str, dict] = {}
    for job in jobs:
        task = tasks.setdefault(job.group, {
            "key": job.group, "label": job.group, "status": "queued",
            "t_start": None, "t_end": None, "codes": {},
        })
        for row in job.rows:
            task["codes"][row.name or row.label] = {
                "total": len(row.items),
                "cached": row.done,
                "computed": 0,
                "running_key": None,
                "status": "queued",
                "t_start": None,
                "t_end": None,
                "obs_times": [],
                "seed_avg": row.seconds_per_item,
            }
    return {"header": list(header), "tasks": list(tasks.values())}


def apply_event(rows: Mapping[str, dict], ev: Mapping) -> None:
    """Fold one progress event into the rows of the job that sent it.

    ``rows`` maps the label the events carry to that row's display state.
    """
    t = ev.get("t")
    if t == "plan":
        # The job's own item count is authoritative: correct the plan's estimate.
        n = len(ev.get("items", []))
        for c in rows.values():
            c["total"] = n
        return
    c = rows.get(ev.get("code"))
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


def drain(progress_file: Path, seen: int, rows: Mapping[str, dict]) -> int:
    """Apply the progress lines after the first ``seen`` complete ones; return the new count.

    Only newline-terminated lines are read, so a line mid-write is picked up on
    the next call.
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
                apply_event(rows, json.loads(ln))
            except json.JSONDecodeError:
                pass
    return len(complete)


def run_jobs(jobs: Sequence[Job], commands: Sequence[Command], *, header: Sequence[str] = (),
             stream: Optional[TextIO] = None) -> list[tuple[Job, Path]]:
    """Run each job's command in turn with the live display; return the failed jobs and logs.

    Each process gets ``VV_PROGRESS`` naming its own progress file, and its
    output goes to a log file so nothing scrolls.
    """
    out = stream or sys.stdout
    state = build_state(jobs, header)
    by_group = {t["label"]: t for t in state["tasks"]}
    last_of_group = {job.group: i for i, job in enumerate(jobs)}
    display = dashboard.LiveDisplay(out)
    display.update(state, time.monotonic())
    logs = Path(tempfile.mkdtemp(prefix="vv-run-"))
    failures: list[tuple[Job, Path]] = []

    for i, (job, cmd) in enumerate(zip(jobs, commands)):
        task = by_group[job.group]
        if task["status"] == "queued":
            task["status"] = "running"
            task["t_start"] = time.monotonic()
        rows = {row.label: task["codes"][row.name or row.label] for row in job.rows}
        # The plan's counts were an estimate; the job's own events rebuild them.
        for c in rows.values():
            c["cached"] = 0
            c["computed"] = 0
            c["obs_times"] = []
        progress_file = logs / f"{i:03d}.jsonl"
        progress_file.write_text("")
        log_path = logs / f"{i:03d}.log"
        env = {**os.environ, **cmd.env, progress.ENV_VAR: str(progress_file)}
        with open(log_path, "w") as log:
            proc = subprocess.Popen([str(a) for a in cmd.argv], cwd=cmd.cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            seen = 0
            while True:
                seen = drain(progress_file, seen, rows)
                display.update(state, time.monotonic())
                if proc.poll() is not None:
                    break
                time.sleep(0.3 if display._tty else 1.0)
            drain(progress_file, seen, rows)
        now = time.monotonic()
        for c in rows.values():
            if c["status"] != "done":
                c["status"] = "done"
                c["t_end"] = now
                c["running_key"] = None
        if last_of_group[job.group] == i:
            task["status"] = "done"
            task["t_end"] = now
        if proc.returncode != 0:
            failures.append((job, log_path))
        display.update(state, time.monotonic())

    print(file=out)
    if failures:
        print(f"{len(failures)} job(s) FAILED:", file=out)
        for job, log_path in failures:
            what = job.code.spec if job.code is not None else "the installed codes"
            print(f"  {job.group} with {what}   (log: {log_path})", file=out)
    else:
        print(f"all {len(jobs)} job(s) completed", file=out)
    return failures
