"""The pre-release check: run a fixed set with a candidate build and compare its grades with the published run.

``vv run --release-check <code>...`` runs the project's :class:`~vv.ReleaseCheck`
set (its models on its libraries) with the candidate codes and, beside them,
the reference codes at the version the newest published run has. Every result
the candidate has is then graded (:meth:`vv.Project.grades`) against that
published run's own reference record where it has one, so a grade can only
move because the candidate did, and compared with the grade the published run
gives the same item.

Flagged:

* ``worse``: a lower band than in the published run;
* ``fails``: the published run has a grade, the candidate has no record, and
  the job that should have made it failed;
* ``missing``: the published run has a grade and the candidate has none.

Listed, not flagged: ``better``, and ``new`` (nothing published to compare
with). The check only reports: whether to release stays a human decision.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, TextIO

from .publish import Run, _load_run, _root, collect, published_runs
from .runner import ALL, Code, Grade, Project, ReleaseCheck, RunResult, run

FLAGGED = ("worse", "fails", "missing")
STATUSES = FLAGGED + ("better", "new", "same")


class ReleaseCheckError(RuntimeError):
    """The release check cannot run; the message says why."""


@dataclass
class Setup:
    """A checked release-check request: what runs, and the published run it is compared with."""

    check: ReleaseCheck
    against: Run
    references: list[str]
    """The reference codes, at the versions ``against`` has."""
    candidates: list[Code]

    @property
    def codes(self) -> list[str]:
        """Every code the check runs: the references, then the candidates."""
        return self.references + [c.spec for c in self.candidates]


def setup(project: Project, codes) -> Setup:
    """Check a release-check request before anything runs; raise :class:`ReleaseCheckError` listing every problem."""
    check = project.release_check
    if check is None:
        raise ReleaseCheckError(f"{type(project).__name__} has no release check: set its release_check")
    runs = published_runs(_root(project))
    if not runs:
        raise ReleaseCheckError("nothing is published yet, so there is nothing to check against: "
                                "publish a run first (vv publish)")
    against = _load_run(_root(project), runs[0])
    problems: list[str] = []
    candidates: list[Code] = []
    if codes is None or isinstance(codes, str) or not list(codes):
        problems.append("name the codes to check as a list, each with an exact version or commit")
    else:
        for spec in codes:
            try:
                code = project.parse_code(spec)
            except ValueError as e:
                problems.append(str(e))
                continue
            if code.name in check.references:
                problems.append(f"{code.spec}: {code.name} is what the check grades against, so it "
                                "comes from the published run; name the codes to check")
            elif any(c.spec == code.spec for c in candidates):
                problems.append(f"code {code.spec} is named twice")
            else:
                candidates.append(code)
    published: dict[str, str] = {}
    for spec in against.codes:
        try:
            published.setdefault(project.parse_code(spec).name, spec)
        except ValueError as e:
            problems.append(f"the published run {against.tag} names {spec}, which this project cannot read: {e}")
    references = []
    for name in check.references:
        if name in published:
            references.append(published[name])
        else:
            problems.append(f"the published run {against.tag} has no {name} results to grade against")
    if problems:
        raise ReleaseCheckError("cannot run this release check:\n" + "\n".join(f"  - {p}" for p in problems))
    return Setup(check, against, references, candidates)


@dataclass
class Finding:
    """How one graded result of the candidate compares with the published run."""

    key: tuple[str, str, str, str]
    """``(model, library, label, item)``."""
    status: str
    old: Optional[Grade] = None
    new: Optional[Grade] = None
    note: str = ""
    log: Optional[Path] = None


@dataclass
class Report:
    """The outcome of a release check, from the stored records."""

    setup: Setup
    models: list[str]
    libraries: list[str]
    findings: list[Finding]
    unpublished: list[Code] = field(default_factory=list)
    """Candidate codes whose name the published run has no results for."""

    def count(self, status: str) -> int:
        return sum(f.status == status for f in self.findings)

    @property
    def flagged(self) -> list[Finding]:
        return [f for f in self.findings if f.status in FLAGGED]

    @property
    def ok(self) -> bool:
        """True when nothing got worse, failed or went missing."""
        return not self.flagged

    def format(self) -> str:
        s = self.setup
        lines = [f"release check of {', '.join(c.spec for c in s.candidates)}",
                 f"against {s.against.tag} (published {s.against.date}), graded with its "
                 f"{', '.join(s.references)} records",
                 f"set: {'every model' if s.check.models == ALL else ', '.join(self.models)} "
                 f"on {', '.join(self.libraries)}" + _environment_note(s.check.environment)]
        for code in self.unpublished:
            lines.append(f"the published run has no {code.name} results, so every grade of {code.spec} is new")
        label_width = max((len(f.key[2]) for f in self.findings), default=0)
        item_width = max((len(f.key[3]) for f in self.findings), default=0)
        for status in STATUSES[:-1]:
            found = [f for f in self.findings if f.status == status]
            if not found:
                continue
            lines += ["", f"{status}: {len(found)}"]
            group = None
            for f in found:
                model, library, label, item = f.key
                if (model, library) != group:
                    group = (model, library)
                    lines.append(f"  {model} {library}")
                lines.append(f"    {label:{label_width}s}  {item:{item_width}s}  {_describe(f)}")
        counts = ", ".join(f"{self.count(st)} {st}" for st in STATUSES)
        lines += ["", f"{len(self.findings)} graded results: {counts}"]
        if self.ok:
            lines.append("nothing got worse, failed or went missing")
        else:
            lines.append(f"{len(self.flagged)} flagged: review them before releasing "
                         "(the check only reports; releasing stays your decision)")
        return "\n".join(lines)


def _environment_note(environment: Mapping[str, Optional[str]]) -> str:
    if not environment:
        return ""
    parts = [f"{k}={v}" if v is not None else f"{k} unset" for k, v in environment.items()]
    return f" ({', '.join(parts)})"


@contextlib.contextmanager
def pinned_environment(environment: Mapping[str, Optional[str]]):
    """Set (or, for ``None``, unset) each variable for the duration, then put them back."""
    before = {k: os.environ.get(k) for k in environment}
    try:
        for k, v in environment.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _describe(f: Finding) -> str:
    if f.status in ("fails", "missing"):
        text = f"{f.old.band} -> {f.note}"
        return text + (f" (log: {f.log})" if f.log is not None else "")
    if f.status == "new":
        return f"{f.new.band}" + (f"   {f.new.note}" if f.new.note else "")
    text = f"{f.old.band} -> {f.new.band}"
    if f.old.note or f.new.note:
        text += f"   {f.old.note} -> {f.new.note}"
    return text


def report(project: Project, codes, failed: Optional[Mapping[tuple[str, str, str], Path]] = None) -> Report:
    """Compare the candidate's stored records with the newest published run; run nothing.

    ``failed`` maps the ``(model, library, code name)`` of each job that failed
    to its log, which tells ``fails`` from ``missing``.
    """
    return _report(project, setup(project, codes), failed)


def _report(project: Project, s: Setup, failed: Optional[Mapping[tuple[str, str, str], Path]]) -> Report:
    failed = dict(failed or {})
    current = collect(project, models=ALL if s.check.models == ALL else list(s.check.models),
                      libraries=list(s.check.libraries), codes=s.codes)
    names = {c.name for c in s.candidates}
    references = set(s.check.references)

    def in_set(e) -> bool:
        return e.model in current.models and e.library in current.libraries

    published_refs = {e.key: e for e in s.against.entries if in_set(e) and e.code in references}
    candidate = [e for e in current.entries if e.code in names]
    # The candidate is graded against the published references wherever there
    # is one, so that only the candidate can move a grade.
    graded_now = Run(current.codes, current.models, current.libraries, sorted(
        candidate + list(published_refs.values())
        + [e for e in current.entries if e.code in references and e.key not in published_refs],
        key=lambda e: e.key))
    graded_then = Run(s.against.codes, current.models, current.libraries,
                      [e for e in s.against.entries if in_set(e) and e.code in names | references],
                      s.against.tag, s.against.date)
    old, new = project.grades(graded_then), project.grades(graded_now)

    bands = list(s.check.bands)

    def rank(grade: Grade) -> int:
        if grade.band not in bands:
            raise ReleaseCheckError(f"{type(project).__name__}.grades gave the band {grade.band!r}, "
                                    f"which is not one of its release check's bands {bands}")
        return bands.index(grade.band)

    code_of = {(e.model, e.library, e.label): e.code for e in s.against.entries + candidate}
    has_record = {e.key for e in candidate}
    findings = []
    for key in sorted(old.keys() | new.keys()):
        o, n = old.get(key), new.get(key)
        if o is not None and n is not None:
            step = rank(n) - rank(o)
            findings.append(Finding(key, "worse" if step > 0 else "better" if step < 0 else "same", o, n))
        elif n is not None:
            rank(n)
            findings.append(Finding(key, "new", new=n))
        else:
            rank(o)
            log = failed.get((key[0], key[1], code_of.get(key[:3])))
            if key in has_record:
                findings.append(Finding(key, "missing", o, note="a record, but nothing to grade it against"))
            elif log is not None:
                findings.append(Finding(key, "fails", o, note="no record: its job failed", log=log))
            else:
                findings.append(Finding(key, "missing", o, note="no record"))
    unpublished = [c for c in s.candidates if not any(e.code == c.name for e in s.against.entries)]
    return Report(s, current.models, current.libraries, findings, unpublished)


def release_check(project: Project, codes, *, stream: Optional[TextIO] = None) -> tuple[RunResult, Report]:
    """Run the project's release-check set with ``codes``, then compare it with the newest published run.

    The references run at the published run's versions, so their records
    usually exist already and only the candidates run. The check's
    :attr:`~vv.ReleaseCheck.environment` is set while it plans and runs, so
    it chooses the same items every time. Returns the run and the
    :class:`Report`; neither stops a release.
    """
    s = setup(project, codes)
    models = ALL if s.check.models == ALL else list(s.check.models)
    with pinned_environment(s.check.environment):
        result = run(project, models=models, libraries=list(s.check.libraries), codes=s.codes, stream=stream)
    failed = {(job.model, job.library, job.code.name): log for job, log in result.failures}
    return result, _report(project, s, failed)


__all__ = ["FLAGGED", "Finding", "ReleaseCheckError", "Report", "STATUSES", "Setup", "pinned_environment",
           "release_check", "report", "setup"]
