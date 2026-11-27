"""Published runs: compare the stored records with what the site shows, publish them, archive them.

A *run* is the record each item has for a set of codes on a set of models and
libraries: what :meth:`vv.Project.records` picks from the stored records, with
no simulation, nuclear data or environment needed. The records stay in git.

Two files in the repository say what has been published:

* ``runs/<tag>.json``, the manifest of one run: its codes, models and
  libraries, and every record in it, each with its path and git blob id, so
  the exact bytes can be read back from git even after a record is replaced;
* ``published.json``, the small file naming the runs, newest first. The site
  shows the first two.

:func:`compare` shows what changed between the records the codes have now and
the newest published run. :func:`publish` refuses dirty records, writes the
manifest and the new ``published.json`` onto a branch from the default branch
(the working tree and the current branch are not touched) and opens a pull
request. When it merges, :func:`release` (run by CI) tags the merge commit and
attaches the run's records as one archive to a GitHub release: a frozen
snapshot that also lets old runs be pruned from git later.
"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .records import decode, read_record
from .runner import Project, select

SCHEMA = 1
PUBLISHED = "published.json"
RUNS = "runs"
SHOWN = 2
"""How many of the newest runs the site shows: the latest and the one before it."""


class PublishError(RuntimeError):
    """The records cannot be published; the message says why."""


def blob_id(data: bytes) -> str:
    """The id git gives ``data`` as a blob, so a record's exact bytes can be found in git."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git(root: Path, *args: str, input: Optional[bytes] = None, env: Optional[dict] = None) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], input=input, capture_output=True,
                         env={**os.environ, **(env or {})})
    if out.returncode != 0:
        raise PublishError(f"git {' '.join(args)} failed:\n{out.stderr.decode().strip()}")
    return out.stdout.decode()


def _root(project: Project) -> Path:
    if project.root is None:
        raise PublishError(f"{type(project).__name__} has no root: set Project.root to the repository folder")
    return Path(project.root).resolve()


@dataclass(frozen=True)
class Entry:
    """One record in a run."""

    model: str
    library: str
    code: str      # the code's name, e.g. yamc-cpu
    label: str     # the name its results are filed under, e.g. yamc-cpu-ttb
    item: str
    build_id: str
    commit: Optional[str]
    dirty: Optional[bool]
    path: str      # relative to the repository, with / separators
    blob: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.model, self.library, self.label, self.item

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Run:
    """A run: its codes, models and libraries, and the record of each item."""

    codes: list[str]
    models: list[str]
    libraries: list[str]
    entries: list[Entry]
    tag: Optional[str] = None
    date: Optional[str] = None
    code_names: Optional[list[str]] = None
    """The codes' names, which say which published records are in scope (from the entries if unset)."""

    def manifest(self) -> dict:
        return {"schema": SCHEMA, "tag": self.tag, "date": self.date, "codes": self.codes,
                "models": self.models, "libraries": self.libraries,
                "records": [e.to_json() for e in self.entries]}

    @classmethod
    def from_manifest(cls, data: dict) -> "Run":
        return cls(data["codes"], data["models"], data["libraries"],
                   [Entry(**e) for e in data["records"]], data.get("tag"), data.get("date"))

    def summary(self) -> dict:
        """The run's line in ``published.json``."""
        return {"tag": self.tag, "date": self.date, "codes": self.codes, "models": self.models,
                "libraries": self.libraries, "records": len(self.entries),
                "manifest": f"{RUNS}/{self.tag}.json"}


def collect(project: Project, *, models, libraries, codes) -> Run:
    """The run the codes have now: the record :meth:`Project.records` picks for each item.

    Every argument is required, as for :func:`vv.plan`, but nothing is installed
    or run and no nuclear data is needed.
    """
    root = _root(project)
    sel = select(project, models=models, libraries=libraries, codes=codes, records_only=True)
    entries = []
    for model, library in sel.pairs:
        for code in sel.codes:
            if not project.runs(model, library, code):
                continue
            for path in project.records(model, library, code):
                path = Path(path).resolve()
                data = path.read_bytes()
                record = read_record(path)
                if record is None:
                    raise PublishError(f"{path} is not a readable record")
                ci = record.get("code_identity", {})
                entries.append(Entry(model, library, code.name, record["code"], record["item"],
                                     ci["build_id"], ci.get("commit"), ci.get("dirty"),
                                     path.relative_to(root).as_posix(), blob_id(data)))
    entries.sort(key=lambda e: e.key)
    return Run([c.spec for c in sel.codes], sel.models, sel.libraries, entries,
               code_names=[c.name for c in sel.codes])


def _published(text: Optional[str]) -> list[dict]:
    return json.loads(text)["runs"] if text else []


def published_runs(root: Path | str) -> list[dict]:
    """The runs ``published.json`` names in the working tree, newest first."""
    path = Path(root) / PUBLISHED
    return _published(path.read_text() if path.exists() else None)


def _show(root: Path, ref: str, path: str) -> Optional[str]:
    """``path`` at ``ref``, or ``None`` if it is not there."""
    out = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{path}"], capture_output=True)
    return out.stdout.decode() if out.returncode == 0 else None


def _record_bytes(root: Path, entry: Entry) -> bytes:
    """The exact bytes of a published record: from the working tree if unchanged, else from git."""
    path = root / entry.path
    if path.exists():
        data = path.read_bytes()
        if blob_id(data) == entry.blob:
            return data
    out = subprocess.run(["git", "-C", str(root), "cat-file", "blob", entry.blob], capture_output=True)
    if out.returncode != 0:
        raise PublishError(f"the published record {entry.path} (blob {entry.blob}) is neither in the "
                           "working tree nor in git; fetch the history or its run's release archive")
    return out.stdout


# --- comparing ---------------------------------------------------------------


def _leaves(obj, prefix=""):
    """``(path, array)`` for every numeric value in a decoded results dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)) and obj and all(isinstance(v, dict) for v in obj):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        try:
            arr = np.asarray(obj, dtype=float)
        except (TypeError, ValueError):
            return
        yield prefix, arr


def results_difference(old: dict, new: dict) -> tuple[float, str]:
    """The largest relative difference between two results dicts, and where it is.

    Relative to the larger magnitude of the two values; values that are zero in
    both count as equal and NaN in both as equal. A value only one of them has,
    or whose shape changed, counts as a difference of ``inf``.
    """
    a, b = dict(_leaves(old)), dict(_leaves(new))
    worst, where = 0.0, ""
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b or a[key].shape != b[key].shape:
            return float("inf"), key
        x, y = a[key], b[key]
        both_nan = np.isnan(x) & np.isnan(y)
        scale = np.maximum(np.abs(x), np.abs(y))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(both_nan | (scale == 0), 0.0, np.abs(x - y) / scale)
        rel = np.nan_to_num(rel, nan=np.inf)
        if rel.size and rel.max() > worst:
            worst, where = float(rel.max()), key
    return worst, where


@dataclass
class Change:
    """How one item's record differs from the published run."""

    key: tuple[str, str, str, str]
    status: str  # new, missing, changed, rerun (same results, new record) or same
    new: Optional[Entry] = None
    old: Optional[Entry] = None
    difference: float = 0.0
    where: str = ""


@dataclass
class Comparison:
    """What changed between the records the codes have now and a published run."""

    run: Run
    against: Optional[Run]
    changes: list[Change]

    def count(self, status: str) -> int:
        return sum(c.status == status for c in self.changes)

    def format(self) -> str:
        if self.against is None:
            lines = ["nothing is published yet: every record is new"]
        else:
            lines = [f"compared with {self.against.tag} (published {self.against.date}, "
                     f"codes {', '.join(self.against.codes)})"]
        group = None
        for c in self.changes:
            if c.status == "same":
                continue
            model, library, label, item = c.key
            if (model, library) != group:
                group = (model, library)
                lines += ["", f"{model} {library}"]
            if c.status == "new":
                note = c.new.build_id
            elif c.status == "missing":
                note = f"was {c.old.build_id}"
            else:
                note = f"{c.new.build_id}, was {c.old.build_id}"
                if c.status == "changed":
                    diff = "a value added, removed or reshaped" if c.difference == float("inf") \
                        else f"largest relative difference {c.difference:.3g}"
                    note += f"; {diff} in {c.where}"
            lines.append(f"    {label:14s} {item:12s} {c.status:8s} {note}")
        counts = ", ".join(f"{self.count(s)} {s}" for s in ("new", "missing", "changed", "rerun", "same"))
        return "\n".join(lines + ["", f"{len(self.changes)} records: {counts}"])


def compare_runs(root: Path, run: Run, against: Optional[Run]) -> Comparison:
    """Compare ``run`` with the published run ``against`` over the models, libraries and codes of ``run``."""
    root = Path(root)
    names = set(run.code_names if run.code_names is not None else (e.code for e in run.entries))
    old = {} if against is None else {e.key: e for e in against.entries
                                      if e.model in run.models and e.library in run.libraries and e.code in names}
    new = {e.key: e for e in run.entries}
    changes = []
    for key in sorted(old.keys() | new.keys()):
        n, o = new.get(key), old.get(key)
        if o is None:
            changes.append(Change(key, "new", new=n))
        elif n is None:
            changes.append(Change(key, "missing", old=o))
        elif n.blob == o.blob:
            changes.append(Change(key, "same", n, o))
        else:
            before = decode(json.loads(_record_bytes(root, o)).get("results", {}))
            after = decode(json.loads((root / n.path).read_bytes()).get("results", {}))
            diff, where = results_difference(before, after)
            changes.append(Change(key, "changed" if diff > 0 else "rerun", n, o, diff, where))
    return Comparison(run, against, changes)


def _load_run(root: Path, summary: dict, ref: Optional[str] = None) -> Run:
    text = _show(root, ref, summary["manifest"]) if ref else (root / summary["manifest"]).read_text()
    if text is None:
        raise PublishError(f"{summary['manifest']} is not on {ref}")
    return Run.from_manifest(json.loads(text))


def compare(project: Project, *, models, libraries, codes) -> Comparison:
    """What changed between the records the codes have now and the newest published run."""
    root = _root(project)
    run = collect(project, models=models, libraries=libraries, codes=codes)
    runs = published_runs(root)
    return compare_runs(root, run, _load_run(root, runs[0]) if runs else None)


# --- publishing --------------------------------------------------------------


def _default_branch(root: Path) -> str:
    out = subprocess.run(["git", "-C", str(root), "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip().split("/", 1)[1] if out.returncode == 0 else "main"


def run_tag(codes, date: str, taken) -> str:
    """``vv-<date>-<codes>``, e.g. ``vv-2026-11-27-openmc-0.16.1.dev0-yamc-cpu-0.24.0``, unique among ``taken``."""
    names = [re.sub(r"[^A-Za-z0-9._-]", "-", c.replace("==", "-").replace("@", "-g")) for c in codes]
    tag = f"vv-{date}-" + "-".join(names)
    out, n = tag, 1
    while out in taken:
        n += 1
        out = f"{tag}.{n}"
    return out


def open_pull_request(root: Path, base: str, branch: str, title: str, body: str) -> str:
    """Open a pull request of ``branch`` into ``base`` with the GitHub CLI; return its URL."""
    out = subprocess.run(["gh", "pr", "create", "--base", base, "--head", branch, "--title", title,
                          "--body", body], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        raise PublishError(f"the branch {branch} is pushed, but opening its pull request failed:\n"
                           f"{out.stderr.strip()}")
    return out.stdout.strip().splitlines()[-1]


def _commit(root: Path, base_ref: str, records: list[str], files: dict[str, bytes], message: str) -> str:
    """A commit on ``base_ref`` adding ``records`` from the working tree and ``files``; nothing checked out changes."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git(root, "read-tree", base_ref, env=env)
        shas = _git(root, "hash-object", "-w", "--stdin-paths",
                    input="\n".join(records).encode() + b"\n").split() if records else []
        index = [f"100644 {sha}\t{path}" for sha, path in zip(shas, records)]
        for path, data in files.items():
            index.append(f"100644 {_git(root, 'hash-object', '-w', '--stdin', input=data).strip()}\t{path}")
        _git(root, "update-index", "--index-info", input=("\n".join(index) + "\n").encode(), env=env)
        tree = _git(root, "write-tree", env=env).strip()
    return _git(root, "commit-tree", tree, "-p", base_ref, "-m", message).strip()


@dataclass
class Published:
    """A publishing pull request, and the records it holds that were not committed in the checkout."""

    url: str
    uncommitted: list[str]

    def format(self) -> str:
        if not self.uncommitted:
            return self.url
        return (f"{self.url}\n{len(self.uncommitted)} record(s) were not committed here and are on its branch: "
                "remove them, or check out that branch, before pulling once it merges")


def publish(project: Project, *, models, libraries, codes) -> Published:
    """Open a pull request publishing the records the codes have now.

    Refuses dirty records, and a run whose records are exactly the newest
    published run's. The branch is made from the default branch on ``origin``
    without touching the working tree, and holds the records (committed or
    not), the run's manifest and the new ``published.json``.
    """
    root = _root(project)
    run = collect(project, models=models, libraries=libraries, codes=codes)
    if not run.entries:
        raise PublishError("these codes have no records on these models and libraries: run them first")
    dirty = [e for e in run.entries if e.dirty]
    if dirty:
        raise PublishError(f"{len(dirty)} record(s) come from builds with uncommitted changes; rerun them "
                           "from a committed build:\n" + "\n".join(f"  - {e.path}" for e in dirty))
    base = _default_branch(root)
    _git(root, "fetch", "--quiet", "origin", base)
    base_ref = f"origin/{base}"
    runs = _published(_show(root, base_ref, PUBLISHED))
    previous = _load_run(root, runs[0], base_ref) if runs else None
    comparison = compare_runs(root, run, previous)
    if previous is not None and all(c.status == "same" for c in comparison.changes) \
            and {e.key for e in run.entries} == {e.key for e in previous.entries}:
        raise PublishError(f"nothing to publish: these are exactly the records of {previous.tag}")

    remote_tags = {line.rsplit("/", 1)[-1] for line in _git(root, "ls-remote", "--tags", "--refs", "origin").splitlines()}
    now = datetime.datetime.now(datetime.timezone.utc)
    run.date = now.isoformat(timespec="seconds")
    run.tag = run_tag(run.codes, now.date().isoformat(), remote_tags | {r["tag"] for r in runs})
    files = {
        f"{RUNS}/{run.tag}.json": (json.dumps(run.manifest(), indent=1) + "\n").encode(),
        PUBLISHED: (json.dumps({"schema": SCHEMA, "runs": [run.summary(), *runs]}, indent=2) + "\n").encode(),
    }
    title = f"Publish {run.tag}"
    commit = _commit(root, base_ref, [e.path for e in run.entries], files, title)
    branch = f"vv-publish/{run.tag}"
    _git(root, "push", "--quiet", "origin", f"{commit}:refs/heads/{branch}")
    url = open_pull_request(root, base, branch, title, _body(run, comparison))
    tracked = set(_git(root, "ls-files", "--", *[e.path for e in run.entries]).splitlines())
    changed = set(_git(root, "diff", "--name-only", "HEAD", "--", *[e.path for e in run.entries]).splitlines())
    return Published(url, [e.path for e in run.entries if e.path not in tracked or e.path in changed])


def _body(run: Run, comparison: Comparison) -> str:
    unknown = sum(e.commit is None for e in run.entries)
    lines = [f"Publishes {len(run.entries)} records as `{run.tag}`, run with {', '.join(f'`{c}`' for c in run.codes)} "
             f"on {', '.join(run.libraries)}.", "",
             "When this merges, CI tags the merge commit and attaches the run's records to a release of that tag.", ""]
    if comparison.against is None:
        lines.append("This is the first published run.")
    else:
        counts = ", ".join(f"{comparison.count(s)} {s}" for s in ("new", "missing", "changed", "rerun", "same"))
        lines.append(f"Against `{comparison.against.tag}`: {counts}.")
    if unknown:
        lines += ["", f"{unknown} record(s) have no recorded commit (their build cannot be traced to a commit)."]
    detail = comparison.format()
    if len(detail) < 60000:
        lines += ["", "<details><summary>Every change</summary>", "", "```text", detail, "```", "", "</details>"]
    return "\n".join(lines)


# --- archiving and releasing ---------------------------------------------------


def archive(root: Path | str, tag: str, out_dir: Path | str = ".") -> Path:
    """Write ``<tag>.tar.gz``, the run's manifest and the exact bytes of its records; return its path."""
    root = Path(root).resolve()
    manifest_path = f"{RUNS}/{tag}.json"
    manifest = (root / manifest_path).read_bytes()
    run = Run.from_manifest(json.loads(manifest))
    out = Path(out_dir) / f"{tag}.tar.gz"
    with open(out, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for name, data in [(manifest_path, manifest)] + [(e.path, _record_bytes(root, e)) for e in run.entries]:
            info = tarfile.TarInfo(f"{tag}/{name}")
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return out


def release(root: Path | str) -> list[str]:
    """Make a GitHub release, with its archive, for every published run that has none; return their tags.

    Run in CI on the default branch after a publishing pull request merges:
    each new release tags the commit checked out.
    """
    root = Path(root).resolve()
    made = []
    for summary in published_runs(root):
        tag = summary["tag"]
        if subprocess.run(["gh", "release", "view", tag], cwd=root, capture_output=True).returncode == 0:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = archive(root, tag, tmp)
            notes = (f"{summary['records']} records run with {', '.join(summary['codes'])} on "
                     f"{', '.join(summary['libraries'])}. The manifest is `{summary['manifest']}`.")
            sha = _git(root, "rev-parse", "HEAD").strip()
            out = subprocess.run(["gh", "release", "create", tag, str(path), "--target", sha,
                                  "--title", tag, "--notes", notes], cwd=root, capture_output=True, text=True)
            if out.returncode != 0:
                raise PublishError(f"creating the release {tag} failed:\n{out.stderr.strip()}")
        made.append(tag)
    return made


__all__ = ["Change", "Comparison", "Entry", "PublishError", "Published", "Run", "SHOWN", "archive", "blob_id",
           "collect", "compare", "compare_runs", "open_pull_request", "publish", "published_runs",
           "release", "results_difference", "run_tag"]
