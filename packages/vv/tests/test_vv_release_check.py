"""The pre-release check: a fixed set run with a candidate build, graded against the published run.

A stand-in project keeps its records in a throwaway git repository whose
``origin`` is a bare repository beside it, as in the publishing tests. ``ref``
is the reference code and ``code-a`` the code under test; a result is graded
by how far its value is from the reference's.
"""
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import vv
from vv import publish, release_check
from vv.cli import main
from vv.records import decode, find_record, save_record

BANDS = ("Excellent", "Good", "Fair", "Poor")


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def band(difference):
    for limit, name in ((0.01, "Excellent"), (0.05, "Good"), (0.2, "Fair")):
        if difference < limit:
            return name
    return "Poor"


WRITE = """
import json, os, sys
import numpy as np
from vv.records import build_id, save_record
job = json.loads(os.environ["JOB"])
for item, value in job["values"].items():
    ident = {"name": job["code"], "version": job["version"], "commit": "c0ffee", "dirty": False,
             "build_id": build_id(job["version"], "c0ffee", False, None)}
    save_record(job["cache"], job["library"], job["code"], item, {"flux": np.array([1.0, value])}, identity=ident)
sys.exit(1 if job["fail"] else 0)
"""


class CheckProject(vv.Project):
    """``sphere`` and ``slab`` on ``lib-a``; ``ref==V`` is the reference and ``code-a==V`` the code under test."""

    name = "check"
    release_check = vv.ReleaseCheck(models=vv.ALL, libraries=["lib-a"], references=["ref"], bands=BANDS)

    def __init__(self, root, values=None, fail=()):
        self.root = root
        self.values = values or {}  # (model, code name) -> {item: value} a job writes
        self.fail = set(fail)       # (model, code name) whose job fails after writing

    def models(self):
        return ["sphere", "slab"]

    def libraries(self):
        return {"lib-a": None}

    def defined(self, model, library):
        return True

    def parse_code(self, spec):
        name, _, version = spec.partition("==")
        if name not in ("ref", "code-a") or not version:
            raise ValueError(f"cannot read code {spec!r}; codes: ['ref', 'code-a']")
        return vv.Code(spec, name, lambda: sys.executable, lambda: True)

    def rows(self, model, library, code, codes):
        return [vv.Row(label=code.name, items=sorted(self.values.get((model, code.name), {})))]

    def command(self, job, codes, python):
        job_data = {"cache": str(self.root / "cache" / job.model), "library": job.library, "code": job.code.name,
                    "version": job.code.spec.split("==")[1], "values": self.values.get((job.model, job.code.name), {}),
                    "fail": (job.model, job.code.name) in self.fail}
        return vv.Command([str(python), "-c", WRITE], env={"JOB": json.dumps(job_data)}, cwd=self.root)

    def records(self, model, library, code):
        version = code.spec.split("==")[1]
        out = []
        for item_dir in sorted((self.root / "cache" / model / library / code.name).glob("*")):
            found = find_record(self.root / "cache" / model, library, code.name, item_dir.name,
                                accept=lambda r: r["code_identity"]["version"] == version)
            if found:
                out.append(found[0])
        return out

    def grades(self, run):
        values = {}
        for e in run.entries:
            data = json.loads(publish.record_bytes(self.root, e))
            values[e.key] = decode(data["results"])["flux"][1]
        out = {}
        for (model, library, label, item), value in values.items():
            ref = values.get((model, library, "ref", item))
            if label != "ref" and ref is not None and np.isfinite(value):
                diff = abs(value - ref) / ref
                out[(model, library, label, item)] = vv.Grade(band(diff), f"{100 * diff:.1f}%")
        return out


def save(root, model, code, item, version, value):
    ident = {"name": code, "version": version, "commit": "abc123", "dirty": False,
             "build_id": vv.records.build_id(version, "abc123", False, None)}
    save_record(root / "cache" / model, "lib-a", code, item, {"flux": np.array([1.0, value])}, identity=ident)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository whose published run has ref 1.0 and code-a 1.0 on both models."""
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{k}_NAME", "tester")
        monkeypatch.setenv(f"GIT_{k}_EMAIL", "tester@example.com")
    remote, root = tmp_path / "remote.git", tmp_path / "repo"
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "clone", "--quiet", str(remote), str(root)], check=True, capture_output=True)
    git(root, "checkout", "--quiet", "-b", "main")
    for item, ref, value in (("Fe56", 10.0, 10.05), ("H1", 10.0, 10.3), ("Li6", 10.0, 13.0), ("Li7", 10.0, 10.0)):
        save(root, "sphere", "ref", item, "1.0", ref)
        save(root, "sphere", "code-a", item, "1.0", value)
    save(root, "slab", "ref", "W", "1.0", 10.0)
    save(root, "slab", "code-a", "W", "1.0", 10.0)
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "records")
    git(root, "push", "--quiet", "-u", "origin", "main")
    git(root, "remote", "set-head", "origin", "main")
    opened = []
    monkeypatch.setattr(publish, "open_pull_request", lambda *a: opened.append(a[2]) or "https://example.com/pull/1")
    publish.publish(CheckProject(root), models=vv.ALL, libraries=["lib-a"], codes=["ref==1.0", "code-a==1.0"])
    git(root, "fetch", "--quiet", "origin")
    git(root, "merge", "--quiet", "--ff-only", f"origin/{opened[0]}")
    return root


def test_the_candidate_is_graded_against_the_published_references(repo):
    # A rerun of the reference since publishing must not move a grade: only the candidate can.
    save(repo, "sphere", "ref", "Fe56", "1.0", 50.0)
    save(repo, "sphere", "code-a", "Fe56", "1.1", 10.0)   # 0.5% -> 0%: same band
    save(repo, "sphere", "code-a", "H1", "1.1", 10.8)     # 3% -> 8%: Good -> Fair
    save(repo, "sphere", "code-a", "Li6", "1.1", 10.2)    # 30% -> 2%: Poor -> Good
    save(repo, "sphere", "ref", "Be9", "1.0", 10.0)       # an item the published run did not have
    save(repo, "sphere", "code-a", "Be9", "1.1", 10.0)
    # Li7 and slab W have no code-a 1.1 record
    report = release_check.report(CheckProject(repo), ["code-a==1.1"],
                                  failed={("slab", "lib-a", "code-a"): repo / "slab.log"})
    status = {f.key[3]: f.status for f in report.findings}
    assert status == {"Fe56": "same", "H1": "worse", "Li6": "better", "Be9": "new", "Li7": "missing", "W": "fails"}
    assert report.setup.references == ["ref==1.0"]
    assert not report.ok and [f.key[3] for f in report.flagged] == ["W", "H1", "Li7"]

    text = report.format()
    assert "release check of code-a==1.1" in text and "graded with its ref==1.0 records" in text
    assert "set: every model on lib-a" in text
    assert "Good -> Fair   3.0% -> 8.0%" in text and "Poor -> Good   30.0% -> 2.0%" in text
    assert "Excellent -> no record: its job failed (log:" in text and "slab.log" in text
    assert "Excellent -> no record\n" in text
    assert "6 graded results: 1 worse, 1 fails, 1 missing, 1 better, 1 new, 1 same" in text
    assert "3 flagged: review them before releasing" in text


def test_a_candidate_as_good_as_the_published_run_passes(repo):
    for item, value in (("Fe56", 10.05), ("H1", 10.3), ("Li6", 13.0), ("Li7", 10.0)):
        save(repo, "sphere", "code-a", item, "1.1", value)
    save(repo, "slab", "code-a", "W", "1.1", 10.0)
    report = release_check.report(CheckProject(repo), ["code-a==1.1"])
    assert report.ok and report.count("same") == 5
    assert "nothing got worse, failed or went missing" in report.format()


def test_a_record_the_project_cannot_grade_is_missing(repo):
    save(repo, "sphere", "code-a", "Li7", "1.1", float("nan"))
    save(repo, "sphere", "code-a", "Ne20", "1.1", 10.0)  # no reference and not published: graded by neither
    findings = {f.key[3]: f for f in release_check.report(CheckProject(repo), ["code-a==1.1"]).findings}
    assert findings["Li7"].status == "missing" and findings["Li7"].note == "a record, but nothing to grade it against"
    assert "Ne20" not in findings


def test_the_check_runs_the_set_then_reports(repo):
    values = {("sphere", "code-a"): {"Fe56": 10.0, "H1": 10.3, "Li6": 13.0, "Li7": 12.0},
              ("slab", "code-a"): {}, ("sphere", "ref"): {}, ("slab", "ref"): {}}
    project = CheckProject(repo, values=values, fail={("slab", "code-a")})
    out = io.StringIO()
    result, report = release_check.release_check(project, ["code-a==1.1"], stream=out)
    assert [c.spec for c in result.plan.codes] == ["ref==1.0", "code-a==1.1"]
    assert [(job.model, job.code.name) for job, _ in result.failures] == [("slab", "code-a")]
    status = {f.key[3]: f.status for f in report.findings}
    assert status == {"Fe56": "same", "H1": "same", "Li6": "same", "Li7": "worse", "W": "fails"}


def test_the_check_pins_its_environment_while_it_runs(repo, monkeypatch):
    project = CheckProject(repo, values={("sphere", "code-a"): {"Fe56": 10.0}})
    project.release_check = vv.ReleaseCheck(models=vv.ALL, libraries=["lib-a"], references=["ref"], bands=BANDS,
                                            environment={"CHECK_ITEMS": "all", "NARROW": None})
    monkeypatch.setenv("NARROW", "Fe56")
    monkeypatch.delenv("CHECK_ITEMS", raising=False)
    seen, rows = [], project.rows

    def recording_rows(*args):
        seen.append((os.environ.get("CHECK_ITEMS"), os.environ.get("NARROW")))
        return rows(*args)

    project.rows = recording_rows
    _, report = release_check.release_check(project, ["code-a==1.1"], stream=io.StringIO())
    assert seen and set(seen) == {("all", None)}
    assert os.environ["NARROW"] == "Fe56" and "CHECK_ITEMS" not in os.environ
    assert "set: every model on lib-a (CHECK_ITEMS=all, NARROW unset)" in report.format()


@pytest.mark.parametrize("codes, problem", [
    (["ref==2.0"], "ref is what the check grades against"),
    (["code-a==1.1", "code-a==1.1"], "named twice"),
    (["serpent==2"], "cannot read code"),
    ([], "name the codes to check"),
    ("code-a==1.1", "name the codes to check"),
])
def test_a_bad_request_is_refused_before_anything_runs(repo, codes, problem):
    with pytest.raises(release_check.ReleaseCheckError, match=problem):
        release_check.setup(CheckProject(repo), codes)


def test_nothing_to_check_against_until_a_run_is_published(tmp_path):
    with pytest.raises(release_check.ReleaseCheckError, match="nothing is published yet"):
        release_check.setup(CheckProject(tmp_path), ["code-a==1.1"])


def test_a_project_without_a_release_check_says_so(repo):
    project = CheckProject(repo)
    project.release_check = None
    with pytest.raises(release_check.ReleaseCheckError, match="has no release check"):
        release_check.setup(project, ["code-a==1.1"])


def test_the_published_run_must_have_the_reference(repo):
    project = CheckProject(repo)
    project.release_check = vv.ReleaseCheck(models=vv.ALL, libraries=["lib-a"], references=["ref", "code-b"],
                                            bands=BANDS)
    with pytest.raises(release_check.ReleaseCheckError, match="has no code-b results to grade against"):
        release_check.setup(project, ["code-a==1.1"])


def test_a_grade_outside_the_bands_is_an_error(repo):
    project = CheckProject(repo)
    project.release_check = vv.ReleaseCheck(models=vv.ALL, libraries=["lib-a"], references=["ref"],
                                            bands=("Excellent", "Good"))
    with pytest.raises(release_check.ReleaseCheckError, match="not one of its release check's bands"):
        release_check.report(project, ["code-a==1.0"])


def test_the_command_line_takes_the_codes_and_nothing_else(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--release-check", "code-a==1.1", "--models", "sphere"])
    assert "leave out --models" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["run", "--models", "sphere"])
    assert "required: --libraries, --codes (or --release-check CODE)" in capsys.readouterr().err
