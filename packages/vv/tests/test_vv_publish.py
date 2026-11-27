"""Comparing stored records with the published run, publishing them and archiving a run.

A stand-in project keeps its records in a throwaway git repository whose
``origin`` is a bare repository beside it; no GitHub is involved.
"""
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import vv
from vv import publish
from vv.records import read_record, save_record


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


class RecordsProject(vv.Project):
    """One model, ``sphere``, on ``lib-a``; ``code-a==V`` has the newest record of version V of each item."""

    def __init__(self, root):
        self.root = root

    def models(self):
        return ["sphere"]

    def libraries(self):
        return {"lib-a": "no data here, which publishing does not need"}

    def defined(self, model, library):
        return True

    def parse_code(self, spec):
        name, _, version = spec.partition("==")
        if name != "code-a" or not version:
            raise ValueError(f"cannot read code {spec!r}")
        return vv.Code(spec, name, lambda: sys.executable, lambda: False,
                       check=lambda: (_ for _ in ()).throw(ValueError("not installable here")))

    def rows(self, model, library, code, codes):
        return []

    def command(self, job, codes, python):
        return vv.Command([str(python)])

    def records(self, model, library, code):
        version = code.spec.split("==")[1]
        out = []
        for item_dir in sorted((self.root / "cache" / library / code.name).glob("*")):
            found = vv.records.find_record(self.root / "cache", library, code.name, item_dir.name,
                                           accept=lambda r: r["code_identity"]["version"] == version)
            if found:
                out.append(found[0])
        return out


def save(root, item, version, value, *, dirty=False, commit="abc123def456"):
    ident = {"name": "code-a", "version": version, "commit": commit, "dirty": dirty,
             "build_id": vv.records.build_id(version, commit, dirty, "ff" if dirty else None)}
    return save_record(root / "cache", "lib-a", "code-a", item, {"flux": np.array([1.0, value])}, identity=ident)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with two records of code-a 1.0, committed and pushed to its bare origin."""
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{k}_NAME", "tester")
        monkeypatch.setenv(f"GIT_{k}_EMAIL", "tester@example.com")
    remote, root = tmp_path / "remote.git", tmp_path / "repo"
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "clone", "--quiet", str(remote), str(root)], check=True, capture_output=True)
    git(root, "checkout", "--quiet", "-b", "main")
    save(root, "Fe56", "1.0", 2.0)
    save(root, "H1", "1.0", 3.0)
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "records")
    git(root, "push", "--quiet", "-u", "origin", "main")
    git(root, "remote", "set-head", "origin", "main")
    return root


@pytest.fixture
def prs(monkeypatch):
    opened = []

    def fake(root, base, branch, title, body):
        opened.append(dict(base=base, branch=branch, title=title, body=body))
        return f"https://example.com/pull/{len(opened)}"

    monkeypatch.setattr(publish, "open_pull_request", fake)
    return opened


def merge(root, branch):
    """Merge a pushed publishing branch into main, as the pull request would, and pull it.

    Records that were published uncommitted are on the branch, so they are removed first.
    """
    git(root, "clean", "--quiet", "-fd")
    git(root, "fetch", "--quiet", "origin")
    git(root, "merge", "--quiet", "--ff-only", f"origin/{branch}")
    git(root, "push", "--quiet", "origin", "main")


KW = dict(models=["sphere"], libraries=["lib-a"])


def test_collecting_needs_no_data_or_install_and_names_each_record_by_its_git_blob(repo):
    run = publish.collect(RecordsProject(repo), codes=["code-a==1.0"], **KW)
    assert [e.item for e in run.entries] == ["Fe56", "H1"]
    for e in run.entries:
        assert e.blob == git(repo, "hash-object", e.path).strip()
        assert e.build_id == "1.0+gabc123def456" and e.code == e.label == "code-a"


def test_before_anything_is_published_every_record_is_new(repo):
    comparison = publish.compare(RecordsProject(repo), codes=["code-a==1.0"], **KW)
    assert comparison.against is None and comparison.count("new") == 2
    assert "nothing is published yet" in comparison.format()


def test_publishing_opens_a_pull_request_without_touching_the_checkout(repo, prs):
    save(repo, "Li6", "1.0", 4.0)  # a new record, not committed
    head, status = git(repo, "rev-parse", "HEAD"), git(repo, "status", "--porcelain")
    done = publish.publish(RecordsProject(repo), codes=["code-a==1.0"], **KW)
    assert done.url == "https://example.com/pull/1"
    assert done.uncommitted == ["cache/lib-a/code-a/Li6/1.0+gabc123def456.json"]
    assert "before pulling" in done.format()
    assert git(repo, "rev-parse", "HEAD") == head and git(repo, "status", "--porcelain") == status

    (pr,) = prs
    assert pr["base"] == "main" and pr["branch"].startswith("vv-publish/vv-")
    assert "3 records" in pr["body"] and "first published run" in pr["body"]
    ref = f"origin/{pr['branch']}"
    git(repo, "fetch", "--quiet", "origin")
    runs = json.loads(git(repo, "show", f"{ref}:published.json"))["runs"]
    tag = runs[0]["tag"]
    assert tag.endswith("-code-a-1.0") and runs[0]["records"] == 3
    manifest = json.loads(git(repo, "show", f"{ref}:runs/{tag}.json"))
    assert {r["item"] for r in manifest["records"]} == {"Fe56", "H1", "Li6"}
    # The uncommitted record went onto the branch with the bytes the manifest names.
    li6 = next(r for r in manifest["records"] if r["item"] == "Li6")
    assert git(repo, "rev-parse", f"{ref}:{li6['path']}").strip() == li6["blob"]


def test_after_publishing_compare_shows_what_a_new_version_changed(repo, prs):
    project = RecordsProject(repo)
    publish.publish(project, codes=["code-a==1.0"], **KW)
    merge(repo, prs[0]["branch"])
    assert publish.compare(project, codes=["code-a==1.0"], **KW).count("same") == 2

    save(repo, "Fe56", "1.1", 2.2)   # results moved by 10%
    save(repo, "H1", "1.1", 3.0)     # a new build, the same results
    save(repo, "Li7", "1.1", 1.0)
    comparison = publish.compare(project, codes=["code-a==1.1"], **KW)
    by_item = {c.key[3]: c for c in comparison.changes}
    assert by_item["Fe56"].status == "changed" and by_item["Fe56"].where == "flux"
    assert by_item["Fe56"].difference == pytest.approx(0.2 / 2.2)
    assert by_item["H1"].status == "rerun" and by_item["Li7"].status == "new"
    assert "1 new, 0 missing, 1 changed, 1 rerun, 0 same" in comparison.format()

    publish.publish(project, codes=["code-a==1.1"], **KW)
    assert "Against `vv-" in prs[1]["body"] and "1 changed" in prs[1]["body"]
    merge(repo, prs[1]["branch"])
    runs = publish.published_runs(repo)
    assert [r["codes"] for r in runs] == [["code-a==1.1"], ["code-a==1.0"]]
    assert len({r["tag"] for r in runs}) == 2


def test_a_record_missing_from_the_new_run_is_listed(repo, prs):
    project = RecordsProject(repo)
    publish.publish(project, codes=["code-a==1.0"], **KW)
    merge(repo, prs[0]["branch"])
    save(repo, "Fe56", "1.1", 2.0)
    by_item = {c.key[3]: c.status for c in publish.compare(project, codes=["code-a==1.1"], **KW).changes}
    assert by_item == {"Fe56": "rerun", "H1": "missing"}


def test_dirty_records_are_refused(repo, prs):
    path = save(repo, "Fe56", "1.0", 2.5, dirty=True)
    record = json.loads(path.read_text())
    record["run"]["date"] = "2999-01-01T00:00:00+00:00"  # newer than the clean record, not a tie
    path.write_text(json.dumps(record))
    with pytest.raises(publish.PublishError, match="uncommitted changes"):
        publish.publish(RecordsProject(repo), codes=["code-a==1.0"], **KW)
    assert not prs


def test_publishing_the_published_records_again_is_refused(repo, prs):
    project = RecordsProject(repo)
    publish.publish(project, codes=["code-a==1.0"], **KW)
    merge(repo, prs[0]["branch"])
    with pytest.raises(publish.PublishError, match="nothing to publish"):
        publish.publish(project, codes=["code-a==1.0"], **KW)


def test_a_code_with_no_records_cannot_be_published(repo, prs):
    with pytest.raises(publish.PublishError, match="no records"):
        publish.publish(RecordsProject(repo), codes=["code-a==9.0"], **KW)


def test_the_archive_holds_the_published_bytes_even_after_a_record_is_replaced(repo, prs, tmp_path):
    project = RecordsProject(repo)
    publish.publish(project, codes=["code-a==1.0"], **KW)
    merge(repo, prs[0]["branch"])
    tag = publish.published_runs(repo)[0]["tag"]
    manifest = json.loads((repo / f"runs/{tag}.json").read_text())
    save(repo, "Fe56", "1.0", 99.0)  # the same build rerun replaces its own record
    out = publish.archive(repo, tag, tmp_path)
    assert out.name == f"{tag}.tar.gz"
    with tarfile.open(out) as tar:
        names = tar.getnames()
        assert names[0] == f"{tag}/runs/{tag}.json"
        for r in manifest["records"]:
            assert publish.blob_id(tar.extractfile(f"{tag}/{r['path']}").read()) == r["blob"]
    assert publish.archive(repo, tag, tmp_path).read_bytes() == out.read_bytes()  # reproducible


def test_release_makes_each_missing_release_once(repo, prs, tmp_path, monkeypatch):
    project = RecordsProject(repo)
    publish.publish(project, codes=["code-a==1.0"], **KW)
    merge(repo, prs[0]["branch"])
    log, bin_dir = tmp_path / "gh.log", tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> {log}\n'
        f'if [ "$1 $2" = "release view" ]; then grep -q "release create $3 " {log}; exit $?; fi\n'
    )
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    tag = publish.published_runs(repo)[0]["tag"]
    assert publish.release(repo) == [tag]
    assert publish.release(repo) == []
    created = [ln for ln in log.read_text().splitlines() if ln.startswith("release create")]
    assert len(created) == 1 and f"{tag}.tar.gz" in created[0]
    assert git(repo, "rev-parse", "HEAD").strip() in created[0]


@pytest.mark.parametrize("old, new, diff, where", [
    ({"a": 1.0}, {"a": 1.0}, 0.0, ""),
    ({"a": [0.0, np.nan]}, {"a": [0.0, np.nan]}, 0.0, ""),
    ({"a": [1.0, 4.0]}, {"a": [1.0, 5.0]}, 0.2, "a"),
    ({"s": {"x": 1.0}}, {"s": {"x": 1.0, "y": 2.0}}, float("inf"), "s.y"),
    ({"a": [1.0]}, {"a": [1.0, 2.0]}, float("inf"), "a"),
    ({"a": 1.0, "name": "Fe"}, {"a": 1.0, "name": "Fe"}, 0.0, ""),
])
def test_results_difference(old, new, diff, where):
    assert publish.results_difference(old, new) == (pytest.approx(diff), where)


def test_run_tags_are_unique():
    assert publish.run_tag(["openmc==0.16.1.dev0", "yamc-cpu@86ef200e"], "2026-11-27", set()) == \
        "vv-2026-11-27-openmc-0.16.1.dev0-yamc-cpu-g86ef200e"
    assert publish.run_tag(["a==1"], "2026-11-27", {"vv-2026-11-27-a-1"}) == "vv-2026-11-27-a-1.2"


def test_fixed_records_are_part_of_the_run_and_published_with_it(repo, prs):
    """A measurement, say: no code makes it, every run of the model carries it."""
    measured = save_record(repo / "measured", "lib-a", "measurement", "Fe56", {"flux": np.array([1.0, 2.1])},
                           identity={"name": "measurement", "version": "report-1", "build_id": "report-1"})

    class WithMeasurement(RecordsProject):
        def fixed_records(self, model, library):
            return [measured]

    project = WithMeasurement(repo)
    run = publish.collect(project, codes=["code-a==1.0"], **KW)
    assert [(e.code, e.item) for e in run.entries] == [("code-a", "Fe56"), ("code-a", "H1"), ("measurement", "Fe56")]
    assert run.code_names == ["code-a", "measurement"]
    publish.publish(project, codes=["code-a==1.0"], **KW)
    merge(repo, prs[0]["branch"])
    by_key = {c.key[2:]: c.status for c in publish.compare(project, codes=["code-a==1.0"], **KW).changes}
    assert by_key == {("code-a", "Fe56"): "same", ("code-a", "H1"): "same", ("measurement", "Fe56"): "same"}
