"""Result records: one per build, never overwritten by another build."""
import json

import pytest

import cache_utils
from cache_utils import (
    build_id,
    cached_items,
    cached_or_run,
    find_record,
    latest_record,
    save_cached,
)


def _yamc(build, commit=None):
    return {"name": "yamc", "version": "0.21.0", "commit": commit, "dirty": False,
            "binary_sha": None, "backend": "cpu", "build_id": build}


@pytest.fixture(autouse=True)
def _no_accept_old(monkeypatch):
    monkeypatch.delenv("YAMC_VERIFY_ACCEPT_OLD_CACHE", raising=False)


def test_build_ids():
    assert build_id("0.21.0", "86ef200e6a0e497a", False, None) == "0.21.0+g86ef200e6a0e"
    # A dirty build also carries the binary hash, so each rebuild is its own record.
    assert build_id("0.21.0", "86ef200e6a0e497a", True, "abc") == "0.21.0+g86ef200e6a0e.dirty.soabc"
    # A wheel without a commit is identified by its binary.
    assert build_id("0.20.0", None, None, "abc") == "0.20.0+soabc"
    assert build_id("0.20.0", None, None, None) == "0.20.0+unknown"
    assert build_id(None, None, None, None) == "unknown"


def test_a_record_describes_itself(tmp_path):
    path = save_cached(tmp_path, "endf-b8.1", "yamc-cpu", "Fe56", {"x": 1.0},
                       settings={"particles": 10}, elapsed_seconds=2.5,
                       identity=_yamc("0.21.0+gaaaa", "aaaa"))
    assert path == tmp_path / "endf-b8.1" / "yamc-cpu" / "Fe56" / "0.21.0+gaaaa.json"
    record = json.loads(path.read_text())
    assert record["schema"] == 1
    assert (record["library"], record["code"], record["item"]) == ("endf-b8.1", "yamc-cpu", "Fe56")
    assert record["code_identity"]["commit"] == "aaaa"
    assert record["settings"] == {"particles": 10}
    assert record["settings_hash"] == cache_utils.settings_hash({"particles": 10})
    assert record["run"]["elapsed_seconds"] == 2.5 and record["run"]["date"]


def test_two_builds_keep_two_records(tmp_path, monkeypatch):
    for build in ("0.20.0+gaaaa", "0.21.0+gbbbb"):
        save_cached(tmp_path, "lib", "yamc-cpu", "Fe56", {"build": build}, settings={}, identity=_yamc(build))
    assert sorted(p.name for p in (tmp_path / "lib" / "yamc-cpu" / "Fe56").iterdir()) == [
        "0.20.0+gaaaa.json", "0.21.0+gbbbb.json"]
    assert cached_items(tmp_path, "lib", "yamc-cpu") == ["Fe56"]
    assert latest_record(tmp_path, "lib", "yamc-cpu", "Fe56")[1]["results"]["build"] in (
        "0.20.0+gaaaa", "0.21.0+gbbbb")


def test_yamc_needs_the_installed_build(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_utils, "code_identity", lambda code: _yamc("0.21.0+gbbbb"))
    save_cached(tmp_path, "lib", "yamc-cpu", "Fe56", {"x": 1}, settings={"p": 1}, identity=_yamc("0.20.0+gaaaa"))
    assert find_record(tmp_path, "lib", "yamc-cpu", "Fe56", settings={"p": 1}) is None

    calls = []
    out = cached_or_run(tmp_path, "lib", "yamc-cpu", "Fe56", lambda: calls.append(1) or {"x": 2},
                        settings={"p": 1}, verbose=False)
    assert out == {"x": 2} and calls == [1]
    # Same build, same settings: the record is used, not recomputed.
    out = cached_or_run(tmp_path, "lib", "yamc-cpu", "Fe56", lambda: calls.append(1) or {"x": 3},
                        settings={"p": 1}, verbose=False)
    assert out == {"x": 2} and calls == [1]
    # Different settings on the same build re-run and replace that build's record.
    cached_or_run(tmp_path, "lib", "yamc-cpu", "Fe56", lambda: {"x": 4}, settings={"p": 2}, verbose=False)
    assert find_record(tmp_path, "lib", "yamc-cpu", "Fe56", settings={"p": 1}) is None


def test_accept_old_cache_takes_any_build(tmp_path, monkeypatch):
    save_cached(tmp_path, "lib", "yamc-cpu", "Fe56", {"x": 1}, settings={"p": 1}, identity=_yamc("0.4.0+soabc"))
    monkeypatch.setenv("YAMC_VERIFY_ACCEPT_OLD_CACHE", "1")
    assert find_record(tmp_path, "lib", "yamc-cpu", "Fe56", settings={"p": 1}) is not None
    assert find_record(tmp_path, "lib", "yamc-cpu", "Fe56", settings={"p": 2}) is None


def test_code_labels():
    assert cache_utils.code_name("openmc-led") == "openmc"
    assert cache_utils.code_name("yamc-cpu-ttb") == "yamc"
    assert cache_utils._backend("yamc-gpu-led") == "gpu"
    assert cache_utils._backend("yamc") is None
    with pytest.raises(ValueError):
        cache_utils.code_name("serpent")


def test_committed_records_are_all_schema_1():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "tasks"
    files = list(root.glob("**/cache/*/*/*/*.json"))
    assert files, "no committed records found"
    for f in files:
        record = json.loads(f.read_text())
        assert record.get("schema") == 1, f
        assert f.stem == record["code_identity"]["build_id"], f
    # Nothing left in the old <library>/<code>/<item>.json layout.
    assert not list(root.glob("**/cache/*/*/*.json"))
