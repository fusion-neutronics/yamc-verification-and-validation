"""Which records the yamc and OpenMC runs accept (the format itself is tested in packages/vv)."""
import json

import pytest

import cache_utils
from cache_utils import cached_or_run, find_cached, save_cached


def _yamc(build, commit=None):
    return {"name": "yamc", "version": "0.21.0", "commit": commit, "dirty": False,
            "binary_sha": None, "backend": "cpu", "build_id": build}


@pytest.fixture(autouse=True)
def _no_accept_old(monkeypatch):
    monkeypatch.delenv("YAMC_VERIFY_ACCEPT_OLD_CACHE", raising=False)


def test_yamc_needs_the_installed_build(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_utils, "code_identity", lambda code: _yamc("0.21.0+gbbbb"))
    save_cached(tmp_path, "lib", "yamc-cpu", "Fe56", {"x": 1}, settings={"p": 1}, identity=_yamc("0.20.0+gaaaa"))
    assert find_cached(tmp_path, "lib", "yamc-cpu", "Fe56", settings={"p": 1}) is None

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
    assert find_cached(tmp_path, "lib", "yamc-cpu", "Fe56", settings={"p": 1}) is None


def test_cached_or_run_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_utils, "code_identity", lambda code: _yamc("0.21.0+gbbbb"))
    events = tmp_path / "progress.jsonl"
    monkeypatch.setenv("VV_PROGRESS", str(events))
    cached_or_run(tmp_path, "lib", "yamc-cpu", "Fe56", lambda: {"x": 1}, settings={}, verbose=False)
    cached_or_run(tmp_path, "lib", "yamc-cpu", "Fe56", lambda: {"x": 1}, settings={}, verbose=False)
    seen = [json.loads(line) for line in events.read_text().splitlines()]
    assert [(e["t"], e.get("status")) for e in seen] == [
        ("item_start", None), ("item", "computed"), ("item", "cached")]


def test_accept_old_cache_takes_any_build(tmp_path, monkeypatch):
    save_cached(tmp_path, "lib", "yamc-cpu", "Fe56", {"x": 1}, settings={"p": 1}, identity=_yamc("0.4.0+soabc"))
    monkeypatch.setenv("YAMC_VERIFY_ACCEPT_OLD_CACHE", "1")
    assert find_cached(tmp_path, "lib", "yamc-cpu", "Fe56", settings={"p": 1}) is not None
    assert find_cached(tmp_path, "lib", "yamc-cpu", "Fe56", settings={"p": 2}) is None


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
