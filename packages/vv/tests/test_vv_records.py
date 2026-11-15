"""The record format: one self-describing record per result and build."""
import json

import numpy as np
import pytest
from vv import records
from vv.records import build_id, cached_items, find_record, latest_record, save_record


def _ident(build, commit=None, version="1.2.0"):
    return {"name": "code-a", "version": version, "commit": commit, "dirty": False,
            "binary_sha": None, "backend": None, "build_id": build}


def test_build_ids():
    assert build_id("0.21.0", "86ef200e6a0e497a", False, None) == "0.21.0+g86ef200e6a0e"
    # A dirty build also carries the binary hash, so each rebuild is its own record.
    assert build_id("0.21.0", "86ef200e6a0e497a", True, "abc") == "0.21.0+g86ef200e6a0e.dirty.soabc"
    # A build without a commit is identified by its binary.
    assert build_id("0.20.0", None, None, "abc") == "0.20.0+soabc"
    assert build_id("0.20.0", None, None, None) == "0.20.0+unknown"
    assert build_id(None, None, None, None) == "unknown"


def test_a_record_describes_itself(tmp_path):
    path = save_record(tmp_path, "lib", "code-a", "Fe56", {"x": 1.0}, identity=_ident("1.2.0+gaaaa", "aaaa"),
                       settings={"particles": 10}, elapsed_seconds=2.5)
    assert path == tmp_path / "lib" / "code-a" / "Fe56" / "1.2.0+gaaaa.json"
    record = json.loads(path.read_text())
    assert record["schema"] == records.SCHEMA == 1
    assert (record["library"], record["code"], record["item"]) == ("lib", "code-a", "Fe56")
    assert record["code_identity"]["commit"] == "aaaa"
    assert record["settings"] == {"particles": 10}
    assert record["settings_hash"] == records.settings_hash({"particles": 10})
    assert record["run"]["elapsed_seconds"] == 2.5 and record["run"]["date"]


def test_an_identity_must_name_its_build(tmp_path):
    with pytest.raises(ValueError, match="build_id"):
        save_record(tmp_path, "lib", "code-a", "Fe56", {}, identity={"name": "code-a"})


def test_numpy_round_trips(tmp_path):
    results = {"flux": np.array([1.0, 2.5]), "n": np.int64(3), "ok": np.bool_(True), "nested": [np.float32(1.5)]}
    path = save_record(tmp_path, "lib", "code-a", "Fe56", results, identity=_ident("b"))
    loaded = records.decode(records.read_record(path)["results"])
    assert loaded["flux"].dtype == np.float64 and loaded["flux"].tolist() == [1.0, 2.5]
    assert loaded["n"] == 3 and loaded["ok"] is True and loaded["nested"] == [1.5]


def test_two_builds_keep_two_records(tmp_path):
    for build in ("0.20.0+gaaaa", "0.21.0+gbbbb"):
        save_record(tmp_path, "lib", "code-a", "Fe56", {"build": build}, identity=_ident(build))
    assert sorted(p.name for p in (tmp_path / "lib" / "code-a" / "Fe56").iterdir()) == [
        "0.20.0+gaaaa.json", "0.21.0+gbbbb.json"]
    assert cached_items(tmp_path, "lib", "code-a") == ["Fe56"]
    assert latest_record(tmp_path, "lib", "code-a", "Fe56")[1]["results"]["build"] in (
        "0.20.0+gaaaa", "0.21.0+gbbbb")


def test_find_by_build_by_predicate_or_newest(tmp_path):
    save_record(tmp_path, "lib", "code-a", "Fe56", {"x": 1}, identity=_ident("1.0+ga", "a", "1.0"),
                settings={"p": 1, "library": "lib"})
    save_record(tmp_path, "lib", "code-a", "Fe56", {"x": 2}, identity=_ident("2.0+gb", "b", "2.0"),
                settings={"p": 1, "library": "lib"})
    # Settings are a subset match.
    assert find_record(tmp_path, "lib", "code-a", "Fe56", settings={"library": "lib"}) is not None
    assert find_record(tmp_path, "lib", "code-a", "Fe56", settings={"p": 2}) is None
    # A named build, or nothing.
    path, record = find_record(tmp_path, "lib", "code-a", "Fe56", build_id="1.0+ga")
    assert record["results"] == {"x": 1}
    assert find_record(tmp_path, "lib", "code-a", "Fe56", build_id="3.0+gc") is None
    # A predicate on the record.
    found = find_record(tmp_path, "lib", "code-a", "Fe56",
                        accept=lambda r: r["code_identity"]["version"] == "2.0")
    assert found[1]["results"] == {"x": 2}
    assert find_record(tmp_path, "lib", "code-a", "Fe56", accept=lambda r: False) is None


def test_unreadable_and_other_schema_files_are_ignored(tmp_path):
    d = records.record_dir(tmp_path, "lib", "code-a", "Fe56")
    d.mkdir(parents=True)
    (d / "broken.json").write_text("{not json")
    (d / "old.json").write_text(json.dumps({"schema": 0}))
    assert list(records.iter_records(tmp_path, "lib", "code-a", "Fe56")) == []


def test_item_names_with_a_slash_are_one_folder(tmp_path):
    assert records.record_dir(tmp_path, "lib", "code-a", "a/b").name == "a_b"


def test_records_without_a_run_date_sort_last(tmp_path):
    save_record(tmp_path, "lib", "code-a", "Fe56", {"x": "new"}, identity=_ident("new"))
    old = save_record(tmp_path, "lib", "code-a", "Fe56", {"x": "old"}, identity=_ident("old"))
    record = json.loads(old.read_text())
    record["run"]["date"] = None
    old.write_text(json.dumps(record))
    assert [r["results"]["x"] for _, r in records.iter_records(tmp_path, "lib", "code-a", "Fe56")] == ["new", "old"]
