"""The site is built from the published runs: the newest run, with the previous one's grades beside it."""
import json

from vv.publish import blob_id, collect

from yamc_verify import index_builder
from yamc_verify.project import PROJECT


def _run(codes, tag, models=("neutron/sphere",)):
    run = collect(PROJECT, models=list(models), libraries=["endf-b8.1"], codes=codes)
    run.tag, run.date = tag, "2026-11-27T00:00:00+00:00"
    return run


def test_with_nothing_published_the_index_names_no_benchmarks():
    index = index_builder.build_index(runs=[])
    assert index["run"] is None and index["benchmarks"] == [] and index["totals"]["simulations"] == 0


def test_the_site_holds_the_newest_run_and_its_exact_records(tmp_path):
    run = _run(["openmc==0.16.1.dev0", "yamc-cpu==0.11.0"], "vv-new")
    index = index_builder.build_index(runs=[run], site=tmp_path)
    assert index["run"]["tag"] == "vv-new" and index["previous_run"] is None
    assert index["totals"]["simulations"] == len(run.entries)
    assert {r["task"] for r in index["benchmarks"]} == {"neutron/sphere/endf-b8.1"}
    blobs = {e.blob for e in run.entries}
    for r in index["benchmarks"]:
        assert "previous" not in r
        for path in r["files"].values():
            data = (tmp_path / path).read_bytes()
            assert path == f"records/{blob_id(data)}.json" and blob_id(data) in blobs
            assert json.loads(data)["schema"] == 1


def test_each_comparison_carries_the_previous_runs_grade():
    old = _run(["openmc==0.16.1.dev0", "yamc-cpu==0.11.0"], "vv-old")
    new = _run(["openmc==0.16.1.dev0", "yamc-cpu==0.11.0"], "vv-new")
    index = index_builder.build_index(runs=[new, old])
    assert index["previous_run"]["tag"] == "vv-old" and index["previous_run"]["only_in_previous"] == []
    compared = [r for r in index["benchmarks"] if r["comparisons"]]
    assert compared
    for r in compared:
        for c in r["comparisons"]:
            assert r["previous"][c["code"]]["quality"] == c["quality"]


def test_a_benchmark_only_in_the_previous_run_is_listed():
    old = _run(["openmc==0.16.1.dev0", "yamc-cpu==0.4.0"], "vv-old", models=("neutron/sphere", "neutron/tokamak"))
    new = _run(["openmc==0.16.1.dev0", "yamc-cpu==0.4.0"], "vv-new")
    index = index_builder.build_index(runs=[new, old])
    gone = index["previous_run"]["only_in_previous"]
    assert gone and all(i.startswith("neutron/tokamak/") for i in gone)
