"""The site is built from the published runs: the newest run, with the previous one's grades beside it."""
import json

import vv
from vv import release_check
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


def test_the_release_check_set_is_every_verification_model_on_endf_b8_1_stable_isotopes():
    check = PROJECT.release_check
    # every model compared with OpenMC; the ungraded validation models stay out
    assert "oktavian/sphere" not in check.models and "neutron/sphere" in check.models
    assert set(check.models) == {m for m in PROJECT.models() if m != "oktavian/sphere"}
    assert list(check.libraries) == ["endf-b8.1"]
    assert list(check.references) == ["openmc"] and tuple(check.bands) == index_builder.QUALITY_ORDER
    # every stable isotope, whatever item variables the caller has set
    assert dict(check.environment) == {"YAMC_VERIFY_NUMBER_OF_SIMULATIONS": "-1",
                                       "YAMC_VERIFY_NUCLIDES": None, "YAMC_VERIFY_ELEMENTS": None}


def test_the_release_check_grades_as_the_site_does():
    run = _run(["openmc==0.16.1.dev0", "yamc-cpu==0.11.0"], "vv-new")
    grades = PROJECT.grades(run)
    site = {(r["task"].rsplit("/", 1)[0], r["library"], c["code"], r["item"]): c["quality"]
            for r in index_builder.build_index(runs=[run])["benchmarks"] for c in r["comparisons"] if c["quality"]}
    assert grades and {k: g.band for k, g in grades.items()} == site
    assert all(g.note.startswith("chi2/N ") for g in grades.values())


def test_a_release_check_against_a_published_run(monkeypatch):
    """yamc 0.11.0 checked against a published 0.4.0 run, from the committed records alone."""
    published = _run(["openmc==0.16.1.dev0", "yamc-cpu==0.4.0"], "vv-old", models=PROJECT.models())
    monkeypatch.setattr(release_check, "published_runs", lambda root: [{"manifest": "runs/vv-old.json"}])
    monkeypatch.setattr(release_check, "_load_run", lambda root, summary: published)
    report = release_check.report(PROJECT, ["yamc-cpu==0.11.0"])
    assert report.setup.references == ["openmc==0.16.1.dev0"]
    bands = list(index_builder.QUALITY_ORDER)
    by_status = {}
    for f in report.findings:
        by_status.setdefault(f.status, []).append(f)
        assert f.key[2].startswith("yamc-cpu")
        if f.status in ("worse", "better", "same"):
            step = bands.index(f.new.band) - bands.index(f.old.band)
            assert {"worse": step > 0, "better": step < 0, "same": step == 0}[f.status]
    # 0.11.0 has records on the neutron sphere only, so the published tokamak and slab grades are missing.
    assert {f.key[0] for f in by_status["missing"]} >= {"neutron/tokamak", "neutron/multimaterial-slab"}
    assert {f.key[0] for s in ("worse", "better", "same", "new") for f in by_status.get(s, [])} == {"neutron/sphere"}
    assert "central_column Excellent -> no record" not in report.format()  # the item column fits its items
