"""The OKTAVIAN validation task: the scanned measurement, its record, and the comparison with yamc.

No yamc runs here and nothing is downloaded: the parser is driven with text
written for the test, and the comparisons with the committed measurement
records and results made from them.
"""
import json
import math
import shutil

import pytest
import vv
from vv.publish import blob_id, collect
from vv.records import save_record

from yamc_verify import index_builder, project, tasks
from yamc_verify.families import oktavian
from yamc_verify.project import PROJECT, YamcProject

TASK = tasks.get_task("oktavian/sphere/endf-b8.1")
ROOT = project.REPO_ROOT


# --- reading the scan ---------------------------------------------------------


def test_the_parser_undoes_safe_ocr_substitutions_and_joins_the_bins():
    text = "\n".join([
        "  upper     lower     value      error",
        "  2.0E+0l   1.5E+00   3.0E-02    1.0E-03",   # l for 1 in the exponent
        "  1.5E+00   1.0E+0O   2.0E-02    1,0E-03",   # O for 0, comma for point
        "  1.0E+00   5.0E-01   1.0E-02    5.0E-04",
        "  5.0E+01   2.5E-01   1E-38      0.0E+00",   # the upper edge lost its exponent sign; no measurement
        "  2.5E-01   1.0E-01   ~4.0E-03   abc",       # its error cannot be read
    ])
    notes = []
    rows = oktavian.read_bins(text, "test.dat", notes=notes)
    uppers = [r[0] for r in rows]
    lowers = [r[1] for r in rows]
    assert uppers[0] == pytest.approx(20.0) and uppers[3] == pytest.approx(0.5)  # 5.0E+01 read as 5.0E-01
    assert all(math.isclose(lo, up) for lo, up in zip(lowers, uppers[1:]))
    assert [r[2] for r in rows] == [pytest.approx(0.03), pytest.approx(0.02), pytest.approx(0.01), None, None]
    assert any("1 bins unreadable" in n for n in notes) and any("1 bins have no measurement" in n for n in notes)


def test_a_missing_row_is_an_error_not_a_wider_bin():
    text = "2.0E+00 1.5E+00 3.0E-02 1.0E-03\n1.0E+00 5.0E-01 1.0E-02 5.0E-04\n"
    with pytest.raises(ValueError, match="a row is missing or misread"):
        oktavian.read_bins(text, "test.dat")


# --- the committed measurement records ----------------------------------------


@pytest.mark.parametrize("item", sorted(oktavian.SPHERES))
def test_each_sphere_has_a_measurement_record_from_the_pinned_files(item):
    record = json.loads(oktavian.measurement_path(TASK, item, ROOT).read_text())
    assert record["schema"] == 1 and record["code"] == oktavian.MEASUREMENT and record["item"] == item
    ident = record["code_identity"]
    sphere = oktavian.SPHERES[item]
    assert ident["version"] == oktavian.REPORT and ident["build_id"].startswith("JAERI-M-94-014+sha")
    assert ident["source"]["files"] == {sphere["neutrons"][0]: sphere["neutrons"][1],
                                        sphere["photons"][0]: sphere["photons"][1], **oktavian.REPORTS}
    r = record["results"]
    for p, bins in (("neutron", "n_energy_bins"), ("photon", "p_energy_bins")):
        edges = r[bins]
        assert len(r[f"{p}_leakage"]) == len(r[f"{p}_leakage_error"]) == len(edges) - 1
        assert all(a < b for a, b in zip(edges, edges[1:]))
        assert sum(v is not None for v in r[f"{p}_leakage"]) > 40
    for spectrum in (r["source_neutrons"], r["target_photons"]):
        assert len(spectrum["probabilities"]) == len(spectrum["energy_bins"]) - 1
        assert sum(spectrum["probabilities"]) == pytest.approx(1.0)
    assert r["target_photons"]["per_source_neutron"] == oktavian.TARGET_PHOTONS_PER_NEUTRON
    # the D-T source peaks near 14 MeV
    probs, edges = r["source_neutrons"]["probabilities"], r["source_neutrons"]["energy_bins"]
    peak = max(range(len(probs)), key=probs.__getitem__)
    assert 13e6 < edges[peak] < 15e6


# --- comparing with the measurement --------------------------------------------


def test_goodness_of_fit_skips_unmeasured_bins():
    fit = oktavian.goodness_of_fit([1.1, 2.0, 3.0], [0.1, 0.1, 0.1], [1.0, None, 3.0], [0.1, None, 0.1],
                                   [1.0, 1.0, 1.0])
    assert fit["bins"] == 2
    assert fit["integral_c_over_e"] == pytest.approx((1.1 + 3.0) / 4.0)
    assert fit["chi2_per_bin"] == pytest.approx((0.1 ** 2 / 0.02 + 0) / 2)
    assert fit["bin_deviation_percent"] == pytest.approx(100 * math.sqrt(math.log(1.1) ** 2 / 2))
    assert fit["c_over_e"] == [pytest.approx(1.1), None, pytest.approx(1.0)]


def _calculated(experiment, scale):
    """A stand-in yamc result: the measurement times ``scale``, 1% Monte Carlo error."""
    out = {}
    for p in ("neutron", "photon"):
        mean = [scale * (v or 0.0) for v in experiment[f"{p}_leakage"]]
        out[f"{p}_leakage_mean"] = mean
        out[f"{p}_leakage_std"] = [0.01 * v for v in mean]
    out["n_energy_bins"], out["p_energy_bins"] = experiment["n_energy_bins"], experiment["p_energy_bins"]
    return out


def test_a_calculation_ten_percent_high_reads_as_c_over_e_1_1():
    experiment = oktavian.load_measurement(TASK, "Al27", ROOT)
    fit = oktavian.compare(experiment, _calculated(experiment, 1.1))
    for p in ("neutron", "photon"):
        assert fit[p]["integral_c_over_e"] == pytest.approx(1.1)
        assert fit[p]["bin_deviation_percent"] == pytest.approx(100 * math.log(1.1))


# --- the task and the project ------------------------------------------------------


def test_the_task_is_validation_run_on_the_cpu_only():
    assert TASK.validation and TASK.family in tasks.VALIDATION_FAMILIES
    assert TASK.codes == ("yamc-cpu",) and TASK.item_kind == "nuclide"
    assert not any(t.validation for t in tasks.all_tasks() if t.family != "oktavian")
    gpu, openmc = PROJECT.parse_code("yamc-gpu==0.28.0"), PROJECT.parse_code("openmc==0.16.1.dev0")
    assert not PROJECT.runs("oktavian/sphere", "endf-b8.1", gpu)
    assert not PROJECT.runs("oktavian/sphere", "endf-b8.1", openmc)


@pytest.mark.parametrize("requested, expected", [("", ["Al27", "Mn55"]), ("Mn55", ["Mn55"]),
                                                 ("Fe56,H1", []), ("Fe56,Al27", ["Al27"])])
def test_a_nuclide_list_runs_the_spheres_it_names_and_no_error_otherwise(monkeypatch, requested, expected):
    monkeypatch.setenv("YAMC_VERIFY_NUCLIDES", requested)
    assert oktavian.discover(TASK) == expected


def test_verification_and_validation_each_take_their_own_models():
    assert "oktavian/sphere" in project._models("all", True)
    assert "oktavian/sphere" not in project._models("all", False)
    assert "neutron/sphere" in project._models("all", False)
    with pytest.raises(vv.SelectionError, match="not validation"):
        project._models(["neutron/sphere"], True)
    with pytest.raises(vv.SelectionError, match="use validation"):
        project._models(["oktavian/sphere"], False)


@pytest.fixture
def repo(tmp_path):
    """A repository with the committed measurement and a yamc 0.28.0 record 10% high on Al27."""
    shutil.copytree(oktavian.measurement_dir(TASK, ROOT), oktavian.measurement_dir(TASK, tmp_path))
    experiment = oktavian.load_measurement(TASK, "Al27", ROOT)
    ident = {"name": "yamc", "version": "0.28.0", "commit": "e" * 40, "dirty": False, "backend": "cpu",
             "build_id": "0.28.0+g" + "e" * 12}
    save_record(tmp_path / TASK.cache_dir, TASK.library, "yamc-cpu", "Al27", _calculated(experiment, 1.1),
                identity=ident, settings={"library": TASK.library})
    p = YamcProject()
    p.root = tmp_path
    return p


def test_the_measurement_is_part_of_every_run_of_the_model(repo):
    run = collect(repo, models=["oktavian/sphere"], libraries=["endf-b8.1"], codes=["yamc-cpu==0.28.0"])
    assert sorted((e.code, e.item) for e in run.entries) == [
        ("measurement", "Al27"), ("measurement", "Mn55"), ("yamc-cpu", "Al27")]
    assert "measurement" in run.code_names
    # a run in which no code runs the model has no measurement either
    none = collect(repo, models=["oktavian/sphere", "neutron/sphere"], libraries=["endf-b8.1"],
                   codes=["openmc==0.16.1.dev0"])
    assert not any(e.code == "measurement" for e in none.entries)


def test_the_site_shows_the_comparison_ungraded_and_outside_the_agreement_figures(repo, monkeypatch):
    run = collect(repo, models=["oktavian/sphere"], libraries=["endf-b8.1"], codes=["yamc-cpu==0.28.0"])
    run.tag, run.date = "vv-test", "2026-11-27T00:00:00+00:00"
    monkeypatch.setattr(index_builder, "_REPO_ROOT", repo.root)
    index = index_builder.build_index(runs=[run])
    assert index["totals"]["simulations"] == 1  # the measurement is not a simulation
    assert index["totals"]["comparisons"] == 0 and index["totals"]["validations"] == 1
    assert index["agreement"]["quality_counts"] == {} and index["agreement"]["worst"] == []
    (al,) = [b for b in index["benchmarks"] if b["item"] == "Al27"]
    (c,) = al["comparisons"]
    assert c["kind"] == "validation" and c["quality"] is None and c["code"] == "yamc-cpu"
    assert c["neutron"]["integral_c_over_e"] == pytest.approx(1.1)
    assert c["photon"]["bin_deviation_percent"] == pytest.approx(100 * math.log(1.1))
    assert "c_over_e" not in c["neutron"]  # per bin, the page works it out from the records
    assert sorted(al["files"]) == ["measurement", "yamc-cpu"]
    record = repo.root / next(e.path for e in run.entries if e.code == "measurement" and e.item == "Al27")
    assert al["files"]["measurement"] == f"records/{blob_id(record.read_bytes())}.json"
