"""Evaluating your own ENDF file on the OKTAVIAN spheres, with a stand-in for NJOY and yamc."""
import sys
import types

import pytest

from yamc_verify import evaluate, tasks
from yamc_verify.families import oktavian

TASK = tasks.get_task("oktavian/sphere/endf-b8.1")


@pytest.fixture
def converter(tmp_path, monkeypatch):
    """A stand-in yamc whose conversion writes ``<nuclide>.arrow`` named by the file's first line."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    calls = []

    def convert_neutron_transport(endf, out_dir, **kw):
        calls.append(endf)
        nuclide = open(endf).readline().strip()
        out = tmp_path / "cache" / out_dir / f"{nuclide}.arrow"
        out.mkdir(parents=True)
        return str(out)

    monkeypatch.setitem(sys.modules, "yamc", types.SimpleNamespace(convert_neutron_transport=convert_neutron_transport))
    return calls


def test_an_evaluation_is_converted_once_per_file_contents(tmp_path, converter):
    endf = tmp_path / "my-Al27.endf"
    endf.write_text("Al27\n")
    first = evaluate.convert_evaluation(endf, "Al27")
    assert first.endswith("Al27.arrow") and evaluate.convert_evaluation(endf, "Al27") == first
    assert len(converter) == 1
    endf.write_text("Al27\nchanged\n")
    assert evaluate.convert_evaluation(endf, "Al27") != first and len(converter) == 2


def test_an_evaluation_of_another_nuclide_is_refused(tmp_path, converter):
    endf = tmp_path / "fe.endf"
    endf.write_text("Fe56\n")
    with pytest.raises(ValueError, match="is an evaluation of Fe56, but the sphere is Al27"):
        evaluate.convert_evaluation(endf, "Al27")


def test_the_cases_put_the_file_first_and_check_their_arguments(tmp_path, converter):
    endf = tmp_path / "mn.endf"
    endf.write_text("Mn55\n")
    got = evaluate.cases("Mn55", endf, "new Mn55", ["endf-b8.1"])
    assert [c[0] for c in got] == ["new Mn55", "ENDF/B-VIII.1"] and got[1][2] == "endf-b8.1"
    assert got[0][1]["sha256"] == evaluate.sha256_of(endf)
    with pytest.raises(ValueError, match="needs --label"):
        evaluate.cases("Mn55", endf, None, [])
    with pytest.raises(ValueError, match="unknown library"):
        evaluate.cases("Mn55", None, None, ["endf-b9"])
    with pytest.raises(ValueError, match="nothing to evaluate"):
        evaluate.cases("Mn55", None, None, [])


def test_the_summary_and_figure_show_every_evaluation():
    experiment = oktavian.load_measurement(TASK, "Al27", evaluate.REPO_ROOT)
    report = {"benchmark": "OKTAVIAN Al27 sphere", "vessel_and_photon_data": "endf-b8.1", "evaluations": []}
    for label, scale in (("mine", 1.1), ("ENDF/B-VIII.1", 1.0)):
        results = {}
        for p in ("neutron", "photon"):
            mean = [scale * (v or 0.0) for v in experiment[f"{p}_leakage"]]
            results[f"{p}_leakage_mean"], results[f"{p}_leakage_std"] = mean, [0.01 * v for v in mean]
        report["evaluations"].append({"label": label, "results": results,
                                      "fit": oktavian.compare(experiment, results)})
    text = evaluate.summary(report)
    assert "mine" in text and "1.100" in text and "1.000" in text
    fig = evaluate.figure(report, experiment)
    assert {t.name for t in fig.data if t.showlegend} == {"mine", "ENDF/B-VIII.1", "OKTAVIAN measurement"}
    assert fig.layout.xaxis.type == fig.layout.xaxis3.type == "log"  # both neutron rows
