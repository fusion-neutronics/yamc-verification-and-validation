"""The scheduled core comparison must reject invalid or disagreeing results."""

import io
import lzma
import tarfile
from xml.etree import ElementTree

import pytest

from yamc_verify.core_references import extract_subset
from yamc_verify.core_validation import checks_pass, compare_case, scalar_check, spectrum_check


def test_scalar_disagreement_fails():
    assert not scalar_check("flux", 1.0, 2.0)["passed"]


def test_statistically_consistent_scalar_passes():
    assert scalar_check("flux", 1.0, 1.2, reference_std=0.1)["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_scalar_is_refused(value):
    with pytest.raises(ValueError, match="non-finite"):
        scalar_check("flux", 1.0, value)


def test_negative_uncertainty_is_refused():
    with pytest.raises(ValueError, match="negative uncertainty"):
        scalar_check("flux", 1.0, 1.0, reference_std=-1.0)


def test_difference_with_zero_uncertainty_fails():
    assert not spectrum_check("spectrum", [1.0], [2.0], [0.0], [0.0])["passed"]


def test_equal_zero_uncertainty_bins_pass():
    assert spectrum_check("spectrum", [0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0])["passed"]


def test_spectrum_length_change_is_refused():
    with pytest.raises(ValueError, match="different spectrum lengths"):
        spectrum_check("spectrum", [1.0], [], [0.1], [])


def test_missing_inventory_product_fails():
    checks = compare_case("transmutation", {"steps": [{"Fe55": 1e-5}]}, {"steps": [{}]})
    assert not checks[0]["passed"]


def test_an_empty_inventory_comparison_is_refused():
    with pytest.raises(ValueError, match="no comparisons"):
        compare_case("transmutation", {"steps": [{}]}, {"steps": [{}]})


def test_decay_shape_gate_uses_the_peak_step_and_retains_cooling_diagnostics():
    expected = {"spectra": [[100.0], [1.0]], "spectra_std": [[1.0], [0.1]]}
    actual = {"spectra": [[100.0], [1.22]], "spectra_std": [[1.0], [0.1]]}
    checks = compare_case("decay_photons", expected, actual)
    assert checks_pass(checks)
    assert any(not check["passed"] and not check["required"] for check in checks if "required" in check)


def test_a_peak_decay_shape_disagreement_still_fails():
    expected = {"spectra": [[100.0, 100.0]], "spectra_std": [[1.0, 1.0]]}
    actual = {"spectra": [[120.0, 80.0]], "spectra_std": [[1.0, 1.0]]}
    assert not checks_pass(compare_case("decay_photons", expected, actual))


def test_no_required_checks_is_not_a_pass():
    assert not checks_pass([])


def _archive(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return io.BytesIO(lzma.compress(stream.getvalue()))


def test_data_subset_rewrites_only_selected_library_paths(tmp_path):
    index = b'<cross_sections><library materials="Fe56" type="neutron" path="neutron/Fe56.h5"/><library materials="U235" type="neutron" path="neutron/U235.h5"/></cross_sections>'
    stream = _archive({
        "data/cross_sections.xml": index,
        "data/neutron/Fe56.h5": b"selected",
        "data/neutron/U235.h5": b"not selected",
    })
    assert extract_subset(stream, tmp_path, {"Fe56"}) == {"Fe56"}
    assert (tmp_path / "Fe56.h5").read_bytes() == b"selected"
    assert not (tmp_path / "U235.h5").exists()
    libraries = ElementTree.parse(tmp_path / "cross_sections.xml").getroot().findall("library")
    assert len(libraries) == 1
    assert libraries[0].attrib["path"] == "Fe56.h5"


def test_data_archive_path_traversal_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unsafe data archive path"):
        extract_subset(_archive({"../Fe56.h5": b"bad"}), tmp_path, {"Fe56"})


def test_a_data_archive_without_an_index_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no cross_sections.xml"):
        extract_subset(_archive({"Fe56.h5": b"data"}), tmp_path, {"Fe56"})