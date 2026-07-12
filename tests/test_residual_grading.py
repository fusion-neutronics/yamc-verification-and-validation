"""Residual-grading mechanism tests (yamc-verification #59).

These lock the anti-masking contract at the metric level: an expected difference
is removed from the grade only in its declared direction and only in its declared
bins; anything else (opposite sign, non-declared bin, continuum) still counts.
"""
from yamc_verify.plotting import calculate_spectral_agreement


def _spec(mean, std):
    return {"flux_spectra_mean": list(mean), "flux_spectra_std_dev": list(std)}


def _pair(ref_mean, oth_mean, sigma=1.0, expected=None):
    n = len(ref_mean)
    return calculate_spectral_agreement(
        _spec(ref_mean, [sigma] * n),
        _spec(oth_mean, [sigma] * n),
        expected=expected,
    )


def test_no_expected_is_behaviour_neutral():
    # residual == full and quality unchanged when nothing is declared expected.
    ref = [100.0, 100.0, 100.0, 100.0, 100.0]
    oth = [100.0, 130.0, 100.0, 100.0, 100.0]  # one bin disagrees 30 sigma
    s = _pair(ref, oth)
    assert s["reduced_chi_squared"] == s["residual_reduced_chi_squared"]
    assert s["quality"] == "Poor"
    assert s["excluded_flux_pct"] == 0.0
    assert s["excluded_bins"] == []


def test_expected_over_excess_is_removed_from_grade():
    # YAMC exceeds OpenMC in the declared bin -> residual clears it -> Excellent,
    # while the full chi-squared still records the difference.
    ref = [100.0, 100.0, 100.0, 100.0, 100.0]
    oth = [100.0, 130.0, 100.0, 100.0, 100.0]
    s = _pair(ref, oth, expected={1: "over"})
    assert s["quality"] == "Excellent"
    assert s["residual_reduced_chi_squared"] < 1.5
    assert s["reduced_chi_squared"] > 5.0  # full still sees it
    assert 1 in s["excluded_bins"]
    assert s["excluded_flux_pct"] > 0.0


def test_opposite_direction_still_counts():
    # Declared "over" but YAMC is BELOW OpenMC in that bin: NOT the expected
    # difference, must still degrade the grade (anti-masking).
    ref = [100.0, 100.0, 100.0, 100.0, 100.0]
    oth = [100.0, 70.0, 100.0, 100.0, 100.0]  # shortfall, not excess
    s = _pair(ref, oth, expected={1: "over"})
    assert s["quality"] == "Poor"
    assert s["residual_reduced_chi_squared"] == s["reduced_chi_squared"]
    assert s["excluded_bins"] == []


def test_non_declared_bin_still_counts():
    # An expected excess in bin 1 is removed, but a real discrepancy in bin 3
    # (not declared) still drives the grade to Poor.
    ref = [100.0, 100.0, 100.0, 100.0, 100.0]
    oth = [100.0, 130.0, 100.0, 130.0, 100.0]
    s = _pair(ref, oth, expected={1: "over"})
    assert 1 in s["excluded_bins"]
    assert 3 not in s["excluded_bins"]
    assert s["quality"] == "Poor"


def test_under_direction():
    # "under": OpenMC expected to exceed YAMC. A YAMC overshoot there still counts.
    ref = [100.0, 100.0, 100.0]
    oth_expected = [100.0, 70.0, 100.0]   # YAMC below -> matches "under" -> removed
    oth_overshoot = [100.0, 130.0, 100.0]  # YAMC above -> opposite -> still counts
    s_ok = _pair(ref, oth_expected, expected={1: "under"})
    s_bad = _pair(ref, oth_overshoot, expected={1: "under"})
    assert s_ok["quality"] == "Excellent"
    assert s_bad["quality"] == "Poor"


# --- index-level anti-masking tests (_compare) ------------------------------
import types  # noqa: E402

from yamc_verify.index_builder import _compare  # noqa: E402

# Coarse edges where Fe's characteristic lines (5.4-7 keV) fall in bin 1.
_EDGES = [1e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7]
_NBINS = len(_EDGES) - 1


def _flux(mean):
    # small, uniform Poisson-like stds so a systematic reads as many sigma
    return {
        "flux_spectra_mean": list(mean),
        "flux_spectra_std_dev": [m * 1e-3 for m in mean],
        "energy_bins": list(_EDGES),
    }


def _photon_task():
    return types.SimpleNamespace(family="photon", item_kind="element")


def test_compare_fluorescence_only_grades_excellent_with_annotation():
    base = [100.0] * _NBINS
    oth = list(base)
    oth[1] = 130.0  # YAMC excess only in Fe's fluorescence bin
    c = _compare(_photon_task(), _flux(base), _flux(oth), element="Fe")
    assert c["quality"] == "Excellent"
    assert c["reduced_chi_squared"] > 5.0          # full still sees it
    assert c["residual_reduced_chi_squared"] < 1.5  # residual clears it
    assert c["expected_difference"]["reason"] == "compton-ionization-fluorescence"
    assert not c["expected_difference"]["needs_review"]


def test_compare_continuum_regression_not_masked():
    base = [100.0] * _NBINS
    oth = list(base)
    oth[1] = 130.0   # expected fluorescence excess (bin 1)
    oth[8] = 130.0   # REAL continuum regression in a non-line bin -> must count
    c = _compare(_photon_task(), _flux(base), _flux(oth), element="Fe")
    assert c["quality"] == "Poor"


def test_compare_fluorescence_bin_shortfall_not_masked():
    # YAMC BELOW OpenMC in the fluorescence bin is not the expected difference.
    base = [100.0] * _NBINS
    oth = list(base)
    oth[1] = 70.0
    c = _compare(_photon_task(), _flux(base), _flux(oth), element="Fe")
    assert c["quality"] == "Poor"


def test_compare_neutron_control_no_exclusion():
    # Neutron has no expected difference: a Fe58-URR-like discrepancy stays Poor.
    base = [100.0] * _NBINS
    oth = list(base)
    oth[5] = 115.0
    task = types.SimpleNamespace(family="neutron", item_kind="nuclide")
    c = _compare(task, _flux(base), _flux(oth), element="Fe58")
    assert c["quality"] == "Poor"
    assert c["residual_reduced_chi_squared"] == c["reduced_chi_squared"]
    assert "expected_difference" not in c
