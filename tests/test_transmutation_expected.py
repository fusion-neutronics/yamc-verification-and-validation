"""Transmutation expected-difference tests (isomeric branching).

Lock the anti-masking contract for the depletion ranking: an isomer
ground/metastable pair is excluded from the quality grade only when the
branching overlay is enabled, and unrelated products always keep counting.
"""
from yamc_verify.expected import transmutation_expected_products
from yamc_verify.families.transmutation import compare_transmutation_steps


def test_expected_products_gated_on_branching():
    prods = {"Ag110", "Ag110_m1", "Nb92_m1", "Fe56", "In116", "In116_m2"}
    # Fixed-BR baseline: nothing excluded, so isomers still grade (no masking).
    assert transmutation_expected_products(prods, branching_enabled=False) == set()
    # Overlay on: each metastable and its ground-state partner are expected.
    got = transmutation_expected_products(prods, branching_enabled=True)
    assert got == {"Ag110", "Ag110_m1", "Nb92", "Nb92_m1", "In116", "In116_m2"}


def _steps(pairs):
    return {"steps": [dict(pairs)], "time_s": 1.0}


def test_isomer_difference_excluded_only_when_expected():
    # An isomer pair disagrees 50%; an unrelated nuclide (Fe56) agrees.
    ymc = _steps({"Ag110_m1": 1.0e-6, "Ag110": 1.0e-6, "Fe56": 1.0e-3})
    omc = _steps({"Ag110_m1": 2.0e-6, "Ag110": 0.5e-6, "Fe56": 1.0e-3})

    base = compare_transmutation_steps(ymc, omc, [1.0], ["irradiation"])
    assert base["max_rel_diff_pct"] > 40.0  # isomer diff counts by default
    assert base["expected_diff_count"] == 0

    withexp = compare_transmutation_steps(
        ymc, omc, [1.0], ["irradiation"],
        expected_products={"Ag110", "Ag110_m1"},
    )
    assert withexp["expected_diff_count"] == 2       # both attributed to branching
    assert withexp["max_rel_diff_pct"] < 1.0         # grade now reflects only Fe56


def test_unexpected_difference_still_counts_with_expected_set():
    # A non-isomer disagreement must survive even when an expected set is given.
    ymc = _steps({"Ag110_m1": 1.0e-6, "Co60": 1.0e-6})
    omc = _steps({"Ag110_m1": 2.0e-6, "Co60": 4.0e-6})
    r = compare_transmutation_steps(
        ymc, omc, [1.0], ["irradiation"], expected_products={"Ag110_m1"}
    )
    assert r["expected_diff_count"] == 1
    assert r["max_rel_diff_pct"] > 40.0  # Co60 (75%) still drives the grade
