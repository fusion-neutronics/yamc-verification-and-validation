"""Generic spectrum-comparison plotting and metrics shared across families.

These helpers operate on the common results shape (``flux_spectra_mean`` /
``flux_spectra_std_dev`` / ``energy_bins``) and the ``OpenMC CPU`` / ``YAMC CPU``
/ ``YAMC GPU`` legend-label convention, so neutron / photon / secondary-photon
notebooks all reuse them.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Per-backend plot styling. Keys are backend labels used in the legend.
BACKEND_STYLES = {
    "OpenMC CPU": dict(color="blue", dash=None, marker="circle"),
    "YAMC CPU": dict(color="red", dash="dash", marker="square"),
    "YAMC GPU": dict(color="orange", dash="dot", marker="diamond"),
}
RATIO_COLORS = {
    "YAMC CPU / OpenMC CPU": "green",
    "YAMC GPU / OpenMC CPU": "purple",
    "YAMC GPU / YAMC CPU": "teal",
}
RATIO_FILLS = {
    "green": "rgba(0, 128, 0, 0.2)",
    "purple": "rgba(128, 0, 128, 0.2)",
    "teal": "rgba(0, 128, 128, 0.2)",
}


def create_step_trace(energy_bins, values, name, color, dash=None, showlegend=True, legend="legend"):
    """Create a step-plot trace for Plotly (like matplotlib step with where='post')."""
    x_step = []
    y_step = []
    for i in range(len(values)):
        x_step.extend([energy_bins[i], energy_bins[i + 1]])
        y_step.extend([values[i], values[i]])
    return go.Scatter(
        x=x_step, y=y_step, mode="lines", name=name,
        line=dict(color=color, width=2, dash=dash),
        showlegend=showlegend, legend=legend,
    )


# Reduced-chi-squared agreement bands: the SINGLE grading rule shared by every
# spectrum family (neutron, photon, secondary-photon, decay-photon). A comparison
# is graded by how many sigma the two codes differ per bin, which separates a
# genuine systematic discrepancy from Monte Carlo noise. Transmutation still grades
# on relative difference because its densities carry no uncertainty (yamc #188).
QUALITY_CHI2_THRESHOLDS = (1.5, 3.0, 5.0)  # Excellent < 1.5 <= Good < 3.0 <= Fair < 5.0 <= Poor


def quality_from_chi2(reduced_chi_squared) -> str:
    """Map a reduced chi-squared to an Excellent / Good / Fair / Poor band."""
    exc, good, fair = QUALITY_CHI2_THRESHOLDS
    if reduced_chi_squared < exc:
        return "Excellent"
    if reduced_chi_squared < good:
        return "Good"
    if reduced_chi_squared < fair:
        return "Fair"
    return "Poor"


def calculate_spectral_agreement(ref_results, other_results, flux_threshold=1e-6, expected=None):
    """Statistical agreement metrics comparing one spectrum to a reference.

    Relative-difference and chi-squared metrics are computed only over bins where
    *both* spectra clear a relative threshold (``peak / 10**6``), so a bin where
    the reference is numerically near-zero cannot produce an astronomical relative
    difference. Bins where one spectrum has flux and the other is effectively zero
    are reported separately as ``*_only_flux_pct`` (the fraction of a code's total
    flux the other code lacks). A large ``other_only_flux_pct`` is the signature of
    one code emitting where the other does not, for example D1S decay-product
    gammas that OpenMC's D1S omits.

    ``expected`` optionally declares bins that carry a known, physics-driven
    directional difference (e.g. Compton-ionization fluorescence lines where YAMC
    exceeds OpenMC). It maps ``bin_index -> "over" | "under"`` where ``"over"``
    means "other is expected to exceed ref" and ``"under"`` the reverse. In such a
    bin only the excess in the *expected* direction is removed for the residual
    chi-squared; a deviation in the opposite direction still counts in full. This
    is deliberately one-sided and localized so that a real regression next to an
    expected difference is never masked (see yamc-verification #59). The full
    chi-squared is always reported unchanged as ``reduced_chi_squared``; the grade
    (``quality``) is taken from ``residual_reduced_chi_squared``. With
    ``expected=None`` the residual equals the full value and the result is
    identical to the pre-residual behaviour.
    """
    ref_flux = np.asarray(ref_results["flux_spectra_mean"], dtype=float)
    ref_std = np.asarray(ref_results["flux_spectra_std_dev"], dtype=float)
    oth_flux = np.asarray(other_results["flux_spectra_mean"], dtype=float)
    oth_std = np.asarray(other_results["flux_spectra_std_dev"], dtype=float)
    expected = expected or {}

    peak = max(float(np.max(ref_flux)) if ref_flux.size else 0.0,
               float(np.max(oth_flux)) if oth_flux.size else 0.0)
    # relative threshold (6 decades below the peak), matching the plot's display
    # floor; falls back to the absolute threshold for an all-zero spectrum.
    thr = peak / 1e6 if peak > 0 else flux_threshold

    chi_squared_terms = []       # full: (o - r) per both-on bin
    residual_chi_squared_terms = []  # residual: expected excess removed
    rel_diffs = []
    weighted_rel_diffs = []
    flux_weights = []
    ref_total = float(np.sum(ref_flux[ref_flux > 0]))
    oth_total = float(np.sum(oth_flux[oth_flux > 0]))
    ref_only_flux = 0.0  # reference flux in bins where the other is ~0
    oth_only_flux = 0.0  # other flux in bins where the reference is ~0
    excluded_flux = 0.0  # flux removed as an expected directional difference
    excluded_bins = []

    for i in range(len(ref_flux)):
        r = ref_flux[i]
        o = oth_flux[i]
        r_on = r >= thr
        o_on = o >= thr
        if r_on and o_on:
            combined_std = np.sqrt(ref_std[i] ** 2 + oth_std[i] ** 2)
            diff = o - r
            # Residual difference: remove only the excess in the expected
            # direction. "over": drop positive (o - r), keep any shortfall.
            # "under": drop negative (o - r), keep any overshoot.
            resid = diff
            direction = expected.get(i)
            if direction == "over" and diff > 0:
                resid = 0.0
            elif direction == "under" and diff < 0:
                resid = 0.0
            if resid != diff:
                removed = abs(diff) - abs(resid)
                if removed > 0:
                    excluded_flux += removed
                    excluded_bins.append(i)
            if combined_std > 0:
                chi_squared_terms.append((diff / combined_std) ** 2)
                residual_chi_squared_terms.append((resid / combined_std) ** 2)
            rel_diff = abs(diff / r) * 100
            if np.isfinite(rel_diff):
                rel_diffs.append(rel_diff)
                weight = max(r, o)
                weighted_rel_diffs.append(rel_diff * weight)
                flux_weights.append(weight)
        elif o_on and not r_on:
            oth_only_flux += o
        elif r_on and not o_on:
            ref_only_flux += r

    dof = len(chi_squared_terms)
    reduced_chi_squared = float(np.sum(chi_squared_terms)) / dof if dof > 0 else 0.0
    residual_reduced_chi_squared = (
        float(np.sum(residual_chi_squared_terms)) / dof if dof > 0 else 0.0
    )

    median_rel_diff = float(np.median(rel_diffs)) if rel_diffs else 0.0
    weighted_avg_rel_diff = (
        float(np.sum(weighted_rel_diffs) / np.sum(flux_weights)) if flux_weights else 0.0
    )
    other_only_flux_pct = (oth_only_flux / oth_total * 100) if oth_total > 0 else 0.0
    ref_only_flux_pct = (ref_only_flux / ref_total * 100) if ref_total > 0 else 0.0
    excluded_flux_pct = (excluded_flux / oth_total * 100) if oth_total > 0 else 0.0

    # Grade on the residual: a known expected difference does not degrade the grade,
    # but anything left after removing it (continuum, opposite-sign, non-declared
    # bins) still does.
    quality = quality_from_chi2(residual_reduced_chi_squared)

    return {
        "reduced_chi_squared": reduced_chi_squared,
        "residual_reduced_chi_squared": residual_reduced_chi_squared,
        "degrees_of_freedom": dof,
        "median_rel_diff": median_rel_diff,
        "weighted_avg_rel_diff": weighted_avg_rel_diff,
        "quality": quality,
        "bins_analyzed": len(rel_diffs),
        "total_bins": len(ref_flux),
        "other_only_flux_pct": other_only_flux_pct,
        "ref_only_flux_pct": ref_only_flux_pct,
        "ref_total_flux": ref_total,
        "other_total_flux": oth_total,
        "excluded_flux_pct": excluded_flux_pct,
        "excluded_bins": excluded_bins,
    }


def create_agreement_table(stats):
    """Create a summary table of spectral agreement statistics."""
    import pandas as pd

    rows = [
        {"Metric": "Reduced Chi-Squared", "Value": f"{stats['reduced_chi_squared']:.2f}",
         "Note": f"(ideal ~ 1.0, dof={stats['degrees_of_freedom']})"},
        {"Metric": "Weighted Avg Rel. Diff", "Value": f"{stats['weighted_avg_rel_diff']:.2f}%",
         "Note": "(flux-weighted)"},
        {"Metric": "Median Rel. Diff", "Value": f"{stats['median_rel_diff']:.2f}%",
         "Note": "(robust to outliers)"},
        {"Metric": "Agreement Quality", "Value": stats["quality"],
         "Note": f"({stats['bins_analyzed']}/{stats['total_bins']} bins analyzed)"},
    ]
    return pd.DataFrame(rows)
