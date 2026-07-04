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


def calculate_spectral_agreement(ref_results, other_results, flux_threshold=1e-6):
    """Statistical agreement metrics comparing one spectrum to a reference.

    Relative-difference and chi-squared metrics are computed only over bins where
    *both* spectra clear a relative threshold (``peak / 10**6``), so a bin where
    the reference is numerically near-zero cannot produce an astronomical relative
    difference. Bins where one spectrum has flux and the other is effectively zero
    are reported separately as ``*_only_flux_pct`` (the fraction of a code's total
    flux the other code lacks). A large ``other_only_flux_pct`` is the signature of
    one code emitting where the other does not, for example D1S decay-product
    gammas that OpenMC's D1S omits.
    """
    ref_flux = np.asarray(ref_results["flux_spectra_mean"], dtype=float)
    ref_std = np.asarray(ref_results["flux_spectra_std_dev"], dtype=float)
    oth_flux = np.asarray(other_results["flux_spectra_mean"], dtype=float)
    oth_std = np.asarray(other_results["flux_spectra_std_dev"], dtype=float)

    peak = max(float(np.max(ref_flux)) if ref_flux.size else 0.0,
               float(np.max(oth_flux)) if oth_flux.size else 0.0)
    # relative threshold (6 decades below the peak), matching the plot's display
    # floor; falls back to the absolute threshold for an all-zero spectrum.
    thr = peak / 1e6 if peak > 0 else flux_threshold

    chi_squared_terms = []
    rel_diffs = []
    weighted_rel_diffs = []
    flux_weights = []
    ref_total = float(np.sum(ref_flux[ref_flux > 0]))
    oth_total = float(np.sum(oth_flux[oth_flux > 0]))
    ref_only_flux = 0.0  # reference flux in bins where the other is ~0
    oth_only_flux = 0.0  # other flux in bins where the reference is ~0

    for i in range(len(ref_flux)):
        r = ref_flux[i]
        o = oth_flux[i]
        r_on = r >= thr
        o_on = o >= thr
        if r_on and o_on:
            combined_std = np.sqrt(ref_std[i] ** 2 + oth_std[i] ** 2)
            if combined_std > 0:
                chi_squared_terms.append(((o - r) / combined_std) ** 2)
            rel_diff = abs((o - r) / r) * 100
            if np.isfinite(rel_diff):
                rel_diffs.append(rel_diff)
                weight = max(r, o)
                weighted_rel_diffs.append(rel_diff * weight)
                flux_weights.append(weight)
        elif o_on and not r_on:
            oth_only_flux += o
        elif r_on and not o_on:
            ref_only_flux += r

    chi_squared = np.sum(chi_squared_terms)
    dof = len(chi_squared_terms)
    reduced_chi_squared = chi_squared / dof if dof > 0 else 0.0

    median_rel_diff = float(np.median(rel_diffs)) if rel_diffs else 0.0
    weighted_avg_rel_diff = (
        float(np.sum(weighted_rel_diffs) / np.sum(flux_weights)) if flux_weights else 0.0
    )
    other_only_flux_pct = (oth_only_flux / oth_total * 100) if oth_total > 0 else 0.0
    ref_only_flux_pct = (ref_only_flux / ref_total * 100) if ref_total > 0 else 0.0

    if reduced_chi_squared < 1.5:
        quality = "Excellent"
    elif reduced_chi_squared < 3.0:
        quality = "Good"
    elif reduced_chi_squared < 5.0:
        quality = "Fair"
    else:
        quality = "Poor"

    return {
        "reduced_chi_squared": reduced_chi_squared,
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
