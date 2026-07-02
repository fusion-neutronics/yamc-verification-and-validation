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
    """Statistical agreement metrics comparing one spectrum to a reference."""
    ref_flux = ref_results["flux_spectra_mean"]
    ref_std = ref_results["flux_spectra_std_dev"]
    oth_flux = other_results["flux_spectra_mean"]
    oth_std = other_results["flux_spectra_std_dev"]

    chi_squared_terms = []
    rel_diffs = []
    weighted_rel_diffs = []
    flux_weights = []

    for i in range(len(ref_flux)):
        r = ref_flux[i]
        o = oth_flux[i]
        r_std = ref_std[i]
        o_std = oth_std[i]

        combined_std = np.sqrt(r_std**2 + o_std**2)
        if combined_std > 0:
            chi_squared_terms.append(((o - r) / combined_std) ** 2)

        if r > flux_threshold or o > flux_threshold:
            if r > 0:
                rel_diff = abs((o - r) / r) * 100
                if np.isfinite(rel_diff):
                    rel_diffs.append(rel_diff)
                    weight = max(r, o)
                    weighted_rel_diffs.append(rel_diff * weight)
                    flux_weights.append(weight)

    chi_squared = np.sum(chi_squared_terms)
    dof = len(chi_squared_terms)
    reduced_chi_squared = chi_squared / dof if dof > 0 else 0.0

    median_rel_diff = np.median(rel_diffs) if rel_diffs else 0.0
    weighted_avg_rel_diff = np.sum(weighted_rel_diffs) / np.sum(flux_weights) if flux_weights else 0.0

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
