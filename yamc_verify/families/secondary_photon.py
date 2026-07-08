"""Secondary-photon (coupled neutron+photon) sphere/broomstick verification family.

One implementation for both coupled neutron-photon notebooks
(sphere/broomstick x endf-b8.1). Geometry comes from
:mod:`yamc_verify.geometry` and all run parameters from the task registry, so
the notebooks reduce to a few lines.

Unlike the neutron/photon-only families, the YAMC model here is driven with
``transport_secondary_photons=True`` (it couples neutron+photon transport);
``verbose=[]`` is added purely to silence YAMC output and is orthogonal to the
coupling. OpenMC is silenced with ``verbosity=1`` + ``output=False`` and uses
``photon_transport=True`` with ``electron_treatment='led'`` so its local energy
deposition matches YAMC's photon KERMA. The photon ``heating`` score is a KERMA
special-case: OpenMC sums a ``ParticleFilter(["photon","electron","positron"])``
tally to compare like-for-like against YAMC's photon-booked charged-particle KE.

Two spectra are produced per item (neutron VITAMIN-J-175, photon VITAMIN-J-42),
and the results dict carries both ``n_*`` and ``p_*`` scalar/spectrum sets.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import discovery
from ..geometry import openmc_region, yamc_region
from ..plotting import BACKEND_STYLES, RATIO_COLORS
from ..runstep import run_step
from ..tasks import Task

# Neutron scalar scores
NEUTRON_SCALAR_SCORES = [
    "flux", "heating", "heating-local", "total", "absorption",
    "H1-production", "H2-production", "H3-production",
    "He3-production", "He4-production",
]

# Photon scalar scores
PHOTON_SCALAR_SCORES = [
    "flux", "coherent-scatter", "incoherent-scatter",
    "photoelectric", "pair-production", "heating",
]

# Score-to-MT mapping for deduplication
SCORE_TO_MT = {
    "total": 1, "absorption": 27, "heating": 301, "heating-local": 901,
    "H1-production": 203, "H2-production": 204, "H3-production": 205,
    "He3-production": 206, "He4-production": 207,
    "coherent-scatter": 502, "incoherent-scatter": 504,
    "photoelectric": 522, "pair-production": 516,
}

# Electron-treatment matching. yamc and OpenMC are compared under the SAME
# treatment; a mismatch (e.g. yamc's default TTB vs OpenMC LED) makes the
# low-energy photon spectrum diverge spuriously (yamc issue #179). Keys are the
# benchmark/OpenMC spelling; values are the yamc spelling (yamc says "local"
# where OpenMC says "led").
_YAMC_ELECTRON_TREATMENT = {"led": "local", "ttb": "ttb"}


def _treatments(task: Task) -> list:
    """Electron treatments to run (matched on both codes). Reads a list from
    ``extras['electron_treatments']``, else falls back to the single
    ``extras['electron_treatment']`` (default ``['led']``)."""
    ts = task.extras.get("electron_treatments")
    if ts:
        return list(ts)
    return [task.extras.get("electron_treatment", "led")]


# --------------------------------------------------------------------------
# Simulation runners
# --------------------------------------------------------------------------
def run_openmc_simulation(nuclide, neutron_path, element, photon_path, task: Task,
                          electron_treatment: str = "led"):
    """Run OpenMC coupled neutron-photon simulation for a single nuclide."""
    import openmc as mc

    neutron_group_structure = task.extras["neutron_group_structure"]
    photon_group_structure = task.extras["photon_group_structure"]

    mc.reset_auto_ids()
    mc.config['cross_sections'] = f"{task.nuclear_data_dir}cross_sections.xml"

    # Geometry (injected)
    region = openmc_region(task.geometry, task.dims)

    # Material
    material = mc.Material()
    material.temperature = 294
    material.add_nuclide(nuclide, 1.0)
    material.set_density("g/cm3", 1.0)

    cell = mc.Cell(name=task.geometry, region=region, fill=material)
    geometry = mc.Geometry([cell])

    # Source: 14.06 MeV isotropic neutrons at origin
    source = mc.IndependentSource(
        space=mc.stats.Point([0, 0, 0]),
        energy=mc.stats.Discrete([task.source_energy], [1.0]),
        angle=mc.stats.Isotropic(),
    )

    settings = mc.Settings(
        particles=task.particles,
        batches=task.batches,
        source=source,
        seed=1,
        run_mode='fixed source',
        output={'tallies': False, 'summary': False},
    )
    settings.verbosity = 1  # quiet OpenMC (combined with output=False on run)
    settings.photon_transport = True
    # Matched to yamc for a like-for-like comparison: 'led' (both deposit e-/e+
    # KE locally, no bremsstrahlung -- isolates photon production + local KERMA)
    # or 'ttb' (both radiate thick-target bremsstrahlung). A mismatch (yamc TTB
    # vs OpenMC LED) makes the low-energy photon spectrum diverge spuriously
    # (yamc issue #179).
    settings.electron_treatment = electron_treatment

    # Filters and tallies
    cell_filter = mc.CellFilter(cell)
    neutron_filter = mc.ParticleFilter("neutron")
    photon_filter = mc.ParticleFilter("photon")
    tallies = []

    # Neutron scalar scores
    for score in NEUTRON_SCALAR_SCORES:
        t = mc.Tally()
        t.filters = [cell_filter, neutron_filter]
        t.scores = [score]
        t.name = f"n_{score}"
        tallies.append(t)

    # Per-MT neutron reactions
    in_neu = mc.data.IncidentNeutron.from_hdf5(neutron_path)
    named_mts = set(SCORE_TO_MT.values())
    mt_list = sorted(in_neu.reactions.keys())
    for mt in mt_list:
        if mt in named_mts:
            continue
        t = mc.Tally()
        t.filters = [cell_filter, neutron_filter]
        t.scores = [str(mt)]
        t.name = str(mt)
        tallies.append(t)

    # Photon scalar scores.
    #
    # The photon "heating" score is special-cased below. YAMC does not transport
    # electrons/positrons: it deposits ALL charged-particle (e-/e+) kinetic energy
    # locally at the photon interaction (KERMA) and books it under the photon. OpenMC
    # instead transports the e-/e+ and books their deposit under the electron/positron
    # particle, so a photon-filtered heating tally in OpenMC EXCLUDES it. To compare
    # like-for-like against YAMC's photon KERMA we filter heating on
    # photon + electron + positron and sum the three bins (see collection loop).
    # All other photon scores stay photon-filtered (they are photon-specific).
    for score in PHOTON_SCALAR_SCORES:
        if score == "heating":
            continue
        t = mc.Tally()
        t.filters = [cell_filter, photon_filter]
        t.scores = [score]
        t.name = f"p_{score}"
        tallies.append(t)

    # Photon heating = KERMA: photon + electron + positron deposit, summed.
    photon_kerma_filter = mc.ParticleFilter(["photon", "electron", "positron"])
    t_p_heating = mc.Tally()
    t_p_heating.filters = [cell_filter, photon_kerma_filter]
    t_p_heating.scores = ["heating"]
    t_p_heating.name = "p_heating"
    tallies.append(t_p_heating)

    # Neutron flux spectrum
    n_energy_filter = mc.EnergyFilter.from_group_structure(neutron_group_structure)
    n_bins_2d = n_energy_filter.bins
    n_energy_bins = np.concatenate([[n_bins_2d[0, 0]], n_bins_2d[:, 1]])
    t_n_spec = mc.Tally()
    t_n_spec.filters = [cell_filter, neutron_filter, n_energy_filter]
    t_n_spec.scores = ["flux"]
    t_n_spec.name = "neutron_spectrum"
    tallies.append(t_n_spec)

    # Photon flux spectrum
    p_energy_filter = mc.EnergyFilter.from_group_structure(photon_group_structure)
    p_bins_2d = p_energy_filter.bins
    p_energy_bins = np.concatenate([[p_bins_2d[0, 0]], p_bins_2d[:, 1]])
    t_p_spec = mc.Tally()
    t_p_spec.filters = [cell_filter, photon_filter, p_energy_filter]
    t_p_spec.scores = ["flux"]
    t_p_spec.name = "photon_spectrum"
    tallies.append(t_p_spec)

    # Run
    model = mc.Model(geometry=geometry, settings=settings, tallies=tallies)
    sp_filename = model.run(apply_tally_results=True, output=False)
    sp = mc.StatePoint(sp_filename)

    transport_time = sp.runtime['transport']
    total_particles = task.particles * task.batches
    particles_per_second = total_particles / transport_time

    # Collect results
    n_scalars = {}
    n_stds = {}
    p_scalars = {}
    p_stds = {}
    mt_scalars = {}
    mt_stds = {}

    for t in tallies:
        if t.name in ("neutron_spectrum", "photon_spectrum"):
            continue
        if t.name == "p_heating":
            # ParticleFilter(photon, electron, positron) -> one bin per particle.
            # Sum the bins to recover the full photon KERMA (matches YAMC); combine
            # the per-bin standard deviations in quadrature (bins are independent).
            mean_bins = t.mean.flatten()
            std_bins = t.std_dev.flatten()
            p_scalars["heating"] = float(np.sum(mean_bins))
            p_stds["heating"] = float(np.sqrt(np.sum(std_bins**2)))
            continue
        val = float(t.mean.flatten()[0])
        std = float(t.std_dev.flatten()[0])
        if t.name.startswith("n_"):
            n_scalars[t.name[2:]] = val
            n_stds[t.name[2:]] = std
        elif t.name.startswith("p_"):
            p_scalars[t.name[2:]] = val
            p_stds[t.name[2:]] = std
        else:
            mt_scalars[t.name] = val
            mt_stds[t.name] = std

    n_scalars.update(mt_scalars)
    n_stds.update(mt_stds)

    return {
        'nuclide': nuclide,
        'code': 'openmc',
        'particles/s': particles_per_second,
        'n_scalars': n_scalars,
        'n_stds': n_stds,
        'p_scalars': p_scalars,
        'p_stds': p_stds,
        'n_spectrum_mean': np.array(t_n_spec.mean).squeeze(),
        'n_spectrum_std': np.array(t_n_spec.std_dev).squeeze(),
        'n_energy_bins': n_energy_bins,
        'p_spectrum_mean': np.array(t_p_spec.mean).squeeze(),
        'p_spectrum_std': np.array(t_p_spec.std_dev).squeeze(),
        'p_energy_bins': p_energy_bins,
    }


def run_yamc_simulation(nuclide, neutron_path, element, photon_path, task: Task, compute="cpu",
                        electron_treatment: str = "led"):
    """Run YAMC coupled neutron-photon simulation for a single nuclide. Returns (results_dict, model)."""
    import yamc as mc

    neutron_group_structure = task.extras["neutron_group_structure"]
    photon_group_structure = task.extras["photon_group_structure"]

    # Geometry (injected)
    region = yamc_region(task.geometry, task.dims)

    # Material with both neutron and photon data
    material = mc.Material(
        composition={nuclide: 1.0},
        density=1.0,
        temperature=294,
        id=1,
    )
    material.read_nuclear_data(task.library)

    cell = mc.Cell(id=1, name=task.geometry, region=region, material=material)
    geometry = mc.Geometry([cell])

    # Source: 14.06 MeV isotropic neutrons at origin
    source = mc.NeutronSource(
        position=(0, 0, 0),
        direction=mc.sources.Isotropic(),
        energy=mc.sources.Discrete([task.source_energy], [1.0]),
    )

    tallies = []

    # Neutron scalar scores
    for score in NEUTRON_SCALAR_SCORES:
        t = mc.Tally(
            scores=[str(score)],
            name=f"n_{score}",
            cells=cell,
            particle="neutron",
        )
        tallies.append(t)

    # Per-MT neutron reactions
    named_mts = set(SCORE_TO_MT.values())
    mt_list = material.reaction_mts
    for mt in mt_list:
        if mt in named_mts:
            continue
        t = mc.Tally(
            scores=[str(mt)],
            name=str(mt),
            cells=cell,
            particle="neutron",
        )
        tallies.append(t)

    # Photon scalar scores
    for score in PHOTON_SCALAR_SCORES:
        t = mc.Tally(
            scores=[str(score)],
            name=f"p_{score}",
            cells=cell,
            particle="photon",
        )
        tallies.append(t)

    # Neutron flux spectrum
    t_n_spec = mc.Tally(
        scores=["flux"],
        name="neutron_spectrum",
        cells=cell,
        particle="neutron",
        energy_group_structure=neutron_group_structure,
    )
    tallies.append(t_n_spec)
    n_energy_bins = np.array(t_n_spec.energy_bins)

    # Photon flux spectrum
    t_p_spec = mc.Tally(
        scores=["flux"],
        name="photon_spectrum",
        cells=cell,
        particle="photon",
        energy_group_structure=photon_group_structure,
    )
    tallies.append(t_p_spec)
    p_energy_bins = np.array(t_p_spec.energy_bins)

    # Run
    model = mc.Model(
        geometry=geometry,
        tallies=tallies,
        source=source,
        transport_secondary_photons=True,
        # Matched to the OpenMC reference (yamc "local" == OpenMC "led").
        electron_treatment=_YAMC_ELECTRON_TREATMENT[electron_treatment],
        verbose=[],
    )
    _t0 = time.perf_counter()
    sim_results = model.simulate_transport(
        total_particles=task.particles * task.batches,
        seed=1,
        compute=compute,
    )
    _elapsed = time.perf_counter() - _t0

    # Collect results
    n_scalars = {}
    n_stds = {}
    p_scalars = {}
    p_stds = {}
    mt_scalars = {}
    mt_stds = {}

    for t in tallies:
        if t.name in ("neutron_spectrum", "photon_spectrum"):
            continue
        r = sim_results[t]
        val = r.mean[0]
        std = r.standard_deviation[0]
        if t.name.startswith("n_"):
            n_scalars[t.name[2:]] = val
            n_stds[t.name[2:]] = std
        elif t.name.startswith("p_"):
            p_scalars[t.name[2:]] = val
            p_stds[t.name[2:]] = std
        else:
            mt_scalars[t.name] = val
            mt_stds[t.name] = std

    n_scalars.update(mt_scalars)
    n_stds.update(mt_stds)

    n_spec = sim_results[t_n_spec]
    p_spec = sim_results[t_p_spec]

    results = {
        'nuclide': nuclide,
        'code': f'yamc-{compute}',
        'particles/s': (
            sim_results.particles_per_second
            if sim_results.particles_per_second is not None
            else (task.particles * task.batches) / _elapsed if _elapsed > 0 else 0
        ),
        'n_scalars': n_scalars,
        'n_stds': n_stds,
        'p_scalars': p_scalars,
        'p_stds': p_stds,
        'n_spectrum_mean': np.array(n_spec.mean).squeeze(),
        'n_spectrum_std': np.array(n_spec.standard_deviation).squeeze(),
        'n_energy_bins': n_energy_bins,
        'p_spectrum_mean': np.array(p_spec.mean).squeeze(),
        'p_spectrum_std': np.array(p_spec.standard_deviation).squeeze(),
        'p_energy_bins': p_energy_bins,
    }
    return results, model


# --------------------------------------------------------------------------
# Plots, metrics and tables (family-specific; moved verbatim from the notebook)
# --------------------------------------------------------------------------
def create_step_trace(energy_bins, values, name, color, dash=None):
    """Create a step plot trace for Plotly (similar to matplotlib step with where='post')."""
    x_step = []
    y_step = []
    for i in range(len(values)):
        x_step.extend([energy_bins[i], energy_bins[i+1]])
        y_step.extend([values[i], values[i]])

    return go.Scatter(
        x=x_step,
        y=y_step,
        mode='lines',
        name=name,
        line=dict(color=color, width=2, dash=dash),
    )


def create_comparison_plot(energy_bins, openmc_flux, openmc_std, yamc_flux, yamc_std, task: Task,
                           title_prefix="Flux", yamc_label='YAMC'):
    """Create a Plotly subplot with flux spectra (top) and ratio (bottom)."""
    flux_decades = task.extras.get("flux_decades", 6)

    # Per-backend colours (shared with neutron/photon) so OpenMC CPU / YAMC CPU /
    # YAMC GPU are distinct: the figure is drawn once per yamc backend, so the
    # yamc series colour is keyed off yamc_label ("YAMC CPU" / "YAMC GPU").
    omc_style = BACKEND_STYLES["OpenMC CPU"]
    ymc_style = BACKEND_STYLES.get(yamc_label, dict(color="red", dash="dash", marker="square"))
    ratio_color = RATIO_COLORS.get(f"{yamc_label} / OpenMC CPU", "green")

    energy_bins = np.asarray(energy_bins)
    bin_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])

    omc_flux = openmc_flux.copy()
    omc_std = openmc_std.copy()
    ymc_flux = yamc_flux.copy()
    ymc_std = yamc_std.copy()

    # Normalize by lethargy bin width
    lethargy_bin_width = np.log10(energy_bins[1:] / energy_bins[:-1])
    omc_flux = omc_flux / lethargy_bin_width
    omc_std = omc_std / lethargy_bin_width
    ymc_flux = ymc_flux / lethargy_bin_width
    ymc_std = ymc_std / lethargy_bin_width

    # Find the lowest energy bin with non-zero flux
    has_flux = (omc_flux > 0) | (ymc_flux > 0)
    if np.any(has_flux):
        first_nonzero_idx = np.argmax(has_flux)
        min_plot_energy = energy_bins[first_nonzero_idx]
    else:
        min_plot_energy = energy_bins[0]

    max_plot_energy = energy_bins[-1] * 1.1

    max_flux = max(
        np.max(omc_flux[omc_flux > 0]) if np.any(omc_flux > 0) else 1e-10,
        np.max(ymc_flux[ymc_flux > 0]) if np.any(ymc_flux > 0) else 1e-10
    )
    flux_threshold = max_flux / (10 ** flux_decades)

    omc_flux_plot = np.where(omc_flux >= flux_threshold, omc_flux, np.nan)
    ymc_flux_plot = np.where(ymc_flux >= flux_threshold, ymc_flux, np.nan)
    omc_std_plot = np.where(omc_flux >= flux_threshold, omc_std, np.nan)
    ymc_std_plot = np.where(ymc_flux >= flux_threshold, ymc_std, np.nan)

    # Calculate ratio with error propagation
    valid = (omc_flux > 0) & (ymc_flux > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(valid, ymc_flux / omc_flux, np.nan)
        rel_err_yamc = np.where(ymc_flux > 0, ymc_std / ymc_flux, 0)
        rel_err_openmc = np.where(omc_flux > 0, omc_std / omc_flux, 0)
    ratio_rel_err = np.sqrt(rel_err_yamc**2 + rel_err_openmc**2)
    ratio_std = np.where(valid, ratio * ratio_rel_err, np.nan)  # noqa: F841 (verbatim from notebook)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{title_prefix} per Unit Lethargy', f'{yamc_label} / OpenMC CPU Ratio')
    )

    fig.add_trace(
        create_step_trace(energy_bins, omc_flux_plot, 'OpenMC CPU', omc_style["color"],
                          dash=omc_style["dash"]),
        row=1, col=1
    )
    fig.add_trace(
        create_step_trace(energy_bins, ymc_flux_plot, yamc_label, ymc_style["color"],
                          dash=ymc_style["dash"]),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=bin_centers, y=omc_flux_plot, mode='markers',
            name='OpenMC CPU error',
            marker=dict(color=omc_style["color"], size=4, symbol=omc_style["marker"]),
            error_y=dict(type='data', array=omc_std_plot, visible=True,
                         color=omc_style["color"], thickness=1),
            showlegend=False,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=bin_centers, y=ymc_flux_plot, mode='markers',
            name=f'{yamc_label} error',
            marker=dict(color=ymc_style["color"], size=4, symbol=ymc_style["marker"]),
            error_y=dict(type='data', array=ymc_std_plot, visible=True,
                         color=ymc_style["color"], thickness=1),
            showlegend=False,
        ),
        row=1, col=1
    )

    fig.add_trace(
        create_step_trace(energy_bins, ratio, f'{yamc_label} / OpenMC CPU', ratio_color),
        row=2, col=1
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_xaxes(type="log", title_text="Energy (eV)",
                     range=[np.log10(min_plot_energy), np.log10(max_plot_energy)],
                     exponentformat='e', row=2, col=1)
    fig.update_xaxes(type="log",
                     range=[np.log10(min_plot_energy), np.log10(max_plot_energy)],
                     exponentformat='e', row=1, col=1)
    fig.update_yaxes(type="log", title_text="Flux / Lethargy", exponentformat='e', row=1, col=1)
    fig.update_yaxes(title_text="Ratio", range=[0.5, 1.5], row=2, col=1)

    fig.update_layout(
        height=700,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=20, t=40, b=40)
    )

    return fig


def calculate_spectral_agreement(openmc_flux, yamc_flux, openmc_std, yamc_std, flux_threshold=1e-6):
    """Calculate statistical agreement metrics for flux spectra comparison."""

    chi_squared_terms = []
    rel_diffs = []
    weighted_rel_diffs = []
    flux_weights = []

    for i in range(len(openmc_flux)):
        omc = openmc_flux[i]
        ymc = yamc_flux[i]
        omc_std = openmc_std[i]
        ymc_std = yamc_std[i]

        combined_std = np.sqrt(omc_std**2 + ymc_std**2)
        if combined_std > 0:
            chi_squared_terms.append(((ymc - omc) / combined_std)**2)

        if omc > flux_threshold or ymc > flux_threshold:
            if omc > 0:
                rel_diff = abs((ymc - omc) / omc) * 100
                if np.isfinite(rel_diff):
                    rel_diffs.append(rel_diff)
                    weight = max(omc, ymc)
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
        'reduced_chi_squared': reduced_chi_squared,
        'degrees_of_freedom': dof,
        'median_rel_diff': median_rel_diff,
        'weighted_avg_rel_diff': weighted_avg_rel_diff,
        'quality': quality,
        'bins_analyzed': len(rel_diffs),
        'total_bins': len(openmc_flux)
    }


def create_agreement_table(stats):
    """Create a summary table of spectral agreement statistics."""
    rows = [
        {'Metric': 'Reduced Chi-Squared', 'Value': f"{stats['reduced_chi_squared']:.2f}",
         'Note': f"(ideal ~ 1.0, dof={stats['degrees_of_freedom']})"},
        {'Metric': 'Weighted Avg Rel. Diff', 'Value': f"{stats['weighted_avg_rel_diff']:.2f}%",
         'Note': '(flux-weighted)'},
        {'Metric': 'Median Rel. Diff', 'Value': f"{stats['median_rel_diff']:.2f}%",
         'Note': '(robust to outliers)'},
        {'Metric': 'Agreement Quality', 'Value': stats['quality'],
         'Note': f"({stats['bins_analyzed']}/{stats['total_bins']} bins analyzed)"},
    ]
    return pd.DataFrame(rows)


def _mt_descriptions():
    import yamc

    return yamc.data.reaction_names()


# Photon interaction MT descriptions
PHOTON_MT_DESCRIPTIONS = {
    'flux': 'Total photon flux',
    'coherent-scatter': 'Coherent (Rayleigh) scatter (MT 502)',
    'incoherent-scatter': 'Incoherent (Compton) scatter (MT 504)',
    'photoelectric': 'Photoelectric absorption (MT 522)',
    'pair-production': 'Pair production (MT 516)',
    # Heating uses DIFFERENT particle filters per code (see the note above the
    # comparison tables): yamc books all charged-particle (e-/e+) KE locally
    # under the photon, so the OpenMC side sums photon+electron+positron to
    # compare like-for-like against yamc's photon KERMA.
    'heating': 'Heating (KERMA) [yamc: photon filter, e-/e+ KE deposited locally | OpenMC: sum of photon+electron+positron filters]',
}


def get_mt_and_reaction(score_name, mt_descriptions):
    """Get MT number and reaction description for a score."""
    try:
        mt = int(score_name)
        return str(mt), mt_descriptions.get(mt, f'MT {mt}')
    except ValueError:
        pass

    if score_name in SCORE_TO_MT:
        mt = SCORE_TO_MT[score_name]
        return str(mt), score_name

    return '-', score_name


def _best(scalars, stds, score_names):
    """Pick the best (non-nan, non-zero preferred) value across alias score names."""
    bv, bs = np.nan, np.nan
    for name in score_names:
        v = scalars.get(name, np.nan)
        s = stds.get(name, np.nan)
        if not np.isnan(v) and v != 0:
            if np.isnan(bv) or bv == 0:
                bv, bs = v, s
        elif np.isnan(bv):
            bv, bs = v, s
    return bv, bs


def _fmt(v, s):
    return 'N/A' if np.isnan(v) else f"{v:.4e} +/- {s:.2e}"


def _reldiff(v, ref):
    if np.isnan(ref) or ref == 0 or np.isnan(v):
        return 'N/A'
    return f"{(v - ref) / ref * 100:.2f}%"


def create_neutron_tally_table(openmc_results, yamc_by_backend, mt_descriptions):
    """Neutron tally comparison, deduplicated by MT, one column per yamc backend."""
    all_tallies = set(openmc_results['n_scalars'])
    for res in yamc_by_backend.values():
        all_tallies |= set(res['n_scalars'])

    mt_to_scores = {}
    for name in all_tallies:
        mt_display, _ = get_mt_and_reaction(name, mt_descriptions)
        mt_to_scores.setdefault(mt_display, []).append(name)

    rows = []
    for mt_display, score_names in mt_to_scores.items():
        ov, osd = _best(openmc_results['n_scalars'], openmc_results['n_stds'], score_names)
        yvals = {b: _best(res['n_scalars'], res['n_stds'], score_names)
                 for b, res in yamc_by_backend.items()}

        reaction = None
        if mt_display != '-':
            try:
                reaction = mt_descriptions.get(int(mt_display))
            except ValueError:
                pass
        if reaction is None:
            _, reaction = get_mt_and_reaction(score_names[0], mt_descriptions)

        ov_zero = ov == 0 or np.isnan(ov)
        all_y_zero = all((v == 0 or np.isnan(v)) for v, _ in yvals.values())
        if ov_zero and all_y_zero:
            continue

        row = {'MT': mt_display, 'Reaction': reaction, 'OpenMC CPU': _fmt(ov, osd)}
        for b, (v, s) in yvals.items():
            row[f'YAMC {b.upper()}'] = _fmt(v, s)
        for b, (v, _s) in yvals.items():
            row[f'Rel. Diff {b} (%)'] = _reldiff(v, ov)
        row['_mt_sort'] = int(mt_display) if mt_display != '-' else 9999
        rows.append(row)

    rows.sort(key=lambda r: r['_mt_sort'])
    for r in rows:
        del r['_mt_sort']
    return pd.DataFrame(rows)


def create_photon_tally_table(openmc_results, yamc_by_backend):
    """Photon tally comparison, one column per yamc backend."""
    all_tallies = set(openmc_results['p_scalars'])
    for res in yamc_by_backend.values():
        all_tallies |= set(res['p_scalars'])

    rows = []
    for name in sorted(all_tallies):
        ov = openmc_results['p_scalars'].get(name, np.nan)
        osd = openmc_results['p_stds'].get(name, np.nan)
        yvals = {b: (res['p_scalars'].get(name, np.nan), res['p_stds'].get(name, np.nan))
                 for b, res in yamc_by_backend.items()}

        ov_zero = ov == 0 or np.isnan(ov)
        all_y_zero = all((v == 0 or np.isnan(v)) for v, _ in yvals.values())
        if ov_zero and all_y_zero:
            continue

        row = {'Score': name, 'Reaction': PHOTON_MT_DESCRIPTIONS.get(name, name),
               'OpenMC CPU': _fmt(ov, osd)}
        for b, (v, s) in yvals.items():
            row[f'YAMC {b.upper()}'] = _fmt(v, s)
        for b, (v, _s) in yvals.items():
            row[f'Rel. Diff {b} (%)'] = _reldiff(v, ov)
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Discovery / run / display
# --------------------------------------------------------------------------
def discover(task: Task, run_openmc: bool) -> list[tuple]:
    """Discover coupled neutron+photon nuclides as (name, npath, element, ppath) tuples."""
    return discovery.discover_coupled_nuclides(task.nuclear_data_dir, task.library, run_openmc)


def _metas(task: Task, electron_treatment: str):
    from cache_utils import yamc_fingerprint

    openmc_meta = {
        "library": task.library,
        "electron_treatment": electron_treatment,
        "particles": task.particles,
        "batches": task.batches,
        "source_energy": task.source_energy,
        "neutron_group_structure": task.extras["neutron_group_structure"],
        "photon_group_structure": task.extras["photon_group_structure"],
    }
    if task.geometry == "sphere":
        openmc_meta["sphere_radius"] = task.dims.sphere_radius
    else:
        openmc_meta["cylinder_radius"] = task.dims.cylinder_radius
        openmc_meta["cylinder_half_height"] = task.dims.cylinder_half_height
    yamc_meta = {**openmc_meta, **yamc_fingerprint()}
    return openmc_meta, yamc_meta


def run_all(task: Task, backends=None, cache_dir: str = "cache"):
    """Run every backend for every discovered coupled nuclide.

    Returns ``(all_results, first_yamc_model)``. ``cache_dir`` is resolved
    relative to the current working directory (the notebook's own directory
    under nbconvert), so the default ``"cache"`` lands in
    ``tasks/secondary-photon/<geometry>/cache`` as before. Both OpenMC and
    YAMC are cached (YAMC keyed by build fingerprint).
    """
    from ..backends import resolve

    b = backends if backends is not None else resolve()
    treatments = _treatments(task)
    nuclides = discover(task, b.run_openmc)
    print(f"Running coupled simulations for {len(nuclides)} nuclides "
          f"x {len(treatments)} electron treatment(s) {treatments}...")
    for nuc, _, elem, _ in nuclides:
        print(f"  {nuc} ({elem})")

    all_results = {}
    first_yamc_model = None
    for nuclide, neutron_path, element, photon_path in nuclides:
        print(f"\nProcessing {nuclide} ({element})...")
        per_nuclide = {'element': element}

        # One matched (openmc, yamc) pair per electron treatment. The cache code
        # is qualified by treatment (openmc-led / openmc-ttb / yamc-cpu-led ...)
        # so both live side by side; the result keys carry the treatment suffix
        # and `display_results` groups by treatment.
        for treatment in treatments:
            openmc_meta, yamc_meta = _metas(task, treatment)

            if b.run_openmc:
                try:
                    per_nuclide[f'openmc-{treatment}'] = run_step(
                        cache_dir, task.library, f"openmc-{treatment}", nuclide,
                        lambda n=nuclide, npath=neutron_path, el=element, ppath=photon_path,
                        t=treatment:
                            run_openmc_simulation(n, npath, el, ppath, task, electron_treatment=t),
                        meta=openmc_meta, cache=True)
                    print(f"  OpenMC ({treatment}): "
                          f"{per_nuclide[f'openmc-{treatment}']['particles/s']:.0f} particles/s")
                except Exception as e:  # noqa: BLE001 - isolate backend failures
                    print(f"  OpenMC ({treatment}) Error: {e}")

            for compute in b.computes:
                captured = {}

                def _run(n=nuclide, npath=neutron_path, el=element, ppath=photon_path,
                         c=compute, t=treatment, cap=captured):
                    res, model = run_yamc_simulation(
                        n, npath, el, ppath, task, compute=c, electron_treatment=t)
                    cap["model"] = model
                    return res

                try:
                    yamc_results = run_step(
                        cache_dir, task.library, f"yamc-{compute}-{treatment}", nuclide, _run,
                        meta=yamc_meta, cache=task.caches_yamc)
                    per_nuclide[f'yamc-{compute}-{treatment}'] = yamc_results
                    if first_yamc_model is None and "model" in captured:
                        first_yamc_model = captured["model"]
                    print(f"  YAMC ({compute}, {treatment}): "
                          f"{yamc_results['particles/s']:.0f} particles/s")
                except Exception as e:  # noqa: BLE001
                    print(f"  YAMC ({compute}, {treatment}) Error: {e}")

        if len(per_nuclide) > 1:  # more than just 'element'
            all_results[nuclide] = per_nuclide

    print(f"\nCompleted {len(all_results)} / {len(nuclides)} nuclides successfully.")
    return all_results, first_yamc_model


def display_geometry(model, task: Task):
    """Embed the yamc geometry plot as an iframe (called from a notebook)."""
    import base64

    from IPython.display import HTML, display

    if model is None:
        print("No successful YAMC simulation to plot.")
        return
    html_plot = model.plot(basis="xz", n_samples=500, axis_units="cm")
    html_plot.save(f"{task.geometry}_model_plot.html")
    encoded = base64.b64encode(html_plot.html.encode()).decode()
    display(HTML(f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="600" frameborder="0"></iframe>'))


def display_results(task: Task, all_results: dict):
    """Render per-nuclide neutron+photon comparison plots, agreement tables, and
    tally tables, grouped by electron treatment (matched yamc/OpenMC pairs)."""
    from IPython.display import Markdown, display

    mt_descriptions = _mt_descriptions()
    treatments = _treatments(task)

    for nuclide, results in all_results.items():
        element = results['element']
        display(Markdown(f"## {nuclide} ({element})"))
        for treatment in treatments:
            openmc_results = results.get(f'openmc-{treatment}')
            # keys look like `yamc-<compute>-<treatment>`; strip both ends.
            yamc_by_backend = {
                k[len("yamc-"):].rsplit(f"-{treatment}", 1)[0]: res
                for k, res in results.items()
                if k.startswith("yamc-") and k.endswith(f"-{treatment}")
            }
            display(Markdown(f"### Electron treatment: {treatment.upper()}"))
            _display_one_treatment(
                nuclide, element, openmc_results, yamc_by_backend, task, mt_descriptions)
        print()  # spacing between nuclides


def _display_one_treatment(nuclide, element, openmc_results, yamc_by_backend, task,
                           mt_descriptions):
    """Render the matched OpenMC-vs-yamc comparison for one electron treatment."""
    from IPython.display import Markdown, display

    # Performance - one entry per code/backend that ran
    perf_parts = []
    if openmc_results is not None:
        perf_parts.append(f"OpenMC: {openmc_results['particles/s']:,.0f} particles/s")
    for b, res in yamc_by_backend.items():
        perf_parts.append(f"YAMC ({b}): {res['particles/s']:,.0f} particles/s")
    display(Markdown("**Performance:** " + " | ".join(perf_parts)))

    # Comparisons need OpenMC plus at least one yamc backend.
    if openmc_results is None or not yamc_by_backend:
        only = "OpenMC" if openmc_results is not None else "YAMC"
        display(Markdown(
            f"*Only {only} ran; comparison plots and tables are skipped.*"
        ))
        return

    # Per-backend spectra plots and spectral agreement (neutron + photon).
    for b, res in yamc_by_backend.items():
        n_fig = create_comparison_plot(
            openmc_results['n_energy_bins'],
            openmc_results['n_spectrum_mean'], openmc_results['n_spectrum_std'],
            res['n_spectrum_mean'], res['n_spectrum_std'], task,
            title_prefix='Neutron Flux', yamc_label=f'YAMC {b.upper()}'
        )
        n_fig.update_layout(title=f"{nuclide} - Neutron Flux Spectra (OpenMC CPU vs YAMC {b.upper()})")
        n_fig.show()
        display(Markdown(f"### Neutron Spectral Agreement (YAMC {b})"))
        n_stats = calculate_spectral_agreement(
            openmc_results['n_spectrum_mean'], res['n_spectrum_mean'],
            openmc_results['n_spectrum_std'], res['n_spectrum_std'],
        )
        display(create_agreement_table(n_stats).style.hide(axis='index'))

        p_fig = create_comparison_plot(
            openmc_results['p_energy_bins'],
            openmc_results['p_spectrum_mean'], openmc_results['p_spectrum_std'],
            res['p_spectrum_mean'], res['p_spectrum_std'], task,
            title_prefix='Photon Flux', yamc_label=f'YAMC {b.upper()}'
        )
        p_fig.update_layout(title=f"{nuclide} ({element}) - Photon Flux Spectra (OpenMC CPU vs YAMC {b.upper()})")
        p_fig.show()
        display(Markdown(f"### Photon Spectral Agreement (YAMC {b})"))
        p_stats = calculate_spectral_agreement(
            openmc_results['p_spectrum_mean'], res['p_spectrum_mean'],
            openmc_results['p_spectrum_std'], res['p_spectrum_std'],
        )
        display(create_agreement_table(p_stats).style.hide(axis='index'))

    # Combined tally tables, one column per backend.
    display(Markdown("### Neutron Tally Comparison"))
    display(create_neutron_tally_table(openmc_results, yamc_by_backend, mt_descriptions).style.hide(axis='index'))
    display(Markdown("### Photon Tally Comparison"))
    display(create_photon_tally_table(openmc_results, yamc_by_backend).style.hide(axis='index'))
