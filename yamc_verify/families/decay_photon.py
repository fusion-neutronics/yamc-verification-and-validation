"""Decay-photon (D1S) sphere/broomstick verification family.

One implementation for both decay-photon notebooks (sphere/broomstick x
endf-b8.1). Geometry comes from :mod:`yamc_verify.geometry` and all run
parameters from the task registry, so the notebooks reduce to a few lines.
YAMC is silenced with ``verbose=[]`` and OpenMC with ``verbosity=1`` +
``output=False``.

The D1S workflow runs a coupled neutron+decay-photon transport, then
time-corrects the parent-nuclide-binned photon flux tally into a per-timestep
shutdown-dose spectrum (OpenMC via ``openmc.deplete.d1s``; YAMC via a
``PulseSchedule`` of ``Pulse``/``Cooldown`` steps). Metrics and plots are
per-timestep and therefore D1S-specific; they live in this module.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import discovery
from ..geometry import openmc_region, yamc_region
from ..runstep import run_step
from ..tasks import Task

FLUX_DECADES = 6


# --------------------------------------------------------------------------
# Simulation runners
# --------------------------------------------------------------------------
def run_openmc_d1s(nuclide, neutron_path, element, photon_path, task: Task):
    """Run OpenMC D1S simulation.

    Returns dict with TCF-corrected photon flux spectra at each timestep.
    """
    import openmc
    from openmc.deplete import d1s

    chain_file = discovery.resolve_openmc_chain(task.library)
    timesteps = task.extras["timesteps"]
    source_rates = task.extras["source_rates"]

    openmc.reset_auto_ids()
    openmc.config['cross_sections'] = os.path.join(task.nuclear_data_dir, 'cross_sections.xml')
    # OpenMC keeps the full chain - its fission-product-yield borrowing needs
    # nuclides outside any small reduce(...) reach.
    openmc.config['chain_file'] = chain_file

    material = openmc.Material()
    material.temperature = 294
    material.add_nuclide(nuclide, 1.0)
    material.set_density('g/cm3', 1.0)

    region = openmc_region(task.geometry, task.dims)

    cell = openmc.Cell(name=task.geometry, region=region, fill=material)
    geometry = openmc.Geometry([cell])

    source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([task.source_energy], [1]),
    )

    settings = openmc.Settings(
        particles=task.particles,
        batches=task.batches,
        source=source,
        seed=1,
        run_mode='fixed source',
        output={'tallies': False, 'summary': False},
    )
    settings.verbosity = 1  # quiet OpenMC (combined with output=False on run)
    settings.photon_transport = True
    settings.use_decay_photons = True

    cell_filter = openmc.CellFilter(cell)
    photon_filter = openmc.ParticleFilter(['photon'])
    energy_filter = openmc.EnergyFilter.from_group_structure(task.extras["photon_group_structure"])
    energy_bins_2d = energy_filter.bins
    energy_bins = np.concatenate([[energy_bins_2d[0, 0]], energy_bins_2d[:, 1]])

    tally = openmc.Tally(name='d1s_spectrum')
    tally.filters = [cell_filter, photon_filter, energy_filter]
    tally.scores = ['flux']

    model = openmc.Model(
        geometry, openmc.Materials([material]), settings,
        openmc.Tallies([tally]),
    )

    d1s.prepare_tallies(model=model)
    radionuclides = d1s.get_radionuclides(model)

    t0 = time.time()
    sp_path = model.run(output=False)
    elapsed = time.time() - t0

    with openmc.StatePoint(sp_path) as sp:
        tally_result = sp.get_tally(name='d1s_spectrum')

    tcf = d1s.time_correction_factors(
        nuclides=radionuclides, timesteps=timesteps,
        source_rates=source_rates, timestep_units='s',
    )

    spectra = []
    spectra_std = []
    for step_idx in range(len(timesteps)):
        corrected = d1s.apply_time_correction(
            tally=tally_result, time_correction_factors=tcf,
            index=step_idx + 1, sum_nuclides=True,
        )
        spectra.append(corrected.mean.flatten().tolist())
        spectra_std.append(corrected.std_dev.flatten().tolist())

    return {
        'radionuclides': radionuclides,
        'energy_bins': energy_bins,
        'spectra': spectra,
        'spectra_std': spectra_std,
        'elapsed': elapsed,
    }


def run_yamc_d1s(nuclide, neutron_path, element, photon_path, task: Task, compute="cpu"):
    """Run YAMC D1S simulation. Returns (results_dict, model)."""
    import yamc as mc

    transmutation_data = discovery.transmutation_data_path(task.library)
    reduce_level = task.extras["reduce_level"]
    timesteps = task.extras["timesteps"]
    source_rates = task.extras["source_rates"]

    material = mc.Material(
        composition={nuclide: 1.0},
        density=1.0,
        temperature=294,
        id=1,
    )
    material.read_nuclear_data(task.library)

    region = yamc_region(task.geometry, task.dims)

    cell = mc.Cell(id=1, name=task.geometry, region=region, material=material)
    geometry = mc.Geometry([cell])

    source = mc.NeutronSource(
        position=(0, 0, 0),
        direction=mc.sources.Isotropic(),
        energy=mc.sources.Discrete([task.source_energy], [1.0]),
    )

    # Reduce the YAMC transmutation network to nuclides reachable from the seed
    # (matches OpenMC's chain.reduce(...)); export to a local arrow dir and point
    # the global at it so radionuclides()/decay-photon channels pick it up.
    yamc_chain = mc.TransmutationChain(transmutation_data)
    yamc_reduced = yamc_chain.reduce([nuclide], level=reduce_level)
    reduced_path = os.path.join(os.getcwd(), '_reduced_transmutation.chain.arrow')
    yamc_reduced.export_to_arrow(reduced_path)
    mc.transmutation_data = reduced_path

    # Unstable activation products reachable from this model's materials.
    radionuclides = mc.Model(
        geometry=geometry, source=source, verbose=[],
    ).radionuclides()

    tally = mc.Tally(
        scores=['flux'],
        name='d1s_spectrum',
        cells=cell,
        particle='photon',
        energy_group_structure=task.extras["photon_group_structure"],
        parent_nuclides=radionuclides,
    )

    energy_bins = np.array(tally.energy_bins)

    model = mc.Model(
        geometry=geometry,
        tallies=[tally],
        source=source,
        transport_secondary_photons=True,
        use_decay_photons=True,
        verbose=[],
    )

    t0 = time.time()
    dp = model.simulate_transport(
        total_particles=task.particles * task.batches,
        seed=1,
        compute=compute,
    )[tally]
    elapsed = time.time() - t0

    # Build the irradiation/cooling schedule (Pulse for irrad, Cooldown for cool)
    # and time-correct the parent-nuclide-binned photon tally into a DoseResult.
    schedule_steps = []
    for dt, rate in zip(timesteps, source_rates):
        if rate > 0:
            schedule_steps.append(mc.Pulse(source=source, rate=rate, duration=dt))
        else:
            schedule_steps.append(mc.Cooldown(duration=dt))
    schedule = mc.PulseSchedule(schedule_steps)
    dose = schedule.time_correct_tally(dp)

    spectra = [list(row) for row in np.array(dose.mean)]
    spectra_std = [list(row) for row in np.array(dose.std_dev)]

    return {
        'backend': f'yamc-{compute}',
        'radionuclides': radionuclides,
        'energy_bins': energy_bins,
        'spectra': spectra,
        'spectra_std': spectra_std,
        'elapsed': elapsed,
    }, model


# --------------------------------------------------------------------------
# Plots and metrics (D1S-specific, per-timestep)
# --------------------------------------------------------------------------
def create_step_trace(energy_bins, values, name, color, dash=None):
    """Create a step plot trace for Plotly."""
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


def create_comparison_plot(energy_bins, openmc_flux, openmc_std, yamc_flux, yamc_std, title="Flux", ref_label='OpenMC CPU', yamc_label='YAMC'):
    """Create a Plotly subplot with photon flux spectra (top) and ratio (bottom)."""

    energy_bins = np.asarray(energy_bins)
    bin_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])

    omc_flux = np.asarray(openmc_flux, dtype=float).copy()
    omc_std = np.asarray(openmc_std, dtype=float).copy()
    ymc_flux = np.asarray(yamc_flux, dtype=float).copy()
    ymc_std = np.asarray(yamc_std, dtype=float).copy()

    # Normalize by lethargy bin width
    lethargy_bin_width = np.log10(energy_bins[1:] / energy_bins[:-1])
    omc_flux /= lethargy_bin_width
    omc_std /= lethargy_bin_width
    ymc_flux /= lethargy_bin_width
    ymc_std /= lethargy_bin_width

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
    flux_threshold = max_flux / (10 ** FLUX_DECADES)

    omc_flux_plot = np.where(omc_flux >= flux_threshold, omc_flux, np.nan)
    ymc_flux_plot = np.where(ymc_flux >= flux_threshold, ymc_flux, np.nan)
    omc_std_plot = np.where(omc_flux >= flux_threshold, omc_std, np.nan)
    ymc_std_plot = np.where(ymc_flux >= flux_threshold, ymc_std, np.nan)

    valid = (omc_flux > 0) & (ymc_flux > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(valid, ymc_flux / omc_flux, np.nan)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=('Photon Flux per Unit Lethargy', f'{yamc_label} / {ref_label} Ratio')
    )

    fig.add_trace(
        create_step_trace(energy_bins, omc_flux_plot, ref_label, 'blue'),
        row=1, col=1
    )
    fig.add_trace(
        create_step_trace(energy_bins, ymc_flux_plot, yamc_label, 'red', dash='dash'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=bin_centers, y=omc_flux_plot, mode='markers',
            marker=dict(color='blue', size=4, symbol='circle'),
            error_y=dict(type='data', array=omc_std_plot, visible=True, color='blue', thickness=1),
            showlegend=False,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=bin_centers, y=ymc_flux_plot, mode='markers',
            marker=dict(color='red', size=4, symbol='square'),
            error_y=dict(type='data', array=ymc_std_plot, visible=True, color='red', thickness=1),
            showlegend=False,
        ),
        row=1, col=1
    )

    fig.add_trace(
        create_step_trace(energy_bins, ratio, f'{yamc_label} / {ref_label}', 'green'),
        row=2, col=1
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_xaxes(type="log", title_text="Energy (eV)",
                     range=[np.log10(min_plot_energy), np.log10(max_plot_energy)],
                     exponentformat='e', row=2, col=1)
    fig.update_xaxes(type="log",
                     range=[np.log10(min_plot_energy), np.log10(max_plot_energy)],
                     exponentformat='e', row=1, col=1)
    fig.update_yaxes(type="log", title_text="Photon Flux / Lethargy", exponentformat='e', row=1, col=1)
    fig.update_yaxes(title_text="Ratio", range=[0.5, 1.5], row=2, col=1)

    fig.update_layout(
        height=700,
        title=title,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=20, t=60, b=40)
    )

    return fig


def calculate_spectral_agreement(openmc_flux, yamc_flux, openmc_std, yamc_std, flux_threshold=1e-6):
    """Calculate statistical agreement metrics for a single photon flux spectrum."""

    openmc_flux = np.asarray(openmc_flux, dtype=float)
    yamc_flux = np.asarray(yamc_flux, dtype=float)
    openmc_std = np.asarray(openmc_std, dtype=float)
    yamc_std = np.asarray(yamc_std, dtype=float)

    chi_squared_terms = []
    rel_diffs = []
    weighted_rel_diffs = []
    flux_weights = []

    for i in range(len(openmc_flux)):
        omc, ymc = openmc_flux[i], yamc_flux[i]
        omc_s, ymc_s = openmc_std[i], yamc_std[i]
        combined_std = np.sqrt(omc_s**2 + ymc_s**2)
        if combined_std > 0:
            chi_squared_terms.append(((ymc - omc) / combined_std) ** 2)
        if omc > flux_threshold or ymc > flux_threshold:
            if omc > 0:
                rd = abs((ymc - omc) / omc) * 100
                if np.isfinite(rd):
                    rel_diffs.append(rd)
                    w = max(omc, ymc)
                    weighted_rel_diffs.append(rd * w)
                    flux_weights.append(w)

    chi_squared = float(np.sum(chi_squared_terms))
    dof = len(chi_squared_terms)
    reduced_chi_squared = chi_squared / dof if dof > 0 else 0.0
    median_rel_diff = float(np.median(rel_diffs)) if rel_diffs else 0.0
    weighted_avg_rel_diff = (
        float(np.sum(weighted_rel_diffs) / np.sum(flux_weights)) if flux_weights else 0.0
    )

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
        'total_bins': len(openmc_flux),
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


def d1s_quality_label(max_rel_diff):
    """Assign quality label based on max relative difference (%) for D1S."""
    if max_rel_diff < 2.0:
        return "Excellent"
    elif max_rel_diff < 5.0:
        return "Good"
    elif max_rel_diff < 10.0:
        return "Fair"
    else:
        return "Poor"


def rel_diff(a, b):
    """Safe relative difference: (b - a) / a * 100, returns NaN if a == 0."""
    if a == 0:
        return np.nan
    return (b - a) / a * 100


def compute_step_metrics(openmc_result, yamc_result, timesteps, phases):
    """Compute per-timestep comparison metrics."""
    step_metrics = []
    for step_idx in range(len(timesteps)):
        omc_spec = np.array(openmc_result['spectra'][step_idx])
        ymc_spec = np.array(yamc_result['spectra'][step_idx])

        omc_total = omc_spec.sum()
        ymc_total = ymc_spec.sum()
        total_rel = rel_diff(omc_total, ymc_total)

        step_metrics.append({
            'step': step_idx + 1,
            'phase': phases[step_idx],
            'omc_total': float(omc_total),
            'ymc_total': float(ymc_total),
            'total_rel_pct': float(total_rel),
        })

    finite_diffs = [
        abs(m['total_rel_pct']) for m in step_metrics
        if np.isfinite(m['total_rel_pct'])
    ]
    max_total_rel = max(finite_diffs) if finite_diffs else 0.0
    quality = d1s_quality_label(max_total_rel)
    speedup = openmc_result['elapsed'] / yamc_result['elapsed'] if yamc_result['elapsed'] > 0 else float('inf')

    return {
        'step_metrics': step_metrics,
        'max_total_rel_pct': max_total_rel,
        'quality': quality,
        'speedup': speedup,
    }


def create_step_metrics_table(metrics, ref_label='OpenMC CPU', yamc_label='YAMC'):
    """Create a table of per-timestep comparison metrics."""
    rows = []
    for m in metrics['step_metrics']:
        rows.append({
            'Step': m['step'],
            'Phase': m['phase'],
            f'{ref_label} Total': f"{m['omc_total']:.4e}",
            f'{yamc_label} Total': f"{m['ymc_total']:.4e}",
            'Rel. Diff (%)': f"{m['total_rel_pct']:+.2f}%" if np.isfinite(m['total_rel_pct']) else 'N/A',
        })

    rows.append({
        'Step': '',
        'Phase': '',
        f'{ref_label} Total': '',
        f'{yamc_label} Total': 'Quality:',
        'Rel. Diff (%)': metrics['quality'],
    })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Discovery / run / display
# --------------------------------------------------------------------------
def discover(task: Task, run_openmc: bool) -> list[tuple]:
    chain_file = discovery.resolve_openmc_chain(task.library)
    transmutation_data = discovery.transmutation_data_path(task.library)
    return discovery.discover_d1s_nuclides(
        task.nuclear_data_dir, task.library, chain_file, transmutation_data)


def _openmc_meta(task: Task) -> dict:
    """OpenMC cache metadata (byte-matching the notebook's OPENMC_CACHE_META)."""
    meta = {
        "library": task.library,
        "particles": task.particles,
        "batches": task.batches,
        "source_energy": task.source_energy,
        "photon_group_structure": task.extras["photon_group_structure"],
    }
    if task.geometry == "sphere":
        meta["sphere_radius"] = task.dims.sphere_radius
    elif task.geometry == "broomstick":
        meta["cylinder_radius"] = task.dims.cylinder_radius
        meta["cylinder_half_height"] = task.dims.cylinder_half_height
    meta["chain_file"] = discovery.resolve_openmc_chain(task.library)
    meta["d1s_timesteps"] = list(task.extras["timesteps"])
    meta["d1s_source_rates"] = list(task.extras["source_rates"])
    return meta


def _yamc_meta(task: Task) -> dict:
    """YAMC D1S cache metadata: the OpenMC meta plus the build fingerprint."""
    from cache_utils import yamc_fingerprint

    return {**_openmc_meta(task), **yamc_fingerprint()}


def run_all(task: Task, backends=None, cache_dir: str = "cache"):
    """Run every backend for every discovered nuclide. Returns (all_results, first_yamc_model).

    ``cache_dir`` is resolved relative to the current working directory, which
    when run under nbconvert is the notebook's own directory (so the default
    ``"cache"`` lands in ``tasks/<family>/<geometry>/cache`` as before). Both
    OpenMC and YAMC are cached (YAMC keyed by build fingerprint).
    """
    from ..backends import resolve

    b = backends if backends is not None else resolve()
    openmc_meta = _openmc_meta(task)
    yamc_meta = _yamc_meta(task)
    timesteps = task.extras["timesteps"]
    phases = task.extras["phases"]

    nuclides = discover(task, b.run_openmc)
    print(f"Running D1S simulations for {len(nuclides)} nuclides...")
    for nuc, _, elem, _ in nuclides:
        print(f"  {nuc} ({elem})")

    all_results = {}
    first_yamc_model = None

    for nuclide, neutron_path, element, photon_path in nuclides:
        print(f"\nProcessing {nuclide} ({element})...")

        openmc_result = None
        if b.run_openmc:
            try:
                print("  Running OpenMC D1S...")
                openmc_result = run_step(
                    cache_dir, task.library, "openmc", nuclide,
                    lambda n=nuclide, npath=neutron_path, el=element, ppath=photon_path:
                        run_openmc_d1s(n, npath, el, ppath, task),
                    meta=openmc_meta, cache=True,
                )
                print(f"    {openmc_result['elapsed']:.1f}s, "
                      f"{len(openmc_result['radionuclides'])} radionuclides")
            except Exception as e:  # noqa: BLE001 - isolate backend failures
                print(f"  OpenMC Error: {e}")

        yamc_by_compute = {}
        for compute in b.computes:
            captured = {}

            def _run(n=nuclide, npath=neutron_path, el=element, ppath=photon_path,
                     c=compute, cap=captured):
                res, model = run_yamc_d1s(n, npath, el, ppath, task, compute=c)
                cap["model"] = model
                return res

            try:
                print(f"  Running YAMC D1S ({compute})...")
                yamc_result = run_step(
                    cache_dir, task.library, f"yamc-{compute}", nuclide, _run,
                    meta=yamc_meta, cache=task.caches_yamc,
                )
                yamc_by_compute[compute] = yamc_result
                if first_yamc_model is None and "model" in captured:
                    first_yamc_model = captured["model"]
                print(f"    {yamc_result['elapsed']:.1f}s, "
                      f"{len(yamc_result['radionuclides'])} radionuclides")
            except Exception as e:  # noqa: BLE001
                print(f"  YAMC ({compute}) Error: {e}")

        # Nothing ran for this nuclide -> skip.
        if openmc_result is None and not yamc_by_compute:
            continue

        # Reference for the metrics: OpenMC if available, else yamc-cpu.
        reference = openmc_result if openmc_result is not None else yamc_by_compute.get("cpu")
        ref_label = "OpenMC" if openmc_result is not None else "yamc-cpu"

        per_compute_metrics = {}
        for compute, yamc_result in yamc_by_compute.items():
            if reference is None or (openmc_result is None and compute == "cpu"):
                # No reference (cpu-only, openmc off) -> nothing to compare against.
                continue
            metrics = compute_step_metrics(reference, yamc_result, timesteps, phases)
            per_compute_metrics[compute] = metrics
            print(f"  yamc-{compute} vs {ref_label}:")
            for m in metrics['step_metrics']:
                print(f"    Step {m['step']} ({m['phase']:>5}): "
                      f"{ref_label}={m['omc_total']:.4e}  yamc-{compute}={m['ymc_total']:.4e}  "
                      f"diff={m['total_rel_pct']:+.2f}%")
            print(f"    Quality: {metrics['quality']} | Speedup: {metrics['speedup']:.2f}x")

        all_results[nuclide] = {
            'openmc': openmc_result,
            'yamc': yamc_by_compute,
            'metrics': per_compute_metrics,
            'reference_label': ref_label,
            'element': element,
        }

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
    html_plot.save("d1s_model_plot.html")
    encoded = base64.b64encode(html_plot.html.encode()).decode()
    display(HTML(f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="600" frameborder="0"></iframe>'))


def display_results(task: Task, all_results: dict):
    """Render per-nuclide per-timestep comparison plots, metrics, and agreement tables."""
    from IPython.display import Markdown, display

    timesteps = task.extras["timesteps"]
    phases = task.extras["phases"]

    for nuclide, results in all_results.items():
        openmc_result = results['openmc']
        yamc_by_compute = results['yamc']
        per_compute_metrics = results['metrics']
        ref_label = results['reference_label']
        element = results['element']

        # Reference spectra: OpenMC if present, else yamc-cpu.
        reference = openmc_result if openmc_result is not None else yamc_by_compute.get('cpu')
        if reference is None:
            continue

        display(Markdown(f"## {nuclide} ({element})"))

        ref_time = reference['elapsed']
        perf_bits = [f"{ref_label}: {ref_time:.1f}s"]
        for compute, yr in yamc_by_compute.items():
            perf_bits.append(f"yamc-{compute}: {yr['elapsed']:.1f}s")
        display(Markdown(
            "**Performance:** " + " | ".join(perf_bits) + "  \n"
            f"**Radionuclides:** {ref_label}: {len(reference['radionuclides'])}"
        ))

        energy_bins = reference['energy_bins']

        for compute, yamc_result in yamc_by_compute.items():
            metrics = per_compute_metrics.get(compute)
            if metrics is None:
                # yamc-cpu IS the reference (openmc off) -> nothing to compare.
                continue
            display(Markdown(f"### yamc-{compute} vs {ref_label}"))
            metrics_table = create_step_metrics_table(metrics, ref_label=("OpenMC CPU" if openmc_result is not None else "YAMC CPU"), yamc_label=f"YAMC {compute.upper()}")
            display(metrics_table.style.hide(axis='index'))

            for step_idx in range(len(timesteps)):
                phase = phases[step_idx]
                display(Markdown(f"#### Step {step_idx + 1} ({phase})"))

                omc_flux = np.array(reference['spectra'][step_idx])
                omc_std = np.array(reference['spectra_std'][step_idx])
                ymc_flux = np.array(yamc_result['spectra'][step_idx])
                ymc_std = np.array(yamc_result['spectra_std'][step_idx])

                fig = create_comparison_plot(
                    energy_bins, omc_flux, omc_std, ymc_flux, ymc_std,
                    title=f"{nuclide} - YAMC {compute.upper()} - Step {step_idx + 1} ({phase})",
                    ref_label=("OpenMC CPU" if openmc_result is not None else "YAMC CPU"),
                    yamc_label=f"YAMC {compute.upper()}",
                )
                fig.show()

                display(Markdown("##### Spectral Agreement"))
                stats = calculate_spectral_agreement(omc_flux, ymc_flux, omc_std, ymc_std)
                display(create_agreement_table(stats).style.hide(axis='index'))

        print()
