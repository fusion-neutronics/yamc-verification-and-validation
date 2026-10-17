"""Photon sphere/broomstick verification family.

One implementation for all four photon notebooks (sphere/broomstick x
endf/fendl). Geometry comes from :mod:`yamc_verify.geometry` and all run
parameters from the task registry, so the notebooks reduce to a few lines.
YAMC is silenced with ``verbose=[]`` and OpenMC with ``verbosity=1`` +
``output=False``.

The item is an *element* (elemental symbol, e.g. ``Fe``), a 1 MeV isotropic
photon point source is placed at the origin, and the photon flux plus the four
photon-interaction reaction rates (coherent/incoherent scatter, photoelectric,
pair-production) are tallied, along with an energy-binned flux spectrum over the
CCFE-24-PHOTON group structure. Only OpenMC is cached; YAMC runs live.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import discovery
from ..geometry import openmc_region, yamc_region
from ..plotting import (
    BACKEND_STYLES,
    RATIO_COLORS,
    RATIO_FILLS,
    calculate_spectral_agreement,
    create_agreement_table,
    create_step_trace,
)
from ..runstep import run_step
from ..tasks import Task

# CCFE-24-PHOTON energy bins (eV) - used to build the OpenMC EnergyFilter.
PHOTON_BINS = [
    1.0e3, 5.0e3, 1.0e4, 2.0e4, 5.0e4, 1.0e5, 2.0e5, 3.0e5, 4.0e5,
    5.0e5, 5.11e5, 6.0e5, 8.0e5, 1.0e6, 1.5e6, 2.0e6, 3.0e6, 4.0e6,
    5.0e6, 6.0e6, 8.0e6, 1.0e7, 1.2e7, 1.5e7, 2.0e7,
]

_PHOTON_SCORES = ["coherent-scatter", "incoherent-scatter", "photoelectric", "pair-production"]

# Photon interaction MT descriptions
PHOTON_MT_DESCRIPTIONS = {
    "flux": "Total photon flux",
    "coherent-scatter": "Coherent (Rayleigh) scatter (MT 502)",
    "incoherent-scatter": "Incoherent (Compton) scatter (MT 504)",
    "photoelectric": "Photoelectric absorption (MT 522)",
    "pair-production": "Pair production (MT 516)",
}


# --------------------------------------------------------------------------
# Simulation runners
# --------------------------------------------------------------------------
def run_openmc_simulation(element: str, task: Task) -> dict:
    """Run the OpenMC photon simulation for one element."""
    import openmc as mc

    mc.reset_auto_ids()
    mc.config["cross_sections"] = f"{task.nuclear_data_dir}cross_sections.xml"

    region = openmc_region(task.geometry, task.dims)

    # Material
    material = mc.Material()
    material.add_element(element, 1.0)
    material.set_density("g/cm3", 1.0)
    material.temperature = 294

    cell = mc.Cell(name=task.geometry, region=region, fill=material)
    geometry = mc.Geometry([cell])

    # Source: 1 MeV isotropic photons at origin
    source = mc.IndependentSource(
        space=mc.stats.Point([0, 0, 0]),
        energy=mc.stats.Discrete([task.source_energy], [1.0]),
        angle=mc.stats.Isotropic(),
    )
    source.particle = "photon"

    settings = mc.Settings(
        particles=task.particles,
        batches=task.batches,
        source=source,
        seed=1,
        run_mode="fixed source",
        output={"tallies": False, "summary": False},
    )
    settings.verbosity = 1  # quiet OpenMC (combined with output=False on run)

    # Filters and tallies
    cell_filter = mc.CellFilter(cell)
    photon_filter = mc.ParticleFilter("photon")
    energy_filter = mc.EnergyFilter(PHOTON_BINS)
    energy_bins_2d = energy_filter.bins
    energy_bins = np.concatenate([[energy_bins_2d[0, 0]], energy_bins_2d[:, 1]])

    tallies = []

    # Photon interaction score tallies
    for score in ["flux"] + _PHOTON_SCORES:
        tally = mc.Tally()
        tally.filters = [cell_filter, photon_filter]
        tally.scores = [score]
        tally.name = score
        tallies.append(tally)

    # Energy-binned photon flux tally
    tally_spectrum = mc.Tally()
    tally_spectrum.filters = [cell_filter, photon_filter, energy_filter]
    tally_spectrum.scores = ["flux"]
    tally_spectrum.name = "flux_energy_binned"
    tallies.append(tally_spectrum)

    # Run
    model = mc.Model(geometry=geometry, settings=settings, tallies=tallies)
    sp_filename = model.run(apply_tally_results=True, output=False)
    sp = mc.StatePoint(sp_filename)

    transport_time = sp.runtime["transport"]
    total_particles = task.particles * task.batches
    particles_per_second = total_particles / transport_time

    results = {
        "element": element,
        "code": "openmc",
        "particles/s": particles_per_second,
        "tallies": {},
        "tally_std": {},
    }

    for tally in tallies:
        if tally.name != "flux_energy_binned":
            results["tallies"][tally.name] = tally.mean.flatten()[0]
            results["tally_std"][tally.name] = tally.std_dev.flatten()[0]

    results["flux_spectra_mean"] = np.array(tally_spectrum.mean).squeeze()
    results["flux_spectra_std_dev"] = np.array(tally_spectrum.std_dev).squeeze()
    results["energy_bins"] = energy_bins

    return results


def run_yamc_simulation(element: str, task: Task, compute: str = "cpu"):
    """Run the YAMC photon simulation for one element. Returns (results, model).

    ``compute`` selects the transport backend ("cpu" or "gpu").
    """
    import yamc as mc

    region = yamc_region(task.geometry, task.dims)

    # Material - photon-only, no neutron data needed
    material = mc.Material(
        composition={element: 1.0},
        density=1.0,
        temperature=294,
        id=1,
    )
    # The photon data come from the task's library too. yamc otherwise takes
    # photon data from endf-b8.1 whatever the neutron library, for its atomic
    # relaxation tables, while OpenMC reads the task library's own photon files:
    # on FENDL-3.2d the two codes would then transport on different photon data.
    material.read_nuclear_data(task.library, photon_data={element: task.library})

    cell = mc.Cell(id=1, name=task.geometry, region=region, material=material)
    geometry = mc.Geometry([cell])

    # Source: 1 MeV isotropic photons at origin
    source = mc.PhotonSource(
        position=(0, 0, 0),
        direction=mc.sources.Isotropic(),
        energy=mc.sources.Discrete([task.source_energy], [1.0]),
    )

    tallies = []

    # Photon interaction score tallies
    for score in ["flux"] + _PHOTON_SCORES:
        t = mc.Tally(
            scores=[str(score)],
            name=str(score),
            cells=cell,
            particle="photon",
        )
        tallies.append(t)

    # Energy-binned photon flux tally
    tally_spectrum = mc.Tally(
        scores=["flux"],
        name="flux_energy_binned",
        cells=cell,
        particle="photon",
        energy_group_structure=task.extras["photon_group_structure"],
    )
    tallies.append(tally_spectrum)

    # Energy group bins (for plotting)
    energy_bins = np.array(tally_spectrum.energy_bins)

    # Run
    model = mc.Model(
        geometry=geometry,
        tallies=tallies,
        source=source,
        verbose=[],
    )
    t0 = time.perf_counter()
    sim_results = model.simulate_transport(
        total_particles=task.particles * task.batches,
        seed=1,
        compute=compute,
    )
    elapsed = time.perf_counter() - t0

    total = task.particles * task.batches
    results = {
        "element": element,
        "code": f"yamc-{compute}",
        "particles/s": (
            sim_results.particles_per_second
            if sim_results.particles_per_second is not None
            else (total / elapsed if elapsed > 0 else 0)
        ),
        "tallies": {},
        "tally_std": {},
    }

    for t in tallies:
        if t.name != "flux_energy_binned":
            r = sim_results[t]
            results["tallies"][t.name] = r.mean[0]
            results["tally_std"][t.name] = r.standard_deviation[0]

    spec = sim_results[tally_spectrum]
    results["flux_spectra_mean"] = np.array(spec.mean).squeeze()
    results["flux_spectra_std_dev"] = np.array(spec.standard_deviation).squeeze()
    results["energy_bins"] = energy_bins

    return results, model


# --------------------------------------------------------------------------
# Plots and tables
# --------------------------------------------------------------------------
def create_comparison_plot(openmc_results, yamc_results_by_backend, *, max_plot_energy,
                           flux_decades=6):
    """Photon flux spectra (top) with per-backend ratios (bottom)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if openmc_results is not None:
        ref_label = "OpenMC CPU"
        ratio_targets = [(f"YAMC {b.upper()}", res) for b, res in yamc_results_by_backend.items()]
    elif len(yamc_results_by_backend) >= 2:
        backends = list(yamc_results_by_backend)
        ref_label = f"YAMC {backends[0].upper()}"
        ratio_targets = [(f"YAMC {b.upper()}", yamc_results_by_backend[b]) for b in backends[1:]]
    else:
        ref_label = None
        ratio_targets = []

    spectra = []
    if openmc_results is not None:
        spectra.append(("OpenMC CPU", openmc_results))
    for backend, res in yamc_results_by_backend.items():
        spectra.append((f"YAMC {backend.upper()}", res))

    if not spectra:
        raise ValueError("create_comparison_plot needs at least one result")

    energy_bins = spectra[0][1]["energy_bins"]
    bin_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
    lethargy_bin_width = np.log10(energy_bins[1:] / energy_bins[:-1])

    flux_by_label = {}
    std_by_label = {}
    for label, res in spectra:
        flux_by_label[label] = res["flux_spectra_mean"].copy() / lethargy_bin_width
        std_by_label[label] = res["flux_spectra_std_dev"].copy() / lethargy_bin_width

    has_flux = np.zeros(len(energy_bins) - 1, dtype=bool)
    for f in flux_by_label.values():
        has_flux = has_flux | (f > 0)
    if np.any(has_flux):
        first_nonzero_idx = int(np.argmax(has_flux))
        min_plot_energy = energy_bins[first_nonzero_idx]
    else:
        min_plot_energy = energy_bins[0]

    all_positive_max = []
    for f in flux_by_label.values():
        if np.any(f > 0):
            all_positive_max.append(np.max(f[f > 0]))
    max_flux = max(all_positive_max) if all_positive_max else 1e-10
    flux_threshold = max_flux / (10 ** flux_decades)

    def _mask(values, std, ref):
        keep = ref >= flux_threshold
        return np.where(keep, values, np.nan), np.where(keep, std, np.nan)

    show_ratio = bool(ratio_targets)
    if show_ratio:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Photon Flux per Unit Lethargy", f"Ratio (vs {ref_label})"),
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=("Photon Flux per Unit Lethargy",),
        )

    for label, _res in spectra:
        f = flux_by_label[label]
        s = std_by_label[label]
        f_plot, s_plot = _mask(f, s, f)
        style = BACKEND_STYLES.get(label, dict(color="gray", dash=None, marker="circle"))
        fig.add_trace(
            create_step_trace(energy_bins, f_plot, label, style["color"], dash=style["dash"]),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bin_centers, y=f_plot, mode="markers",
                name=f"{label} error",
                marker=dict(color=style["color"], size=4, symbol=style["marker"]),
                error_y=dict(type="data", array=s_plot, visible=True,
                             color=style["color"], thickness=1),
                showlegend=False,
            ),
            row=1, col=1,
        )

    if show_ratio:
        ref_flux = flux_by_label[ref_label]
        ref_std = std_by_label[ref_label]
        for label, _res in ratio_targets:
            f = flux_by_label[label]
            s = std_by_label[label]
            valid = (ref_flux > 0) & (f > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(valid, f / ref_flux, np.nan)
                rel_err_top = np.where(f > 0, s / f, 0)
                rel_err_ref = np.where(ref_flux > 0, ref_std / ref_flux, 0)
            ratio_rel_err = np.sqrt(rel_err_top**2 + rel_err_ref**2)
            ratio_std = np.where(valid, ratio * ratio_rel_err, np.nan)  # noqa: F841

            ratio_label = f"{label} / {ref_label}"
            ratio_color = RATIO_COLORS.get(ratio_label, "green")
            rgba_fill = RATIO_FILLS.get(ratio_color, "rgba(0, 128, 0, 0.2)")  # noqa: F841

            fig.add_trace(
                create_step_trace(energy_bins, ratio, ratio_label, ratio_color, legend="legend2"),
                row=2, col=1,
            )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_xaxes(type="log", title_text="Energy (eV)",
                     range=[np.log10(min_plot_energy), np.log10(max_plot_energy)],
                     exponentformat="e", row=(2 if show_ratio else 1), col=1)
    if show_ratio:
        fig.update_xaxes(type="log",
                         range=[np.log10(min_plot_energy), np.log10(max_plot_energy)],
                         exponentformat="e", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Photon Flux / Lethargy", exponentformat="e", row=1, col=1)
    if show_ratio:
        fig.update_yaxes(title_text="Ratio", range=[0.5, 1.5], row=2, col=1)

    fig.update_layout(
        height=700 if show_ratio else 500,
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        legend2=dict(x=0.02, y=0.28, xanchor="left", yanchor="top"),
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return fig


def _format_tally(value, std):
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.4e} +/- {std:.2e}"


def _format_rel_diff(value, ref_val):
    if ref_val is None or np.isnan(ref_val) or ref_val == 0 or value is None or np.isnan(value):
        return "N/A"
    return f"{(value - ref_val) / ref_val * 100:.2f}%"


def create_tally_table(openmc_results, yamc_results_by_backend):
    """Photon tally comparison table with one column per code/backend."""
    all_tallies = set()
    if openmc_results is not None:
        all_tallies |= set(openmc_results["tallies"].keys())
    for res in yamc_results_by_backend.values():
        all_tallies |= set(res["tallies"].keys())

    yamc_backends = list(yamc_results_by_backend)
    if openmc_results is not None:
        ref_results = openmc_results
        rel_diff_targets = yamc_backends
    elif yamc_backends:
        ref_results = yamc_results_by_backend[yamc_backends[0]]
        rel_diff_targets = yamc_backends[1:]
    else:
        return pd.DataFrame()

    rows = []
    for name in sorted(all_tallies):
        if openmc_results is not None:
            openmc_val = openmc_results["tallies"].get(name, np.nan)
            openmc_std = openmc_results["tally_std"].get(name, np.nan)
        else:
            openmc_val = openmc_std = np.nan
        yamc_vals = {}
        for backend, res in yamc_results_by_backend.items():
            yamc_vals[backend] = (
                res["tallies"].get(name, np.nan),
                res["tally_std"].get(name, np.nan),
            )
        ref_val = ref_results["tallies"].get(name, np.nan)

        all_zero = (
            (openmc_results is None or openmc_val == 0 or np.isnan(openmc_val))
            and all((v[0] == 0 or np.isnan(v[0])) for v in yamc_vals.values())
        )
        if all_zero:
            continue

        reaction = PHOTON_MT_DESCRIPTIONS.get(name, name)
        row = {"Score": name, "Reaction": reaction}
        if openmc_results is not None:
            row["OpenMC CPU"] = _format_tally(openmc_val, openmc_std)
        for backend, (val, std) in yamc_vals.items():
            row[f"YAMC {backend.upper()}"] = _format_tally(val, std)
        for backend in rel_diff_targets:
            val, _ = yamc_vals[backend]
            row[f"Rel. Diff {backend} (%)"] = _format_rel_diff(val, ref_val)
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Discovery / run / display
# --------------------------------------------------------------------------
def discover(task: Task, run_openmc: bool) -> list[str]:
    return discovery.discover_photon_elements(task.nuclear_data_dir, task.library, run_openmc)


def _metas(task: Task, *, openmc_build: bool):
    """OpenMC + YAMC cache metas, geometry-dependent to byte-match the notebooks.

    The YAMC meta folds in ``yamc_fingerprint()`` so the YAMC cache auto-
    invalidates on a new yamc build (same policy as neutron). With
    ``openmc_build`` the OpenMC meta also folds in ``openmc_fingerprint()``, so
    a reference made by another OpenMC build re-runs. It is asked for only when
    OpenMC runs, so a yamc-only run needs no OpenMC.
    """
    from cache_utils import openmc_fingerprint, yamc_fingerprint

    openmc_meta = {
        "library": task.library,
        "particles": task.particles,
        "batches": task.batches,
        "source_energy": task.source_energy,
    }
    if task.geometry == "sphere":
        openmc_meta["sphere_radius"] = task.dims.sphere_radius
    elif task.geometry == "broomstick":
        openmc_meta["cylinder_radius"] = task.dims.cylinder_radius
        openmc_meta["cylinder_half_height"] = task.dims.cylinder_half_height
    yamc_meta = {**openmc_meta, **yamc_fingerprint()}
    if openmc_build:
        openmc_meta = {**openmc_meta, **openmc_fingerprint()}
    return openmc_meta, yamc_meta


def run_all(task: Task, backends=None, cache_dir: str = "cache"):
    """Run every backend for every discovered element. Returns (all_results, first_yamc_model).

    ``cache_dir`` is resolved relative to the current working directory, which
    when run under nbconvert is the notebook's own directory (so the default
    ``"cache"`` lands in ``tasks/photon/<geometry>/cache`` as before). Both
    OpenMC and YAMC are cached (YAMC keyed by build fingerprint).
    """
    from ..backends import resolve

    b = backends if backends is not None else resolve()
    openmc_meta, yamc_meta = _metas(task, openmc_build=b.run_openmc)
    elements = discover(task, b.run_openmc)
    print(f"Running photon simulations for {len(elements)} elements: {elements}")

    all_results = {}
    first_yamc_model = None
    for element in elements:
        per = {}
        for compute in b.computes:
            captured = {}

            def _run(el=element, c=compute, cap=captured):
                res, model = run_yamc_simulation(el, task, compute=c)
                cap["model"] = model
                return res

            try:
                per[f"yamc-{compute}"] = run_step(
                    cache_dir, task.library, f"yamc-{compute}", element, _run,
                    meta=yamc_meta, cache=task.caches_yamc)
                if first_yamc_model is None and "model" in captured:
                    first_yamc_model = captured["model"]
            except Exception as e:  # noqa: BLE001 - isolate backend failures
                print(f"  YAMC ({compute}) Error ({element}): {e}")
        if b.run_openmc:
            try:
                per["openmc"] = run_step(
                    cache_dir, task.library, "openmc", element,
                    lambda el=element: run_openmc_simulation(el, task),
                    meta=openmc_meta, cache=True)
            except Exception as e:  # noqa: BLE001
                print(f"  OpenMC Error ({element}): {e}")
        if per:
            all_results[element] = per
    print(f"Completed {len(all_results)} / {len(elements)} elements successfully.")
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
    """Render per-element comparison plots, agreement tables, and tally tables."""
    from IPython.display import Markdown, display

    max_plot_energy = (task.source_energy or 0) * 1.1
    for element, results in all_results.items():
        openmc_results = results.get("openmc")
        yamc_results_by_backend = {}
        for key, res in results.items():
            if key.startswith("yamc-"):
                yamc_results_by_backend[key[len("yamc-"):]] = res

        display(Markdown(f"## {element}"))

        perf_parts = []
        if openmc_results is not None:
            perf_parts.append(f"OpenMC: {openmc_results['particles/s']:,.0f} particles/s")
        for compute, yres in yamc_results_by_backend.items():
            perf_parts.append(f"YAMC ({compute}): {yres['particles/s']:,.0f} particles/s")
        display(Markdown("**Performance:** " + " | ".join(perf_parts)))

        fig = create_comparison_plot(openmc_results, yamc_results_by_backend,
                                     max_plot_energy=max_plot_energy)
        fig.update_layout(title=f"{element} - Photon Flux Spectra Comparison")
        fig.show()

        if openmc_results is not None:
            ref_label = "OpenMC"
            ref_results = openmc_results
            comparisons = list(yamc_results_by_backend.items())
        elif len(yamc_results_by_backend) >= 2:
            bks = list(yamc_results_by_backend)
            ref_label = f"YAMC ({bks[0]})"
            ref_results = yamc_results_by_backend[bks[0]]
            comparisons = [(b, yamc_results_by_backend[b]) for b in bks[1:]]
        else:
            comparisons = []

        for compute, yres in comparisons:
            heading = (
                f"#### Spectral Agreement (YAMC {compute} vs {ref_label})"
                if len(comparisons) > 1 or openmc_results is None
                else "#### Spectral Agreement"
            )
            display(Markdown(heading))
            stats = calculate_spectral_agreement(ref_results, yres)
            display(create_agreement_table(stats).style.hide(axis="index"))

        display(Markdown("#### Photon Tally Comparison"))
        table = create_tally_table(openmc_results, yamc_results_by_backend)
        display(table.style.hide(axis="index"))
