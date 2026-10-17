"""Neutron sphere/broomstick verification family.

One implementation for all four neutron notebooks (sphere/broomstick x
endf/fendl). Geometry comes from :mod:`yamc_verify.geometry` and all run
parameters from the task registry, so the notebooks reduce to a few lines.
YAMC is silenced with ``verbose=[]`` and OpenMC with ``verbosity=1`` +
``output=False``.
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
from ..plotting import (
    BACKEND_STYLES,
    RATIO_COLORS,
    calculate_spectral_agreement,
    create_agreement_table,
    create_step_trace,
)
from ..runstep import run_step
from ..tasks import Task

FLUX_DECADES = 6
_SPECIAL_SCORES = [
    "total", "absorption", "flux", "heating", "heating-local",
    "H1-production", "H2-production", "H3-production", "He3-production", "He4-production",
]


# --------------------------------------------------------------------------
# Simulation runners
# --------------------------------------------------------------------------
def _seed() -> int:
    """Base RNG seed for both codes (``YAMC_VERIFY_SEED``, default 1). A run
    with another seed is an independent sample of the same problem and caches
    under its own key, so N-seed re-measurements of a single-bin claim (issue
    fusion-neutronics/core#34 entry 6) do not disturb the seed-1 references."""
    return int(os.environ.get("YAMC_VERIFY_SEED", "1"))


def _cache_key(nuclide: str) -> str:
    """Cache file key: the nuclide, suffixed with the seed when it is not 1."""
    seed = _seed()
    return nuclide if seed == 1 else f"{nuclide}-seed{seed}"


def run_openmc_simulation(nuclide: str, task: Task) -> dict:
    """Run the OpenMC neutron simulation for one nuclide."""
    import openmc as mc

    mc.config["cross_sections"] = f"{task.nuclear_data_dir}cross_sections.xml"

    region1 = openmc_region(task.geometry, task.dims)
    material1 = mc.Material()
    material1.material_id = 1
    material1.add_nuclide(nuclide, 1.0)
    material1.set_density("g/cm3", 1.0)
    material1.temperature = 294
    cell1 = mc.Cell(cell_id=1, name="inner_sphere", region=region1, fill=material1)
    geometry = mc.Geometry([cell1])

    in_neu = mc.data.IncidentNeutron.from_hdf5(
        discovery.openmc_library(task.nuclear_data_dir, "neutron")[nuclide])
    mts_int = [mt for mt in in_neu.reactions.keys() if mt not in (1, 3, 4, 27, 101)]
    mts = [str(mt) for mt in mts_int]

    cell_filter = mc.CellFilter(cell1)
    energy_filter = mc.EnergyFilter.from_group_structure("VITAMIN-J-175")
    energy_bins_2d = energy_filter.bins
    energy_bins = np.concatenate([[energy_bins_2d[0, 0]], energy_bins_2d[:, 1]])

    tallies = []
    for score in mts + _SPECIAL_SCORES:
        tally = mc.Tally()
        tally.filters = [cell_filter]
        tally.scores = [score]
        tally.name = str(score)
        tallies.append(tally)

    tally_spectrum = mc.Tally()
    tally_spectrum.filters = [cell_filter, energy_filter]
    tally_spectrum.scores = ["flux"]
    tally_spectrum.name = "flux_energy_binned"
    tallies.append(tally_spectrum)

    source = mc.IndependentSource(
        angle=mc.stats.Isotropic(),
        energy=mc.stats.Discrete([task.source_energy], [1.0]),
        space=mc.stats.Point([0, 0, 0]),
    )
    settings = mc.Settings(
        particles=task.particles, batches=task.batches, source=source, seed=_seed(),
        run_mode="fixed source", output={"tallies": False, "summary": False},
    )
    settings.verbosity = 1  # quiet OpenMC (combined with output=False on run)

    model = mc.Model(geometry=geometry, settings=settings, tallies=tallies)
    sp_filename = model.run(apply_tally_results=True, output=False)
    sp = mc.StatePoint(sp_filename)

    transport_time = sp.runtime["transport"]
    total_particles = task.particles * task.batches
    results = {
        "nuclide": nuclide,
        "code": "openmc",
        "particles/s": total_particles / transport_time,
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


def run_yamc_simulation(nuclide: str, task: Task, compute: str = "cpu"):
    """Run the YAMC neutron simulation for one nuclide. Returns (results, model)."""
    import yamc as mc

    region1 = yamc_region(task.geometry, task.dims)
    material1 = mc.Material(composition={nuclide: 1.0}, density=1.0, temperature=294, id=1)
    material1.read_nuclear_data(task.library)

    mts = [mt for mt in material1.reaction_mts if mt not in (1, 3, 4, 27, 101)]

    cell1 = mc.Cell(id=1, name="inner_sphere", region=region1, material=material1)
    geometry = mc.Geometry([cell1])

    tallies = []
    for score in mts + _SPECIAL_SCORES:
        tallies.append(mc.Tally(scores=[str(score)], name=str(score), cells=cell1, particle="neutron"))
    tally_spectrum = mc.Tally(
        scores=["flux"], name="flux_energy_binned", cells=cell1, particle="neutron",
        energy_group_structure="VITAMIN-J-175",
    )
    tallies.append(tally_spectrum)
    energy_bins = np.array(tally_spectrum.energy_bins)

    source = mc.NeutronSource(
        position=(0, 0, 0),
        direction=mc.sources.Isotropic(),
        energy=mc.sources.Discrete([task.source_energy], [1.0]),
    )
    model = mc.Model(
        geometry=geometry, tallies=tallies, source=source, verbose=[],
    )
    t0 = time.perf_counter()
    sim_results = model.simulate_transport(
        total_particles=task.particles * task.batches, seed=_seed(), compute=compute,
    )
    elapsed = time.perf_counter() - t0

    total = task.particles * task.batches
    results = {
        "nuclide": nuclide,
        "code": f"yamc-{compute}",
        "particles/s": (
            sim_results.particles_per_second
            if sim_results.particles_per_second is not None
            else (total / elapsed if elapsed > 0 else 0)
        ),
        "tallies": {},
        "tally_std": {},
    }
    for tally in tallies:
        if tally.name != "flux_energy_binned":
            r = sim_results[tally]
            results["tallies"][tally.name] = r.mean[0]
            results["tally_std"][tally.name] = r.standard_deviation[0]
    spec = sim_results[tally_spectrum]
    results["flux_spectra_mean"] = np.array(spec.mean).squeeze()
    results["flux_spectra_std_dev"] = np.array(spec.standard_deviation).squeeze()
    results["energy_bins"] = energy_bins
    return results, model


# --------------------------------------------------------------------------
# MT descriptions / tally-table helpers
# --------------------------------------------------------------------------
def _mt_descriptions():
    import yamc

    return yamc.data.reaction_names()


def _score_to_mt(mt_descriptions):
    m = {name: mt for mt, name in mt_descriptions.items()}
    m.update({
        "total": 1, "elastic": 2, "absorption": 27, "fission": 18,
        "heating": 301, "heating-local": 901, "H1-production": 203,
        "H2-production": 204, "H3-production": 205, "He3-production": 206,
        "He4-production": 207,
    })
    return m


def get_mt_and_reaction(score_name, mt_descriptions, score_to_mt):
    """Return ``(mt_display, reaction_description)`` for a score name."""
    try:
        mt = int(score_name)
        return str(mt), mt_descriptions.get(mt, f"MT {mt}")
    except ValueError:
        pass
    if score_name in score_to_mt:
        return str(score_to_mt[score_name]), score_name
    return "-", score_name


def _best_value(results, score_names):
    best_val = np.nan
    best_std = np.nan
    for name in score_names:
        val = results["tallies"].get(name, np.nan)
        std = results["tally_std"].get(name, np.nan)
        if not np.isnan(val) and val != 0:
            if np.isnan(best_val) or best_val == 0:
                best_val = val
                best_std = std
        elif np.isnan(best_val):
            best_val = val
            best_std = std
    return best_val, best_std


def _is_zero_or_nan(value):
    return value == 0 or np.isnan(value)


def _format_tally(value, std):
    if np.isnan(value):
        return "N/A"
    return f"{value:.4e} +/- {std:.2e}"


def _format_rel_diff(value, ref_val):
    if np.isnan(ref_val) or ref_val == 0 or np.isnan(value):
        return "N/A"
    return f"{(value - ref_val) / ref_val * 100:.2f}%"


# --------------------------------------------------------------------------
# Plots and tables
# --------------------------------------------------------------------------
def create_comparison_plot(openmc_results, yamc_results_by_backend, *, max_plot_energy,
                           flux_decades=FLUX_DECADES):
    """Overlaid flux spectra (top) with per-backend ratios (bottom)."""
    if openmc_results is not None:
        ref_label = "OpenMC CPU"
        ratio_targets = [(f"YAMC {b.upper()}", res) for b, res in yamc_results_by_backend.items()]
    elif len(yamc_results_by_backend) >= 2:
        bks = list(yamc_results_by_backend)
        ref_label = f"YAMC {bks[0].upper()}"
        ratio_targets = [(f"YAMC {b.upper()}", yamc_results_by_backend[b]) for b in bks[1:]]
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
        min_plot_energy = energy_bins[int(np.argmax(has_flux))]
    else:
        min_plot_energy = energy_bins[0]

    all_positive_max = [np.max(f[f > 0]) for f in flux_by_label.values() if np.any(f > 0)]
    max_flux = max(all_positive_max) if all_positive_max else 1e-10
    flux_threshold = max_flux / (10 ** flux_decades)

    def _mask(values, std, ref):
        keep = ref >= flux_threshold
        return np.where(keep, values, np.nan), np.where(keep, std, np.nan)

    show_ratio = bool(ratio_targets)
    if show_ratio:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Flux per Unit Lethargy", f"Ratio (vs {ref_label})"),
        )
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Flux per Unit Lethargy",))

    for label, _res in spectra:
        f = flux_by_label[label]
        s = std_by_label[label]
        f_plot, s_plot = _mask(f, s, f)
        style = BACKEND_STYLES.get(label, dict(color="gray", dash=None, marker="circle"))
        fig.add_trace(create_step_trace(energy_bins, f_plot, label, style["color"], dash=style["dash"]),
                      row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=bin_centers, y=f_plot, mode="markers", name=f"{label} error",
                marker=dict(color=style["color"], size=4, symbol=style["marker"]),
                error_y=dict(type="data", array=s_plot, visible=True, color=style["color"], thickness=1),
                showlegend=False,
            ),
            row=1, col=1,
        )

    if show_ratio:
        ref_flux = flux_by_label[ref_label]
        for label, _res in ratio_targets:
            f = flux_by_label[label]
            valid = (ref_flux > 0) & (f > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(valid, f / ref_flux, np.nan)
            ratio_label = f"{label} / {ref_label}"
            ratio_color = RATIO_COLORS.get(ratio_label, "green")
            fig.add_trace(create_step_trace(energy_bins, ratio, ratio_label, ratio_color, legend="legend2"),
                          row=2, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_xaxes(type="log", title_text="Energy (eV)",
                     range=[np.log10(min_plot_energy), np.log10(max_plot_energy)],
                     exponentformat="e", row=(2 if show_ratio else 1), col=1)
    if show_ratio:
        fig.update_xaxes(type="log", range=[np.log10(min_plot_energy), np.log10(max_plot_energy)],
                         exponentformat="e", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Flux / Lethargy", exponentformat="e", row=1, col=1)
    if show_ratio:
        fig.update_yaxes(title_text="Ratio", range=[0.5, 1.5], row=2, col=1)

    fig.update_layout(
        height=700 if show_ratio else 500,
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        legend2=dict(x=0.02, y=0.27, xanchor="left", yanchor="top"),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def create_tally_table(openmc_results, yamc_results_by_backend):
    """Comparison table with one column per code/backend."""
    mt_descriptions = _mt_descriptions()
    score_to_mt = _score_to_mt(mt_descriptions)

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

    mt_to_scores = {}
    for name in all_tallies:
        mt_display, _ = get_mt_and_reaction(name, mt_descriptions, score_to_mt)
        mt_to_scores.setdefault(mt_display, []).append(name)

    rows = []
    for mt_display, score_names in mt_to_scores.items():
        if openmc_results is not None:
            openmc_val, openmc_std = _best_value(openmc_results, score_names)
        else:
            openmc_val, openmc_std = np.nan, np.nan
        yamc_vals = {b: _best_value(res, score_names) for b, res in yamc_results_by_backend.items()}
        ref_val, _ = _best_value(ref_results, score_names)

        all_zero = (
            (openmc_results is None or _is_zero_or_nan(openmc_val))
            and all(_is_zero_or_nan(v[0]) for v in yamc_vals.values())
        )
        if all_zero:
            continue

        reaction = None
        if mt_display != "-":
            try:
                reaction = mt_descriptions.get(int(mt_display))
            except ValueError:
                pass
        if reaction is None:
            _, reaction = get_mt_and_reaction(score_names[0], mt_descriptions, score_to_mt)

        row = {"MT": mt_display, "Reaction": reaction}
        if openmc_results is not None:
            row["OpenMC CPU"] = _format_tally(openmc_val, openmc_std)
        for backend, (val, std) in yamc_vals.items():
            row[f"YAMC {backend.upper()}"] = _format_tally(val, std)
        for backend in rel_diff_targets:
            val, _ = yamc_vals[backend]
            row[f"Rel. Diff {backend} (%)"] = _format_rel_diff(val, ref_val)

        row["_mt_sort"] = int(mt_display) if mt_display != "-" else 9999
        rows.append(row)

    rows.sort(key=lambda r: r["_mt_sort"])
    for row in rows:
        del row["_mt_sort"]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Discovery / run / display
# --------------------------------------------------------------------------
def discover(task: Task, run_openmc: bool) -> list[str]:
    return discovery.discover_neutron_nuclides(task.nuclear_data_dir, task.library, run_openmc)


def _metas(task: Task):
    from cache_utils import yamc_fingerprint

    openmc_meta = {
        "library": task.library, "particles": task.particles,
        "batches": task.batches, "source_energy": task.source_energy,
        # Only a non-default seed enters the key, so the seed-1 references
        # already cached stay valid.
        **({"seed": _seed()} if _seed() != 1 else {}),
    }
    yamc_meta = {**openmc_meta, **yamc_fingerprint()}
    return openmc_meta, yamc_meta


def run_all(task: Task, backends=None, cache_dir: str = "cache"):
    """Run every backend for every discovered nuclide. Returns (all_results, first_yamc_model).

    ``cache_dir`` is resolved relative to the current working directory, which
    when run under nbconvert is the notebook's own directory (so the default
    ``"cache"`` lands in ``tasks/<family>/<geometry>/cache`` as before).
    """
    if task.geometry == "tokamak":
        from . import tokamak

        return tokamak.run_all(task, backends, cache_dir=cache_dir)
    if task.geometry == "multimaterial-slab":
        from . import multimaterial_slab

        return multimaterial_slab.run_all(task, backends, cache_dir=cache_dir)

    from ..backends import resolve

    b = backends if backends is not None else resolve()
    openmc_meta, yamc_meta = _metas(task)
    nuclides = discover(task, b.run_openmc)
    print(f"Running simulations for {len(nuclides)} nuclides: {nuclides}")

    all_results = {}
    first_yamc_model = None
    for nuclide in nuclides:
        per = {}
        if b.run_openmc:
            try:
                per["openmc"] = run_step(
                    cache_dir, task.library, "openmc", _cache_key(nuclide),
                    lambda n=nuclide: run_openmc_simulation(n, task), meta=openmc_meta, cache=True)
            except Exception as e:  # noqa: BLE001 - isolate backend failures
                print(f"  OpenMC Error ({nuclide}): {e}")
        for compute in b.computes:
            captured = {}

            def _run(n=nuclide, c=compute, cap=captured):
                res, model = run_yamc_simulation(n, task, compute=c)
                cap["model"] = model
                return res

            try:
                per[f"yamc-{compute}"] = run_step(
                    cache_dir, task.library, f"yamc-{compute}", _cache_key(nuclide), _run,
                    meta=yamc_meta, cache=task.caches_yamc)
                if first_yamc_model is None and "model" in captured:
                    first_yamc_model = captured["model"]
            except Exception as e:  # noqa: BLE001
                print(f"  YAMC ({compute}) Error ({nuclide}): {e}")
        if per:
            all_results[nuclide] = per
    print(f"Completed {len(all_results)} / {len(nuclides)} nuclides successfully.")
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
    """Render per-nuclide comparison plots, agreement tables, and tally tables."""
    from IPython.display import Markdown, display

    max_plot_energy = (task.source_energy or 0) * 1.1
    for nuclide, results in all_results.items():
        openmc_results = results.get("openmc")
        yamc_results_by_backend = {}
        for key, res in results.items():
            if key.startswith("yamc-"):
                yamc_results_by_backend[key[len("yamc-"):]] = res

        display(Markdown(f"## {nuclide}"))
        perf_parts = []
        if openmc_results is not None:
            perf_parts.append(f"OpenMC: {openmc_results['particles/s']:,.0f} particles/s")
        for compute, yres in yamc_results_by_backend.items():
            perf_parts.append(f"YAMC ({compute}): {yres['particles/s']:,.0f} particles/s")
        display(Markdown("**Performance:** " + " | ".join(perf_parts)))

        fig = create_comparison_plot(openmc_results, yamc_results_by_backend,
                                     max_plot_energy=max_plot_energy)
        fig.update_layout(title=f"{nuclide} - Flux Spectra Comparison")
        fig.show()

        if openmc_results is not None:
            ref_label, ref_results = "OpenMC", openmc_results
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

        display(Markdown("#### Tally Comparison"))
        table = create_tally_table(openmc_results, yamc_results_by_backend)
        display(table.style.hide(axis="index"))
