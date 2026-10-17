"""Transmutation verification family (two distinct notebooks).

This family has TWO structurally different notebooks that share their
comparison/plot/metrics code but not their physics:

- ``transmutation-reaction-rate`` (task family): transport-COUPLED depletion.
  A Monte-Carlo transport solve runs at every irradiation step on a solid
  sphere (r=5 cm, vacuum) with a 14.06 MeV point source; the flux-weighted
  reaction rates drive the burnup ODE. OpenMC uses
  ``openmc.deplete.CoupledOperator`` + ``PredictorIntegrator``; YAMC uses
  ``Model.simulate_transmutation(method="coupled", ...)``. Both caches are
  written (``openmc`` and ``yamc-cpu``).

- ``transmutation-flux-spectrum`` (task family): flux-DRIVEN material
  depletion with NO geometry/transport. OpenMC uses ``material.deplete(
  multigroup_flux=..., energy_group_structure=...)``; YAMC uses
  ``material.transmute(schedule=...)`` with a ``mc.sources.Histogram`` flux
  shape. Both caches are written (``openmc`` and ``yamc``).

Both codes solve every task on the library's full depletion chain; neither
reduces it.

The two ``run_*_transmutation`` runners are intentionally NOT unified. Only the
Pulse/Cooldown schedule builder, the step-comparison metrics, the density
tables/plot, and the display loop are shared. Run parameters come from the task
registry (``task.dims``, ``task.particles``, ``task.source_energy``,
``task.extras``); YAMC ``mc.Model`` in the reaction-rate runner is silenced with
``verbose=[]`` and OpenMC with ``verbosity=1`` + ``output=False``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import discovery
from ..runstep import run_step
from ..tasks import Task

DENSITY_THRESHOLD = 1e-15

REACTION_RATE = "transmutation-reaction-rate"
FLUX_SPECTRUM = "transmutation-flux-spectrum"


# --------------------------------------------------------------------------
# Shared schedule builder
# --------------------------------------------------------------------------
def build_pulse_schedule(timesteps, source_rates, source_obj, rate_scale: float = 1.0):
    """Build a yamc ``PulseSchedule`` of ``Pulse``/``Cooldown`` steps.

    Irradiation steps (``rate > 0``) become ``mc.Pulse(source=source_obj,
    rate=rate*rate_scale, duration=dt)``; cooling steps (``rate == 0``) become
    ``mc.Cooldown(duration=dt)``. ``rate_scale`` is 1.0 for the reaction-rate
    notebook (source-rate in n/s) and the total flux magnitude for the
    flux-spectrum notebook.
    """
    import yamc as mc

    schedule_steps = []
    for dt, rate in zip(timesteps, source_rates):
        if rate > 0:
            schedule_steps.append(mc.Pulse(source=source_obj, rate=rate_scale * rate, duration=dt))
        else:
            schedule_steps.append(mc.Cooldown(duration=dt))
    return mc.PulseSchedule(schedule_steps)


# --------------------------------------------------------------------------
# Reaction-rate runners (transport-coupled depletion)
# --------------------------------------------------------------------------
def run_openmc_reaction_rate(nuclide: str, task: Task, chain_file: str) -> dict:
    """Run OpenMC transport-coupled depletion for a single seed nuclide.

    Builds a sphere model with a 14.06 MeV point source, wraps it in a
    ``CoupledOperator`` (``normalization_mode='source-rate'`` so the prescribed
    ``source_rates`` are emission rates in n/s), and integrates the schedule with
    a predictor integrator on the library's full depletion chain, as yamc does.

    Returns dict with deplete_s, time_s and a ``steps`` list of
    ``{nuclide: atom_density}`` dicts (one per timestep, skipping the initial
    state) to match the comparison helpers.
    """
    import os
    import tempfile

    import openmc
    import openmc.deplete

    timesteps = task.extras["timesteps"]
    source_rates = task.extras["source_rates"]
    sphere_radius = task.dims.sphere_radius

    openmc.reset_auto_ids()
    openmc.config['cross_sections'] = os.path.join(task.nuclear_data_dir, 'cross_sections.xml')
    openmc.config['chain_file'] = chain_file

    material = openmc.Material(material_id=1, name=nuclide)
    material.add_nuclide(nuclide, 1.0, percent_type='ao')
    material.set_density('g/cm3', 1.0)
    material.temperature = 294.0
    material.depletable = True
    material.volume = 4.0 / 3.0 * np.pi * sphere_radius ** 3

    sphere = openmc.Sphere(r=sphere_radius, boundary_type='vacuum')
    cell = openmc.Cell(name='sphere', region=-sphere, fill=material)
    geometry = openmc.Geometry([cell])

    source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([task.source_energy], [1.0]),
    )
    settings = openmc.Settings(
        particles=task.particles,
        batches=10,
        source=source,
        seed=1,
        run_mode='fixed source',
        output={'tallies': False, 'summary': False},
    )
    settings.verbosity = 1
    model = openmc.model.Model(geometry, openmc.Materials([material]), settings)

    # Run inside a scratch directory so the statepoint / results files written by
    # the integrator do not litter the notebook folder.
    work = tempfile.mkdtemp(prefix="openmc_coupled_")
    results_h5 = os.path.join(work, "depletion_results.h5")

    cwd = os.getcwd()
    t1 = time.perf_counter()
    try:
        os.chdir(work)
        operator = openmc.deplete.CoupledOperator(
            model, chain_file=chain_file, normalization_mode='source-rate',
        )
        integrator = openmc.deplete.PredictorIntegrator(
            operator,
            timesteps=timesteps,
            source_rates=source_rates,
            timestep_units='s',
        )
        integrator.integrate(path=results_h5)
    finally:
        os.chdir(cwd)
    deplete_s = time.perf_counter() - t1

    results = openmc.deplete.Results(results_h5)
    # Results hold n+1 step materials (index 0 = initial state); skip index 0.
    steps = []
    for step in range(1, len(results)):
        mat = results[step].get_material("1")
        steps.append(mat.get_nuclide_atom_densities())

    return {
        'deplete_s': deplete_s,
        'time_s': deplete_s,
        'steps': steps,
    }


def run_yamc_reaction_rate(nuclide: str, task: Task):
    """Run YAMC transport-coupled depletion for a single seed nuclide.

    Builds the same sphere geometry with a 14.06 MeV point source, then calls
    ``Model.simulate_transmutation(method="coupled", ...)``, which re-runs
    transport at every irradiation step and feeds the flux-weighted reaction
    rates into the burnup ODE. Cooling steps (rate 0) become ``Cooldown`` entries
    that decay only. The network is the library's full chain, as OpenMC's is.
    Which products carry cross sections into transport is yamc's own screening
    (``product_threshold``, default 1e-12): a product is loaded once it reaches
    that atom fraction somewhere in the schedule.

    Returns dict with deplete_s, time_s and a ``steps`` list of
    ``{nuclide: atom_density}`` dicts (one per timestep, skipping the initial
    state) to match the comparison helpers.
    """
    import yamc as mc

    timesteps = task.extras["timesteps"]
    source_rates = task.extras["source_rates"]
    sphere_radius = task.dims.sphere_radius

    material = mc.Material(
        composition={nuclide: 1.0},
        density=1.0,
        temperature=294,
        transmutable=True,
        id=1,
        name=nuclide,
    )
    material.read_nuclear_data(task.library)

    sphere = mc.Sphere(x0=0.0, y0=0.0, z0=0.0, radius=sphere_radius, boundary='vacuum')
    cell = mc.Cell(id=1, name='sphere', region=sphere.below, material=material)
    geometry = mc.Geometry([cell])
    # Stochastic cell volume (sets both cell.volume and material.volume).
    geometry.calculate_volume(samples=1_000_000)

    source = mc.NeutronSource(
        position=(0, 0, 0),
        direction=mc.sources.Isotropic(),
        energy=mc.sources.Discrete([task.source_energy], [1.0]),
    )
    model = mc.Model(
        geometry=geometry, source=source, verbose=[],
    )

    # Irradiation pulses (rate in n/s) and cooling steps, matching OpenMC.
    schedule = build_pulse_schedule(timesteps, source_rates, source, rate_scale=1.0)

    t1 = time.perf_counter()
    with discovery.transmutation_network(task.library):
        results = model.simulate_transmutation(
            method="coupled",
            schedule=schedule,
            total_particles=task.particles,
            seed=1,
        )
    deplete_s = time.perf_counter() - t1

    # num_steps step materials (index 0 = initial state); skip index 0.
    steps = []
    for step in range(1, results.num_steps + 1):
        d = results.get_material_nuclides(material_id=1, step=step)
        steps.append(dict(d) if d else {})

    return {
        'deplete_s': deplete_s,
        'time_s': deplete_s,
        'steps': steps,
    }, model


# --------------------------------------------------------------------------
# Flux-spectrum runners (flux-driven material depletion, no transport)
# --------------------------------------------------------------------------
def run_openmc_flux_spectrum(nuclide: str, task: Task, chain_file: str) -> dict:
    """Run OpenMC ``material.deplete()`` for a single nuclide.

    Depletes on the library's full chain (``chain_file``, used whole, as yamc
    uses its full chain). OpenMC parses the file once per process and reuses it.

    Returns dict with deplete_s, time_s and steps.
    """
    import os

    import openmc

    multigroup_flux = task.extras["multigroup_flux"]
    energy_groups = task.extras["energy_groups"]
    timesteps = task.extras["timesteps"]
    openmc_source_rates = task.extras["openmc_source_rates"]

    openmc.config['cross_sections'] = os.path.join(task.nuclear_data_dir, 'cross_sections.xml')

    material = openmc.Material(name=nuclide)
    material.add_nuclide(nuclide, 1.0, percent_type='ao')
    material.set_density('g/cm3', 1.0)
    material.volume = 1.0
    material.temperature = 294.0
    material.depletable = True

    t0 = time.perf_counter()
    results = material.deplete(
        multigroup_flux=multigroup_flux,
        energy_group_structure=energy_groups,
        timesteps=timesteps,
        source_rates=openmc_source_rates,
        timestep_units='s',
        chain_file=chain_file,
    )
    deplete_s = time.perf_counter() - t0

    # OpenMC returns n+1 materials (index 0 = initial state); skip index 0
    steps = []
    for mat in results[1:]:
        steps.append(mat.get_nuclide_atom_densities())

    return {
        'deplete_s': deplete_s,
        'time_s': deplete_s,
        'steps': steps,
    }


def run_yamc_flux_spectrum(nuclide: str, task: Task):
    """Run YAMC ``material.transmute()`` for a single nuclide.

    Solves on the library's full chain, as OpenMC does: the transmutation
    settings point at the library keyword for the solve. Which products carry
    cross sections is yamc's ``product_threshold`` screening (default 1e-12).
    The 3-group flux shape is carried as a Histogram source; the Pulse rate is
    the total flux magnitude [n/cm^2/s] (sum of the multigroup flux), matching
    the OpenMC source_rates. Cooling steps (rate 0) become Cooldown entries.

    Returns dict with deplete_s, time_s and steps.
    """
    import yamc as mc

    energy_groups = task.extras["energy_groups"]
    multigroup_flux = task.extras["multigroup_flux"]
    timesteps = task.extras["timesteps"]
    yamc_source_rates = task.extras["yamc_source_rates"]

    material = mc.Material(
        composition={nuclide: 1.0},
        density=1.0,
        temperature=294,
        volume=1.0,
        transmutable=True,
        name=nuclide,
    )
    material.read_nuclear_data(task.library)

    # Spectrum shape as a Histogram; pulse rate carries the absolute magnitude.
    spectrum = mc.NeutronSource(
        energy=mc.sources.Histogram(energy_groups, multigroup_flux)
    )
    flux_magnitude = sum(multigroup_flux)
    schedule = build_pulse_schedule(timesteps, yamc_source_rates, spectrum, rate_scale=flux_magnitude)

    t1 = time.perf_counter()
    with discovery.transmutation_network(task.library):
        results = material.transmute(schedule=schedule)
    deplete_s = time.perf_counter() - t1

    # The composition at the end of each schedule step (no initial state).
    steps = [dict(mat.nuclides) for mat in results.step_materials(material.id or 0)]

    return {
        'deplete_s': deplete_s,
        'time_s': deplete_s,
        'steps': steps,
    }


# --------------------------------------------------------------------------
# Shared metrics / tables / plot (byte-identical across both notebooks)
# --------------------------------------------------------------------------
def transmute_quality_label(max_rel_diff):
    """Assign quality label based on max relative difference (%) for transmutation.

    Transmutation is the one family NOT graded on reduced chi-squared (the shared
    rule the spectrum families use via plotting.quality_from_chi2): its depletion
    densities are point values with no uncertainty, so chi-squared is not defined.
    It stays on relative difference until per-nuclide density uncertainties exist
    (yamc issue #188), at which point it should move onto quality_from_chi2 too.
    """
    if max_rel_diff < 0.1:
        return "Excellent"
    elif max_rel_diff < 1.0:
        return "Good"
    elif max_rel_diff < 5.0:
        return "Fair"
    else:
        return "Poor"


def compare_transmutation_steps(yamc_result, openmc_result, timesteps, phases):
    """Compare transmutation results step-by-step.

    Returns dict with detail_rows, summary metrics, and quality.
    """
    detail_rows = []
    shared_rel_diffs = []
    yamc_only_count = 0
    openmc_only_count = 0

    for step_idx in range(len(timesteps)):
        ymc_step = yamc_result['steps'][step_idx] if step_idx < len(yamc_result['steps']) else {}
        omc_step = openmc_result['steps'][step_idx] if step_idx < len(openmc_result['steps']) else {}

        all_products = set(ymc_step.keys()) | set(omc_step.keys())

        for product in sorted(all_products):
            y = ymc_step.get(product, 0.0)
            o = omc_step.get(product, 0.0)
            max_val = max(abs(y), abs(o))
            if max_val < DENSITY_THRESHOLD:
                continue

            y_above = abs(y) >= DENSITY_THRESHOLD
            o_above = abs(o) >= DENSITY_THRESHOLD
            shared = y_above and o_above

            if max_val > 0:
                rd = abs(y - o) / max_val * 100.0
            else:
                rd = 0.0

            detail_rows.append({
                'product': product,
                'step': step_idx + 1,
                'phase': phases[step_idx],
                'yamc': y,
                'openmc': o,
                'rel_diff_pct': rd,
                'shared': shared,
            })

            if shared:
                shared_rel_diffs.append(rd)
            elif y_above and not o_above:
                yamc_only_count += 1
            else:
                openmc_only_count += 1

    if shared_rel_diffs:
        max_rel_diff_pct = max(shared_rel_diffs)
        rms_rel_diff_pct = float(np.sqrt(np.mean(np.array(shared_rel_diffs) ** 2)))
    else:
        max_rel_diff_pct = 0.0
        rms_rel_diff_pct = 0.0

    n_shared = len({dr['product'] for dr in detail_rows if dr['shared']})
    n_total = len({dr['product'] for dr in detail_rows})

    quality = transmute_quality_label(max_rel_diff_pct)
    speedup = openmc_result['time_s'] / yamc_result['time_s'] if yamc_result['time_s'] > 0 else float('inf')

    return {
        'detail_rows': detail_rows,
        'max_rel_diff_pct': max_rel_diff_pct,
        'rms_rel_diff_pct': rms_rel_diff_pct,
        'n_shared': n_shared,
        'n_total': n_total,
        'yamc_only_count': yamc_only_count,
        'openmc_only_count': openmc_only_count,
        'quality': quality,
        'speedup': speedup,
    }


def create_density_comparison_table(detail_rows, step_idx):
    """Create a density comparison table for a specific timestep."""
    step_rows = [dr for dr in detail_rows if dr['step'] == step_idx + 1]

    rows = []
    for dr in sorted(step_rows, key=lambda x: -max(abs(x['yamc']), abs(x['openmc']))):
        rows.append({
            'Nuclide': dr['product'],
            'YAMC CPU': f"{dr['yamc']:.6e}",
            'OpenMC CPU': f"{dr['openmc']:.6e}",
            'Rel. Diff (%)': f"{dr['rel_diff_pct']:.4f}%",
            'Status': 'Shared' if dr['shared'] else ('YAMC only' if abs(dr['yamc']) >= DENSITY_THRESHOLD else 'OpenMC only'),
        })

    return pd.DataFrame(rows)


def create_density_evolution_plot(detail_rows, nuclide, timesteps, phases):
    """Density evolution of the top products with YAMC and OpenMC on one shared axis.

    Each product gets a single colour; YAMC is drawn as a solid line with round
    markers and OpenMC as a dashed line with x markers, so the two codes overlay
    for a direct comparison instead of sitting in separate side-by-side panels.
    """

    # Get all products and their max densities
    product_max = {}
    for dr in detail_rows:
        p = dr['product']
        val = max(abs(dr['yamc']), abs(dr['openmc']))
        if p not in product_max or val > product_max[p]:
            product_max[p] = val

    # Top 10 products by max density (excluding the starting nuclide itself)
    top_products = sorted(
        [(p, v) for p, v in product_max.items() if p != nuclide],
        key=lambda x: -x[1]
    )[:10]

    if not top_products:
        return None

    steps = list(range(1, len(timesteps) + 1))
    step_labels = [f"Step {s}\n({phases[s-1]})" for s in steps]

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    fig = go.Figure()

    for idx, (product, _) in enumerate(top_products):
        yamc_densities = []
        openmc_densities = []
        for step in steps:
            ymc_val = 0.0
            omc_val = 0.0
            for dr in detail_rows:
                if dr['product'] == product and dr['step'] == step:
                    ymc_val = dr['yamc']
                    omc_val = dr['openmc']
                    break
            yamc_densities.append(ymc_val)
            openmc_densities.append(omc_val)

        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=steps, y=yamc_densities, mode='lines+markers',
            name=f"YAMC CPU - {product}", legendgroup=product,
            line=dict(color=color),
            marker=dict(color=color, symbol='circle', size=6),
        ))
        fig.add_trace(go.Scatter(
            x=steps, y=openmc_densities, mode='lines+markers',
            name=f"OpenMC CPU - {product}", legendgroup=product,
            line=dict(color=color, dash='dash'),
            marker=dict(color=color, symbol='x', size=7),
        ))

    fig.update_xaxes(title_text="Timestep", tickvals=steps, ticktext=step_labels)
    fig.update_yaxes(type="log", title_text="Atom Density (atoms/b-cm)", exponentformat='e')

    fig.update_layout(
        height=550,
        title=f"{nuclide} - Product Density Evolution (Top 10), solid=YAMC CPU dashed=OpenMC CPU",
        legend=dict(x=1.02, y=1.0),
        margin=dict(l=60, r=170, t=60, b=60),
    )

    return fig


def create_quality_metrics_table(comparison):
    """Create a summary table of quality metrics."""
    rows = [
        {'Metric': 'Max Rel. Diff (shared)', 'Value': f"{comparison['max_rel_diff_pct']:.4f}%"},
        {'Metric': 'RMS Rel. Diff (shared)', 'Value': f"{comparison['rms_rel_diff_pct']:.4f}%"},
        {'Metric': 'Shared Products', 'Value': str(comparison['n_shared'])},
        {'Metric': 'Total Products', 'Value': str(comparison['n_total'])},
        {'Metric': 'YAMC-only entries', 'Value': str(comparison['yamc_only_count'])},
        {'Metric': 'OpenMC-only entries', 'Value': str(comparison['openmc_only_count'])},
        {'Metric': 'Quality', 'Value': comparison['quality']},
        {'Metric': 'Speedup', 'Value': f"{comparison['speedup']:.1f}x"},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Discovery / run / display
# --------------------------------------------------------------------------
def discover(task: Task, run_openmc: bool):
    """Discover stable seed nuclides in the depletion chain.

    Reaction-rate wants bare name strings; flux-spectrum wants
    ``(name, neutron_path)`` tuples (``return_paths=True``).
    """
    chain_file = discovery.resolve_openmc_chain(task.library)
    tdata = discovery.transmutation_data_path(task.library)
    return_paths = task.family == FLUX_SPECTRUM
    return discovery.discover_stable_nuclides(
        task.nuclear_data_dir, task.library, chain_file, tdata, run_openmc,
        return_paths=return_paths,
    )


def _metas(task: Task, chain_file: str):
    """Reproduce the notebook's cache-meta dicts exactly (byte-match on disk).

    ``"chain": "full"`` records that both codes solved on the library's full
    chain. Results cached when the chain was reduced carry ``reduce_level``
    instead, so they do not match and re-run.
    """
    if task.family == REACTION_RATE:
        from cache_utils import yamc_fingerprint

        common = {
            "library": task.library,
            "method": "coupled",
            "chain_file": chain_file,
            "sphere_radius": task.dims.sphere_radius,
            "source_energy": task.source_energy,
            "material_density": 1.0,
            "timesteps": list(task.extras["timesteps"]),
            "source_rates": list(task.extras["source_rates"]),
            "chain": "full",
        }
        openmc_meta = {**common, "total_particles": task.particles}
        yamc_meta = {**common, "total_particles": task.particles, **yamc_fingerprint()}
        return openmc_meta, yamc_meta
    # flux-spectrum: flux-driven material.transmute() (no transport).
    from cache_utils import yamc_fingerprint

    openmc_meta = {
        "library": task.library,
        "chain_file": chain_file,
        "energy_groups": list(task.extras["energy_groups"]),
        "multigroup_flux": list(task.extras["multigroup_flux"]),
        "timesteps": list(task.extras["timesteps"]),
        "openmc_source_rates": list(task.extras["openmc_source_rates"]),
        "chain": "full",
    }
    yamc_meta = {
        "library": task.library,
        "energy_groups": list(task.extras["energy_groups"]),
        "multigroup_flux": list(task.extras["multigroup_flux"]),
        "timesteps": list(task.extras["timesteps"]),
        "yamc_source_rates": list(task.extras["yamc_source_rates"]),
        "chain": "full",
        **yamc_fingerprint(),
    }
    return openmc_meta, yamc_meta


def run_all(task: Task, backends=None, cache_dir: str = "cache"):
    """Run every backend for every discovered nuclide. Returns (all_results, first_yamc_model).

    Dispatches on ``task.family``: the transport-coupled reaction-rate flow or
    the flux-driven flux-spectrum flow, both on the library's full chain. Both
    backends are cached in both flows.

    ``cache_dir`` is resolved relative to the current working directory, which
    when run under nbconvert is the notebook's own directory.
    """
    from ..backends import resolve

    b = backends if backends is not None else resolve()
    timesteps = task.extras["timesteps"]
    phases = task.extras["phases"]

    chain_file = discovery.resolve_openmc_chain(task.library)
    openmc_meta, yamc_meta = _metas(task, chain_file)

    items = discover(task, b.run_openmc)
    if task.family == FLUX_SPECTRUM:
        names = [t[0] for t in items]
    else:
        names = list(items)
    print(f"Running {len(names)} nuclides: {names}")

    all_results = {}
    first_yamc_model = None

    if task.family == REACTION_RATE:
        for nuclide in names:
            per = {}
            if b.run_openmc:
                try:
                    per["openmc"] = run_step(
                        cache_dir, task.library, "openmc", nuclide,
                        lambda n=nuclide: run_openmc_reaction_rate(n, task, chain_file),
                        meta=openmc_meta, cache=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  OpenMC Error ({nuclide}): {e}")
            if b.computes:
                captured = {}

                def _run(n=nuclide, cap=captured):
                    res, model = run_yamc_reaction_rate(n, task)
                    cap["model"] = model
                    return res

                try:
                    per["yamc"] = run_step(
                        cache_dir, task.library, "yamc-cpu", nuclide, _run,
                        meta=yamc_meta, cache=task.caches_yamc)
                    if first_yamc_model is None and "model" in captured:
                        first_yamc_model = captured["model"]
                except Exception as e:  # noqa: BLE001
                    print(f"  YAMC Error ({nuclide}): {e}")
            _finish_nuclide(per, all_results, nuclide, timesteps, phases)
    else:  # FLUX_SPECTRUM
        for nuclide, _neutron_path in items:
            per = {}
            if b.run_openmc:
                try:
                    per["openmc"] = run_step(
                        cache_dir, task.library, "openmc", nuclide,
                        lambda n=nuclide: run_openmc_flux_spectrum(n, task, chain_file),
                        meta=openmc_meta, cache=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  OpenMC Error ({nuclide}): {e}")
            if b.computes:
                try:
                    per["yamc"] = run_step(
                        cache_dir, task.library, "yamc", nuclide,
                        lambda n=nuclide: run_yamc_flux_spectrum(n, task),
                        meta=yamc_meta, cache=task.caches_yamc)
                except Exception as e:  # noqa: BLE001
                    print(f"  YAMC Error ({nuclide}): {e}")
            _finish_nuclide(per, all_results, nuclide, timesteps, phases)

    print(f"Completed {len(all_results)} / {len(names)} nuclides successfully.")
    return all_results, first_yamc_model


def _finish_nuclide(per, all_results, nuclide, timesteps, phases):
    """Compute the comparison (when both backends ran) and store the result."""
    if 'yamc' in per and 'openmc' in per:
        comparison = compare_transmutation_steps(per['yamc'], per['openmc'], timesteps, phases)
        per['comparison'] = comparison
        print(f"  {nuclide}: max_rel_diff={comparison['max_rel_diff_pct']:.4f}%, "
              f"quality={comparison['quality']}")
    if per:
        all_results[nuclide] = per


def display_geometry(model, task: Task):
    """Embed the yamc geometry plot as an iframe (reaction-rate only).

    The flux-spectrum notebook has no geometry/transport (``first_yamc_model``
    is ``None``), so this simply notes that there is nothing to plot.
    """
    import base64

    from IPython.display import HTML, display

    if model is None:
        print("No geometry to plot (flux-driven depletion has no transport model).")
        return
    html_plot = model.plot(basis="xz", n_samples=500, axis_units="cm")
    html_plot.save(f"{task.geometry}_model_plot.html")
    encoded = base64.b64encode(html_plot.html.encode()).decode()
    display(HTML(f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="600" frameborder="0"></iframe>'))


def display_results(task: Task, all_results: dict):
    """Render per-nuclide quality metrics, density-evolution plots and step tables."""
    from IPython.display import Markdown, display

    timesteps = task.extras["timesteps"]
    phases = task.extras["phases"]

    for nuclide, results in all_results.items():
        yr = results.get('yamc')        # may be None when no yamc-* backend ran
        or_ = results.get('openmc')     # may be None when openmc-cpu not selected
        comparison = results.get('comparison')  # only present when both ran

        # Header
        display(Markdown(f"## {nuclide}"))

        # Performance comparison (depletion time)
        perf_parts = []
        if or_ is not None:
            perf_parts.append(f"OpenMC: {or_['deplete_s']:.3f}s")
        if yr is not None:
            perf_parts.append(f"YAMC: {yr['deplete_s']:.3f}s")
        if or_ is not None and yr is not None and yr['deplete_s'] > 0:
            perf_parts.append(f"Speedup: {or_['deplete_s'] / yr['deplete_s']:.1f}x")
        display(Markdown("**Performance:** " + " | ".join(perf_parts)))

        # Quality metrics and density tables compare OpenMC vs YAMC; without both
        # there is nothing to compare, so note it and continue rather than KeyError.
        if comparison is None:
            only = "OpenMC" if or_ is not None else "YAMC"
            display(Markdown(
                f"*Only {only} ran; comparison metrics and density tables are skipped.*"
            ))
            print()
            continue

        # Quality metrics table
        display(Markdown("### Quality Metrics"))
        metrics_table = create_quality_metrics_table(comparison)
        display(metrics_table.style.hide(axis='index'))

        # Density evolution plot
        fig = create_density_evolution_plot(comparison['detail_rows'], nuclide, timesteps, phases)
        if fig is not None:
            fig.show()

        # Density comparison tables per timestep
        for step_idx in range(len(timesteps)):
            phase = phases[step_idx]
            display(Markdown(f"### Step {step_idx + 1} ({phase})"))
            table = create_density_comparison_table(comparison['detail_rows'], step_idx)
            if not table.empty:
                display(table.style.hide(axis='index'))
            else:
                display(Markdown("*No products above density threshold.*"))

        print()  # Add spacing between nuclides
