"""Neutron multimaterial-slab benchmark (family ``neutron``, geometry ``multimaterial-slab``).

A single fixed geometry: a stack of dissimilar material layers, with a 14 MeV
monodirectional beam entering the front face and the neutron flux spectrum
(VITAMIN-J-175) tallied **in every layer**. One simulation per code; each layer
is cached as its own item so the database shows one row per layer with a
YAMC-vs-OpenMC spectrum comparison.

Why this, alongside ``sphere`` and ``broomstick`` (both single-material leakage
tests): those span *collision count* (sphere = many, isotropic; broomstick =
few, forward) but are blind to the thing every real fusion/fission model is
made of -- **stacked dissimilar materials**. This model exercises the axes they
cannot reach:

- **material interfaces / cell handoff** -- correct material at each point,
  current continuity across boundaries (the class of logic where yamc#204 and
  #206 lived, both boundary/re-entry bugs);
- **spectral coupling** -- the flux entering layer N is the *output* of layer
  N-1, so a 14 MeV beam hardens/scatters in W, softens through the breeder,
  thermalises in water/graphite; errors compound at interfaces and localise to
  a specific material transition instead of a whole ball.

The layer stack is a fusion-blanket-like sequence (plasma-facing outward):
void (source gap) -> tungsten (armour) -> steel (structure) -> Li4SiO4 breeder
-> water (coolant) -> graphite (reflector) -> vanadium (low-activation structure).

Geometry is fully bounded (finite boxes, vacuum implicit complement) so it is
GPU-ready. ``yamc-gpu`` is nonetheless excluded from the task for now: the GPU
backend currently panics on multi-material geometries whose materials have
different nuclide sets (yamc#208, an XS-extraction buffer bug, not a geometry
limitation). Add ``yamc-gpu`` back to the task's ``codes`` once that is fixed.
"""
from __future__ import annotations

import time

import numpy as np

from ..runstep import run_step
from ..tasks import Task

GROUP_STRUCTURE = "VITAMIN-J-175"

# Transverse half-width of the (square) stack cross-section, cm. Large enough
# that a forward 14 MeV beam is well contained through the stack; identical in
# both codes, so any residual transverse leakage cancels in the ratio.
HALF_WIDTH = 40.0

# Ordered layer stack: (name, material-name or None for void, thickness cm).
# Materials chosen for diversity: high-Z strong-URR (W, and the Fe/Cr/Ni/Mo of
# steel), light moderators (H2O, graphite), the multi-isotope Li breeder, and a
# distinct low-activation structural metal (V).
LAYERS = [
    ("void", None, 10.0),
    ("tungsten", "tungsten", 2.0),
    ("steel", "steel", 3.0),
    ("breeder", "breeder", 30.0),
    ("water", "water", 5.0),
    ("reflector", "reflector", 20.0),
    ("vanadium", "vanadium", 3.0),
]

# Compositions (atom fractions) + mass density (g/cm3), shared by both codes.
_MATERIALS = {
    "tungsten": ({"W": 1.0}, 19.3),                                   # first-wall armour
    "steel": ({"Fe": 0.695, "Cr": 0.175, "Ni": 0.10, "Mo": 0.03}, 8.0),  # SS316-like
    "breeder": ({"Li6": 2.4, "Li7": 1.6, "Si": 1.0, "O": 4.0}, 2.4),  # Li4SiO4 (Li-6 enriched)
    "water": ({"H": 2.0, "O": 1.0}, 1.0),                            # H2O coolant
    "reflector": ({"C": 1.0}, 1.7),                                  # graphite reflector
    "vanadium": ({"V": 1.0}, 6.1),                                   # V-alloy structure
}

# Layer names that carry material (compared); the void layer is tallied too (it
# sees the incident beam + backscatter) but carries no material.
TALLY_LAYERS = [name for name, _mat, _t in LAYERS]


def _is_nuclide(name: str) -> bool:
    return any(ch.isdigit() for ch in name)


def _edges():
    """Cumulative x-plane offsets [0, t0, t0+t1, ...] for the layer stack."""
    xs = [0.0]
    for _name, _mat, thick in LAYERS:
        xs.append(xs[-1] + thick)
    return xs


# --------------------------------------------------------------------------
# yamc geometry
# --------------------------------------------------------------------------
def build_yamc(task: Task):
    """Build the yamc (geometry, {layer_name: cell}) for the multimaterial slab."""
    import yamc as mc

    mats = {}
    for mname, (comp, dens) in _MATERIALS.items():
        m = mc.Material(composition=dict(comp), density=dens, temperature=294, name=mname)
        m.read_nuclear_data(task.library)
        mats[mname] = m

    xs = _edges()
    h = HALF_WIDTH
    # x-planes at every layer boundary; front and back faces are vacuum.
    xplanes = []
    for i, x in enumerate(xs):
        boundary = "vacuum" if (i == 0 or i == len(xs) - 1) else None
        kw = {"boundary": boundary} if boundary else {}
        xplanes.append(mc.Plane(axis="x", offset=x, **kw))
    ymin = mc.Plane(axis="y", offset=-h, boundary="vacuum")
    ymax = mc.Plane(axis="y", offset=h, boundary="vacuum")
    zmin = mc.Plane(axis="z", offset=-h, boundary="vacuum")
    zmax = mc.Plane(axis="z", offset=h, boundary="vacuum")
    transverse = ymin.above & ymax.below & zmin.above & zmax.below

    cells = {}
    for i, (name, mname, _t) in enumerate(LAYERS):
        region = xplanes[i].above & xplanes[i + 1].below & transverse
        kwargs = {"region": region, "name": name}
        if mname is not None:
            kwargs["material"] = mats[mname]
        cells[name] = mc.Cell(**kwargs)
    geometry = mc.Geometry(list(cells.values()))
    return geometry, cells


def _yamc_source(task: Task):
    """14 MeV monodirectional beam entering the front face (+x)."""
    import yamc as mc

    return mc.NeutronSource(
        position=(0.01, 0.0, 0.0),
        direction=mc.sources.Monodirectional((1.0, 0.0, 0.0)),
        energy=mc.sources.Discrete([task.source_energy], [1.0]),
    )


def run_slab_yamc(task: Task, compute: str = "cpu") -> dict:
    """Run the yamc slab model once; return {layer_name: results_dict}."""
    import yamc as mc

    geometry, cells = build_yamc(task)
    tallies = {}
    for name in TALLY_LAYERS:
        tallies[name] = mc.Tally(
            scores=["flux"], name=f"flux_{name}", cells=cells[name],
            particle="neutron", energy_group_structure=GROUP_STRUCTURE,
        )
    energy_bins = np.array(next(iter(tallies.values())).energy_bins)

    model = mc.Model(geometry=geometry, tallies=list(tallies.values()),
                     source=_yamc_source(task), verbose=[])
    total = task.particles * task.batches
    t0 = time.perf_counter()
    sim = model.simulate_transport(total_particles=total, seed=1, compute=compute)
    elapsed = time.perf_counter() - t0
    pps = sim.particles_per_second if sim.particles_per_second is not None else (
        total / elapsed if elapsed > 0 else 0)

    out = {}
    for name, tally in tallies.items():
        r = sim[tally]
        out[name] = {
            "layer": name, "code": f"yamc-{compute}", "particles/s": pps,
            "flux_spectra_mean": np.array(r.mean).squeeze(),
            "flux_spectra_std_dev": np.array(r.standard_deviation).squeeze(),
            "energy_bins": energy_bins,
        }
    return out


# --------------------------------------------------------------------------
# OpenMC geometry
# --------------------------------------------------------------------------
def build_openmc(task: Task):
    """Build the OpenMC (geometry, {layer_name: cell}) for the multimaterial slab."""
    import openmc

    mats = {}
    for mname, (comp, dens) in _MATERIALS.items():
        m = openmc.Material(name=mname)
        for elem, frac in comp.items():
            if _is_nuclide(elem):
                m.add_nuclide(elem, frac, "ao")
            else:
                m.add_element(elem, frac, "ao")
        m.set_density("g/cm3", dens)
        m.temperature = 294
        mats[mname] = m

    xs = _edges()
    h = HALF_WIDTH
    xplanes = []
    for i, x in enumerate(xs):
        kw = {"boundary_type": "vacuum"} if (i == 0 or i == len(xs) - 1) else {}
        xplanes.append(openmc.XPlane(x0=x, **kw))
    ymin = openmc.YPlane(y0=-h, boundary_type="vacuum")
    ymax = openmc.YPlane(y0=h, boundary_type="vacuum")
    zmin = openmc.ZPlane(z0=-h, boundary_type="vacuum")
    zmax = openmc.ZPlane(z0=h, boundary_type="vacuum")
    transverse = +ymin & -ymax & +zmin & -zmax

    cells = {}
    for i, (name, mname, _t) in enumerate(LAYERS):
        region = +xplanes[i] & -xplanes[i + 1] & transverse
        cell = openmc.Cell(name=name, region=region)
        if mname is not None:
            cell.fill = mats[mname]
        cells[name] = cell
    return openmc.Geometry(list(cells.values())), cells


def run_slab_openmc(task: Task) -> dict:
    """Run the OpenMC slab model once; return {layer_name: results_dict}."""
    import openmc

    openmc.config["cross_sections"] = f"{task.nuclear_data_dir}cross_sections.xml"
    geometry, cells = build_openmc(task)

    energy_filter = openmc.EnergyFilter.from_group_structure(GROUP_STRUCTURE)
    ebins_2d = energy_filter.bins
    energy_bins = np.concatenate([[ebins_2d[0, 0]], ebins_2d[:, 1]])

    tallies = {}
    for name in TALLY_LAYERS:
        t = openmc.Tally(name=f"flux_{name}")
        t.filters = [openmc.CellFilter(cells[name]), energy_filter]
        t.scores = ["flux"]
        tallies[name] = t

    source = openmc.IndependentSource(
        space=openmc.stats.Point((0.01, 0.0, 0.0)),
        angle=openmc.stats.Monodirectional((1.0, 0.0, 0.0)),
        energy=openmc.stats.Discrete([task.source_energy], [1.0]),
    )
    settings = openmc.Settings(
        particles=task.particles, batches=task.batches, source=source, seed=1,
        run_mode="fixed source", output={"tallies": False, "summary": False},
    )
    settings.verbosity = 1

    model = openmc.Model(geometry=geometry, settings=settings,
                         tallies=openmc.Tallies(list(tallies.values())))
    sp = openmc.StatePoint(model.run(apply_tally_results=True, output=False))
    total = task.particles * task.batches
    pps = total / sp.runtime["transport"]

    out = {}
    for name, tally in tallies.items():
        out[name] = {
            "layer": name, "code": "openmc", "particles/s": pps,
            "flux_spectra_mean": np.array(tally.mean).squeeze(),
            "flux_spectra_std_dev": np.array(tally.std_dev).squeeze(),
            "energy_bins": energy_bins,
        }
    return out


# --------------------------------------------------------------------------
# discovery / run
# --------------------------------------------------------------------------
def discover(task: Task, run_openmc: bool) -> list[str]:
    """The compared layers (fixed for the slab geometry)."""
    return list(TALLY_LAYERS)


def _metas(task: Task):
    from cache_utils import yamc_fingerprint

    openmc_meta = {
        "library": task.library, "particles": task.particles,
        "batches": task.batches, "source_energy": task.source_energy,
    }
    yamc_meta = {**openmc_meta, **yamc_fingerprint()}
    return openmc_meta, yamc_meta


def run_all(task: Task, backends=None, cache_dir: str = "cache"):
    """Run the slab model once per code; cache one item per layer.

    The model is memoized per code: the first uncached layer triggers the single
    simulation and the rest reuse its per-layer results. Returns (all_results,
    None) to match the other families' run_all signature.
    """
    from ..backends import resolve

    b = backends if backends is not None else resolve()
    openmc_meta, yamc_meta = _metas(task)
    layers = discover(task, b.run_openmc)
    print(f"Running multimaterial-slab neutron model for {len(layers)} layers: {layers}")

    memo: dict = {}  # code -> {layer_name: result_dict}

    def layer_fn(code, layer_name, runner):
        def _fn():
            if code not in memo:
                memo[code] = runner()
            return memo[code][layer_name]
        return _fn

    all_results: dict = {}
    for layer_name in layers:
        per = {}
        if b.run_openmc:
            try:
                per["openmc"] = run_step(
                    cache_dir, task.library, "openmc", layer_name,
                    layer_fn("openmc", layer_name, lambda: run_slab_openmc(task)),
                    meta=openmc_meta, cache=True)
            except Exception as e:  # noqa: BLE001
                print(f"  OpenMC Error ({layer_name}): {e}")
        for compute in b.computes:
            code = f"yamc-{compute}"
            try:
                per[code] = run_step(
                    cache_dir, task.library, code, layer_name,
                    layer_fn(code, layer_name, lambda c=compute: run_slab_yamc(task, compute=c)),
                    meta=yamc_meta, cache=task.caches_yamc)
            except Exception as e:  # noqa: BLE001
                print(f"  YAMC ({compute}) Error ({layer_name}): {e}")
        if per:
            all_results[layer_name] = per
    print(f"Completed {len(all_results)} / {len(layers)} layers.")
    return all_results, None
