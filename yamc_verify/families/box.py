"""Neutron big-box leakage benchmark (family ``neutron``, geometry ``box``).

A deliberately *thick* transmission / self-shielding test, complementary to the
thin ``sphere`` geometry. One natural element at a time: a large cube of the
element at a realistic solid atom density, with a 14 MeV point source at the
centre, and the neutron flux spectrum tallied in a thin **outer shell** (a
leakage proxy). Measuring the flux after transmission through many
mean-free-paths at a realistic number density exposes resonance self-shielding
(including the multi-isotope self-shielding of a natural element) and
deep-penetration differences that the thin, near-infinite-dilution ``sphere``
is blind to.

Per **element**, not per nuclide, on purpose: single isotopes transport nearly
identically between the codes, and the self-shielding differences show up only
in the multi-isotope natural element (see yamc#204). Read across the suite:
- ``sphere`` agrees, ``box`` disagrees -> a self-shielding / transport difference.
- both disagree                        -> a fundamental cross-section difference.

Actinide elements are excluded: a large dense block of a fissile element would
go critical and a fixed-source run would diverge. Fission (the only route to
criticality) is confined to the actinides.
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np

from cache_utils import select_requested

from .. import discovery
from ..discovery import element_from_nuclide
from ..runstep import run_step
from ..tasks import Task

GROUP_STRUCTURE = "VITAMIN-J-175"

# Big cube (cm) and a realistic solid atom density. Resonance self-shielding
# scales with the atom density, so it is fixed here and the per-element mass
# density is derived from the element's atomic weight.
SPEC = {
    "outer_half_width": 30.0,   # cube spans -30..30 cm (a "big box")
    "inner_half_width": 25.0,   # leakage tally shell is 25..30 cm
    "atom_density": 0.06,       # atoms/b-cm (a typical solid; drives self-shielding)
}

_AVOGADRO_BARN = 0.6022140  # N_A * 1e-24 (atoms.barn / mol.cm)

# Actinide element symbols (excluded so the big dense box can't go critical).
_ACTINIDES = frozenset("Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())


def _mass_density(element: str) -> float:
    """g/cm3 giving ``SPEC['atom_density']`` atoms/b-cm for natural ``element``."""
    import openmc.data

    return SPEC["atom_density"] * openmc.data.atomic_weight(element) / _AVOGADRO_BARN


# --------------------------------------------------------------------------
# geometry (nested cubes; leakage tallied in the outer shell)
# --------------------------------------------------------------------------
def build_yamc(element: str, task: Task):
    import yamc as mc

    mat = mc.Material(composition={element: 1.0}, density=_mass_density(element), temperature=294)
    mat.read_nuclear_data(task.library)
    b, a = SPEC["outer_half_width"], SPEC["inner_half_width"]

    def planes(h, boundary=None):
        kw = {"boundary": boundary} if boundary else {}
        return {ax: (mc.Plane(axis=ax, offset=-h, **kw), mc.Plane(axis=ax, offset=h, **kw))
                for ax in ("x", "y", "z")}

    o, i = planes(b, boundary="vacuum"), planes(a)
    outer = (o["x"][0].above & o["x"][1].below & o["y"][0].above
             & o["y"][1].below & o["z"][0].above & o["z"][1].below)
    inner = (i["x"][0].above & i["x"][1].below & i["y"][0].above
             & i["y"][1].below & i["z"][0].above & i["z"][1].below)
    outside = (i["x"][0].below | i["x"][1].above | i["y"][0].below
               | i["y"][1].above | i["z"][0].below | i["z"][1].above)

    core = mc.Cell(region=inner, name="core", material=mat)
    shell = mc.Cell(region=outer & outside, name="leakage_shell", material=mat)
    geometry = mc.Geometry([core, shell])
    return geometry, shell


def build_openmc(element: str, task: Task):
    import openmc

    mat = openmc.Material()
    mat.add_element(element, 1.0)
    mat.set_density("g/cm3", _mass_density(element))
    mat.temperature = 294
    b, a = SPEC["outer_half_width"], SPEC["inner_half_width"]

    def planes(h, boundary=None):
        kw = {"boundary_type": boundary} if boundary else {}
        return {
            "x": (openmc.XPlane(x0=-h, **kw), openmc.XPlane(x0=h, **kw)),
            "y": (openmc.YPlane(y0=-h, **kw), openmc.YPlane(y0=h, **kw)),
            "z": (openmc.ZPlane(z0=-h, **kw), openmc.ZPlane(z0=h, **kw)),
        }

    o, i = planes(b, boundary="vacuum"), planes(a)
    outer = (+o["x"][0] & -o["x"][1] & +o["y"][0] & -o["y"][1] & +o["z"][0] & -o["z"][1])
    inner = (+i["x"][0] & -i["x"][1] & +i["y"][0] & -i["y"][1] & +i["z"][0] & -i["z"][1])
    outside = (-i["x"][0] | +i["x"][1] | -i["y"][0] | +i["y"][1] | -i["z"][0] | +i["z"][1])

    core = openmc.Cell(name="core", region=inner, fill=mat)
    shell = openmc.Cell(name="leakage_shell", region=outer & outside, fill=mat)
    return openmc.Geometry([core, shell]), shell, mat


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------
def run_yamc_simulation(element: str, task: Task, compute: str = "cpu") -> dict:
    import yamc as mc

    geometry, shell = build_yamc(element, task)
    tally = mc.Tally(scores=["flux"], name="flux_energy_binned", cells=shell,
                     particle="neutron", energy_group_structure=GROUP_STRUCTURE)
    energy_bins = np.array(tally.energy_bins)
    source = mc.NeutronSource(position=(0, 0, 0), direction=mc.sources.Isotropic(),
                              energy=mc.sources.Discrete([task.source_energy], [1.0]))
    model = mc.Model(geometry=geometry, tallies=[tally], source=source, verbose=[])
    total = task.particles * task.batches
    t0 = time.perf_counter()
    sim = model.simulate_transport(total_particles=total, seed=1, compute=compute)
    elapsed = time.perf_counter() - t0
    pps = sim.particles_per_second if sim.particles_per_second is not None else (
        total / elapsed if elapsed > 0 else 0)
    spec = sim[tally]
    return {
        "element": element, "code": f"yamc-{compute}", "particles/s": pps,
        "flux_spectra_mean": np.array(spec.mean).squeeze(),
        "flux_spectra_std_dev": np.array(spec.standard_deviation).squeeze(),
        "energy_bins": energy_bins,
    }


def run_openmc_simulation(element: str, task: Task) -> dict:
    import openmc

    openmc.config["cross_sections"] = f"{task.nuclear_data_dir}cross_sections.xml"
    geometry, shell, _mat = build_openmc(element, task)
    energy_filter = openmc.EnergyFilter.from_group_structure(GROUP_STRUCTURE)
    ebins_2d = energy_filter.bins
    energy_bins = np.concatenate([[ebins_2d[0, 0]], ebins_2d[:, 1]])
    tally = openmc.Tally(name="flux_energy_binned")
    tally.filters = [openmc.CellFilter(shell), energy_filter]
    tally.scores = ["flux"]
    source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        energy=openmc.stats.Discrete([task.source_energy], [1.0]),
        angle=openmc.stats.Isotropic(),
    )
    settings = openmc.Settings(
        particles=task.particles, batches=task.batches, source=source, seed=1,
        run_mode="fixed source", output={"tallies": False, "summary": False},
    )
    settings.verbosity = 1
    model = openmc.Model(geometry=geometry, settings=settings, tallies=openmc.Tallies([tally]))
    sp = openmc.StatePoint(model.run(apply_tally_results=True, output=False))
    total = task.particles * task.batches
    return {
        "element": element, "code": "openmc",
        "particles/s": total / sp.runtime["transport"],
        "flux_spectra_mean": np.array(tally.mean).squeeze(),
        "flux_spectra_std_dev": np.array(tally.std_dev).squeeze(),
        "energy_bins": energy_bins,
    }


# --------------------------------------------------------------------------
# discovery / run
# --------------------------------------------------------------------------
def discover(task: Task, run_openmc: bool) -> list[str]:
    """Natural elements yamc can build for neutron transport, minus actinides."""
    import yamc

    neutron_dir = os.path.join(task.nuclear_data_dir, "neutron")
    nuclides = [os.path.splitext(os.path.basename(f))[0]
                for f in sorted(glob.glob(os.path.join(neutron_dir, "*.h5")))]
    elements = sorted({element_from_nuclide(n) for n in nuclides if "_" not in n})
    runnable = []
    for e in elements:
        if e in _ACTINIDES:
            continue
        try:
            m = yamc.Material(composition={e: 1.0}, density=1.0)
            m.read_nuclear_data(task.library)
            runnable.append(e)
        except Exception:  # noqa: BLE001 - element yamc cannot build from natural abundance
            continue
    if not runnable:
        raise FileNotFoundError(f"No box-runnable elements found for library {task.library!r}.")
    return select_requested(
        runnable, kind="element",
        stable_getter=lambda: {element_from_nuclide(n)
                               for n in discovery.stable_nuclide_names(task.library, run_openmc)},
    )


def _metas(task: Task):
    from cache_utils import yamc_fingerprint

    openmc_meta = {
        "library": task.library, "particles": task.particles,
        "batches": task.batches, "source_energy": task.source_energy,
    }
    yamc_meta = {**openmc_meta, **yamc_fingerprint()}
    return openmc_meta, yamc_meta


def run_all(task: Task, backends=None, cache_dir: str = "cache"):
    """Run every backend for every discovered (non-actinide) element; cache."""
    from ..backends import resolve

    b = backends if backends is not None else resolve()
    openmc_meta, yamc_meta = _metas(task)
    elements = discover(task, b.run_openmc)
    print(f"Running box leakage model for {len(elements)} elements: {elements}")

    all_results: dict = {}
    for element in elements:
        per = {}
        if b.run_openmc:
            try:
                per["openmc"] = run_step(
                    cache_dir, task.library, "openmc", element,
                    lambda e=element: run_openmc_simulation(e, task), meta=openmc_meta, cache=True)
            except Exception as e:  # noqa: BLE001
                print(f"  OpenMC Error ({element}): {e}")
        for compute in b.computes:
            code = f"yamc-{compute}"
            try:
                per[code] = run_step(
                    cache_dir, task.library, code, element,
                    lambda el=element, c=compute: run_yamc_simulation(el, task, compute=c),
                    meta=yamc_meta, cache=task.caches_yamc)
            except Exception as e:  # noqa: BLE001
                print(f"  YAMC ({compute}) Error ({element}): {e}")
        if per:
            all_results[element] = per
    print(f"Completed {len(all_results)} / {len(elements)} elements.")
    return all_results, None
