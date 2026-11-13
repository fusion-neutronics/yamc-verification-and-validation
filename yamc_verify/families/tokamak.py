"""Neutron tokamak verification geometry (family ``neutron``, geometry ``tokamak``).

A fixed multi-material tokamak model (the geometry of yamc's
``examples/python/simple_tokamak.py``) run as a neutron-transport benchmark: one
simulation per code tallies the neutron flux spectrum (VITAMIN-J-175) in each
named component cell, and every cell is cached as its own item so the benchmark
database shows one row per cell with a YAMC-vs-OpenMC spectrum comparison.

Unlike the sphere/broomstick neutron family (one single-nuclide sphere per
item), this is one fixed geometry with realistic per-component fusion materials.
The geometry is defined once as plain numbers (:data:`SPEC`) and built
identically in yamc and OpenMC so the comparison is like-for-like.

Only the flux-seeing component cells are compared (:data:`TALLY_CELLS`); the
bioshield / air / outer-void cells receive essentially no neutrons under analog
transport at these statistics, so their spectra would be pure noise. The
whole-geometry flux **mesh** comparison is a planned follow-up.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..runstep import run_step
from ..tasks import Task

# --------------------------------------------------------------------------
# Geometry specification (cm) -- the numbers from simple_tokamak.py, shared by
# both builders so yamc and OpenMC get an identical geometry.
# --------------------------------------------------------------------------
SPEC = {
    "central_column_radius": 100.0,
    "inner_sphere": 480.0,
    "middle_sphere": 500.0,
    "outer_sphere": 600.0,
    "floor_top": -650.0,
    "floor_bot": -950.0,
    "wall_in": 1675.0,      # room interior half-width (x and y)
    "wall_out": 1875.0,     # bioshield outer half-width
    "ceiling_bot": 1175.0,
    "ceiling_top": 1325.0,
    "bound_xy": 1975.0,     # vacuum bounding box half-width (x and y)
    "bound_zmin": -1050.0,
    "bound_zmax": 1425.0,
    # PF coils: (r_center, z_center, r_width, z_height)
    "pf_coils": [
        (150.0, 650.0, 50.0, 50.0),    # pf_coil_1 -- upper inboard
        (150.0, -620.0, 50.0, 50.0),   # pf_coil_2 -- lower inboard
        (650.0, 350.0, 50.0, 50.0),    # pf_coil_3 -- upper outboard
        (650.0, -350.0, 50.0, 50.0),   # pf_coil_4 -- lower outboard
    ],
    "source_ring_radius": 300.0,
}

# Component cells whose neutron flux spectrum is tallied and compared. The
# bioshield/air/void cells see ~no flux under analog transport, so they are
# built (to keep the geometry identical) but not compared.
TALLY_CELLS = [
    "central_column", "firstwall", "blanket", "inner_vessel",
    "pf_coil_1", "pf_coil_2", "pf_coil_3", "pf_coil_4",
]

GROUP_STRUCTURE = "VITAMIN-J-175"

# --------------------------------------------------------------------------
# Materials (atom fractions) -- ITER/DEMO-like, mapped onto the components the
# geometry provides. Compositions are shared so both codes build them the same.
# --------------------------------------------------------------------------
# Li4SiO4 breeder, 60% Li-6 enriched: Li4 (Li6 2.4 / Li7 1.6) Si1 O4.
_MATERIALS = {
    "tungsten": ({"W": 1.0}, 19.3),                       # first-wall armor
    "breeder": ({"Li6": 2.4, "Li7": 1.6, "Si": 1.0, "O": 4.0}, 2.4),  # Li4SiO4 (Li-6 enriched)
    "eurofer": ({"Fe": 0.885, "Cr": 0.094, "W": 0.003, "Mn": 0.004, "C": 0.014}, 7.8),  # RAFM steel
    "magnet": ({"Cu": 0.6, "Nb": 0.3, "Sn": 0.1}, 8.9),   # Cu + Nb3Sn
    # Concrete slimmed to light constituents (Ca/Fe dropped -- see simple_tokamak.py).
    "concrete": ({"O": 0.532, "Si": 0.337, "Al": 0.034, "H": 0.023, "Na": 0.016}, 2.3),
    "air": ({"N": 0.7847, "O": 0.2105}, 0.001225),
}

# cell name -> material name (cells absent here are true void).
_CELL_MATERIAL = {
    "central_column": "eurofer",
    "firstwall": "tungsten",
    "blanket": "breeder",
    "floor": "concrete",
    "bioshield_wall": "concrete",
    "bioshield_ceiling": "concrete",
    "pf_coil_1": "magnet", "pf_coil_2": "magnet",
    "pf_coil_3": "magnet", "pf_coil_4": "magnet",
    "air_a": "air", "air_b": "air", "air_c": "air", "air_d": "air", "air_e": "air",
    "air_f": "air", "air_g": "air", "air_h": "air", "air_i": "air",
    # inner_vessel, void_gap_* are true void (no material).
}


def _is_nuclide(name: str) -> bool:
    return any(ch.isdigit() for ch in name)


# --------------------------------------------------------------------------
# yamc geometry
# --------------------------------------------------------------------------
def build_yamc(task: Task):
    """Build the yamc (geometry, {cell_name: cell}) for the tokamak."""
    import yamc as mc

    s = SPEC
    mats = {}
    for mname, (comp, dens) in _MATERIALS.items():
        m = mc.Material(composition=dict(comp), density=dens, temperature=294, name=mname)
        m.read_nuclear_data(task.library)
        mats[mname] = m

    cc = mc.Cylinder(axis="z", radius=s["central_column_radius"])
    inner = mc.Sphere(radius=s["inner_sphere"])
    middle = mc.Sphere(radius=s["middle_sphere"])
    outer = mc.Sphere(radius=s["outer_sphere"])
    floor_top = mc.Plane(axis="z", offset=s["floor_top"])
    floor_bot = mc.Plane(axis="z", offset=s["floor_bot"])
    win = s["wall_in"]
    wout = s["wall_out"]
    wall_in_xmin = mc.Plane(axis="x", offset=-win)
    wall_in_xmax = mc.Plane(axis="x", offset=win)
    wall_in_ymin = mc.Plane(axis="y", offset=-win)
    wall_in_ymax = mc.Plane(axis="y", offset=win)
    wall_out_xmin = mc.Plane(axis="x", offset=-wout)
    wall_out_xmax = mc.Plane(axis="x", offset=wout)
    wall_out_ymin = mc.Plane(axis="y", offset=-wout)
    wall_out_ymax = mc.Plane(axis="y", offset=wout)
    ceiling_bot = mc.Plane(axis="z", offset=s["ceiling_bot"])
    ceiling_top = mc.Plane(axis="z", offset=s["ceiling_top"])
    bxy = s["bound_xy"]
    bound_xmin = mc.Plane(axis="x", offset=-bxy, boundary="vacuum")
    bound_xmax = mc.Plane(axis="x", offset=bxy, boundary="vacuum")
    bound_ymin = mc.Plane(axis="y", offset=-bxy, boundary="vacuum")
    bound_ymax = mc.Plane(axis="y", offset=bxy, boundary="vacuum")
    bound_zmin = mc.Plane(axis="z", offset=s["bound_zmin"], boundary="vacuum")
    bound_zmax = mc.Plane(axis="z", offset=s["bound_zmax"], boundary="vacuum")

    pf = []
    for rc, zc, rw, zh in s["pf_coils"]:
        pf.append((
            mc.Cylinder(axis="z", radius=rc - rw / 2),
            mc.Cylinder(axis="z", radius=rc + rw / 2),
            mc.Plane(axis="z", offset=zc + zh / 2),
            mc.Plane(axis="z", offset=zc - zh / 2),
        ))

    room_xy = wall_in_xmin.above & wall_in_xmax.below & wall_in_ymin.above & wall_in_ymax.below
    shield_xy = wall_out_xmin.above & wall_out_xmax.below & wall_out_ymin.above & wall_out_ymax.below
    outer_xy = bound_xmin.above & bound_xmax.below & bound_ymin.above & bound_ymax.below
    outside_room = wall_in_xmin.below | wall_in_xmax.above | wall_in_ymin.below | wall_in_ymax.above
    outside_shield = wall_out_xmin.below | wall_out_xmax.above | wall_out_ymin.below | wall_out_ymax.above

    regions = {
        "central_column": cc.below & outer.below,
        "firstwall": middle.below & inner.above & cc.above,
        "blanket": middle.above & outer.below & cc.above,
        "inner_vessel": cc.above & inner.below,
        "floor": shield_xy & floor_top.below & floor_bot.above,
        "bioshield_wall": shield_xy & outside_room & ceiling_bot.below & floor_top.above,
        "bioshield_ceiling": shield_xy & ceiling_top.below & ceiling_bot.above,
        "void_gap_side": outer_xy & outside_shield & ceiling_top.below & floor_bot.above,
        "void_gap_top": outer_xy & bound_zmax.below & ceiling_top.above,
        "void_gap_bottom": outer_xy & floor_bot.below & bound_zmin.above,
    }
    (p1i, p1o, p1t, p1b), (p2i, p2o, p2t, p2b), (p3i, p3o, p3t, p3b), (p4i, p4o, p4t, p4b) = pf
    regions.update({
        "air_a": room_xy & ceiling_bot.below & p1t.above,
        "air_b": room_xy & p1t.below & p1b.above & (p1o.above | p1i.below),
        "air_c": room_xy & p1b.below & p3t.above & outer.above,
        "air_d": room_xy & p3t.below & p3b.above & outer.above & (p3o.above | p3i.below),
        "air_e": room_xy & p3b.below & p4t.above & outer.above,
        "air_f": room_xy & p4t.below & p4b.above & outer.above & (p4o.above | p4i.below),
        "air_g": room_xy & p4b.below & p2t.above & outer.above,
        "air_h": room_xy & p2t.below & p2b.above & outer.above & (p2o.above | p2i.below),
        "air_i": room_xy & p2b.below & floor_top.above,
    })
    for i, (pi, po, pt, pb) in enumerate(pf, start=1):
        regions[f"pf_coil_{i}"] = pi.above & po.below & pt.below & pb.above

    cells = {}
    for name, region in regions.items():
        mname = _CELL_MATERIAL.get(name)
        kwargs = {"region": region, "name": name}
        if mname is not None:
            kwargs["material"] = mats[mname]
        cells[name] = mc.Cell(**kwargs)
    geometry = mc.Geometry(list(cells.values()))
    return geometry, cells


def _yamc_source():
    import yamc as mc

    return mc.NeutronSource(
        energy=mc.sources.Discrete([14.06e6], [1.0]),
        direction=mc.sources.Isotropic(),
        position=mc.sources.CylindricalRing(
            radius=mc.sources.Discrete([SPEC["source_ring_radius"]], [1.0]),
            phi=mc.sources.Uniform(0.0, 2 * math.pi),
            z=mc.sources.Discrete([0.0], [1.0]),
        ),
    )


def run_tokamak_yamc(task: Task, compute: str = "cpu") -> dict:
    """Run the yamc tokamak model once; return {cell_name: results_dict}."""
    import yamc as mc

    geometry, cells = build_yamc(task)
    tallies = {}
    for name in TALLY_CELLS:
        tallies[name] = mc.Tally(
            scores=["flux"], name=f"flux_{name}", cells=cells[name],
            particle="neutron", energy_group_structure=GROUP_STRUCTURE,
        )
    energy_bins = np.array(next(iter(tallies.values())).energy_bins)

    # surface tracking: the model has large void/air regions where delta
    # tracking would churn fictitious collisions (see simple_tokamak.py).
    model = mc.Model(geometry=geometry, tallies=list(tallies.values()),
                     source=_yamc_source(), tracking_mode="surface", verbose=[])
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
            "cell": name, "code": f"yamc-{compute}", "particles/s": pps,
            "flux_spectra_mean": np.array(r.mean).squeeze(),
            "flux_spectra_std_dev": np.array(r.standard_deviation).squeeze(),
            "energy_bins": energy_bins,
        }
    return out


# --------------------------------------------------------------------------
# OpenMC geometry
# --------------------------------------------------------------------------
def build_openmc(task: Task):
    """Build the OpenMC (geometry, {cell_name: cell}) for the tokamak."""
    import openmc

    s = SPEC
    mats = {}
    mat_objs = []
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
        mat_objs.append(m)

    cc = openmc.ZCylinder(r=s["central_column_radius"])
    inner = openmc.Sphere(r=s["inner_sphere"])
    middle = openmc.Sphere(r=s["middle_sphere"])
    outer = openmc.Sphere(r=s["outer_sphere"])
    floor_top = openmc.ZPlane(z0=s["floor_top"])
    floor_bot = openmc.ZPlane(z0=s["floor_bot"])
    win, wout = s["wall_in"], s["wall_out"]
    wall_in_xmin = openmc.XPlane(x0=-win)
    wall_in_xmax = openmc.XPlane(x0=win)
    wall_in_ymin = openmc.YPlane(y0=-win)
    wall_in_ymax = openmc.YPlane(y0=win)
    wall_out_xmin = openmc.XPlane(x0=-wout)
    wall_out_xmax = openmc.XPlane(x0=wout)
    wall_out_ymin = openmc.YPlane(y0=-wout)
    wall_out_ymax = openmc.YPlane(y0=wout)
    ceiling_bot = openmc.ZPlane(z0=s["ceiling_bot"])
    ceiling_top = openmc.ZPlane(z0=s["ceiling_top"])
    bxy = s["bound_xy"]
    bound_xmin = openmc.XPlane(x0=-bxy, boundary_type="vacuum")
    bound_xmax = openmc.XPlane(x0=bxy, boundary_type="vacuum")
    bound_ymin = openmc.YPlane(y0=-bxy, boundary_type="vacuum")
    bound_ymax = openmc.YPlane(y0=bxy, boundary_type="vacuum")
    bound_zmin = openmc.ZPlane(z0=s["bound_zmin"], boundary_type="vacuum")
    bound_zmax = openmc.ZPlane(z0=s["bound_zmax"], boundary_type="vacuum")

    pf = []
    for rc, zc, rw, zh in s["pf_coils"]:
        pf.append((
            openmc.ZCylinder(r=rc - rw / 2),
            openmc.ZCylinder(r=rc + rw / 2),
            openmc.ZPlane(z0=zc + zh / 2),
            openmc.ZPlane(z0=zc - zh / 2),
        ))

    room_xy = +wall_in_xmin & -wall_in_xmax & +wall_in_ymin & -wall_in_ymax
    shield_xy = +wall_out_xmin & -wall_out_xmax & +wall_out_ymin & -wall_out_ymax
    outer_xy = +bound_xmin & -bound_xmax & +bound_ymin & -bound_ymax
    outside_room = -wall_in_xmin | +wall_in_xmax | -wall_in_ymin | +wall_in_ymax
    outside_shield = -wall_out_xmin | +wall_out_xmax | -wall_out_ymin | +wall_out_ymax

    regions = {
        "central_column": -cc & -outer,
        "firstwall": -middle & +inner & +cc,
        "blanket": +middle & -outer & +cc,
        "inner_vessel": +cc & -inner,
        "floor": shield_xy & -floor_top & +floor_bot,
        "bioshield_wall": shield_xy & outside_room & -ceiling_bot & +floor_top,
        "bioshield_ceiling": shield_xy & -ceiling_top & +ceiling_bot,
        "void_gap_side": outer_xy & outside_shield & -ceiling_top & +floor_bot,
        "void_gap_top": outer_xy & -bound_zmax & +ceiling_top,
        "void_gap_bottom": outer_xy & -floor_bot & +bound_zmin,
    }
    (p1i, p1o, p1t, p1b), (p2i, p2o, p2t, p2b), (p3i, p3o, p3t, p3b), (p4i, p4o, p4t, p4b) = pf
    regions.update({
        "air_a": room_xy & -ceiling_bot & +p1t,
        "air_b": room_xy & -p1t & +p1b & (+p1o | -p1i),
        "air_c": room_xy & -p1b & +p3t & +outer,
        "air_d": room_xy & -p3t & +p3b & +outer & (+p3o | -p3i),
        "air_e": room_xy & -p3b & +p4t & +outer,
        "air_f": room_xy & -p4t & +p4b & +outer & (+p4o | -p4i),
        "air_g": room_xy & -p4b & +p2t & +outer,
        "air_h": room_xy & -p2t & +p2b & +outer & (+p2o | -p2i),
        "air_i": room_xy & -p2b & +floor_top,
    })
    for i, (pi, po, pt, pb) in enumerate(pf, start=1):
        regions[f"pf_coil_{i}"] = +pi & -po & -pt & +pb

    cells = {}
    cid = 1
    for name, region in regions.items():
        mname = _CELL_MATERIAL.get(name)
        cell = openmc.Cell(cell_id=cid, name=name, region=region)
        if mname is not None:
            cell.fill = mats[mname]
        cells[name] = cell
        cid += 1
    geometry = openmc.Geometry(list(cells.values()))
    return geometry, cells, mat_objs


def run_tokamak_openmc(task: Task) -> dict:
    """Run the OpenMC tokamak model once; return {cell_name: results_dict}."""
    import openmc

    openmc.config["cross_sections"] = f"{task.nuclear_data_dir}cross_sections.xml"
    geometry, cells, _mats = build_openmc(task)

    energy_filter = openmc.EnergyFilter.from_group_structure(GROUP_STRUCTURE)
    ebins_2d = energy_filter.bins
    energy_bins = np.concatenate([[ebins_2d[0, 0]], ebins_2d[:, 1]])

    tallies = {}
    for name in TALLY_CELLS:
        t = openmc.Tally(name=f"flux_{name}")
        t.filters = [openmc.CellFilter(cells[name]), energy_filter]
        t.scores = ["flux"]
        tallies[name] = t

    r = openmc.stats.Discrete([SPEC["source_ring_radius"]], [1.0])
    phi = openmc.stats.Uniform(0.0, 2 * math.pi)
    z = openmc.stats.Discrete([0.0], [1.0])
    source = openmc.IndependentSource(
        space=openmc.stats.CylindricalIndependent(r=r, phi=phi, z=z, origin=(0.0, 0.0, 0.0)),
        energy=openmc.stats.Discrete([14.06e6], [1.0]),
        angle=openmc.stats.Isotropic(),
    )
    settings = openmc.Settings(
        particles=task.particles, batches=task.batches, source=source, seed=1,
        run_mode="fixed source", output={"tallies": False, "summary": False},
    )
    settings.verbosity = 1

    model = openmc.Model(geometry=geometry, settings=settings,
                         tallies=openmc.Tallies(list(tallies.values())))
    sp_filename = model.run(apply_tally_results=True, output=False)
    sp = openmc.StatePoint(sp_filename)
    total = task.particles * task.batches
    pps = total / sp.runtime["transport"]

    out = {}
    for name, tally in tallies.items():
        out[name] = {
            "cell": name, "code": "openmc", "particles/s": pps,
            "flux_spectra_mean": np.array(tally.mean).squeeze(),
            "flux_spectra_std_dev": np.array(tally.std_dev).squeeze(),
            "energy_bins": energy_bins,
        }
    return out


# --------------------------------------------------------------------------
# discovery / run
# --------------------------------------------------------------------------
def discover(task: Task, run_openmc: bool) -> list[str]:
    """The compared component cells (fixed for the tokamak geometry)."""
    return list(TALLY_CELLS)


def _metas(task: Task):
    openmc_meta = {
        "library": task.library, "particles": task.particles,
        "batches": task.batches, "source_energy": task.source_energy,
    }
    yamc_meta = dict(openmc_meta)
    return openmc_meta, yamc_meta


def run_all(task: Task, backends=None, cache_dir: str = "cache"):
    """Run the tokamak model once per code; cache one item per component cell.

    The model is expensive, so each code's run is memoized: the first uncached
    cell triggers the single simulation and the rest reuse its per-cell results.
    Returns (all_results, None) to match the other families' run_all signature.
    """
    from ..backends import resolve

    b = backends if backends is not None else resolve()
    openmc_meta, yamc_meta = _metas(task)
    cells = discover(task, b.run_openmc)
    print(f"Running tokamak neutron model for {len(cells)} cells: {cells}")

    memo: dict = {}  # code -> {cell_name: result_dict}

    def cell_fn(code, cell_name, runner):
        def _fn():
            if code not in memo:
                memo[code] = runner()
            return memo[code][cell_name]
        return _fn

    all_results: dict = {}
    for cell_name in cells:
        per = {}
        if b.run_openmc:
            try:
                per["openmc"] = run_step(
                    cache_dir, task.library, "openmc", cell_name,
                    cell_fn("openmc", cell_name, lambda: run_tokamak_openmc(task)),
                    settings=openmc_meta, cache=True)
            except Exception as e:  # noqa: BLE001
                print(f"  OpenMC Error ({cell_name}): {e}")
        for compute in b.computes:
            code = f"yamc-{compute}"
            try:
                per[code] = run_step(
                    cache_dir, task.library, code, cell_name,
                    cell_fn(code, cell_name, lambda c=compute: run_tokamak_yamc(task, compute=c)),
                    settings=yamc_meta, cache=task.caches_yamc)
            except Exception as e:  # noqa: BLE001
                print(f"  YAMC ({compute}) Error ({cell_name}): {e}")
        if per:
            all_results[cell_name] = per
    print(f"Completed {len(all_results)} / {len(cells)} cells.")
    return all_results, None
