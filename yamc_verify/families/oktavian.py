"""OKTAVIAN spheres: yamc's neutron and photon leakage spectra against the measurement.

The first experiment validation task: every other family compares yamc with
OpenMC, this one compares yamc with what was measured. Each item is a
single-isotope sphere around a D-T neutron source, held in its steel vessel,
from Osaka University's OKTAVIAN facility (JAERI-M 94-014, Maekawa et al.,
1994):

    Al27  1.22 g/cm3, the 40 cm Type-II vessel (SUS-304)
    Mn55  4.37 g/cm3, the 61 cm Type-I vessel (SS-41)

The Co and Nb spheres are also single isotopes but are left out: the Co
measurement reports more leaking neutrons than its physics allows, and for Nb
every library misses by the same amount, which points at the vessel model
rather than the data (fusion-neutronics/example-endf-to-transport-validation#1).

The measurement is a record like any code's (see :func:`write_measurement`),
committed under ``tasks/oktavian/sphere/measurement/`` and part of every run
of the task (:meth:`yamc_verify.project.YamcProject.fixed_records`). It holds
the measured neutron leakage per unit lethargy and photon leakage per MeV, per
source neutron, with their errors, and the experiment's own source neutron
spectrum and the tritium target's gamma rays, so a yamc run reads everything
it needs from the record and downloads nothing.

Everything comes from the FENDL-2 benchmark sublibrary on the NEA mirror,
checked against sha256 pins:

* Geometry: the vessel descriptions and Table 4.1 of n-leak/report.txt. For
  the Al sphere they match the authors' own MCNP deck (Fig. 4.21) surface for
  surface. The Type-I vessel has a 5.1 cm re-entrant hole to the centre whose
  tube thickness the report does not give, so the hole is modelled as empty.
  The sample masses in Table 4.1 agree with these geometries to within 3%.
* Source neutrons: for Al the SI1/SP1 histogram of the MCNP deck, which
  resolves the D-T peak more finely; for Mn its Table 4.2.
* Target gammas: Table 3.1 of g-leak/report.txt, the same for every sphere.
  The tritium target emits 0.0862 gammas per source neutron itself; they are
  transported in a separate photon-only run and added to the photon leakage.
* Measured leakage: n-leak/<sphere>.dat and g-leak/<sphere>.dat.

The files are scanned report pages. The parsers undo the OCR substitutions
they can undo safely (O for 0, l for 1, a stray ~ or comma) and repair lost
exponent signs only where the bin edges must join up. A value that cannot be
read safely is left out of the comparison (``None`` in the record) and
reported, never guessed.

Known differences from the experiment, kept rather than tuned away:

* Gamma gate: the photon measurement counted photons arriving within about
  65 ns of the source pulse. yamc runs to steady state, so slow capture gammas
  are included.
* Mn beam hole: modelled empty (above).
* Sample masses: within 3% of the report's (above).

Leakage is the ``current-above`` surface current on the vessel's outer sphere,
which is also the vacuum boundary, so every particle crossing it leaves and
none comes back. It agrees with the void-shell track-length estimate the
example repository used to within 0.015% in every bin holding more than 0.1%
of the leakage, on both spheres, at 10 million histories.

yamc's GPU transport has no surface or photon tallies yet, so the task runs on
the CPU only. The comparison is not graded: the site shows C/E per bin,
chi2/N with experimental and Monte Carlo errors in quadrature, the integral
C/E and the rms of ln(C/E), and the release check leaves it out.

To (re)write the measurement records, which downloads the report files once
into ``~/.cache/yamc-verification/oktavian``::

    python -m yamc_verify.families.oktavian
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from pathlib import Path

from vv import progress
from vv.records import SCHEMA, encode

from ..runstep import run_step
from ..tasks import Task

# --- the experiment ----------------------------------------------------------

# The spheres, by sample nuclide. To add one, take its density from Table 4.1,
# its vessel type and source table from n-leak/report.txt, and pin its two
# .dat files.
SPHERES = {
    "Al27": {"element": "Al", "density": 1.223, "vessel": "Type-II", "steel": "SUS-304",
             "source": "MCNP deck",
             "neutrons": ("n-leak/al40.dat", "3fdefe48a9a45157b7b1109fd387e7b1841460cadc77ca4f13022c56007e4aaf"),
             "photons": ("g-leak/al40.dat", "bfba2b0b0e164f04328165dbcbaa856d23095230cea850ca881541eb3288437f")},
    "Mn55": {"element": "Mn", "density": 4.37, "vessel": "Type-I", "steel": "SS-41",
             "source": "Table 4.2 Neutron source spectrum for LiF, Mn, Cu, Mo and W.",
             "neutrons": ("n-leak/mn60.dat", "51aaf22e1a735248e5e2156b4df025ffd39498fb5a0cc55d83995b80104f5464"),
             "photons": ("g-leak/mn60.dat", "09d2dd28dfd0936a08dce6f15e4ec526222f118d88069be87999a11e0fd293fb")},
}

# Vessel geometry (cm). Type-II is the MCNP deck of Fig. 4.21: a 20 cm central
# void, 0.2 cm walls, and an 11 cm duct (a tube of radius 5.55 to 5.75) along
# +x from x = 8.32. Type-I (60 cm inside, 0.5 cm walls) has no central void,
# only a 5.1 cm hole along +x to the centre.
VESSELS = {
    "Type-II": {"void": 10.0, "inner": 10.2, "outer": 19.75, "vessel": 19.95,
                "bore": 5.55, "tube": 5.75, "duct_start": 8.32},
    "Type-I": {"void": None, "inner": None, "outer": 30.0, "vessel": 30.5,
               "bore": 2.55, "tube": 2.55, "duct_start": 0.0},
}

# Vessel steels: mass fractions and density. SUS-304 is card M2 of the MCNP
# deck. SS-41 (JIS general structural steel) is taken as iron.
STEELS = {
    "SUS-304": ({"Cr": 0.185, "Fe": 0.704, "Ni": 0.111}, 7.824),
    "SS-41": ({"Fe": 1.0}, 7.85),
}

REPORT = "JAERI-M 94-014"
SOURCE_URL = ("https://www.oecd-nea.org/dbforms/data/eva/evatapes/fendl_20/"
              "validation/benchmarks/jaerim94014/oktavian/")
REPORTS = {
    "n-leak/report.txt": "39c07f467e6208532cc96f151160be9df94b6e6ea3fb3c7f68725e867e963b91",
    "g-leak/report.txt": "30969b2ddcf928e7b37ef48487545ff1f27063c0edc512f311b909f863bd24c0",
}
TARGET_PHOTONS_PER_NEUTRON = 0.0862  # Table 3.1, the same target for every sphere

MEASUREMENT = "measurement"
"""The name the measurement is filed under, beside the codes' names."""


def _download_dir() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "yamc-verification" / "oktavian"


def download(name: str, sha256: str) -> str:
    """The text of one report file, downloaded once and checked against its pin."""
    path = _download_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(SOURCE_URL + name, headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(urllib.request.urlopen(request).read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"{path} does not match its sha256 pin; delete it and run again")
    return path.read_text().replace("\r", "")


# --- reading the scanned report ----------------------------------------------


def number(token: str) -> float:
    """A number from the scan, with the substitutions that are safe to undo."""
    return float(token.replace("O", "0").replace("l", "1").replace("~", "")
                 .replace(",", ".").replace("Ee", "E+"))


def flips(token: str) -> list[float]:
    """The value as read, and with its exponent sign flipped either way."""
    value = number(token)
    match = re.search(r"E[-+]?(\d+)$", token)
    k = int(match.group(1)) if match else 0
    return [value] if k == 0 else [value, value * 10 ** (2 * k), value / 10 ** (2 * k)]


def read_bins(text: str, name: str, need_error: bool = True, notes: list | None = None):
    """The (upper, lower, value, error) rows of a scanned spectrum in MeV, descending.

    Edges are repaired only where they must join up: each bin's upper edge is
    the previous bin's lower edge, so a lost exponent sign is put back and a
    missing row is an error, not a silently widened bin. A value (or, with
    ``need_error``, an error) that cannot be read is ``None``, and that bin is
    left out of the comparison. The files mark an empty bin with 1E-38 or 0.
    What was left out is appended to ``notes``.
    """
    rows = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 3 or not re.match(r"[~\d.]", tokens[0]):
            continue
        try:
            first, second = number(tokens[0]), number(tokens[1])
        except ValueError:
            continue
        # Some files list the lower edge first and some the upper.
        up_tok, lo_tok = (tokens[0], tokens[1]) if first > second else (tokens[1], tokens[0])
        rows.append((up_tok, lo_tok, tokens[2:]))
    if len(rows) > 1 and number(rows[0][0]) < number(rows[1][0]):
        rows.reverse()  # ascending in the file
    out, unreadable, empty = [], [], []
    for up_tok, lo_tok, rest in rows:
        if out:
            upper = out[-1][1]
            printed = min(flips(up_tok), key=lambda c: abs(math.log(c / upper)))
            if not math.isclose(printed, upper, rel_tol=2e-3):
                raise ValueError(f"{name}: the bin below {upper:.5g} MeV starts at {up_tok}: "
                                 "a row is missing or misread")
        else:
            upper = number(up_tok)
        lower = min(flips(lo_tok), key=lambda c: abs(math.log(c * 1.08 / upper)))
        values = []
        for token in (rest + [None, None])[:2]:
            try:
                values.append(None if token is None else number(token))
            except ValueError:
                values.append(None)
        value, error = values
        where = f"{lower:.4g}-{upper:.4g} MeV"
        if value is None or (need_error and error is None):
            unreadable.append(f"{where} ({' '.join(rest)})")
            value = None
        elif value < 1e-30:
            empty.append(where)
            value = None if need_error else 0.0
        out.append((upper, lower, value, error))
    if notes is not None:
        if unreadable:
            notes.append(f"{name}: {len(unreadable)} bins unreadable in the scan, left out: "
                         + "; ".join(unreadable))
        if empty and need_error:
            notes.append(f"{name}: {len(empty)} bins have no measurement (1E-38 or 0), left out: "
                         + ", ".join(empty))
    return out


def source_spectrum(sphere: dict) -> tuple[list[float], list[float]]:
    """Ascending source bin edges (eV) and the fraction of neutrons in each."""
    text = download("n-leak/report.txt", REPORTS["n-leak/report.txt"])
    if sphere["source"] == "MCNP deck":
        # The SI1/SP1 histogram of Fig. 4.21. One SP1 entry, 8.581E-03 at
        # 0.50-0.57 MeV, is ten times its neighbours and makes the
        # probabilities sum to 1.007; read as 8.581E-04 they sum to 0.9993, so
        # that is taken as a digit lost in the scan.
        def values(s):
            return [number(t) for t in s.split()]
        edges = values(text.split("SI1")[1].split("C    *** SOURCE")[0])
        probs = values(text.split("SP1")[1].split("C     *****")[0])[1:]
        probs = [8.581e-4 if p == 8.581e-3 else p for p in probs]
        return [e * 1e6 for e in edges], [p / sum(probs) for p in probs]
    # A table of lethargy flux: each bin's share is the flux times ln(up/low).
    body = re.split(r"\nTable 4\.\d|\nFig", text.split(sphere["source"])[-1])[0]
    bins = read_bins(body, sphere["source"], need_error=False)[::-1]
    if any(v is None for _, _, v, _ in bins):
        raise ValueError(f"{sphere['source']}: a source bin is unreadable in the scan")
    probs = [v * math.log(up / lo) for up, lo, v, _ in bins]
    edges = [bins[0][1]] + [up for up, _, _, _ in bins]
    return [e * 1e6 for e in edges], [p / sum(probs) for p in probs]


def target_photons() -> tuple[list[float], list[float]]:
    """Table 3.1: the target's own gammas, 0.1 MeV groups given by upper edge (eV)."""
    text = download("g-leak/report.txt", REPORTS["g-leak/report.txt"])
    body = text.split("Table 3.1")[-1].split("Total number")[0]
    table = []
    for line in body.splitlines():
        tokens = line.split()
        if len(tokens) == 2:
            try:
                table.append((number(tokens[0]), number(tokens[1])))
            except ValueError:
                pass
    edges = [(table[0][0] - 0.1) * 1e6] + [upper * 1e6 for upper, _ in table]
    return edges, [p / sum(p for _, p in table) for _, p in table]


def measured(sphere: dict, particle: str, notes: list | None = None):
    """Ascending edges (eV), values and errors; ``None`` marks an unusable bin."""
    name, sha256 = sphere[f"{particle}s"]
    bins = read_bins(download(name, sha256), name, notes=notes)[::-1]
    edges = [bins[0][1] * 1e6] + [up * 1e6 for up, _, _, _ in bins]
    return edges, [b[2] for b in bins], [b[3] for b in bins]


# --- the measurement record --------------------------------------------------


def measurement_dir(task: Task, root: Path | str) -> Path:
    """Where the task's measurement records are committed."""
    return Path(root) / os.path.dirname(task.notebook) / MEASUREMENT


def measurement_path(task: Task, item: str, root: Path | str) -> Path:
    return measurement_dir(task, root) / f"{item}.json"


def measurement(item: str) -> tuple[dict, dict, list[str]]:
    """The measured results of one sphere, the record's identity, and what was left out."""
    sphere = SPHERES[item]
    notes: list[str] = []
    n_edges, n_value, n_error = measured(sphere, "neutron", notes)
    p_edges, p_value, p_error = measured(sphere, "photon", notes)
    source_edges, source_probs = source_spectrum(sphere)
    target_edges, target_probs = target_photons()
    files = {name: sha for name, sha in (sphere["neutrons"], sphere["photons"], *REPORTS.items())}
    digest = hashlib.sha256("".join(f"{k}={v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    identity = {"name": MEASUREMENT, "version": REPORT, "commit": None, "dirty": None, "binary_sha": None,
                "backend": None, "build_id": f"{REPORT.replace(' ', '-')}+sha{digest[:12]}",
                "source": {"url": SOURCE_URL, "files": files}}
    results = {
        "n_energy_bins": n_edges, "neutron_leakage": n_value, "neutron_leakage_error": n_error,
        "p_energy_bins": p_edges, "photon_leakage": p_value, "photon_leakage_error": p_error,
        "units": {"neutron": "leakage per unit lethargy per source neutron",
                  "photon": "leakage per MeV per source neutron"},
        "source_neutrons": {"energy_bins": source_edges, "probabilities": source_probs},
        "target_photons": {"energy_bins": target_edges, "probabilities": target_probs,
                           "per_source_neutron": TARGET_PHOTONS_PER_NEUTRON},
        "left_out": notes,
    }
    return results, identity, notes


def write_measurement(task: Task, item: str, root: Path | str) -> Path:
    """Write the measurement record of one sphere; return its path.

    The same shape as a code's record (:mod:`vv.records`), with no library: a
    measurement is the same whatever data a code runs with.
    """
    results, identity, _ = measurement(item)
    sphere = SPHERES[item]
    settings = {"sphere": item, "vessel": sphere["vessel"], "steel": sphere["steel"],
                "density": sphere["density"], "report": REPORT}
    from vv.records import settings_hash

    record = {"schema": SCHEMA, "library": None, "code": MEASUREMENT, "item": item,
              "code_identity": identity, "settings": settings, "settings_hash": settings_hash(settings),
              "run": {"date": None, "host": None, "elapsed_seconds": None},
              "results": encode(results)}
    path = measurement_path(task, item, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    return path


def load_measurement(task: Task, item: str, root: Path | str) -> dict:
    """The measured results of one sphere, from its committed record."""
    path = measurement_path(task, item, root)
    if not path.is_file():
        raise FileNotFoundError(f"no measurement record {path}: write it with "
                                "python -m yamc_verify.families.oktavian")
    return json.loads(path.read_text())["results"]


# --- the model -----------------------------------------------------------------


def build_model(item: str, library: str, experiment: dict, sample_data: str | None = None):
    """The sample in its vessel, with the neutron and photon leakage tallies.

    Returns ``(geometry, neutron_tally, photon_tally)``. The vessel's outer
    sphere is the vacuum boundary and the tallies are the outward current
    through it, binned as the measurement is. Every nuclear data set comes
    from ``library``, except the sample's neutron data when ``sample_data`` (a
    library keyword or an Arrow path) is given.
    """
    import yamc

    sphere = SPHERES[item]
    v = VESSELS[sphere["vessel"]]
    sample = yamc.Material({item: 1.0}, sphere["density"], temperature=294, name=item)
    sample.read_nuclear_data({item: sample_data or library}, photon_data={sphere["element"]: library})
    fractions, density = STEELS[sphere["steel"]]
    steel = yamc.Material(fractions, density, fraction_type="mass", temperature=294, name=sphere["steel"])
    natural = yamc.data.element_nuclides()
    steel.read_nuclear_data({n: library for e in fractions for n in natural[e]},
                            photon_data={e: library for e in fractions})

    outer = yamc.Sphere(radius=v["outer"])
    vessel = yamc.Sphere(radius=v["vessel"], boundary="vacuum")
    bore = yamc.Cylinder("x", y0=0.0, z0=0.0, radius=v["bore"])
    tube = yamc.Cylinder("x", y0=0.0, z0=0.0, radius=v["tube"])
    plane = yamc.Plane("x", offset=v["duct_start"])
    behind, beside = plane.below, plane.above
    duct = beside & tube.below  # the duct and its tube, out through the vessel

    cavity = beside & bore.below & vessel.below
    steel_parts = vessel.below & outer.above & ~duct
    if v["tube"] > v["bore"]:
        steel_parts = steel_parts | (beside & tube.below & bore.above & vessel.below)
    if v["void"]:
        void, inner = yamc.Sphere(radius=v["void"]), yamc.Sphere(radius=v["inner"])
        cavity = cavity | (void.below & behind)
        steel_parts = steel_parts | (inner.below & void.above & behind)
        sample_region = (outer.below & inner.above & behind) | (outer.below & tube.above & beside)
    else:
        sample_region = outer.below & ~duct
    geometry = yamc.Geometry([
        yamc.Cell(cavity, name="cavity and duct"),
        yamc.Cell(steel_parts, material=steel, name="vessel"),
        yamc.Cell(sample_region, material=sample, name="sample"),
    ])
    neutron = yamc.Tally(scores=["current-above"], surfaces=vessel, particle="neutron",
                         energy_bins=experiment["n_energy_bins"])
    photon = yamc.Tally(scores=["current-above"], surfaces=vessel, particle="photon",
                        energy_bins=experiment["p_energy_bins"])
    return geometry, neutron, photon


def _widths(experiment: dict) -> tuple[list[float], list[float]]:
    """Neutron bin widths in lethargy and photon bin widths in MeV, as the measurement is given."""
    n = experiment["n_energy_bins"]
    p = experiment["p_energy_bins"]
    return ([math.log(hi / lo) for lo, hi in zip(n, n[1:])],
            [(hi - lo) / 1e6 for lo, hi in zip(p, p[1:])])


def _per_width(result, widths, scale=1.0) -> tuple[list[float], list[float]]:
    return ([scale * m / w for m, w in zip(result.mean, widths)],
            [scale * d / w for d, w in zip(result.standard_deviation, widths)])


def run_yamc(item: str, task: Task, experiment: dict, compute: str = "cpu", sample_data: str | None = None) -> dict:
    """yamc's leakage spectra of one sphere, in the measurement's bins and units.

    Two runs: the neutron source with secondary photons, and a photon-only run
    of the target's own gammas, scaled to gammas per source neutron and added
    to the photon leakage (and kept on its own as ``target_photon_*``).
    ``sample_data`` replaces the sample's neutron data (see :func:`build_model`).
    """
    import time

    import yamc

    n_width, p_width = _widths(experiment)
    source = experiment["source_neutrons"]
    target = experiment["target_photons"]
    t0 = time.perf_counter()
    geometry, neutron, photon = build_model(item, task.library, experiment, sample_data)
    run = yamc.Model(
        geometry=geometry, tallies=[neutron, photon], transport_secondary_photons=True, verbose=[],
        source=yamc.NeutronSource(position=(0, 0, 0),
                                  energy=yamc.sources.Histogram(source["energy_bins"], source["probabilities"])),
    ).simulate_transport(total_particles=task.particles, compute=compute)
    n_mean, n_std = _per_width(run[neutron], n_width)
    p_mean, p_std = _per_width(run[photon], p_width)

    geometry, _, target_tally = build_model(item, task.library, experiment, sample_data)
    target_run = yamc.Model(
        geometry=geometry, tallies=[target_tally], verbose=[],
        source=yamc.PhotonSource(position=(0, 0, 0),
                                 energy=yamc.sources.Histogram(target["energy_bins"], target["probabilities"])),
    ).simulate_transport(total_particles=max(task.particles // 5, 1000), compute=compute)
    t_mean, t_std = _per_width(target_run[target_tally], p_width, target["per_source_neutron"])
    seconds = time.perf_counter() - t0
    return {
        "n_energy_bins": list(experiment["n_energy_bins"]),
        "neutron_leakage_mean": n_mean, "neutron_leakage_std": n_std,
        "p_energy_bins": list(experiment["p_energy_bins"]),
        "photon_leakage_mean": [a + b for a, b in zip(p_mean, t_mean)],
        "photon_leakage_std": [math.hypot(a, b) for a, b in zip(p_std, t_std)],
        "target_photon_mean": t_mean, "target_photon_std": t_std,
        "particles/s": task.particles / seconds if seconds > 0 else None,
    }


# --- comparing with the measurement --------------------------------------------


def goodness_of_fit(calc, calc_err, meas, meas_err, widths) -> dict:
    """How well a calculated spectrum matches the measured one. Not a grade.

    * ``chi2_per_bin``: the mean over bins of ((C - E) / sigma)^2, sigma the
      experimental and Monte Carlo errors in quadrature. The neutron errors are
      counting statistics only, so it is large even where C/E is close to 1.
    * ``integral_c_over_e``: total calculated over total measured leakage, over
      the bins with a measurement.
    * ``bin_deviation_percent``: the rms of ln(C/E) over the bins, as a
      percentage, so a bin 20% high and one 20% low count the same.
    * ``mc_error_max_percent``: the largest per-bin Monte Carlo relative error.
    * ``c_over_e``: per bin, ``None`` where there is no measurement.

    Bins without a usable measurement, or with no calculated leakage, are skipped.
    """
    pairs = [(c, ce, e, ee) for c, ce, e, ee in zip(calc, calc_err, meas, meas_err)
             if e is not None and ee is not None and c > 0 and e > 0]
    measured_bins = [(c, e, w) for c, e, w in zip(calc, meas, widths) if e is not None]
    if not pairs:
        return {"bins": 0, "c_over_e": [None] * len(calc)}
    return {
        "chi2_per_bin": sum((c - e) ** 2 / (ce ** 2 + ee ** 2) for c, ce, e, ee in pairs) / len(pairs),
        "integral_c_over_e": (sum(c * w for c, _, w in measured_bins)
                              / sum(e * w for _, e, w in measured_bins)),
        "bin_deviation_percent": 100 * math.sqrt(sum(math.log(c / e) ** 2 for c, _, e, _ in pairs) / len(pairs)),
        "mc_error_max_percent": 100 * max(ce / c for c, ce, _, _ in pairs),
        "bins": len(pairs),
        "c_over_e": [c / e if e else None for c, e in zip(calc, meas)],
    }


def compare(experiment: dict, results: dict) -> dict:
    """The neutron and photon goodness of fit of one calculation against the measurement."""
    n_width, p_width = _widths(experiment)
    out = {}
    for particle, width in (("neutron", n_width), ("photon", p_width)):
        out[particle] = goodness_of_fit(
            list(results[f"{particle}_leakage_mean"]), list(results[f"{particle}_leakage_std"]),
            experiment[f"{particle}_leakage"], experiment[f"{particle}_leakage_error"], width)
    return out


# --- the run loop ----------------------------------------------------------------


def discover(task: Task, discover_with_openmc: bool = False) -> list[str]:
    """The spheres, narrowed to those ``YAMC_VERIFY_NUCLIDES`` names when it is set.

    Unlike the other nuclide models, naming no sphere there runs none rather
    than failing, since a list of nuclides for the code comparisons (Fe56,
    say) need not name an OKTAVIAN sample. The count cap does not apply: there
    are two spheres, both stable isotopes.
    """
    from cache_utils import requested_items

    requested = requested_items("nuclide")
    items = [i for i in sorted(SPHERES) if not requested or i in requested]
    progress.report_plan(items, kind="nuclide")
    return items


def run_all(task: Task, backends=None, cache_dir: str = "cache"):
    """Run yamc on every sphere and cache its leakage spectra. Returns ``(all_results, None)``."""
    from ..backends import resolve

    b = backends if backends is not None else resolve()
    root = Path(__file__).resolve().parents[2]
    items = discover(task)
    print(f"Running the OKTAVIAN spheres: {items}")
    all_results = {}
    for item in items:
        experiment = load_measurement(task, item, root)
        settings = {"library": task.library, "particles": task.particles,
                    "measurement": json.loads(measurement_path(task, item, root).read_text())
                    ["code_identity"]["build_id"]}
        per = {}
        for compute in b.computes:
            if compute != "cpu":
                print(f"  {item}: yamc-{compute} has no surface or photon tallies; the task runs on the CPU only")
                continue
            try:
                per[f"yamc-{compute}"] = run_step(
                    cache_dir, task.library, f"yamc-{compute}", item,
                    lambda i=item, c=compute: run_yamc(i, task, experiment, compute=c),
                    settings=settings, cache=task.caches_yamc)
            except Exception as e:  # noqa: BLE001 - isolate one sphere's failure
                print(f"  yamc ({compute}) error ({item}): {e}")
        if per:
            all_results[item] = per
    print(f"Completed {len(all_results)} / {len(items)} spheres.")
    return all_results, None


def display_results(task: Task, all_results: dict) -> None:
    """Print each sphere's comparison with the measurement (the page's reproducer calls this)."""
    root = Path(__file__).resolve().parents[2]
    print(f"{'':16s} {'':8s} {'chi2/N':>9s} {'C/E':>7s} {'rms ln(C/E)':>12s} {'max MC err':>11s}")
    for item, per in all_results.items():
        experiment = load_measurement(task, item, root)
        for code, results in per.items():
            for particle, fit in compare(experiment, results).items():
                if not fit["bins"]:
                    print(f"{item + ' ' + code:16s} {particle:8s} no measured bins")
                    continue
                print(f"{item + ' ' + code:16s} {particle:8s} {fit['chi2_per_bin']:9.3g} "
                      f"{fit['integral_c_over_e']:7.3f} {fit['bin_deviation_percent']:11.1f}% "
                      f"{fit['mc_error_max_percent']:10.1f}%")


def main(argv=None) -> int:
    """Write the measurement record of every sphere."""
    import argparse

    from .. import tasks

    ap = argparse.ArgumentParser(prog="python -m yamc_verify.families.oktavian",
                                 description="Write the OKTAVIAN measurement records from the pinned report files.")
    ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    task = next(t for t in tasks.all_tasks() if t.family == "oktavian")
    for item in sorted(SPHERES):
        path = write_measurement(task, item, root)
        _, _, notes = measurement(item)
        print(f"wrote {path.relative_to(root)}")
        for note in notes:
            print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
