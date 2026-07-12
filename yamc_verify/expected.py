"""Physics-derived expected-difference signatures for residual grading (#59).

Each helper turns a known, code-independent physics fact (a characteristic
fluorescence line energy, a daughter-gamma line, a reduced-chain membership) into
a per-bin ``{bin_index -> "over" | "under"}`` spec that
``plotting.calculate_spectral_agreement`` uses to remove ONLY that expected
directional difference from the residual grade. The signatures never look at the
observed YAMC-vs-OpenMC discrepancy, which is what keeps residual grading from
masking a real difference.

Fluorescence line energies are precomputed from the photon atomic-relaxation data
into ``data/fluorescence_lines.json`` so the index build has no runtime dependency
on the nuclear-data files (regenerate with ``python -m yamc_verify.expected``).
"""
from __future__ import annotations

import bisect
import functools
import json
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "data"
_FLUOR_PATH = _DATA / "fluorescence_lines.json"
_DECAY_PATH = _DATA / "decay_gamma_lines.json"

# Ignore lines below this energy: they sit at or under the photon transport cutoff
# (~1 keV) and below the low edge of the photon tally group structures, so they
# never land in a graded bin anyway.
_MIN_LINE_EV = 1_000.0


@functools.lru_cache(maxsize=1)
def _fluorescence_lines() -> dict[str, list[float]]:
    """Element -> sorted characteristic fluorescence line energies (eV)."""
    try:
        with _FLUOR_PATH.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _bin_index(edges: list[float], energy: float) -> int | None:
    """Index of the bin ``[edges[i], edges[i+1])`` containing ``energy``, or None."""
    if len(edges) < 2 or energy < edges[0] or energy >= edges[-1]:
        return None
    i = bisect.bisect_right(edges, energy) - 1
    return i if 0 <= i < len(edges) - 1 else None


def fluorescence_line_bins(element: str, energy_bins) -> set[int]:
    """Bin indices of ``energy_bins`` that contain a characteristic fluorescence line.

    ``energy_bins`` is the tally group-structure edge array (length n_bins + 1),
    matching the spectrum arrays. Returns an empty set when the element has no
    tabulated lines (e.g. very low Z) or the data file is absent (CI), in which
    case the comparison simply grades on its full spectrum -- conservative, never
    masking.
    """
    lines = _fluorescence_lines().get(element)
    if not lines or energy_bins is None or len(energy_bins) < 2:
        return set()
    edges = [float(e) for e in energy_bins]
    bins = set()
    for e in lines:
        idx = _bin_index(edges, e)
        if idx is not None:
            bins.add(idx)
    return bins


def photon_expected_spec(element: str, energy_bins) -> dict[int, str]:
    """Expected-difference spec for a photon / secondary-photon flux spectrum.

    YAMC emits the real Compton-ionization fluorescence lines OpenMC drops on
    ENDF/B-VIII.1, so it EXCEEDS OpenMC in the characteristic-line bins: direction
    ``"over"``. Only that one-sided excess is removed from the residual grade; a
    shortfall in a line bin, or any difference in a non-line bin, still counts.
    """
    return {i: "over" for i in fluorescence_line_bins(element, energy_bins)}


@functools.lru_cache(maxsize=1)
def _decay_gamma_lines() -> dict[str, list[float]]:
    """Radionuclide -> sorted discrete decay gamma line energies (eV)."""
    try:
        with _DECAY_PATH.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def decay_gamma_bins(radionuclides, energy_bins) -> set[int]:
    """Bin indices containing a decay gamma line of any of ``radionuclides``."""
    table = _decay_gamma_lines()
    if not table or energy_bins is None or len(energy_bins) < 2:
        return set()
    edges = [float(e) for e in energy_bins]
    bins = set()
    for rn in radionuclides or []:
        for e in table.get(rn, ()):  # unknown nuclide -> no lines -> nothing excluded
            idx = _bin_index(edges, e)
            if idx is not None:
                bins.add(idx)
    return bins


def decay_expected_spec(radionuclides, energy_bins) -> dict[int, str]:
    """Expected-difference spec for a decay-photon (D1S) step spectrum.

    YAMC's D1S follows the activation daughters and emits their decay gammas;
    OpenMC's D1S attributes gammas only to the directly activated nuclide, so YAMC
    EXCEEDS OpenMC at the decay gamma lines: direction ``"over"``. A shortfall at a
    line, or any difference away from a known decay gamma, still counts.
    """
    return {i: "over" for i in decay_gamma_bins(radionuclides, energy_bins)}


def generate_decay_gamma_lines(chain_file: str | Path, out_path: Path = _DECAY_PATH) -> dict:
    """Read an OpenMC depletion chain and write nuclide -> decay gamma line energies.

    Only discrete (line) photon emissions are kept; continuous decay-photon
    distributions are not line signatures and are left to grade normally.
    """
    import openmc.deplete

    chain = openmc.deplete.Chain.from_xml(str(chain_file))
    out: dict[str, list[float]] = {}
    for nuc in chain.nuclides:
        # Only the photon emission spectrum carries gamma line energies; the alpha
        # and electron sources carry particle energies, not photon lines.
        dist = (getattr(nuc, "sources", None) or {}).get("photon")
        xs = getattr(dist, "x", None)
        if xs is None:
            continue
        energies = {round(float(e), 1) for e in np.atleast_1d(xs) if float(e) >= _MIN_LINE_EV}
        if energies:
            out[nuc.name] = sorted(energies)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, separators=(",", ":"), sort_keys=True)
    return out


# --------------------------------------------------------------------------
# Generator: rebuild data/fluorescence_lines.json from photon atomic-relaxation
# data. Run as: python -m yamc_verify.expected [photon_dir]
# --------------------------------------------------------------------------
def generate_fluorescence_lines(photon_dir: str | Path, out_path: Path = _FLUOR_PATH) -> dict:
    """Read every ``<Element>.h5`` photon file and write element -> line energies.

    A radiative (fluorescence) transition is one with no tertiary subshell; its
    ``energy (eV)`` is the emitted photon energy. Auger transitions (tertiary set)
    emit an electron, not a photon, and are excluded.
    """
    import openmc.data
    import pandas as pd

    photon_dir = Path(photon_dir)
    out: dict[str, list[float]] = {}
    for h5 in sorted(photon_dir.glob("*.h5")):
        element = h5.stem
        try:
            ar = openmc.data.IncidentPhoton.from_hdf5(h5).atomic_relaxation
        except Exception:
            continue
        if ar is None:
            continue
        energies: set[float] = set()
        for ss in ar.subshells:
            trans = ar.transitions.get(ss)
            if trans is None or len(trans) == 0:
                continue
            for _, row in trans.iterrows():
                # Radiative (fluorescence) transitions have no tertiary subshell
                # (a photon is emitted); Auger transitions set tertiary (an
                # electron is ejected, no photon). pd.isna handles None/NaN/NA.
                if not pd.isna(row.get("tertiary")):
                    continue
                e = float(row["energy (eV)"])
                if e >= _MIN_LINE_EV:
                    energies.add(round(e, 1))
        if energies:
            out[element] = sorted(energies)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, separators=(",", ":"), sort_keys=True)
    return out


def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Rebuild fluorescence_lines.json from photon data.")
    ap.add_argument(
        "photon_dir",
        nargs="?",
        default="/home/jon/nuclear_data/endf-b8.1-hdf5/endfb-viii.1-hdf5/photon",
        help="directory of <Element>.h5 photon files",
    )
    args = ap.parse_args(argv)
    out = generate_fluorescence_lines(args.photon_dir)
    print(f"wrote {_FLUOR_PATH} ({len(out)} elements)")


if __name__ == "__main__":
    _main()
