"""Item discovery for the verification models.

Each physics family iterates over a discovered list of nuclides or elements.
The discovery logic (which was duplicated in every notebook's discovery cell)
lives here so the thin notebooks and the orchestrator's planner agree on
exactly what will run. All functions end by routing the discovered list
through :func:`cache_utils.select_requested`, which honours the
``YAMC_VERIFY_NUCLIDES`` / ``YAMC_VERIFY_ELEMENTS`` / ``YAMC_VERIFY_NUMBER_OF_SIMULATIONS``
environment variables.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

from cache_utils import select_requested

_SINGLE_NUCLIDE_RE = re.compile(r"[A-Z][a-z]?\d+")


def _arrow_cache_nuclides(library: str) -> list[str]:
    """Single-nuclide names present in the local yamc Arrow cache for ``library``."""
    yamc_cache = Path.home() / ".cache" / "yamc"
    prefix = f"{library}-"
    names = []
    for p in sorted(yamc_cache.glob(f"{prefix}*.arrow")):
        if not p.is_dir():
            continue
        name = p.name[len(prefix):-len(".arrow")]
        # Single-nuclide entries only (Ac225, Fe56, ...); drop natural-element
        # entries (Fe, U, ...) that have no mass number.
        if _SINGLE_NUCLIDE_RE.fullmatch(name):
            names.append(name)
    return names


def discover_neutron_nuclides(nuclear_data_dir: str, library: str, run_openmc: bool) -> list[str]:
    """Discover the neutron-transport nuclide list.

    When OpenMC runs, the HDF5 directory is the source of truth (OpenMC needs
    data too). Otherwise fall back to the local yamc Arrow cache.
    """
    if run_openmc:
        h5_files = sorted(glob.glob(os.path.join(nuclear_data_dir, "neutron", "*.h5")))
        nuclides = [os.path.splitext(os.path.basename(f))[0] for f in h5_files]
        if not nuclides:
            raise FileNotFoundError(
                f"No HDF5 files found in {nuclear_data_dir}neutron/\n"
                f"Check the nuclear data directory for library {library!r}."
            )
    else:
        nuclides = _arrow_cache_nuclides(library)
        if not nuclides:
            raise FileNotFoundError(
                f"No yamc Arrow cache found at {Path.home() / '.cache' / 'yamc'} "
                f"for library {library!r}. Run yamc once on any nuclide to populate it."
            )
    # Skip metastable nuclides for transport.
    nuclides = [n for n in nuclides if "_" not in n]
    return select_requested(
        nuclides, kind="nuclide",
        stable_getter=lambda: stable_nuclide_names(library, run_openmc),
    )


def element_from_nuclide(nuclide: str) -> str:
    """Extract the element symbol from a nuclide name: ``'Fe56' -> 'Fe'``."""
    return re.sub(r"\d+", "", nuclide)


def discover_photon_elements(nuclear_data_dir: str, library: str, run_openmc: bool) -> list[str]:
    """Discover elements with photon data that yamc can build from natural abundance."""
    import yamc

    photon_dir = os.path.join(nuclear_data_dir, "photon")
    photon_files = sorted(glob.glob(os.path.join(photon_dir, "*.h5")))
    elements = [os.path.splitext(os.path.basename(f))[0] for f in photon_files]
    if not elements:
        raise FileNotFoundError(f"No photon data files found in {photon_dir}")
    # Drop synthetic / no-natural-abundance elements yamc cannot expand.
    supported = []
    for e in elements:
        try:
            yamc.Material(composition={e: 1.0}, density=1.0)
            supported.append(e)
        except Exception:
            pass
    # For stable-only mode an element is kept if it has at least one stable
    # isotope (i.e. some stable nuclide reduces to its symbol).
    return select_requested(
        supported, kind="element",
        stable_getter=lambda: {element_from_nuclide(n)
                               for n in stable_nuclide_names(library, run_openmc)},
    )


def discover_coupled_nuclides(nuclear_data_dir: str, library: str, run_openmc: bool) -> list[tuple]:
    """Discover coupled neutron+photon nuclides as ``(name, npath, element, ppath)`` tuples."""
    neutron_dir = os.path.join(nuclear_data_dir, "neutron")
    photon_dir = os.path.join(nuclear_data_dir, "photon")
    nuclides: list[tuple] = []
    if run_openmc:
        for f in sorted(glob.glob(os.path.join(neutron_dir, "*.h5"))):
            name = os.path.splitext(os.path.basename(f))[0]
            if "_m" in name:
                continue
            element = element_from_nuclide(name)
            photon_path = os.path.join(photon_dir, f"{element}.h5")
            if os.path.exists(photon_path):
                nuclides.append((name, f, element, photon_path))
        if not nuclides:
            raise FileNotFoundError(f"No coupled nuclides found in {neutron_dir}")
    else:
        for name in _arrow_cache_nuclides(library):
            if "_m" in name:
                continue
            element = element_from_nuclide(name)
            nuclides.append((name, os.path.join(neutron_dir, f"{name}.h5"),
                             element, os.path.join(photon_dir, f"{element}.h5")))
        if not nuclides:
            raise FileNotFoundError(
                f"No yamc Arrow cache found for library {library!r}."
            )
    return select_requested(
        nuclides, kind="nuclide", key=lambda t: t[0],
        stable_getter=lambda: stable_nuclide_names(library, run_openmc),
    )


# ---- transmutation-chain-backed discovery (decay-photon, transmutation) ----
def resolve_openmc_chain(library: str) -> str:
    """Resolve the OpenMC depletion-chain XML path for ``library`` (best effort).

    Uses the *SFR* chain variant: the yamc side is hardwired to the SFR
    transmutation network (:func:`transmutation_data_path`), and every cached
    OpenMC reference was produced with ``chain_endf_b8.1_sfr.xml``. The
    original notebooks resolved the SFR chain; the ``yamc_verify`` extraction
    accidentally dropped the ``_sfr`` suffix, silently switching new runs to
    the full chain and invalidating comparisons against the SFR-chain caches.
    """
    stem = f"chain_{library.replace('-', '_')}_sfr.xml"
    candidates = [
        f"/home/jon/yamc-org/openmc_data/src/openmc_data/depletion/{stem}",
        "/home/jon/yamc-org/openmc_data/src/openmc_data/depletion/chain_endf_b8.1_sfr.xml",
    ]
    try:
        import openmc_data as _omc_data

        candidates.append(os.path.join(
            os.path.dirname(_omc_data.__file__), "depletion", "chain_endf_b8.1_sfr.xml"))
    except Exception:
        pass
    candidates += sorted(glob.glob(
        "/home/jon/*/.venv/lib/python*/site-packages/openmc_data/depletion/chain_endf_b8.1_sfr.xml"))
    return next((p for p in candidates if os.path.exists(p)), candidates[0])


def transmutation_data_path(library: str) -> str:
    """Path to the yamc transmutation-network Arrow file for ``library``."""
    return os.path.expanduser(f"~/.cache/yamc/transmutation-{library}-sfr.arrow")


def _d1s_chain_maps(chain_file: str, transmutation_data: str):
    """Return (half_lives, photon_sources, reactions_map) from OpenMC XML or yamc chain."""
    import xml.etree.ElementTree as ET

    half_lives: dict = {}
    reactions_map: dict = {}
    photon_sources: set = set()
    if os.path.exists(chain_file):
        root = ET.parse(chain_file).getroot()
        for nuc in root.findall("nuclide"):
            name = nuc.get("name")
            hl = nuc.get("half_life")
            if hl is not None:
                half_lives[name] = float(hl)
            for source in nuc.findall("source"):
                if source.get("particle") == "photon":
                    photon_sources.add(name)
            for rx in nuc.findall("reaction"):
                reactions_map.setdefault(name, []).append(rx.get("target"))
    else:
        import yamc

        chain = yamc.TransmutationChain(transmutation_data)
        half_lives = dict(chain.half_lives)
        photon_sources = set(chain.photon_sources)
        reactions_map = {name: [t[1] for t in entries]
                         for name, entries in chain.reactions.items()}
    return half_lives, photon_sources, reactions_map


def discover_d1s_nuclides(nuclear_data_dir: str, library: str, chain_file: str,
                          transmutation_data: str) -> list[tuple]:
    """Discover D1S parent nuclides as ``(name, npath, element, ppath)`` tuples."""
    half_lives, photon_sources, reactions_map = _d1s_chain_maps(chain_file, transmutation_data)
    d1s_parents = set()
    for parent, targets in reactions_map.items():
        for target in targets:
            if target and target != "None" and target in half_lives and target in photon_sources:
                d1s_parents.add(parent)

    neutron_dir = os.path.join(nuclear_data_dir, "neutron")
    photon_dir = os.path.join(nuclear_data_dir, "photon")
    result: list[tuple] = []
    for f in sorted(glob.glob(os.path.join(neutron_dir, "*.h5"))):
        name = os.path.splitext(os.path.basename(f))[0]
        if "_m" in name or name not in d1s_parents:
            continue
        element = element_from_nuclide(name)
        photon_path = os.path.join(photon_dir, f"{element}.h5")
        if os.path.exists(photon_path):
            result.append((name, f, element, photon_path))
    if not result:
        raise FileNotFoundError("No D1S-relevant nuclides found.")
    return select_requested(result, kind="nuclide", key=lambda t: t[0])


def stable_nuclide_names(library: str, run_openmc: bool, chain_file: str | None = None,
                         transmutation_data: str | None = None) -> set[str]:
    """Return the set of stable (non-radioactive) nuclide names for ``library``.

    Stability comes from the depletion chain: OpenMC's chain XML when OpenMC is
    running (and the file exists), otherwise the yamc transmutation network. A
    nuclide is stable when it has no half-life; metastable states are excluded.

    This backs the ``YAMC_VERIFY_NUMBER_OF_SIMULATIONS=-1`` (stable-only)
    discovery mode. It is only evaluated when that mode is active, so the chain
    load stays off the normal (all-isotope / capped) discovery path.
    """
    if chain_file is None:
        chain_file = resolve_openmc_chain(library)
    if transmutation_data is None:
        transmutation_data = transmutation_data_path(library)
    if run_openmc and os.path.exists(chain_file):
        import openmc.deplete

        chain = openmc.deplete.Chain.from_xml(chain_file)
        return {nuc.name for nuc in chain.nuclides
                if nuc.half_life is None and "_m" not in nuc.name}
    import yamc

    yamc_chain = yamc.TransmutationChain(transmutation_data)
    hl = yamc_chain.half_lives
    return {n for n in yamc_chain.nuclide_names if "_m" not in n and hl.get(n) is None}


def discover_stable_nuclides(nuclear_data_dir: str, library: str, chain_file: str,
                             transmutation_data: str, run_openmc: bool,
                             return_paths: bool = False):
    """Discover stable nuclides present in the depletion chain.

    Restricted to stable nuclides deliberately (an unreduced chain pulls in
    ~1000 fission-yield cross sections). Returns bare names, or ``(name, path)``
    tuples when ``return_paths`` is set (the flux-spectrum notebook's form).
    """
    stable = stable_nuclide_names(library, run_openmc, chain_file, transmutation_data)
    neutron_dir = os.path.join(nuclear_data_dir, "neutron")
    if run_openmc and os.path.exists(chain_file):
        names = []
        for f in sorted(glob.glob(os.path.join(neutron_dir, "*.h5"))):
            name = os.path.splitext(os.path.basename(f))[0]
            if "_m" in name or name not in stable:
                continue
            names.append((name, f) if return_paths else name)
        if not names:
            raise FileNotFoundError(f"No stable nuclides found in chain from {neutron_dir}")
    else:
        names = []
        for name in _arrow_cache_nuclides(library):
            if "_m" in name or name not in stable:
                continue
            names.append((name, os.path.join(neutron_dir, f"{name}.h5")) if return_paths else name)
        if not names:
            raise FileNotFoundError(f"No stable nuclides found in yamc chain/cache for {library!r}.")

    key = (lambda t: t[0]) if return_paths else None
    return select_requested(names, kind="nuclide", key=key)
