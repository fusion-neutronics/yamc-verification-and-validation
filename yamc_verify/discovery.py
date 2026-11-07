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

import contextlib
import functools
import glob
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from cache_utils import requested_items, select_requested

_SINGLE_NUCLIDE_RE = re.compile(r"[A-Z][a-z]?\d+")


@functools.lru_cache(maxsize=None)
def _cross_sections_index(nuclear_data_dir: str) -> dict:
    """``{data type: {material name: absolute HDF5 path}}`` from ``cross_sections.xml``."""
    path = os.path.join(nuclear_data_dir, "cross_sections.xml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No cross_sections.xml in {nuclear_data_dir}. OpenMC finds a library's "
            f"data through that file; check NUCLEAR_DATA_DIR in yamc_verify/tasks.py."
        )
    index: dict = {}
    for library in ET.parse(path).getroot().findall("library"):
        kind = index.setdefault(library.get("type"), {})
        full = os.path.join(nuclear_data_dir, library.get("path"))
        for name in (library.get("materials") or "").split():
            # A name listed twice keeps its first file. The IAEA FENDL-3.2d
            # library lists H1 and Li7 twice, each time with identical files.
            kind.setdefault(name, full)
    return index


def openmc_library(nuclear_data_dir: str, kind: str) -> dict[str, str]:
    """The ``{name: HDF5 path}`` entries of one data type in an OpenMC library.

    OpenMC finds a nuclide's or element's data through the library's
    ``cross_sections.xml``, not by file name, and libraries lay their files out
    differently: the OpenMC-built ENDF/B-VIII.1 keeps ``neutron/Fe56.h5``, while
    the IAEA's FENDL-3.2d keeps ``neutron/hdf5/Fe56.h5`` beside
    ``neutron/hdf5/01H_001.h5``. So discovery reads the same file OpenMC does.
    ``kind`` is a ``cross_sections.xml`` type: ``"neutron"`` or ``"photon"``.
    """
    return dict(_cross_sections_index(os.path.abspath(nuclear_data_dir)).get(kind, {}))


def _yamc_cache_dir() -> Path:
    """yamc's nuclear-data cache directory (``yamc.cache_dir()``)."""
    import yamc

    return Path(yamc.cache_dir())


def _cached_releases(library: str) -> list[Path]:
    """The releases of ``library`` in yamc's cache, oldest first.

    yamc caches each published release of a library in its own folder,
    ``<yamc.cache_dir()>/<library>/<release>/``, laid out as the published tree
    (``neutron/<Name>.arrow/``, ``photon/<El>.arrow/``,
    ``transmutation/<subsection>.arrow/``) beside that release's
    ``manifest.json``. Releases are named by date (``2026-10-10.1``), so the
    lexically last folder holding a manifest is the newest.
    """
    return sorted(
        d for d in (_yamc_cache_dir() / library).glob("*")
        if d.is_dir() and (d / "manifest.json").is_file()
    )


def _arrow_cache_nuclides(library: str) -> list[str]:
    """Single-nuclide names downloaded into the newest cached release of ``library``."""
    releases = _cached_releases(library)
    if not releases:
        return []
    names = []
    for p in sorted((releases[-1] / "neutron").glob("*.arrow")):
        if not p.is_dir():
            continue
        name = p.name[:-len(".arrow")]
        # Single-nuclide entries only (Ac225, Fe56, ...); drop natural-element
        # entries (Fe, U, ...) that have no mass number.
        if _SINGLE_NUCLIDE_RE.fullmatch(name):
            names.append(name)
    return names


def discover_neutron_nuclides(nuclear_data_dir: str, library: str, run_openmc: bool) -> list[str]:
    """Discover the neutron-transport nuclide list.

    When OpenMC runs, its library is the source of truth (OpenMC needs data
    too). Otherwise fall back to the local yamc Arrow cache.
    """
    if run_openmc:
        nuclides = sorted(openmc_library(nuclear_data_dir, "neutron"))
        if not nuclides:
            raise FileNotFoundError(
                f"No neutron data listed in {nuclear_data_dir}cross_sections.xml\n"
                f"Check the nuclear data directory for library {library!r}."
            )
    else:
        nuclides = _arrow_cache_nuclides(library)
        if not nuclides:
            raise FileNotFoundError(
                f"No yamc Arrow cache found at {_yamc_cache_dir() / library} "
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

    elements = sorted(openmc_library(nuclear_data_dir, "photon"))
    if not elements:
        raise FileNotFoundError(
            f"No photon data listed in {nuclear_data_dir}cross_sections.xml")
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
        photon = openmc_library(nuclear_data_dir, "photon")
        for name, npath in sorted(openmc_library(nuclear_data_dir, "neutron").items()):
            if "_m" in name:
                continue
            element = element_from_nuclide(name)
            if element in photon:
                nuclides.append((name, npath, element, photon[element]))
        if not nuclides:
            raise FileNotFoundError(
                f"No coupled nuclides listed in {nuclear_data_dir}cross_sections.xml")
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
    """Resolve the OpenMC depletion-chain XML path for ``library`` (best effort)."""
    candidates = [
        f"/home/jon/yamc-org/openmc_data/src/openmc_data/depletion/chain_{library.replace('-', '_')}.xml",
        "/home/jon/yamc-org/openmc_data/src/openmc_data/depletion/chain_endf_b8.1.xml",
    ]
    try:
        import openmc_data as _omc_data

        candidates.append(os.path.join(
            os.path.dirname(_omc_data.__file__), "depletion", "chain_endf_b8.1.xml"))
    except Exception:
        pass
    candidates += sorted(glob.glob(
        "/home/jon/*/.venv/lib/python*/site-packages/openmc_data/depletion/chain_endf_b8.1.xml"))
    return next((p for p in candidates if os.path.exists(p)), candidates[0])


# The transmutation subsections ``yamc.TransmutationChain`` reads from a chain
# root, each a folder of that name, with the section each must hold. yamc
# downloads a release's files one by one as runs need them, so a subsection
# folder can exist with only some of its files. ``branching`` is optional.
_CHAIN_SUBSECTIONS = {
    "decay": "nuclides.arrow",
    "reactions": "reactions.arrow",
    "fission_yields": "fission_yields.arrow",
}
_OPTIONAL_CHAIN_SUBSECTIONS = ("branching",)


def transmutation_data_path(library: str) -> str:
    """Chain root of ``library``'s transmutation network, from yamc's cache.

    yamc caches a release's transmutation subsections as
    ``<yamc.cache_dir()>/<library>/<release>/transmutation/<subsection>.arrow/``,
    while ``yamc.TransmutationChain`` reads a chain root holding ``decay/``,
    ``reactions/``, ``fission_yields/`` and optionally ``branching/``. So this
    links the newest cached release holding all three required subsections into
    such a root, under the temporary directory and named by library and release,
    and returns it. The path is stable for a given release, which matters
    because yamc caches a parsed chain by its path.

    The subsections are downloaded by any yamc run that transmutes with the
    library's keyword. When no cached release holds them yet, this returns the
    library's cache folder, which ``TransmutationChain`` refuses with a message
    naming it; only the callers that read the yamc chain need it.
    """
    releases = [
        d for d in _cached_releases(library)
        if all((d / "transmutation" / f"{sub}.arrow" / section).is_file()
               for sub, section in _CHAIN_SUBSECTIONS.items())
    ]
    if not releases:
        return str(_yamc_cache_dir() / library)
    release = releases[-1]
    root = Path(tempfile.gettempdir()) / "yamc-verify-chains" / library / release.name
    root.mkdir(parents=True, exist_ok=True)
    for sub in (*_CHAIN_SUBSECTIONS, *_OPTIONAL_CHAIN_SUBSECTIONS):
        target = release / "transmutation" / f"{sub}.arrow"
        link = root / sub
        if target.is_dir() and not link.exists():
            link.symlink_to(target, target_is_directory=True)
    return str(root)


# The yamc settings the transmutation network is assembled from.
_NETWORK_SETTINGS = (
    "transmutation_decay_data",
    "transmutation_reactions",
    "transmutation_fission_yields",
)


@contextlib.contextmanager
def transmutation_network(source: str):
    """Point yamc's transmutation network at ``source`` for the duration.

    yamc takes no per-call chain: ``Model.simulate_transmutation``,
    ``Material.transmute``, ``Model.radionuclides`` and the D1S time correction
    all assemble the network from the global decay, reactions and fission-yield
    settings. ``source`` is a library keyword, or a chain root holding
    ``decay/``, ``reactions/`` and ``fission_yields/`` (what
    ``TransmutationChain.export_to_arrow`` writes). All three settings are
    pointed at it and unset afterwards. yamc has no default for them (since
    yamc-core 0.17.0 a calculation that reads an unset one is refused), so a
    run outside this context cannot read a chain nobody chose.
    """
    import yamc as mc

    for name in _NETWORK_SETTINGS:
        setattr(mc, name, source)
    try:
        yield
    finally:
        for name in _NETWORK_SETTINGS:
            setattr(mc, name, None)


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

    neutron = openmc_library(nuclear_data_dir, "neutron")
    photon = openmc_library(nuclear_data_dir, "photon")
    result: list[tuple] = []
    for name, npath in sorted(neutron.items()):
        if "_m" in name or name not in d1s_parents:
            continue
        element = element_from_nuclide(name)
        if element in photon:
            result.append((name, npath, element, photon[element]))

    # Explicitly requested nuclides bypass the D1S-signature filter: naming a
    # nuclide means "run it", even one whose activation products emit no decay
    # photons (e.g. Li6/Li7). Those yield a legitimate all-zero spectrum rather
    # than a hard discovery failure. Only add one whose transport + element
    # photon data exist; a requested nuclide with no data still raises in
    # select_requested (so typos are still caught).
    have = {t[0] for t in result}
    for name in requested_items("nuclide"):
        if name in have or "_m" in name:
            continue
        element = element_from_nuclide(name)
        if name in neutron and element in photon:
            result.append((name, neutron[name], element, photon[element]))
            have.add(name)

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
    # The OpenMC library's neutron entries, when this machine has the library.
    if os.path.isfile(os.path.join(nuclear_data_dir, "cross_sections.xml")):
        neutron = openmc_library(nuclear_data_dir, "neutron")
    else:
        neutron = {}
    if run_openmc and os.path.exists(chain_file):
        names = []
        for name, npath in sorted(neutron.items()):
            if "_m" in name or name not in stable:
                continue
            names.append((name, npath) if return_paths else name)
        if not names:
            raise FileNotFoundError(
                f"No stable nuclides of the chain listed in {nuclear_data_dir}cross_sections.xml")
    else:
        names = []
        for name in _arrow_cache_nuclides(library):
            if "_m" in name or name not in stable:
                continue
            names.append((name, os.path.join(neutron_dir, f"{name}.h5")) if return_paths else name)
        if not names:
            raise FileNotFoundError(f"No stable nuclides found in yamc chain/cache for {library!r}.")

    # Explicitly requested nuclides bypass the stable-only filter: naming an
    # (effectively) unstable nuclide such as W180 or W186 means "irradiate it
    # and report the result", which is valid physics; the stable-only default
    # is just a scope choice to avoid pulling in fission-yield data. Add a
    # requested name that has transport data but was filtered out; a requested
    # name with no data still raises in select_requested.
    have = {(n[0] if return_paths else n) for n in names}
    for name in requested_items("nuclide"):
        if name in have or "_m" in name:
            continue
        if name in neutron:
            names.append((name, neutron[name]) if return_paths else name)
            have.add(name)

    key = (lambda t: t[0]) if return_paths else None
    return select_requested(names, kind="nuclide", key=key)
