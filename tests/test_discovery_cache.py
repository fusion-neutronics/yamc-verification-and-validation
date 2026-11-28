"""The yamc cache lookups in discovery follow yamc's release layout.

yamc caches each published release of a library under
``<yamc.cache_dir()>/<library>/<release>/``. These tests build that layout in a
temporary directory and point a stand-in ``yamc.cache_dir()`` at it, so they
need neither yamc nor nuclear data.
"""
import sys
import tempfile
import types
from pathlib import Path

import pytest

from yamc_verify import discovery


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "yamc-cache"
    root.mkdir()
    monkeypatch.setitem(sys.modules, "yamc", types.SimpleNamespace(cache_dir=lambda: str(root)))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    return root


def _release(root, library, release, nuclides=(), chain=True):
    d = root / library / release
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    for name in nuclides:
        (d / "neutron" / f"{name}.arrow").mkdir(parents=True)
    if chain:
        for sub, section in discovery._CHAIN_SUBSECTIONS.items():
            folder = d / "transmutation" / f"{sub}.arrow"
            folder.mkdir(parents=True)
            (folder / section).write_text("")
    return d


def test_nuclides_come_from_the_newest_release(cache):
    _release(cache, "endf-b8.1", "2026-10-01", nuclides=["Fe56", "Li6"])
    _release(cache, "endf-b8.1", "2026-10-15.1", nuclides=["Fe56", "Fe", "W186"])
    assert discovery._arrow_cache_nuclides("endf-b8.1") == ["Fe56", "W186"]


def test_no_cached_release_lists_nothing(cache):
    assert discovery._arrow_cache_nuclides("endf-b8.1") == []


def test_chain_root_links_the_newest_complete_release(cache):
    _release(cache, "endf-b8.1", "2026-10-01")
    newest = _release(cache, "endf-b8.1", "2026-10-15.1")
    # A newer release whose chain is only partly downloaded is passed over.
    partial = _release(cache, "endf-b8.1", "2026-11-01", chain=False)
    (partial / "transmutation" / "decay.arrow").mkdir(parents=True)

    root = Path(discovery.transmutation_data_path("endf-b8.1"))
    assert root.name == "2026-10-15.1"
    for sub in discovery._CHAIN_SUBSECTIONS:
        assert (root / sub).resolve() == (newest / "transmutation" / f"{sub}.arrow").resolve()
    assert not (root / "branching").exists()
    # Asking again gives the same path, which yamc's chain cache is keyed on.
    assert discovery.transmutation_data_path("endf-b8.1") == str(root)


def test_chain_root_links_branching_when_cached(cache):
    d = _release(cache, "endf-b8.1", "2026-10-15.1")
    (d / "transmutation" / "branching.arrow").mkdir()
    root = Path(discovery.transmutation_data_path("endf-b8.1"))
    assert (root / "branching").resolve() == (d / "transmutation" / "branching.arrow").resolve()


def test_no_cached_chain_names_the_library_cache(cache):
    assert discovery.transmutation_data_path("endf-b8.1") == str(cache / "endf-b8.1")


def test_without_an_openmc_chain_the_yamc_chain_is_read_with_from_arrow(tmp_path, monkeypatch):
    """yamc 0.26.0 replaced ``TransmutationChain(path)`` with ``TransmutationChain.from_arrow(path)``."""
    read = []

    class TransmutationChain:
        def __init__(self, *args, **kwargs):
            raise TypeError("cannot create 'builtins.TransmutationChain' instances")

        @classmethod
        def from_arrow(cls, path):
            read.append(path)
            return types.SimpleNamespace(
                half_lives={"Fe56": None, "Mn56": 9284.0, "Fe55": 8.6e7},
                photon_sources=["Mn56"],
                reactions={"Fe56": [("(n,p)", "Mn56")], "Fe54": [("(n,g)", "Fe55")]},
                nuclide_names=["Fe54", "Fe56", "Fe55", "Mn56", "Fe56_m1"],
            )

    monkeypatch.setitem(sys.modules, "yamc", types.SimpleNamespace(TransmutationChain=TransmutationChain))
    missing = str(tmp_path / "no-chain.xml")
    half_lives, sources, reactions = discovery._d1s_chain_maps(missing, "chain-root")
    assert sources == {"Mn56"} and reactions == {"Fe56": ["Mn56"], "Fe54": ["Fe55"]}
    assert discovery.stable_nuclide_names("endf-b8.1", run_openmc=False, chain_file=missing,
                                          transmutation_data="chain-root") == {"Fe54", "Fe56"}
    assert read[0] == "chain-root" and len(read) == 2
