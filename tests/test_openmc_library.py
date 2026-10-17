"""Discovery finds an OpenMC library's data through its cross_sections.xml.

Libraries lay out their HDF5 files differently. The IAEA's FENDL-3.2d keeps
them under ``neutron/hdf5/`` and ``atom/hdf5/``, some named like
``01H_001.h5``, and lists H1 twice. OpenMC follows ``cross_sections.xml``, so
discovery must too; globbing ``neutron/*.h5`` finds nothing there. These build
such a library from empty files, so they need neither OpenMC nor nuclear data.
"""
import os
import sys
import types

import pytest

from yamc_verify import discovery

XML = """<?xml version='1.0' encoding='UTF-8'?>
<cross_sections>
  <library materials="Ag" path="atom/hdf5/Ag.h5" type="photon"/>
  <library materials="Fe" path="atom/hdf5/Fe.h5" type="photon"/>
  <library materials="H1" path="neutron/hdf5/01H_001.h5" type="neutron"/>
  <library materials="Fe56" path="neutron/hdf5/Fe56.h5" type="neutron"/>
  <library materials="H1" path="neutron/hdf5/H1.h5" type="neutron"/>
  <library materials="Ta180" path="neutron/hdf5/Ta180_m1.h5" type="neutron"/>
  <library materials="Am242_m1" path="neutron/hdf5/Am242_m1.h5" type="neutron"/>
  <library materials="c_H_in_H2O" path="thermal/c_H_in_H2O.h5" type="thermal"/>
</cross_sections>
"""
FILES = ("atom/hdf5/Ag.h5", "atom/hdf5/Fe.h5", "neutron/hdf5/01H_001.h5", "neutron/hdf5/Fe56.h5",
         "neutron/hdf5/H1.h5", "neutron/hdf5/Ta180_m1.h5", "neutron/hdf5/Am242_m1.h5",
         "thermal/c_H_in_H2O.h5")


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A library laid out like the IAEA FENDL-3.2d one, as NUCLEAR_DATA_DIR names it."""
    (tmp_path / "cross_sections.xml").write_text(XML)
    for path in FILES:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).touch()
    for name in ("YAMC_VERIFY_NUCLIDES", "YAMC_VERIFY_ELEMENTS", "YAMC_VERIFY_NUMBER_OF_SIMULATIONS"):
        monkeypatch.delenv(name, raising=False)
    return f"{tmp_path}/"


def test_names_and_paths_come_from_cross_sections_xml(library):
    neutron = discovery.openmc_library(library, "neutron")
    assert sorted(neutron) == ["Am242_m1", "Fe56", "H1", "Ta180"]
    # A name listed twice keeps its first file.
    assert neutron["H1"] == os.path.join(library, "neutron/hdf5/01H_001.h5")
    assert neutron["Ta180"] == os.path.join(library, "neutron/hdf5/Ta180_m1.h5")
    assert sorted(discovery.openmc_library(library, "photon")) == ["Ag", "Fe"]


def test_neutron_discovery_lists_the_library_not_its_file_names(library):
    # Metastable states are not transported, as before.
    assert discovery.discover_neutron_nuclides(library, "fendl-3.2d", run_openmc=True) == [
        "Fe56", "H1", "Ta180"]


def test_photon_discovery_lists_the_library(library, monkeypatch):
    # A stand-in yamc that can build every element from natural abundance.
    monkeypatch.setitem(sys.modules, "yamc", types.SimpleNamespace(Material=lambda **kw: None))
    assert discovery.discover_photon_elements(library, "fendl-3.2d", run_openmc=True) == ["Ag", "Fe"]


def test_coupled_discovery_carries_the_listed_paths(library):
    found = discovery.discover_coupled_nuclides(library, "fendl-3.2d", run_openmc=True)
    assert found == [("Fe56", os.path.join(library, "neutron/hdf5/Fe56.h5"),
                      "Fe", os.path.join(library, "atom/hdf5/Fe.h5"))]


def test_a_directory_without_cross_sections_xml_names_the_setting(tmp_path):
    with pytest.raises(FileNotFoundError, match="NUCLEAR_DATA_DIR"):
        discovery.openmc_library(str(tmp_path), "neutron")
