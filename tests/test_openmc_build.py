"""The photon-bearing OpenMC references are keyed by the OpenMC build.

openmc#4036 corrected Compton shell selection and Doppler broadening, which
moved the photon references. So the photon, secondary-photon and decay-photon
OpenMC results record the version and commit ``openmc --version`` prints, and a
reference made by another build re-runs. The yamc metadata must not carry them
(a new OpenMC must not re-run yamc), and a yamc-only run must not need OpenMC.
A stand-in ``openmc`` executable on ``PATH`` prints the version, so these need
no OpenMC.
"""
import os
import stat
import sys
import types

import pytest

import cache_utils
from cache_utils import load_cached, save_cached
from yamc_verify import tasks
from yamc_verify.families import decay_photon, photon, secondary_photon

BUILD = {"openmc_version": "0.16.1-dev7", "openmc_commit": "0123abc"}


@pytest.fixture(autouse=True)
def _fresh_fingerprint(monkeypatch):
    # Keep yamc out of the yamc metadata: these tests are about OpenMC's build.
    monkeypatch.setenv("YAMC_VERIFY_ACCEPT_OLD_CACHE", "1")
    cache_utils.openmc_fingerprint.cache_clear()
    yield
    cache_utils.openmc_fingerprint.cache_clear()


@pytest.fixture
def openmc_on_path(tmp_path, monkeypatch):
    exe = tmp_path / "openmc"
    exe.write_text("#!/bin/sh\necho 'OpenMC version 0.16.1-dev7'\necho 'Commit hash: 0123abc'\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


def _metas(family, openmc_build):
    """(OpenMC meta, yamc meta) of the family's first endf-b8.1 task."""
    task = next(t for t in tasks.all_tasks() if t.family == family and t.library == "endf-b8.1")
    if family == "photon":
        return photon._metas(task, openmc_build=openmc_build)
    if family == "secondary-photon":
        return secondary_photon._metas(task, "led", openmc_build=openmc_build)
    return decay_photon._openmc_meta(task, openmc_build=openmc_build), decay_photon._yamc_meta(task)


def test_the_fingerprint_is_what_openmc_version_prints(openmc_on_path):
    assert cache_utils.openmc_fingerprint() == BUILD


@pytest.mark.parametrize("family", tasks.OPENMC_BUILD_KEYED_FAMILIES)
def test_openmc_metas_record_the_build_and_yamc_metas_do_not(family, openmc_on_path):
    openmc_meta, yamc_meta = _metas(family, openmc_build=True)
    assert {k: openmc_meta.get(k) for k in BUILD} == BUILD
    assert not set(BUILD) & set(yamc_meta)


@pytest.mark.parametrize("family", tasks.OPENMC_BUILD_KEYED_FAMILIES)
def test_a_yamc_only_run_needs_no_openmc(family, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # no openmc executable anywhere
    openmc_meta, _ = _metas(family, openmc_build=False)
    assert not set(BUILD) & set(openmc_meta)
    with pytest.raises(FileNotFoundError, match="openmc"):
        cache_utils.openmc_fingerprint()


@pytest.mark.parametrize("family", tasks.OPENMC_BUILD_KEYED_FAMILIES)
def test_a_reference_from_another_build_reruns(family, openmc_on_path, tmp_path):
    meta, _ = _metas(family, openmc_build=True)
    unrecorded = {k: v for k, v in meta.items() if k not in BUILD}
    for stored in (unrecorded, {**meta, "openmc_commit": "fffffff"}):
        save_cached(tmp_path, "endf-b8.1", "openmc", "Fe", {"x": 1.0}, meta=stored)
        assert load_cached(tmp_path, "endf-b8.1", "openmc", "Fe", expected_meta=meta) is None
    save_cached(tmp_path, "endf-b8.1", "openmc", "Fe", {"x": 1.0}, meta=meta)
    assert load_cached(tmp_path, "endf-b8.1", "openmc", "Fe", expected_meta=meta) is not None


def test_the_planner_keys_exactly_those_families_by_the_build(openmc_on_path, monkeypatch):
    # The planner imports yamc at module level; a stand-in is enough here.
    monkeypatch.setitem(sys.modules, "yamc", sys.modules.get("yamc") or types.ModuleType("yamc"))
    from yamc_verify import plan

    for task in tasks.all_tasks():
        meta = plan._code_meta("openmc", task, {})
        keyed = task.family in tasks.OPENMC_BUILD_KEYED_FAMILIES
        assert (set(BUILD) <= set(meta)) == keyed, task.key
