"""The photon-bearing OpenMC references are keyed by the OpenMC build.

openmc#4036 corrected Compton shell selection and Doppler broadening, which
moved the photon references. So every OpenMC record stores the version and
commit ``openmc --version`` prints, and the photon, secondary-photon and
decay-photon families only accept a reference from the installed OpenMC while
OpenMC runs. Other OpenMC references, and a yamc-only run, take any build.
A stand-in ``openmc`` executable on ``PATH`` prints the version, so these need
no OpenMC.
"""
import os
import stat
import sys
import types

import pytest

import cache_utils
from cache_utils import find_record, save_cached
from yamc_verify import tasks

BUILD = "0.16.1-dev7+g0123abc"


@pytest.fixture(autouse=True)
def _fresh_identity(monkeypatch):
    monkeypatch.delenv("YAMC_VERIFY_ACCEPT_OLD_CACHE", raising=False)
    cache_utils.openmc_identity.cache_clear()
    yield
    cache_utils.openmc_identity.cache_clear()


@pytest.fixture
def openmc_on_path(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "openmc"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\necho 'OpenMC version 0.16.1-dev7'\necho 'Commit hash: 0123abc'\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{exe.parent}{os.pathsep}{os.environ['PATH']}")


def test_the_identity_is_what_openmc_version_prints(openmc_on_path):
    ident = cache_utils.code_identity("openmc-led")
    assert (ident["name"], ident["version"], ident["commit"]) == ("openmc", "0.16.1-dev7", "0123abc")
    assert ident["build_id"] == BUILD


def test_a_missing_openmc_is_an_error_only_when_its_identity_is_needed(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # no openmc executable anywhere
    old = {"name": "openmc", "build_id": "unknown"}
    save_cached(tmp_path, "endf-b8.1", "openmc", "Fe", {"x": 1.0}, settings={"a": 1}, identity=old)
    # A yamc-only run reads the reference without asking which OpenMC made it.
    assert find_record(tmp_path, "endf-b8.1", "openmc", "Fe", settings={"a": 1}) is not None
    with pytest.raises(FileNotFoundError, match="openmc"):
        find_record(tmp_path, "endf-b8.1", "openmc", "Fe", settings={"a": 1}, build_keyed=True)


def test_a_build_keyed_reference_from_another_build_reruns(openmc_on_path, tmp_path):
    other = {"name": "openmc", "build_id": "0.15.0+gfffffff"}
    save_cached(tmp_path, "endf-b8.1", "openmc", "Fe", {"x": 1.0}, settings={"a": 1}, identity=other)
    assert find_record(tmp_path, "endf-b8.1", "openmc", "Fe", settings={"a": 1}, build_keyed=True) is None
    # Not build-keyed, the other build's reference is used.
    assert find_record(tmp_path, "endf-b8.1", "openmc", "Fe", settings={"a": 1}) is not None
    save_cached(tmp_path, "endf-b8.1", "openmc", "Fe", {"x": 2.0}, settings={"a": 1})
    path, record = find_record(tmp_path, "endf-b8.1", "openmc", "Fe", settings={"a": 1}, build_keyed=True)
    assert path.name == f"{BUILD}.json" and record["results"] == {"x": 2.0}


def test_the_planner_keys_exactly_those_families_while_openmc_runs(monkeypatch):
    # The planner imports yamc at module level; a stand-in is enough here.
    monkeypatch.setitem(sys.modules, "yamc", sys.modules.get("yamc") or types.ModuleType("yamc"))
    from yamc_verify import plan
    from yamc_verify.backends import resolve

    running, not_running = resolve("openmc-cpu,yamc-cpu"), resolve("yamc-cpu")
    for task in tasks.all_tasks():
        keyed = task.family in tasks.OPENMC_BUILD_KEYED_FAMILIES
        assert plan._build_keyed("openmc", task, running) == keyed, task.key
        assert plan._build_keyed("openmc", task, not_running) is False, task.key
        assert plan._build_keyed("yamc-cpu", task, not_running) is True, task.key
