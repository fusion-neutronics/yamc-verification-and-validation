"""The decay-photon (D1S) benchmarks run both codes on the library's full chain.

OpenMC's side reads the chain XML as published. yamc takes its chain from the
global transmutation settings, so these check that every call of a yamc D1S run
that reads the chain (the radionuclide list, the decay-photon transport and the
time correction) sees the library keyword, which is its full published chain,
and that the run never builds a reduced one. A stand-in ``yamc`` records the
settings at each of those calls, so this needs neither yamc nor nuclear data.
"""
import sys
import types

import pytest

from cache_utils import load_cached, save_cached
from yamc_verify import tasks
from yamc_verify.families import decay_photon

SETTINGS = ("transmutation_decay_data", "transmutation_reactions", "transmutation_fission_yields")
D1S_TASKS = ("decay-photon/sphere/endf-b8.1", "decay-photon/broomstick/endf-b8.1")


class _Part:
    """Any geometry, material or source object: takes anything, combines with anything."""

    def __init__(self, *args, **kwargs):
        pass

    below = above = property(lambda self: self)

    def __and__(self, other):
        return self

    def read_nuclear_data(self, library):
        pass


@pytest.fixture
def fake_yamc(monkeypatch, tmp_path):
    """A stand-in ``yamc`` that logs the network settings each chain read sees."""
    mod = types.SimpleNamespace(**{name: None for name in SETTINGS})
    mod.reads = []

    def read(call):
        mod.reads.append((call, tuple(getattr(mod, name) for name in SETTINGS)))

    class Tally(_Part):
        energy_bins = [1.0e4, 1.0e5, 1.0e6]

    class Model(_Part):
        def radionuclides(self):
            read("radionuclides")
            return ["Mn56", "Mn54"]

        def simulate_transport(self, **kwargs):
            read("simulate_transport")
            return {tally: object() for tally in self.tallies}

        def __init__(self, tallies=(), **kwargs):
            self.tallies = list(tallies)

    class PulseSchedule(_Part):
        def time_correct_tally(self, tally_result):
            read("time_correct_tally")
            steps = len(tasks.get_task(D1S_TASKS[0]).extras["timesteps"])
            return types.SimpleNamespace(mean=[[1.0, 2.0]] * steps, std_dev=[[0.1, 0.2]] * steps)

    class TransmutationChain:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the D1S yamc side built a chain of its own instead of "
                                 "reading the library's full chain")

    mod.Model, mod.Tally, mod.PulseSchedule = Model, Tally, PulseSchedule
    mod.TransmutationChain = TransmutationChain
    for name in ("Material", "Cell", "Geometry", "NeutronSource", "Sphere", "Cylinder",
                 "Plane", "Pulse", "Cooldown"):
        setattr(mod, name, _Part)
    mod.sources = types.SimpleNamespace(Isotropic=_Part, Discrete=_Part)
    mod.cache_dir = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "yamc", mod)
    return mod


@pytest.mark.parametrize("key", D1S_TASKS)
def test_yamc_reads_the_full_chain_through_the_whole_run(fake_yamc, key):
    task = tasks.get_task(key)
    result, _ = decay_photon.run_yamc_d1s("Fe56", None, "Fe", None, task)

    assert [call for call, _ in fake_yamc.reads] == [
        "radionuclides", "simulate_transport", "time_correct_tally"]
    for call, settings in fake_yamc.reads:
        assert settings == (task.library,) * 3, call
    # Unset again afterwards, so nothing leaks into the next run.
    assert tuple(getattr(fake_yamc, name) for name in SETTINGS) == (None,) * 3
    assert result["radionuclides"] == ["Mn56", "Mn54"]


@pytest.mark.parametrize("key", D1S_TASKS)
def test_no_reduction_level_is_configured(key):
    assert "reduce_level" not in tasks.get_task(key).extras


def test_a_result_cached_without_the_chain_record_reruns(tmp_path, monkeypatch):
    """A yamc result cached before the chain was recorded may come from the
    level-4 reduced chain, so the cache must not hand it back."""
    # Keep yamc out of the build fingerprint: this test is about the chain key.
    monkeypatch.setenv("YAMC_VERIFY_ACCEPT_OLD_CACHE", "1")
    task = tasks.get_task(D1S_TASKS[0])
    meta = decay_photon._yamc_meta(task)
    assert meta["yamc_chain"] == "full"

    before = {k: v for k, v in meta.items() if k != "yamc_chain"}
    save_cached(tmp_path, task.library, "yamc-cpu", "Ac225", {"spectra": []}, meta=before)
    assert load_cached(tmp_path, task.library, "yamc-cpu", "Ac225", expected_meta=meta) is None

    save_cached(tmp_path, task.library, "yamc-cpu", "Ac225", {"spectra": []}, meta=meta)
    assert load_cached(tmp_path, task.library, "yamc-cpu", "Ac225", expected_meta=meta) is not None
