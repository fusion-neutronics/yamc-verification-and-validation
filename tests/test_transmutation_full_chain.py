"""The transmutation benchmarks solve on the library's full chain in both codes.

OpenMC is handed the chain file as published, for the coupled operator and for
``material.deplete``; yamc's transmutation settings point at the library
keyword, its full published chain, for each solve. Nothing reduces a chain, and
results cached when the chain was reduced are not reused. Stand-in ``yamc`` and
``openmc`` modules record what each solve was given, so these need neither code
nor nuclear data.
"""
import sys
import tempfile
import types

import pytest

from cache_utils import load_cached, save_cached
from yamc_verify import tasks
from yamc_verify.families import transmutation

SETTINGS = ("transmutation_decay_data", "transmutation_reactions", "transmutation_fission_yields")
REACTION_RATE = "transmutation/material-reaction-rate/endf-b8.1"
FLUX_SPECTRUM = "transmutation/material-flux-spectrum/endf-b8.1"
CHAIN_FILE = "/chains/chain_endf_b8.1.xml"


class _Anything:
    """Any object a runner builds and only passes on: takes and returns anything."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, name):
        return _Anything()

    def __call__(self, *args, **kwargs):
        return _Anything()

    def __neg__(self):
        return self


def _no_reduction(*args, **kwargs):
    raise AssertionError("a transmutation runner built or reduced a chain of its own")


@pytest.fixture
def fake_yamc(monkeypatch):
    """A stand-in ``yamc`` that logs the network settings each solve sees."""
    mod = types.SimpleNamespace(**{name: None for name in SETTINGS})
    mod.solves = []
    steps = len(tasks.get_task(REACTION_RATE).extras["timesteps"])

    def solve(call):
        mod.solves.append((call, tuple(getattr(mod, name) for name in SETTINGS)))

    class Results:
        num_steps = steps

        def get_material_nuclides(self, material_id, step):
            return {"Ag108": 1e-9}

        def step_materials(self, material_id):
            return [types.SimpleNamespace(nuclides={"Ag108": 1e-9})] * steps

    class Material(_Anything):
        id = 1

        def transmute(self, schedule):
            solve("transmute")
            return Results()

    class Model(_Anything):
        def simulate_transmutation(self, **kwargs):
            solve("simulate_transmutation")
            return Results()

    mod.Material, mod.Model = Material, Model
    for name in ("Sphere", "Cell", "Geometry", "NeutronSource", "Pulse", "Cooldown",
                 "PulseSchedule"):
        setattr(mod, name, _Anything)
    mod.sources = _Anything()
    mod.TransmutationChain = _no_reduction
    monkeypatch.setitem(sys.modules, "yamc", mod)
    return mod


@pytest.fixture
def fake_openmc(monkeypatch, tmp_path):
    """Stand-in ``openmc`` and ``openmc.deplete`` that log the chain each solve is given."""
    chains = []
    steps = len(tasks.get_task(REACTION_RATE).extras["timesteps"])
    densities = types.SimpleNamespace(get_nuclide_atom_densities=lambda: {"Ag108": 1e-9})

    class Material(_Anything):
        def deplete(self, **kwargs):
            chains.append(("deplete", kwargs["chain_file"]))
            return [densities] * (steps + 1)

    def coupled_operator(model, chain_file, **kwargs):
        chains.append(("CoupledOperator", chain_file))
        return _Anything()

    class Results:
        def __init__(self, path):
            pass

        def __len__(self):
            return steps + 1

        def __getitem__(self, step):
            return types.SimpleNamespace(get_material=lambda mat_id: densities)

    deplete = types.ModuleType("openmc.deplete")
    deplete.CoupledOperator = coupled_operator
    deplete.PredictorIntegrator = _Anything
    deplete.Results = Results
    deplete.Chain = types.SimpleNamespace(from_xml=_no_reduction)
    openmc = types.ModuleType("openmc")
    openmc.config = {}
    openmc.deplete = deplete
    openmc.Material = Material
    for name in ("reset_auto_ids", "Sphere", "Cell", "Geometry", "IndependentSource",
                 "Settings", "Materials"):
        setattr(openmc, name, _Anything)
    openmc.stats = openmc.model = _Anything()
    monkeypatch.setitem(sys.modules, "openmc", openmc)
    monkeypatch.setitem(sys.modules, "openmc.deplete", deplete)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return chains


def test_yamc_reaction_rate_solves_on_the_full_chain(fake_yamc):
    task = tasks.get_task(REACTION_RATE)
    transmutation.run_yamc_reaction_rate("Ag107", task)
    assert fake_yamc.solves == [("simulate_transmutation", (task.library,) * 3)]
    assert tuple(getattr(fake_yamc, name) for name in SETTINGS) == (None,) * 3


def test_yamc_flux_spectrum_solves_on_the_full_chain(fake_yamc):
    task = tasks.get_task(FLUX_SPECTRUM)
    transmutation.run_yamc_flux_spectrum("Ag107", task)
    assert fake_yamc.solves == [("transmute", (task.library,) * 3)]
    assert tuple(getattr(fake_yamc, name) for name in SETTINGS) == (None,) * 3


def test_openmc_reaction_rate_reads_the_chain_file_whole(fake_openmc):
    transmutation.run_openmc_reaction_rate("Ag107", tasks.get_task(REACTION_RATE), CHAIN_FILE)
    assert fake_openmc == [("CoupledOperator", CHAIN_FILE)]


def test_openmc_flux_spectrum_reads_the_chain_file_whole(fake_openmc):
    transmutation.run_openmc_flux_spectrum("Ag107", tasks.get_task(FLUX_SPECTRUM), CHAIN_FILE)
    assert fake_openmc == [("deplete", CHAIN_FILE)]


@pytest.mark.parametrize("key", [REACTION_RATE, FLUX_SPECTRUM])
def test_no_reduction_level_is_configured(key):
    assert "reduce_level" not in tasks.get_task(key).extras


@pytest.mark.parametrize("key", [REACTION_RATE, FLUX_SPECTRUM])
def test_a_result_cached_on_a_reduced_chain_reruns(key, tmp_path, monkeypatch):
    # Keep yamc out of the build fingerprint: this test is about the chain key.
    monkeypatch.setenv("YAMC_VERIFY_ACCEPT_OLD_CACHE", "1")
    task = tasks.get_task(key)
    for code, meta in zip(("openmc", "yamc"), transmutation._metas(task, CHAIN_FILE)):
        assert meta["chain"] == "full" and "reduce_level" not in meta
        reduced = {k: v for k, v in meta.items() if k != "chain"} | {"reduce_level": 4}
        save_cached(tmp_path, task.library, code, "Ag107", {"steps": []}, meta=reduced)
        assert load_cached(tmp_path, task.library, code, "Ag107", expected_meta=meta) is None
        save_cached(tmp_path, task.library, code, "Ag107", {"steps": []}, meta=meta)
        assert load_cached(tmp_path, task.library, code, "Ag107", expected_meta=meta) is not None
