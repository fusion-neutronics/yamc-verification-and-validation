"""yamc's transmutation network points at a chain only for the call that needs it.

yamc takes no per-call chain, so the transmutation families set its global
decay, reactions and fission-yield settings around a solve. A setting left
behind would hand the next run a chain it did not ask for without any error,
so these check the reset, including when the solve raises.
A stand-in ``yamc`` records the settings, so they need neither yamc nor nuclear
data.
"""
import sys
import types

import pytest

from yamc_verify import discovery

SETTINGS = ("transmutation_decay_data", "transmutation_reactions", "transmutation_fission_yields")


@pytest.fixture
def fake_yamc(monkeypatch):
    mod = types.SimpleNamespace(**{name: None for name in SETTINGS})
    monkeypatch.setitem(sys.modules, "yamc", mod)
    return mod


def _settings(mod):
    return [getattr(mod, name) for name in SETTINGS]


def test_settings_point_at_the_source_and_reset_after(fake_yamc):
    with discovery.transmutation_network("/chains/Fe56"):
        assert _settings(fake_yamc) == ["/chains/Fe56"] * 3
    assert _settings(fake_yamc) == [None] * 3


def test_settings_reset_when_the_solve_raises(fake_yamc):
    with pytest.raises(RuntimeError):
        with discovery.transmutation_network("endf-b8.1"):
            raise RuntimeError("solve failed")
    assert _settings(fake_yamc) == [None] * 3
