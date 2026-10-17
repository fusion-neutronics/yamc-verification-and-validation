"""yamc's transmutation network points at a chain only for the call that needs it.

yamc takes no per-call chain, so the transmutation families set its global
decay, reactions and fission-yield settings around a solve. A setting left
behind would hand the next nuclide's run the previous one's reduced chain
without any error, so these check the reset, including when the solve raises.
A stand-in ``yamc`` records the settings, so they need neither yamc nor nuclear
data.
"""
import os
import sys
import tempfile
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


class _Chain:
    """Writes the subsection folders ``TransmutationChain.export_to_arrow`` does."""

    def export_to_arrow(self, path):
        for sub in ("decay", "reactions", "fission_yields"):
            os.makedirs(os.path.join(path, sub))


def test_each_export_gets_its_own_root_removed_after(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with discovery.exported_chain(_Chain(), "Fe56") as first:
        assert os.path.isdir(os.path.join(first, "decay"))
        # yamc caches a parsed chain by path, so a second export must not reuse it.
        with discovery.exported_chain(_Chain(), "Fe56") as second:
            assert second != first
    assert not os.path.exists(first)
    assert not os.path.exists(second)
