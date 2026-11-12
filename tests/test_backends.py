"""MC_BACKENDS has no default: the caller must name the backends."""
import pytest

from yamc_verify import backends


@pytest.mark.parametrize("raw", ["", "  ", ",", " , "])
def test_empty_selection_is_rejected(raw):
    with pytest.raises(ValueError, match="MC_BACKENDS is not set"):
        backends.resolve(raw)


def test_unset_env_var_is_rejected(monkeypatch):
    monkeypatch.delenv("MC_BACKENDS", raising=False)
    with pytest.raises(ValueError, match="MC_BACKENDS is not set"):
        backends.resolve()


def test_env_var_is_read(monkeypatch):
    monkeypatch.setenv("MC_BACKENDS", "yamc-gpu,openmc-cpu")
    b = backends.resolve()
    assert b.tokens == ["yamc-gpu", "openmc-cpu"]
    assert b.run_openmc
    assert b.computes == ["gpu"]
    assert b.codes == ["openmc", "yamc-gpu"]


def test_unknown_token_is_rejected():
    with pytest.raises(ValueError, match="unknown token"):
        backends.resolve("openmc-cpu,serpent")
