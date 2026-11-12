"""Code specs and environment checks (no environment is built here)."""
import pytest

from yamc_verify import environments as envs
from yamc_verify.environments import CodeSpec, parse_code


@pytest.mark.parametrize("spec, expected", [
    ("openmc==0.16.1.dev0", CodeSpec("openmc", "release", "0.16.1.dev0")),
    ("yamc==0.21.0", CodeSpec("yamc", "release", "0.21.0")),
    (" yamc @ 86ef200e ", CodeSpec("yamc", "commit", "86ef200e")),
])
def test_parse(spec, expected):
    assert parse_code(spec) == expected
    assert parse_code(str(expected)) == expected


@pytest.mark.parametrize("spec", ["yamc", "openmc", "yamc>=0.20", "serpent==2", "openmc@abc123"])
def test_parse_rejects_anything_without_an_exact_version(spec):
    with pytest.raises(ValueError):
        parse_code(spec)


def test_env_dirs_name_the_version_or_commit():
    assert envs.env_dir(parse_code("yamc==0.21.0")).name == "yamc-0.21.0"
    assert envs.env_dir(parse_code("yamc@86ef"), "86ef200e6a0e497a").name == "yamc-g86ef200e6a0e"


def test_base_requirements_drop_both_codes_and_the_index_lines():
    reqs = envs._base_requirements()
    assert reqs and not any(r.startswith(("openmc", "yamc-core", "yamc", "-")) and not r.startswith("openmc_data")
                            for r in reqs)
    assert any(r.startswith("openmc_data") for r in reqs)
    assert envs._index_args()[0] == "--extra-index-url"


def test_check_rejects_the_wrong_version_or_a_dirty_or_other_commit():
    ident = {"yamc": {"version": "0.21.0", "commit": "aaa", "dirty": False},
             "openmc": {"version": "0.16.1.dev0", "commit": "x"}}
    envs._check(parse_code("yamc==0.21.0"), ident, None)
    envs._check(parse_code("openmc==0.16.1.dev0"), ident, None)
    envs._check(parse_code("yamc@aaa"), ident, "aaa")
    with pytest.raises(RuntimeError, match="0.21.0"):
        envs._check(parse_code("yamc==0.20.0"), ident, None)
    with pytest.raises(RuntimeError, match="reports commit"):
        envs._check(parse_code("yamc@bbb"), ident, "bbb")
    with pytest.raises(RuntimeError, match="uncommitted"):
        envs._check(parse_code("yamc@aaa"), {**ident, "yamc": {**ident["yamc"], "dirty": True}}, "aaa")
