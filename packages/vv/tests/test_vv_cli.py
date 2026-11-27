"""The vv command finds its project through vv.toml and requires every option."""
import sys
import textwrap

import pytest
from vv import cli

PROJECT = textwrap.dedent('''
    import sys
    import vv

    class P(vv.Project):
        name = "cli verification"
        def models(self): return ["sphere"]
        def libraries(self): return {"lib-a": None}
        def defined(self, model, library): return True
        def parse_code(self, spec):
            if spec != "code-a==1.0":
                raise ValueError(f"cannot read code {spec!r}")
            return vv.Code(spec, "code-a", lambda: sys.executable, lambda: True)
        def rows(self, model, library, code, codes):
            return [vv.Row("code-a", ["Fe56"], done=1, seconds_per_item=5.0)]
        def command(self, job, codes, python):
            return vv.Command([str(python), "-c", "pass"])

    PROJECT = P()
''')


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "cliproject.py").write_text(PROJECT)
    (tmp_path / "vv.toml").write_text('project = "cliproject:PROJECT"\n')
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")  # found from a subfolder too
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "cliproject", raising=False)
    return tmp_path


def test_plan_prints_the_plan(repo, capsys):
    assert cli.main(["plan", "--models", "all", "--libraries", "lib-a", "--codes", "code-a==1.0"]) == 0
    out = capsys.readouterr().out
    assert "code-a==1.0  environment ready" in out and "1/1 cached" in out


def test_run_runs_it(repo, capsys):
    assert cli.main(["run", "--models", "sphere", "--libraries", "lib-a", "--codes", "code-a==1.0"]) == 0
    assert "all 1 job(s) completed" in capsys.readouterr().out


def test_a_bad_request_exits_2_with_the_problems(repo, capsys):
    assert cli.main(["plan", "--models", "cube", "--libraries", "lib-a", "--codes", "code-a==1.0"]) == 2
    assert "unknown models ['cube']. Available: ['sphere']" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["--models", "--libraries", "--codes"])
def test_every_option_is_required(repo, missing):
    args = {"--models": ["sphere"], "--libraries": ["lib-a"], "--codes": ["code-a==1.0"]}
    argv = ["plan"] + [x for k, v in args.items() if k != missing for x in (k, *v)]
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2


def test_no_vv_toml_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "CONFIG", "vv-test-absent.toml")
    assert cli.main(["plan", "--models", "a", "--libraries", "b", "--codes", "c"]) == 2
    assert "no vv-test-absent.toml" in capsys.readouterr().err


def test_find_config_walks_up(tmp_path):
    (tmp_path / "vv.toml").write_text('project = "x:Y"\n')
    (tmp_path / "a" / "b").mkdir(parents=True)
    assert cli.find_config(tmp_path / "a" / "b") == tmp_path / "vv.toml"


def test_a_malformed_project_reference_is_rejected(tmp_path):
    (tmp_path / "vv.toml").write_text('project = "no-colon"\n')
    with pytest.raises(ValueError, match="module:attribute"):
        cli.load_project(tmp_path / "vv.toml")


def test_release_with_nothing_published_does_nothing(repo, capsys):
    assert cli.main(["release"]) == 0
    assert capsys.readouterr().out == ""
