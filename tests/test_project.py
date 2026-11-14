"""The yamc project for vv: code names, libraries, models, records counted as done, commands.

Nothing here builds an environment or runs a code: the records are written by
the tests and the environments are stood in for.
"""
from pathlib import Path

import pytest
import vv
from vv.records import save_record

from yamc_verify import environments, plan, project, run_task, tasks
from yamc_verify.backends import resolve
from yamc_verify.project import PROJECT, YamcProject


@pytest.fixture
def libraries(tmp_path, monkeypatch):
    """endf-b8.1 has its OpenMC data; every other library is missing."""
    data = tmp_path / "endf"
    data.mkdir()
    (data / "cross_sections.xml").write_text("<cross_sections/>")
    dirs = {lib: str(tmp_path / f"absent-{lib}") for lib in tasks.NUCLEAR_DATA_DIR}
    dirs["endf-b8.1"] = str(data)
    monkeypatch.setattr(tasks, "NUCLEAR_DATA_DIR", dirs)
    return dirs


@pytest.fixture
def no_environments(monkeypatch):
    """No environment is built and every version is published."""
    monkeypatch.setattr(environments, "is_ready", lambda spec, **kw: False)
    monkeypatch.setattr(environments, "ready_identity", lambda spec, **kw: None)
    monkeypatch.setattr(environments, "check_available", lambda spec, **kw: None)


@pytest.mark.parametrize("spec, name, env", [
    ("openmc==0.16.1.dev0", "openmc", "openmc==0.16.1.dev0"),
    ("yamc-cpu==0.23.0", "yamc-cpu", "yamc==0.23.0"),
    (" yamc-gpu @ 86ef200e ", "yamc-gpu", "yamc@86ef200e"),
])
def test_codes_name_a_backend_and_an_exact_version(spec, name, env):
    code = PROJECT.parse_code(spec)
    assert code.name == name and code.spec == spec.replace(" ", "")
    assert str(project._environment_spec(code)) == env


def test_both_yamc_backends_of_one_version_share_an_environment():
    cpu, gpu = (project._environment_spec(PROJECT.parse_code(f"yamc-{b}==0.23.0")) for b in ("cpu", "gpu"))
    assert environments.env_dir(cpu) == environments.env_dir(gpu)


@pytest.mark.parametrize("spec", ["yamc", "yamc==0.23.0", "yamc-cpu", "yamc-cpu>=0.20", "openmc@abc123",
                                  "serpent==2", "openmc-cpu==0.16.1"])
def test_anything_else_is_rejected_naming_the_codes(spec):
    with pytest.raises(ValueError, match="openmc"):
        PROJECT.parse_code(spec)


def test_models_libraries_and_where_they_are_defined(libraries):
    models = PROJECT.models()
    assert models[:2] == ["neutron/sphere", "neutron/broomstick"] and len(models) == len(set(models))
    libs = PROJECT.libraries()
    assert libs["endf-b8.1"] is None
    assert "no cross_sections.xml" in libs["fendl-3.2d"]
    assert PROJECT.defined("neutron/sphere", "fendl-3.2d")
    assert not PROJECT.defined("neutron/tokamak", "fendl-3.2d")


def test_which_tasks_run_which_codes():
    gpu, cpu = PROJECT.parse_code("yamc-gpu==0.23.0"), PROJECT.parse_code("yamc-cpu==0.23.0")
    assert not PROJECT.runs("neutron/tokamak", "endf-b8.1", gpu)
    assert PROJECT.runs("neutron/tokamak", "endf-b8.1", cpu)
    # The flux-spectrum task has no transport: its yamc results have no backend label.
    flux = tasks.get_task("transmutation/material-flux-spectrum/endf-b8.1")
    assert PROJECT._label(flux, cpu) == "yamc" and PROJECT._label(flux, gpu) is None


def test_unknown_and_missing_names_stop_it_listing_what_is_available(libraries, no_environments):
    with pytest.raises(vv.SelectionError) as e:
        project.verification(models=["neutron/cube"], libraries=["fendl-3.2d"], codes=["yamc==0.23.0"])
    text = str(e.value)
    assert "unknown models ['neutron/cube']. Available: ['neutron/sphere'" in text
    assert "library fendl-3.2d is missing: no cross_sections.xml" in text and "Available: ['endf-b8.1']" in text
    assert "cannot read code 'yamc==0.23.0'" in text


def test_every_argument_is_required():
    with pytest.raises(TypeError):
        project.verification(models=["neutron/sphere"], libraries=["endf-b8.1"])
    with pytest.raises(TypeError):
        project.verification_plan(libraries=["endf-b8.1"], codes=["yamc-cpu==0.23.0"])


def _identity(name, version, commit, dirty=False, backend=None):
    from vv.records import build_id

    return {"name": name, "version": version, "commit": commit, "dirty": dirty, "binary_sha": None,
            "backend": backend, "build_id": build_id(version, commit, dirty, None)}


@pytest.fixture
def records(tmp_path, monkeypatch, libraries, no_environments):
    """Records under a stand-in repository, and discovery that finds Fe56 and H1."""
    monkeypatch.setattr(plan, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(plan, "discover_items", lambda task, with_openmc: ["Fe56", "H1"])

    def write(task_key, code, item, identity, elapsed=10.0):
        task = tasks.get_task(task_key)
        save_record(tmp_path / task.cache_dir, task.library, code, item, {"x": 1}, identity=identity,
                    settings={"library": task.library}, elapsed_seconds=elapsed)
    return write


def _plan(codes, models=("neutron/sphere",)):
    return project.verification_plan(models=list(models), libraries=["endf-b8.1"], codes=codes)


def test_a_yamc_release_counts_records_of_that_version(records):
    records("neutron/sphere/endf-b8.1", "yamc-cpu", "Fe56", _identity("yamc", "0.23.0", "a" * 40), elapsed=20.0)
    records("neutron/sphere/endf-b8.1", "yamc-cpu", "H1", _identity("yamc", "0.22.0", "b" * 40), elapsed=40.0)
    (row,) = _plan(["yamc-cpu==0.23.0"]).jobs[0].rows
    assert (row.label, row.items, row.done, row.seconds_per_item) == ("yamc-cpu", ["Fe56", "H1"], 1, 30.0)
    # A dirty build of that version does not count.
    records("neutron/sphere/endf-b8.1", "yamc-cpu", "H1", _identity("yamc", "0.23.0", "c" * 40, dirty=True))
    assert _plan(["yamc-cpu==0.23.0"]).jobs[0].rows[0].done == 1


def test_a_built_environment_pins_the_exact_build(records, monkeypatch):
    records("neutron/sphere/endf-b8.1", "yamc-cpu", "Fe56", _identity("yamc", "0.23.0", "a" * 40))
    records("neutron/sphere/endf-b8.1", "yamc-cpu", "H1", _identity("yamc", "0.23.0", "b" * 40))
    monkeypatch.setattr(environments, "ready_identity", lambda spec, **kw: {
        "commit": None, "yamc": {"version": "0.23.0", "commit": "a" * 40, "dirty": False}})
    assert _plan(["yamc-cpu==0.23.0"]).jobs[0].rows[0].done == 1


def test_openmc_references_count_from_any_build_except_for_photon_families(records, monkeypatch):
    omc = _identity("openmc", "0.15.3-dev707", "687bb4fa")
    records("neutron/sphere/endf-b8.1", "openmc", "Fe56", omc)
    records("photon/sphere/endf-b8.1", "openmc", "Fe56", omc)
    p = _plan(["openmc==0.16.1.dev0"], models=["neutron/sphere", "photon/sphere"])
    assert [job.rows[0].done for job in p.jobs] == [1, 0]
    # Once the environment is built, the photon reference from its OpenMC counts.
    monkeypatch.setattr(environments, "ready_identity", lambda spec, **kw: {
        "openmc": {"version": "0.16.1.dev0", "reports": "0.15.3-dev707", "commit": "687bb4fa"}})
    p = _plan(["openmc==0.16.1.dev0"], models=["neutron/sphere", "photon/sphere"])
    assert [job.rows[0].done for job in p.jobs] == [1, 1]


def test_secondary_photon_rows_are_per_electron_treatment(records):
    rows = _plan(["yamc-cpu==0.23.0"], models=["secondary-photon/sphere"]).jobs[0].rows
    assert [r.label for r in rows] == ["yamc-cpu-led", "yamc-cpu-ttb"]
    assert [r.name for r in rows] == ["yamc-cpu==0.23.0 (yamc-cpu-led)", "yamc-cpu==0.23.0 (yamc-cpu-ttb)"]


def test_codes_a_task_does_not_run_are_listed_not_run(records):
    p = _plan(["yamc-gpu==0.23.0", "yamc-cpu==0.23.0"], models=["neutron/tokamak"])
    assert [j.code.spec for j in p.jobs] == ["yamc-cpu==0.23.0"]
    assert p.skipped == ["neutron/tokamak on endf-b8.1 does not run yamc-gpu"]


def test_the_command_runs_the_task_in_the_code_environment(tmp_path):
    python = tmp_path / "env" / "bin" / "python"
    yamc, omc = PROJECT.parse_code("yamc-gpu==0.23.0"), PROJECT.parse_code("openmc==0.16.1.dev0")
    job = vv.Job("neutron/sphere", "endf-b8.1", yamc, [])
    cmd = PROJECT.command(job, [omc, yamc], python)
    assert cmd.argv == [str(python), "-m", "yamc_verify.run_task", "neutron/sphere/endf-b8.1",
                        "--discover-with-openmc"]
    assert cmd.env["MC_BACKENDS"] == "yamc-gpu"
    assert cmd.env["PATH"].startswith(str(python.parent))
    assert str(project.VV_SRC) in cmd.env["PYTHONPATH"] and Path(cmd.cwd) == project.REPO_ROOT
    # Without OpenMC in the verification, yamc finds its items in its own data.
    assert "--discover-with-openmc" not in PROJECT.command(job, [yamc], python).argv
    omc_cmd = PROJECT.command(vv.Job("neutron/sphere", "endf-b8.1", omc, []), [omc, yamc], python)
    assert omc_cmd.env["MC_BACKENDS"] == "openmc-cpu" and "--discover-with-openmc" not in omc_cmd.argv


def test_run_task_can_discover_with_openmc_without_running_it(monkeypatch):
    seen = []
    monkeypatch.setattr(run_task, "run_task", lambda task, b: seen.append((task.key, b)))
    monkeypatch.setenv("MC_BACKENDS", "yamc-cpu")
    run_task.main(["neutron/sphere/endf-b8.1", "--discover-with-openmc"])
    run_task.main(["neutron/sphere/endf-b8.1"])
    (_, with_flag), (_, without) = seen
    assert with_flag.discover_with_openmc and not with_flag.run_openmc
    assert not without.discover_with_openmc


def test_running_openmc_always_discovers_with_openmc():
    assert resolve("openmc-cpu").discover_with_openmc
    assert not resolve("yamc-cpu").discover_with_openmc


def test_a_custom_core_checkout_is_passed_to_the_environment():
    code = YamcProject(core_dir="/somewhere/core").parse_code("yamc-cpu@86ef200e")
    assert code.environment.keywords == {"core_dir": "/somewhere/core"}


def test_the_fixed_geometry_models_plan_their_cells():
    from yamc_verify.families import multimaterial_slab, tokamak

    assert plan.discover_items(tasks.get_task("neutron/tokamak/endf-b8.1"), True) == list(tokamak.TALLY_CELLS)
    slab = tasks.get_task("neutron/multimaterial-slab/endf-b8.1")
    assert plan.discover_items(slab, False) == list(multimaterial_slab.TALLY_LAYERS)
