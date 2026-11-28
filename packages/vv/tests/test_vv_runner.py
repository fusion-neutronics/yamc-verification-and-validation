"""Checking, planning and running a verification through a stand-in project."""
import io
import sys
from pathlib import Path

import pytest
import vv
from vv import dashboard
from vv.runner import apply_event, build_state


class FakeProject(vv.Project):
    """Two models on two libraries; ``slab`` is not defined on ``lib-b``; ``lib-c`` has no data."""

    name = "fake verification"

    def __init__(self, tmp_path, *, script="pass", ready=False, done=None, seconds=None):
        self.tmp_path = Path(tmp_path)
        self.script = script
        self.ready = ready
        self.done = done or {}
        self.seconds = seconds
        self.built = []

    def models(self):
        return ["sphere", "slab"]

    def libraries(self):
        return {"lib-a": None, "lib-b": None, "lib-c": "no data in /nowhere"}

    def defined(self, model, library):
        return not (model == "slab" and library == "lib-b")

    def parse_code(self, spec):
        name, sep, version = spec.partition("==")
        if name not in ("code-a", "code-b") or not version:
            raise ValueError(f"cannot read code {spec!r}; codes: ['code-a', 'code-b']")
        if version == "9.9":
            def check():
                raise ValueError(f"{spec}: version 9.9 is not published; available: ['1.0']")
        else:
            check = None

        def environment():
            self.built.append(spec)
            return sys.executable

        return vv.Code(spec=spec, name=name, environment=environment, is_ready=lambda: self.ready,
                       check=check, build_seconds=60.0)

    def runs(self, model, library, code):
        return not (model == "slab" and code.name == "code-b")

    def rows(self, model, library, code, codes):
        return [vv.Row(label=code.name, items=["Fe56", "H1"], done=self.done.get((model, code.name), 0),
                       seconds_per_item=self.seconds)]

    def command(self, job, codes, python):
        return vv.Command([str(python), "-c", self.script],
                          env={"FAKE_ITEMS": "Fe56,H1", "FAKE_CODE": job.rows[0].label}, cwd=self.tmp_path)


def _problems(excinfo):
    return str(excinfo.value)


@pytest.mark.parametrize("models, libraries, codes", [
    (None, ["lib-a"], ["code-a==1.0"]),
    (["sphere"], None, ["code-a==1.0"]),
    (["sphere"], ["lib-a"], None),
    (["sphere"], ["lib-a"], []),
    ("sphere", ["lib-a"], ["code-a==1.0"]),
])
def test_every_argument_is_required(tmp_path, models, libraries, codes):
    with pytest.raises(vv.SelectionError, match="Available|name each code"):
        vv.plan(FakeProject(tmp_path), models=models, libraries=libraries, codes=codes)


def test_unknown_names_list_what_is_available(tmp_path):
    with pytest.raises(vv.SelectionError) as e:
        vv.plan(FakeProject(tmp_path), models=["sphere", "cube"], libraries=["lib-z"], codes=["serpent==2"])
    text = _problems(e)
    assert "unknown models ['cube']. Available: ['sphere', 'slab']" in text
    assert "unknown libraries ['lib-z']. Available: ['lib-a', 'lib-b', 'lib-c']" in text
    assert "cannot read code 'serpent==2'" in text


def test_a_missing_library_says_why_and_what_is_available(tmp_path):
    with pytest.raises(vv.SelectionError) as e:
        vv.plan(FakeProject(tmp_path), models=["sphere"], libraries=["lib-c"], codes=["code-a==1.0"])
    assert "library lib-c is missing: no data in /nowhere. Available: ['lib-a', 'lib-b']" in _problems(e)


def test_a_code_that_cannot_be_installed_stops_it_unless_already_built(tmp_path):
    with pytest.raises(vv.SelectionError, match="9.9 is not published; available"):
        vv.plan(FakeProject(tmp_path), models=["sphere"], libraries=["lib-a"], codes=["code-a==9.9"])
    vv.plan(FakeProject(tmp_path, ready=True), models=["sphere"], libraries=["lib-a"], codes=["code-a==9.9"])


def test_a_code_named_twice_is_an_error(tmp_path):
    with pytest.raises(vv.SelectionError, match="named twice"):
        vv.plan(FakeProject(tmp_path), models=["sphere"], libraries=["lib-a"], codes=["code-a==1.0"] * 2)


def test_a_model_on_none_of_the_libraries_is_an_error(tmp_path):
    with pytest.raises(vv.SelectionError, match=r"model slab is not defined on \['lib-b'\]; it is defined on"):
        vv.plan(FakeProject(tmp_path), models=["slab"], libraries=["lib-b"], codes=["code-a==1.0"])


def test_the_plan_lists_jobs_skips_and_an_estimate(tmp_path):
    project = FakeProject(tmp_path, done={("sphere", "code-a"): 2}, seconds=30.0)
    p = vv.plan(project, models=vv.ALL, libraries=["lib-a", "lib-b"], codes=["code-a==1.0", "code-b==2.0"])
    assert [(j.model, j.library, j.code.spec) for j in p.jobs] == [
        ("sphere", "lib-a", "code-a==1.0"), ("sphere", "lib-a", "code-b==2.0"),
        ("sphere", "lib-b", "code-a==1.0"), ("sphere", "lib-b", "code-b==2.0"),
        ("slab", "lib-a", "code-a==1.0")]
    assert "slab is not defined on lib-b" in p.skipped
    assert "slab on lib-a does not run code-b" in p.skipped
    # 10 items, 4 done (sphere code-a on both libraries), 30 s each, plus two environments.
    assert p.estimate() == (6 * 30.0 + 2 * 60.0, 0)
    text = p.format()
    assert "code-a==1.0  environment to build (about 60 secs)" in text
    assert "    code-a==1.0 - 2/2 cached" in text
    assert "    code-b==2.0 - 2/2 todo" in text
    assert "not run:" in text and "estimate: about 5 mins, including building 2 environment(s)" in text
    assert project.built == []  # planning builds nothing


def test_an_estimate_without_timings_is_a_lower_bound(tmp_path):
    p = vv.plan(FakeProject(tmp_path, ready=True), models=["sphere"], libraries=["lib-a"], codes=["code-a==1.0"])
    assert p.estimate() == (0.0, 1)
    assert "lower bound" in p.format()


def test_events_correct_the_plan(tmp_path):
    p = vv.plan(FakeProject(tmp_path, done={("sphere", "code-a"): 1}), models=["sphere"], libraries=["lib-a"],
                codes=["code-a==1.0"])
    row = build_state(p.jobs)["tasks"][0]["codes"]["code-a==1.0"]
    rows = {"code-a": row}
    apply_event(rows, {"t": "plan", "items": ["Fe56", "H1", "Li6"]})
    assert row["total"] == 3
    apply_event(rows, {"t": "item_start", "code": "code-a", "key": "Fe56"})
    assert row["status"] == "running" and row["running_key"] == "Fe56"
    apply_event(rows, {"t": "item", "code": "code-a", "key": "Fe56", "status": "computed", "seconds": 2.0})
    apply_event(rows, {"t": "item", "code": "other", "key": "Fe56", "status": "computed"})  # not this job's
    assert (row["computed"], row["obs_times"], row["running_key"]) == (2 - 1, [2.0], None)


EMIT = """
import os, sys
from vv import progress
items = os.environ["FAKE_ITEMS"].split(",")
progress.report_plan(items, kind="nuclide")
for item in items:
    progress.report_item_start(os.environ["FAKE_CODE"], item)
    progress.report_item(os.environ["FAKE_CODE"], item, status="computed", seconds=0.01)
print("ran in", os.getcwd())
"""


def test_run_builds_every_environment_first_then_runs_each_job(tmp_path):
    project = FakeProject(tmp_path, script=EMIT)
    out = io.StringIO()
    result = vv.run(project, models=["sphere"], libraries=["lib-a"], codes=["code-a==1.0", "code-b==2.0"],
                    stream=out)
    assert result.ok and len(result.plan.jobs) == 2
    assert project.built == ["code-a==1.0", "code-b==2.0"]
    text = out.getvalue()
    assert "building the environment for code-a==1.0" in text
    assert "fake verification: code-a==1.0, code-b==2.0" in text
    assert "code-a==1.0 - 2/2 completed" in text and "code-b==2.0 - 2/2 completed" in text
    assert "all 2 job(s) completed" in text


def test_a_failed_job_is_reported_with_its_log(tmp_path):
    project = FakeProject(tmp_path, script="import sys; print('boom'); sys.exit(3)")
    out = io.StringIO()
    result = vv.run(project, models=["sphere"], libraries=["lib-a"], codes=["code-a==1.0"], stream=out)
    assert not result.ok
    job, log = result.failures[0]
    assert job.code.spec == "code-a==1.0" and log.read_text().strip() == "boom"
    assert "1 job(s) FAILED" in out.getvalue()


def test_a_bad_request_runs_nothing(tmp_path):
    project = FakeProject(tmp_path, script=EMIT)
    with pytest.raises(vv.SelectionError):
        vv.run(project, models=["sphere"], libraries=["lib-c"], codes=["code-a==1.0"], stream=io.StringIO())
    assert project.built == []


class SharedProject(FakeProject):
    """code-a and code-b of one version run in one environment."""

    def parse_code(self, spec):
        import dataclasses

        code = super().parse_code(spec)
        return dataclasses.replace(code, environment_key="env-" + spec.partition("==")[2])


def test_codes_sharing_an_environment_build_and_count_it_once(tmp_path):
    project = SharedProject(tmp_path, script=EMIT, seconds=1.0)
    p = vv.plan(project, models=["sphere"], libraries=["lib-a"], codes=["code-a==1.0", "code-b==1.0"])
    assert [c.key for c in p.environments_to_build()] == ["env-1.0"]
    assert p.estimate() == (4 * 1.0 + 60.0, 0)  # four items to run, one environment
    assert "shared as env-1.0" in p.format()
    assert vv.run(project, models=["sphere"], libraries=["lib-a"], codes=["code-a==1.0", "code-b==1.0"],
                  stream=io.StringIO()).ok
    assert project.built == ["code-a==1.0"]


# --- the display off a terminal ------------------------------------------------


def _state(rows):
    return {"header": ["fake verification: code-a==1.0"], "tasks": [{
        "label": "sphere lib-a", "status": "running", "t_start": 0.0, "t_end": None,
        "codes": {name: {"total": 2, "cached": cached, "computed": computed, "running_key": None,
                         "status": status, "t_start": 0.0, "t_end": 5.0 if status == "done" else None,
                         "obs_times": [], "seed_avg": None}
                  for name, (status, cached, computed) in rows.items()}}]}


def test_off_a_terminal_the_display_writes_events_not_snapshots():
    out = io.StringIO()
    display = dashboard.LiveDisplay(out, tty=False, summary_seconds=60)
    display.update(_state({"code-a==1.0": ("queued", 0, 0), "code-b==2.0": ("queued", 2, 0)}), 0.0)
    plan = out.getvalue()
    assert "fake verification: code-a==1.0" in plan and "    code-a==1.0 - 2/2 todo" in plan
    for second in range(1, 30):  # nothing happens: nothing is written
        display.update(_state({"code-a==1.0": ("queued", 0, 0), "code-b==2.0": ("queued", 2, 0)}), float(second))
    assert out.getvalue() == plan
    display.update(_state({"code-a==1.0": ("running", 0, 1), "code-b==2.0": ("queued", 2, 0)}), 30.0)
    display.update(_state({"code-a==1.0": ("done", 0, 2), "code-b==2.0": ("done", 2, 0)}), 31.0)
    events = out.getvalue()[len(plan):].splitlines()
    assert events == ["sphere lib-a: code-a==1.0 started",
                      "sphere lib-a: code-a==1.0 - 2/2 completed 5 secs",
                      "sphere lib-a: code-b==2.0 - 2/2 cached"]


def test_off_a_terminal_a_quiet_stretch_gets_a_summary_line_at_most_every_interval():
    out = io.StringIO()
    display = dashboard.LiveDisplay(out, tty=False, summary_seconds=60)
    running = _state({"code-a==1.0": ("running", 0, 1)})
    display.update(running, 0.0)
    display.update(running, 1.0)  # the start event
    before = len(out.getvalue().splitlines())
    for second in range(2, 200):
        display.update(running, float(second))
    summaries = out.getvalue().splitlines()[before:]
    assert summaries == ["0/2 cached, 1/2 completed"] * 3  # at 61, 121 and 181 s


def test_on_a_terminal_every_update_redraws_in_place():
    out = io.StringIO()
    display = dashboard.LiveDisplay(out, tty=True)
    state = _state({"code-a==1.0": ("queued", 0, 0)})
    display.update(state, 0.0)
    display.update(state, 1.0)
    assert out.getvalue().count("\033[2K" + "fake verification: code-a==1.0") == 2 and "\033[" in out.getvalue()


SLOW = EMIT.replace('print("ran in", os.getcwd())', 'import time; time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))')


def test_a_run_writes_as_many_lines_off_a_terminal_however_long_it_takes(tmp_path, monkeypatch):
    lengths = []
    for sleep in ("0", "2.5"):
        monkeypatch.setenv("FAKE_SLEEP", sleep)
        out = io.StringIO()
        result = vv.run(FakeProject(tmp_path, script=SLOW), models=["sphere"], libraries=["lib-a"],
                        codes=["code-a==1.0", "code-b==2.0"], stream=out)
        assert result.ok and "all 2 job(s) completed" in out.getvalue()
        lengths.append(len(out.getvalue().splitlines()))
    assert lengths[0] == lengths[1]
