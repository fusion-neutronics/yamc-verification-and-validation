"""``verification()``: run named yamc and OpenMC versions over models and libraries.

This is the yamc project for the shared ``vv`` package: it says which models
and libraries exist, how a code is named, and how its environment is built (one
per code version, :mod:`yamc_verify.environments`). ``vv plan`` and ``vv run``
find it through ``vv.toml`` at the repository root.

::

    from yamc_verify import verification

    verification(models=["neutron/sphere"], libraries=["endf-b8.1"],
                 codes=["openmc==0.16.1.dev0", "yamc-cpu==0.23.0"])

Every argument is required. ``models`` is a list of model names (a task key
without its library, such as ``neutron/sphere``) or ``"all"``. A code is one of
:data:`CODE_NAMES` with an exact version or commit:

* ``openmc==<version>``: an OpenMC wheel from the wheels index in requirements.txt;
* ``yamc-cpu==<version>`` / ``yamc-gpu==<version>``: a yamc-core release from
  PyPI, run on that backend;
* ``yamc-cpu@<commit>`` / ``yamc-gpu@<commit>``: yamc-core built at that commit
  of a local core checkout (``$YAMC_CORE_ROOT``, else ``../core``).

A library is available when its OpenMC data (``cross_sections.xml``) is where
``YAMC_VERIFY_NUCLEAR_DATA`` (see :func:`yamc_verify.tasks.nuclear_data_dir`) says. Which items run is chosen
as before, by ``YAMC_VERIFY_NUCLIDES``, ``YAMC_VERIFY_ELEMENTS`` and
``YAMC_VERIFY_NUMBER_OF_SIMULATIONS``.
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Sequence

import vv
from vv.records import build_id, cached_items, find_record

from cache_utils import accept_old_cache

from . import environments, plan, tasks
from .tasks import OPENMC_BUILD_KEYED_FAMILIES

REPO_ROOT = Path(plan.REPO_ROOT)
VV_SRC = REPO_ROOT / "packages" / "vv" / "src"

CODE_NAMES = ("openmc", "yamc-cpu", "yamc-gpu")
_BACKEND_TOKEN = {"openmc": "openmc-cpu", "yamc-cpu": "yamc-cpu", "yamc-gpu": "yamc-gpu"}
# Rough times to build an environment: installing wheels, or compiling yamc-core.
_BUILD_SECONDS = {"release": 300.0, "commit": 1800.0}
_CODE_RE = re.compile(r"\s*(" + "|".join(CODE_NAMES) + r")\s*(==|@)\s*([A-Za-z0-9._+-]+)\s*")


def _environment_spec(code: vv.Code) -> environments.CodeSpec:
    """The environment a code runs in: yamc-cpu and yamc-gpu of one version share it."""
    m = _CODE_RE.fullmatch(code.spec)
    name, op, ref = m.groups()
    return environments.parse_code(f"{name.split('-', 1)[0]}{op}{ref}")


class YamcProject(vv.Project):
    """The models, libraries and codes of yamc verification, for :mod:`vv`."""

    name = "yamc verification"
    root = REPO_ROOT

    def __init__(self, core_dir: Path | str | None = None):
        self.core_dir = core_dir

    def models(self) -> list[str]:
        """Every task key without its library, in registry order."""
        out: list[str] = []
        for task in tasks.all_tasks():
            model = task.key.rsplit("/", 1)[0]
            if model not in out:
                out.append(model)
        return out

    def libraries(self) -> dict[str, str | None]:
        """Every library a task uses, mapped to why it is missing (``None`` if present)."""
        out: dict[str, str | None] = {}
        for task in tasks.all_tasks():
            if task.library in out:
                continue
            folder = tasks.nuclear_data_dir(task.library)
            if folder is None:
                out[task.library] = (f"no folder known; set {tasks.library_env(task.library)} "
                                     "or add it to NUCLEAR_DATA_SUBDIRS in yamc_verify/tasks.py")
            elif not os.path.isfile(os.path.join(folder, "cross_sections.xml")):
                out[task.library] = f"no cross_sections.xml in {folder}"
            else:
                out[task.library] = None
        return out

    def defined(self, model: str, library: str) -> bool:
        """Whether a task is registered for ``model`` on ``library``."""
        return any(t.key == f"{model}/{library}" for t in tasks.all_tasks())

    def parse_code(self, spec: str) -> vv.Code:
        """Read ``openmc==X``, ``yamc-cpu==X``, ``yamc-gpu==X`` or ``yamc-<backend>@<commit>``."""
        m = _CODE_RE.fullmatch(str(spec))
        if m is None:
            raise ValueError(
                f"cannot read code {spec!r}: name one of {list(CODE_NAMES)} with an exact "
                "version or commit, e.g. 'openmc==0.16.1.dev0', 'yamc-cpu==0.23.0' or "
                "'yamc-gpu@86ef200e'"
            )
        name, op, ref = m.groups()
        if name == "openmc" and op == "@":
            raise ValueError(f"{spec!r}: building OpenMC from a commit is not supported; use openmc==<version>")
        env = environments.parse_code(f"{name.split('-', 1)[0]}{op}{ref}")
        kw = {"core_dir": self.core_dir}
        return vv.Code(
            spec=f"{name}{op}{ref}",
            name=name,
            environment=functools.partial(environments.python_for, env, **kw),
            is_ready=functools.partial(environments.is_ready, env, **kw),
            check=functools.partial(self._check, f"{name}{op}{ref}", env),
            build_seconds=_BUILD_SECONDS[env.kind],
            environment_key=str(env),
        )

    def _check(self, spec: str, env: environments.CodeSpec) -> None:
        """Raise ``ValueError`` naming ``spec`` if its version or commit cannot be installed."""
        try:
            environments.check_available(env, core_dir=self.core_dir)
        except ValueError as e:
            raise ValueError(f"{spec}: {e}") from None

    def _label(self, task: tasks.Task, code: vv.Code) -> str | None:
        """The label ``task`` files ``code``'s results under, or ``None`` if it does not run it."""
        if code.name in task.codes:
            return code.name
        if code.name == "yamc-cpu" and "yamc" in task.codes:
            return "yamc"  # a task with no transport files its yamc results without a backend
        return None

    def runs(self, model: str, library: str, code: vv.Code) -> bool:
        """Whether the task runs ``code`` at all (the tokamak has no GPU run, say)."""
        return self._label(tasks.get_task(f"{model}/{library}"), code) is not None

    def _find(self, task: tasks.Task, code: vv.Code):
        """``find(label)`` for :func:`plan.task_rows`: which records count as done for ``code``."""
        env = _environment_spec(code)
        if accept_old_cache():
            return lambda label: {}
        if code.name == "openmc" and task.family not in OPENMC_BUILD_KEYED_FAMILIES:
            return lambda label: {}  # any OpenMC build's reference with matching settings
        ident = environments.ready_identity(env, core_dir=self.core_dir)
        if code.name == "openmc":
            # The record stores what the executable reports, known once the environment is built.
            omc = (ident or {}).get("openmc") or {}
            if not omc.get("reports") or not omc.get("commit"):
                return lambda label: {"accept": lambda record: False}
            return lambda label: {"build_id": build_id(omc["reports"], omc["commit"], None, None)}
        if env.kind == "release":
            if ident is not None and ident["yamc"].get("commit"):
                y = ident["yamc"]
                return lambda label: {"build_id": build_id(y["version"], y["commit"], y["dirty"], None)}

            def accept(record):
                ci = record.get("code_identity", {})
                return ci.get("version") == env.ref and not ci.get("dirty")
            return lambda label: {"accept": accept}
        commit = (ident or {}).get("commit") or environments.resolve_commit(
            environments._core_dir(self.core_dir), env.ref)

        def accept_commit(record):
            ci = record.get("code_identity", {})
            return ci.get("commit") == commit and not ci.get("dirty")
        return lambda label: {"accept": accept_commit}

    def _accepts(self, task: tasks.Task, code: vv.Code):
        """Which stored records belong to ``code``, judged from the records alone, or ``None`` for any.

        yamc: its version, or its commit. OpenMC: any build's reference with
        matching settings, except the families whose references must come from
        the named OpenMC version (see :data:`OPENMC_BUILD_KEYED_FAMILIES`).
        """
        env = _environment_spec(code)
        if code.name == "openmc":
            if task.family not in OPENMC_BUILD_KEYED_FAMILIES:
                return None
            return lambda r: _same_version(r.get("code_identity", {}).get("version"), env.ref)
        if env.kind == "release":
            return lambda r: r.get("code_identity", {}).get("version") == env.ref
        commit = env.ref if re.fullmatch(r"[0-9a-f]{7,40}", env.ref) else environments.resolve_commit(
            environments._core_dir(self.core_dir), env.ref)
        return lambda r: (r.get("code_identity", {}).get("commit") or "").startswith(commit)

    def records(self, model: str, library: str, code: vv.Code) -> list[Path]:
        """The newest record of each stored item that belongs to ``code`` (see :meth:`_accepts`)."""
        task = tasks.get_task(f"{model}/{library}")
        cache_dir = Path(self.root) / task.cache_dir
        accept = self._accepts(task, code)
        out = []
        for label in plan.real_codes([self._label(task, code)], task):
            for item in cached_items(cache_dir, library, label):
                found = find_record(cache_dir, library, label, item, settings={"library": library}, accept=accept)
                if found is not None:
                    out.append(found[0])
        return out

    def rows(self, model: str, library: str, code: vv.Code, codes: Sequence[vv.Code]) -> list[vv.Row]:
        """The items of one task for ``code``, how many have a record it accepts, and earlier times."""
        task = tasks.get_task(f"{model}/{library}")
        with_openmc = any(c.name == "openmc" for c in codes)
        items = plan.discover_items(task, with_openmc)
        labels = plan.real_codes([self._label(task, code)], task)
        return plan.task_rows(task, labels, items, self._find(task, code))

    def command(self, job: vv.Job, codes: Sequence[vv.Code], python: Path) -> vv.Command:
        """``python -m yamc_verify.run_task <task>`` in the code's environment, on its backend."""
        argv = [str(python), "-m", "yamc_verify.run_task", f"{job.model}/{job.library}"]
        if job.code.name != "openmc" and any(c.name == "openmc" for c in codes):
            argv.append("--discover-with-openmc")
        bin_dir = Path(python).parent
        env = {
            "MC_BACKENDS": _BACKEND_TOKEN[job.code.name],
            # The environment's own openmc executable is the one OpenMC runs and records.
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "VIRTUAL_ENV": str(bin_dir.parent),
            # The shared package and this repository come from the checkout, not the environment.
            "PYTHONPATH": os.pathsep.join([str(VV_SRC), str(REPO_ROOT)]),
        }
        return vv.Command(argv, env, cwd=REPO_ROOT)


def _same_version(reported, version: str) -> bool:
    """Whether the version an executable reports (``0.16.1-dev0``) is the package version (``0.16.1.dev0``)."""
    if reported is None:
        return False
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return str(reported) == version
    try:
        return Version(str(reported)) == Version(version)
    except InvalidVersion:
        return str(reported) == version


PROJECT = YamcProject()
"""The project ``vv.toml`` names, with the core checkout from ``$YAMC_CORE_ROOT`` or ``../core``."""


def verification(*, models, libraries, codes) -> vv.RunResult:
    """Run the models on the libraries with each code, in its own environment, with live progress.

    Every argument is required: ``models`` (a list of names, or ``"all"``),
    ``libraries`` and ``codes`` (lists, see the module docstring). Anything
    unknown or missing stops it before anything runs, listing what is
    available. Returns the plan it ran and the jobs that failed.
    """
    return vv.run(PROJECT, models=models, libraries=libraries, codes=codes)


def verification_plan(*, models, libraries, codes) -> vv.Plan:
    """What :func:`verification` would run, what already has records and a rough time.

    Takes the same required arguments; builds no environment and runs nothing.
    """
    return vv.plan(PROJECT, models=models, libraries=libraries, codes=codes)
