"""One Python environment per code version, so two builds can be compared.

A code is named explicitly, with no default version:

* ``openmc==0.16.1.dev0``: an OpenMC wheel from the wheels index in
  ``requirements.txt`` (``--extra-index-url``). Only the versions published
  there can be installed; asking for another lists them.
* ``yamc==0.21.0``: the ``yamc-core`` release from PyPI.
* ``yamc@<commit>``: ``yamc-core`` built from that commit of a local core
  checkout (``core_dir``, else ``$YAMC_CORE_ROOT``, else ``../core``).

Each environment lives under ``~/.cache/yamc-verification/envs/`` and holds the
verification dependencies from ``requirements.txt``, the requested code at the
requested version, and the other code at its latest release. The verification
tools use the other code only to find nuclear data; results are recorded only
for the code the environment was made for.

An environment is checked after it is built (the version, and for a commit the
commit the build reports and that it is clean) and only then marked ready, so a
half-built one is rebuilt rather than used.

    python -m yamc_verify.environments ensure yamc@86ef200e openmc==0.16.1.dev0
    python -m yamc_verify.environments list
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVS_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "yamc-verification" / "envs"
_READY = "ready.json"
_PACKAGES = {"openmc": "openmc", "yamc": "yamc-core"}


@dataclass(frozen=True)
class CodeSpec:
    """A code at an exact version (``kind="release"``) or commit (``kind="commit"``)."""

    name: str
    kind: str
    ref: str

    def __str__(self) -> str:
        return f"{self.name}=={self.ref}" if self.kind == "release" else f"{self.name}@{self.ref}"


def parse_code(spec: str) -> CodeSpec:
    """Parse ``openmc==X``, ``yamc==X`` or ``yamc@<commit>``.

    Raises ``ValueError`` for anything else, including a bare name: there is no
    default version.
    """
    m = re.fullmatch(r"\s*(openmc|yamc)\s*(==|@)\s*([A-Za-z0-9._+-]+)\s*", spec)
    if m is None:
        raise ValueError(
            f"cannot read code {spec!r}: name a code and an exact version or commit, "
            "e.g. 'openmc==0.16.1.dev0', 'yamc==0.21.0' or 'yamc@86ef200e'"
        )
    name, op, ref = m.groups()
    if op == "@" and name != "yamc":
        raise ValueError(f"{spec!r}: building OpenMC from a commit is not supported; use openmc==<version>")
    return CodeSpec(name, "release" if op == "==" else "commit", ref)


def _run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], check=True, text=True, **kw)


def _python(env: Path) -> Path:
    return env / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")


def _index_args() -> list[str]:
    """The ``--extra-index-url`` lines of requirements.txt, as pip arguments."""
    args = []
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        if line.strip().startswith("--extra-index-url"):
            args += line.split()
    return args


def _base_requirements() -> list[str]:
    """requirements.txt without the index lines and without either code."""
    out = []
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        req = line.split("#", 1)[0].strip()
        if not req or req.startswith("-"):
            continue
        if re.match(r"(openmc|yamc-core|yamc)\b(?![_-])", req):
            continue
        out.append(req)
    return out


def resolve_commit(core_dir: Path, ref: str) -> str:
    """The full hash of ``ref`` in the core checkout, fetching once if unknown."""
    def rev_parse():
        return subprocess.run(
            ["git", "-C", str(core_dir), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            capture_output=True, text=True,
        )

    out = rev_parse()
    if out.returncode != 0:
        _run(["git", "-C", core_dir, "fetch", "--quiet", "origin"])
        out = rev_parse()
    if out.returncode != 0:
        raise ValueError(f"commit {ref!r} is not in the core checkout at {core_dir}")
    return out.stdout.strip()


def env_dir(spec: CodeSpec, commit: str | None = None) -> Path:
    """Where the environment for ``spec`` lives (a commit by its first 12 characters)."""
    tag = spec.ref if spec.kind == "release" else f"g{(commit or spec.ref)[:12]}"
    return ENVS_ROOT / f"{spec.name}-{tag}"


def _core_dir(core_dir: Path | str | None) -> Path:
    path = Path(core_dir or os.environ.get("YAMC_CORE_ROOT") or REPO_ROOT.parent / "core")
    if not (path / "packages" / "yamc-core").is_dir():
        raise FileNotFoundError(
            f"no core checkout at {path}: pass core_dir or set YAMC_CORE_ROOT to build a yamc commit"
        )
    return path.resolve()


def _identity(env: Path) -> dict:
    """What the environment actually has installed, read from inside it."""
    probe = (
        "import json, re, shutil, subprocess\n"
        "out = {}\n"
        "import yamc\n"
        "out['yamc'] = {'version': yamc.__version__,\n"
        "               'commit': getattr(yamc, '__commit__', None),\n"
        "               'dirty': getattr(yamc, '__commit_dirty__', None)}\n"
        "exe = shutil.which('openmc')\n"
        "v = subprocess.run([exe, '--version'], capture_output=True, text=True).stdout if exe else ''\n"
        "m = re.search(r'OpenMC version (\\S+)', v); c = re.search(r'Commit hash: (\\S+)', v)\n"
        "from importlib.metadata import version\n"
        "out['openmc'] = {'version': version('openmc'), 'reports': m and m.group(1),\n"
        "                 'commit': c and c.group(1)}\n"
        "print(json.dumps(out))\n"
    )
    env_vars = {**os.environ, "PATH": f"{_python(env).parent}{os.pathsep}{os.environ['PATH']}"}
    out = _run([_python(env), "-c", probe], capture_output=True, env=env_vars).stdout
    return json.loads(out.strip().splitlines()[-1])


def _check(spec: CodeSpec, identity: dict, commit: str | None) -> None:
    """Refuse an environment that does not hold what was asked for.

    A release is checked against the installed package's version, which is what
    ``openmc==X`` / ``yamc==X`` name. An OpenMC wheel's executable may report a
    different version (its build's own), and that is what each record stores.
    """
    got = identity[spec.name]
    if spec.kind == "release":
        if got["version"] != spec.ref:
            raise RuntimeError(f"{spec}: the environment has {spec.name} {got['version']}")
        return
    if got["commit"] != commit:
        raise RuntimeError(
            f"{spec}: the built yamc reports commit {got['commit']!r}, not {commit}. "
            "A yamc-core older than the release that adds yamc.__commit__ cannot be "
            "built by commit."
        )
    if got["dirty"]:
        raise RuntimeError(f"{spec}: the built yamc reports uncommitted changes")


def _available_openmc_versions() -> list[str]:
    out = subprocess.run(
        [sys.executable, "-m", "pip", "index", "versions", "openmc", *_index_args()],
        capture_output=True, text=True,
    ).stdout
    m = re.search(r"Available versions: (.*)", out)
    return [v.strip() for v in m.group(1).split(",")] if m else []


def _build_yamc_wheel(core_dir: Path, commit: str, env: Path, work: Path) -> Path:
    """Build yamc-core at ``commit`` in a throwaway worktree; return the wheel."""
    src = work / "src"
    _run(["git", "-C", core_dir, "worktree", "add", "--detach", "--force", src, commit])
    try:
        _run([_python(env), "-m", "pip", "install", "--quiet", "maturin"])
        maturin = _python(env).parent / ("maturin.exe" if os.name == "nt" else "maturin")
        # Its own target directory: two checkouts sharing one reuse each other's
        # crates, because the path remapping makes them look identical to cargo.
        build_env = {**os.environ, "CARGO_TARGET_DIR": str(work / "target")}
        _run([maturin, "build", "--release", "-o", work / "dist", "-i", _python(env)],
             cwd=src / "packages" / "yamc-core", env=build_env)
    finally:
        _run(["git", "-C", core_dir, "worktree", "remove", "--force", src])
    wheels = sorted((work / "dist").glob("yamc_core-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one yamc-core wheel from the build, found {wheels}")
    return wheels[0]


def ensure(spec: CodeSpec | str, *, core_dir: Path | str | None = None, rebuild: bool = False) -> Path:
    """Return the ready environment for ``spec``, building it first if needed."""
    spec = parse_code(spec) if isinstance(spec, str) else spec
    commit = resolve_commit(_core_dir(core_dir), spec.ref) if spec.kind == "commit" else None
    env = env_dir(spec, commit)
    if (env / _READY).exists() and not rebuild:
        return env
    if spec.name == "openmc":
        available = _available_openmc_versions()
        if available and spec.ref not in available:
            raise ValueError(f"openmc {spec.ref} is not on the wheels index; available: {available}")

    shutil.rmtree(env, ignore_errors=True)
    env.parent.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "venv", env])
    pip = [_python(env), "-m", "pip", "install", "--quiet", *_index_args()]
    _run([*pip, "--upgrade", "pip"])
    _run([*pip, *_base_requirements()])
    other = "yamc" if spec.name == "openmc" else "openmc"
    if spec.kind == "release":
        _run([*pip, f"{_PACKAGES[spec.name]}=={spec.ref}", _PACKAGES[other]])
    else:
        _run([*pip, _PACKAGES[other]])
        with tempfile.TemporaryDirectory(prefix="yamc-build-") as work:
            wheel = _build_yamc_wheel(_core_dir(core_dir), commit, env, Path(work))
            _run([*pip, "--force-reinstall", "--no-deps", wheel])

    identity = _identity(env)
    _check(spec, identity, commit)
    (env / _READY).write_text(json.dumps({"spec": str(spec), "commit": commit, **identity}, indent=2))
    return env


def python_for(spec: CodeSpec | str, **kw) -> Path:
    """The interpreter of the ready environment for ``spec``."""
    return _python(ensure(spec, **kw))


def list_environments() -> list[dict]:
    """Every ready environment, as recorded when it was checked."""
    if not ENVS_ROOT.is_dir():
        return []
    return [json.loads((d / _READY).read_text()) | {"path": str(d)}
            for d in sorted(ENVS_ROOT.iterdir()) if (d / _READY).exists()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("ensure", help="build (if needed) and check environments")
    e.add_argument("codes", nargs="+", help="openmc==X, yamc==X or yamc@<commit>")
    e.add_argument("--core-dir", help="core checkout for yamc@<commit> (default $YAMC_CORE_ROOT or ../core)")
    e.add_argument("--rebuild", action="store_true", help="rebuild even if ready")
    sub.add_parser("list", help="list ready environments")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for env in list_environments():
            print(f"{env['spec']:32s} {env['path']}")
        return 0
    for code in args.codes:
        env = ensure(code, core_dir=args.core_dir, rebuild=args.rebuild)
        print(f"{code}: ready at {env}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
