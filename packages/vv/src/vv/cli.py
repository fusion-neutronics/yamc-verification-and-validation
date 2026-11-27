"""The ``vv`` command: plan, run, compare and publish a verification.

``vv plan`` shows what would run and ``vv run`` runs it. ``vv compare`` shows
what changed against the run the site shows, and ``vv publish`` opens a pull
request publishing the records (see :mod:`vv.publish`). ``vv release``, run by
CI after that merges, makes the release of each published run that has none.

The project comes from the nearest ``vv.toml`` in the current directory or a
parent, which names it as ``module:attribute``::

    project = "my_verification.project:PROJECT"

The folder holding ``vv.toml`` is put on the import path first, so the module
can live in that repository without being installed. Every option is required::

    vv plan --models sphere slab --libraries endf-b8.1 --codes code-a==1.2.0 code-b==0.4.0
    vv run  --models all --libraries endf-b8.1 --codes code-a==1.2.0
    vv compare --models all --libraries endf-b8.1 --codes code-a==1.2.0
    vv publish --models all --libraries endf-b8.1 --codes code-a==1.2.0
    vv release
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import publish as published
from .runner import ALL, Project, SelectionError, plan, run

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG = "vv.toml"


def find_config(start: Path) -> Path:
    """The nearest ``vv.toml`` in ``start`` or a parent folder."""
    start = start.resolve()
    for folder in (start, *start.parents):
        if (folder / CONFIG).is_file():
            return folder / CONFIG
    raise FileNotFoundError(
        f"no {CONFIG} in {start} or any parent folder: run vv inside a repository that "
        f"names its project in {CONFIG}, e.g. project = \"package.module:PROJECT\""
    )


def load_project(config: Path) -> Project:
    """Import the :class:`~vv.runner.Project` that ``config`` names."""
    data = tomllib.loads(Path(config).read_text())
    ref = data.get("project")
    if not isinstance(ref, str) or ref.count(":") != 1:
        raise ValueError(f"{config}: set project = \"module:attribute\", got {ref!r}")
    module_name, attribute = ref.split(":")
    root = str(Path(config).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    project = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(project, Project):
        raise TypeError(f"{config}: {ref} is a {type(project).__name__}, not a vv Project")
    if project.root is None:
        project.root = Path(root)
    return project


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vv", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, text in (("plan", "show what would run, what is already done and a rough time"),
                       ("run", "run it, each code in its own environment, with a live display"),
                       ("compare", "show what changed against the run the site shows"),
                       ("publish", "open a pull request publishing the records these codes have")):
        p = sub.add_parser(name, help=text, description=text)
        p.add_argument("--models", nargs="+", required=True, metavar="MODEL",
                       help=f"models to run, or '{ALL}' for every model on the libraries")
        p.add_argument("--libraries", nargs="+", required=True, metavar="LIBRARY",
                       help="nuclear data libraries to run them on")
        p.add_argument("--codes", nargs="+", required=True, metavar="CODE",
                       help="codes, each with an exact version or commit")
    sub.add_parser("release", help="make the release of every published run that has none (run by CI)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``vv`` command line; return the exit status."""
    args = _parser().parse_args(argv)
    try:
        config = find_config(Path.cwd())
        if args.command == "release":
            for tag in published.release(config.parent):
                print(f"released {tag}")
            return 0
        project = load_project(config)
        kw = dict(models=ALL if args.models == [ALL] else args.models, libraries=args.libraries, codes=args.codes)
        if args.command == "plan":
            print(plan(project, **kw).format())
            return 0
        if args.command == "compare":
            print(published.compare(project, **kw).format())
            return 0
        if args.command == "publish":
            print(published.publish(project, **kw).format())
            return 0
        result = run(project, **kw)
    except (SelectionError, FileNotFoundError, published.PublishError) as e:
        print(f"vv: {e}", file=sys.stderr)
        return 2
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
