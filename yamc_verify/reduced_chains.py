"""Precompute OpenMC's reduced depletion-chain membership per transmutation seed.

The verification's transmutation benchmarks run OpenMC on a chain reduced to
``reduce_level`` hops from the seed (``openmc.deplete.Chain.reduce``), which
prunes the whole network (reactions, decays, and the Bateman matrix). YAMC
instead keeps the full decay network. So a nuclide that YAMC produces but that
lies **outside** OpenMC's reduced chain is one OpenMC cannot produce by
construction: the difference is OpenMC truncation, not a YAMC error.

Deciding that analytically needs OpenMC's reduced-chain node set. That set is a
pure graph reduction of the chain (no simulation), but it needs ``openmc`` and
the chain file, which are not available in the Pages CI that builds ``index.json``.
So we compute it here, where openmc is available, and write it to a committed
sidecar (``<cache_dir>/<library>/reduced_chains.json``) that the index builder
reads with no extra dependencies.

Usage::

    python -m yamc_verify.reduced_chains          # (re)build sidecars for all transmutation tasks
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from . import tasks as _tasks
from .discovery import resolve_openmc_chain

_REPO_ROOT = Path(__file__).resolve().parents[1]

SIDECAR_NAME = "reduced_chains.json"


def sidecar_path(task) -> Path:
    """Repo-relative path of the reduced-chain sidecar for ``task``."""
    return _REPO_ROOT / task.cache_dir / task.library / SIDECAR_NAME


def _cached_seeds(task) -> list[str]:
    """Seed nuclide names that have an OpenMC cache entry for ``task``."""
    d = _REPO_ROOT / task.cache_dir / task.library / "openmc"
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(str(d / "*.json")))


def build_for_task(task) -> dict[str, list[str]]:
    """Compute OpenMC's reduced-chain nuclide set for every cached seed.

    Returns ``{seed: [nuclide names in OpenMC's reduced chain]}``. Requires
    ``openmc`` and the chain file. Raises if either is unavailable.
    """
    import openmc.deplete

    reduce_level = task.extras["reduce_level"]
    chain_file = resolve_openmc_chain(task.library)
    chain = openmc.deplete.Chain.from_xml(chain_file)

    out: dict[str, list[str]] = {}
    for seed in _cached_seeds(task):
        reduced = chain.reduce([seed], reduce_level)
        out[seed] = sorted(n.name for n in reduced.nuclides)
    return out


def load(task) -> dict[str, set] | None:
    """Load ``{seed: set(nuclide names)}`` from the sidecar, or None if absent."""
    p = sidecar_path(task)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    payload = data.get("reduced_chains", data)
    return {seed: set(names) for seed, names in payload.items()}


def main(argv=None):
    transmutation_tasks = [t for t in _tasks.all_tasks() if t.family.startswith("transmutation")]
    for task in transmutation_tasks:
        seeds = _cached_seeds(task)
        if not seeds:
            print(f"{task.key}: no OpenMC cache, skipping")
            continue
        mapping = build_for_task(task)
        p = sidecar_path(task)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(
                {"reduce_level": task.extras["reduce_level"], "reduced_chains": mapping},
                f, indent=1, sort_keys=True,
            )
        sizes = sorted(len(v) for v in mapping.values())
        print(f"{task.key}: wrote {len(mapping)} seeds "
              f"(reduced-chain sizes {sizes[0]}..{sizes[-1]}) -> {p.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
