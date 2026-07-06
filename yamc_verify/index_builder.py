"""Emit the static ``index.json`` consumed by the verification website.

Phase 1 of the website plan (issue #25): walk the on-disk results cache,
compute per-benchmark OpenMC-vs-YAMC agreement, and write a single
``index.json`` holding the headline aggregates plus one record per benchmark
(unique family x geometry x library x nuclide/element) for client-side
filtering. Agreement reuses :func:`yamc_verify.plotting.calculate_spectral_agreement`
(flux-weighted + reduced chi-squared, not a naive per-bin mean) and the
transmutation step comparator, mirroring :mod:`yamc_verify.audit`.

Two denominators are reported deliberately because they differ: ``simulations``
counts every cached run (OpenMC-only runs included) while ``comparisons``
counts only (benchmark, backend) pairs where both an OpenMC and a YAMC result
exist. ``floats_compared`` totals the values examined per comparison:
spectrum bins, shared scalar tallies, and per-timestep nuclide densities.
The global deviation is published as a distribution (mean / median / p90 /
max) plus a worst-N list and per-quality counts -- never a single number.

Runs anywhere the repo is checked out (CI included): no yamc install or
nuclear data needed, only the cached JSON.

Usage::

    python -m yamc_verify.index_builder                 # writes ./index.json
    python -m yamc_verify.index_builder --out _build/html/index.json
    python -m yamc_verify.index_builder --pretty        # indented (debugging)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from cache_utils import _decode, yamc_fingerprint

from . import tasks as _tasks
from .discovery import element_from_nuclide
from .plotting import calculate_spectral_agreement
from .reduced_chains import load as load_reduced_chains

_REPO_ROOT = Path(__file__).resolve().parents[1]

_QUALITY_ORDER = ("Excellent", "Good", "Fair", "Poor")

# results-dict scalar tallies: (means key, prefix for the published name)
_SCALAR_KEYS = (("tallies", ""), ("n_scalars", "n:"), ("p_scalars", "p:"))


def _worst_quality(qualities):
    ranked = [q for q in qualities if q in _QUALITY_ORDER]
    return max(ranked, key=_QUALITY_ORDER.index) if ranked else None


# Electron-treatment suffixes on secondary-photon codes (openmc-led, yamc-cpu-ttb,
# ...). Splitting lets the index pair OpenMC and YAMC of the SAME treatment;
# codes without a suffix (all other families) fall in the "" group unchanged.
_TREATMENTS = ("led", "ttb")


def _split_treatment(code: str):
    """Return (base_code, treatment) e.g. ('openmc','led') or ('yamc-cpu','')."""
    for t in _TREATMENTS:
        if code.endswith(f"-{t}"):
            return code[: -(len(t) + 1)], t
    return code, ""


def _jsonable(obj):
    """Plain JSON types only; numpy scalars unwrapped, non-finite floats -> None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _load(path: Path):
    """Return (meta, results) for one cache file, or None if unreadable."""
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("_meta", {}), _decode(data.get("results") or {})


def _spectrum_pairs(results):
    """Yield every ``(name, mean, std)`` spectrum a cached results dict holds.

    Covers the per-family shapes: ``flux_spectra_*`` (neutron / photon),
    ``n_spectrum_*`` + ``p_spectrum_*`` (secondary-photon) and the
    per-timestep ``spectra`` list (decay-photon).
    """
    if "flux_spectra_mean" in results:
        yield "flux", results["flux_spectra_mean"], results["flux_spectra_std_dev"]
    if "n_spectrum_mean" in results:
        yield "neutron", results["n_spectrum_mean"], results["n_spectrum_std"]
    if "p_spectrum_mean" in results:
        yield "photon", results["p_spectrum_mean"], results["p_spectrum_std"]
    steps = results.get("spectra") or []
    stds = results.get("spectra_std") or []
    for i, (mean, std) in enumerate(zip(steps, stds)):
        yield f"step{i + 1}", mean, std


def _compare_spectra(omc, ymc):
    """Per-spectrum agreement stats for every spectrum both codes cached."""
    o_map = {name: (m, s) for name, m, s in _spectrum_pairs(omc)}
    y_map = {name: (m, s) for name, m, s in _spectrum_pairs(ymc)}
    spectra = []
    for name, (o_mean, o_std) in o_map.items():
        if name not in y_map or len(y_map[name][0]) != len(o_mean):
            continue
        y_mean, y_std = y_map[name]
        stats = calculate_spectral_agreement(
            {"flux_spectra_mean": o_mean, "flux_spectra_std_dev": o_std},
            {"flux_spectra_mean": y_mean, "flux_spectra_std_dev": y_std},
        )
        spectra.append({"name": name, **stats})
    return spectra


def _scalar_rel_diffs(omc, ymc):
    """(count, worst_pct, worst_name) over the shared scalar tallies."""
    count, worst, worst_name = 0, 0.0, None
    for means_key, prefix in _SCALAR_KEYS:
        o_scalars = omc.get(means_key) or {}
        y_scalars = ymc.get(means_key) or {}
        for name in sorted(set(o_scalars) & set(y_scalars)):
            o, y = o_scalars[name], y_scalars[name]
            if not isinstance(o, (int, float)) or not isinstance(y, (int, float)):
                continue
            if not (math.isfinite(o) and math.isfinite(y)) or o == 0:
                continue
            count += 1
            d = abs((y - o) / o) * 100.0
            if d > worst:
                worst, worst_name = d, f"{prefix}{name}"
    return count, worst, worst_name


EXPECTED = "Expected differences"


def _transmutation_extra_nuclides(ymc, reduced_set):
    """Nuclides YAMC produces (above threshold) that lie outside OpenMC's reduced chain.

    These are the nuclides OpenMC cannot produce by construction at this
    ``reduce_level`` (chain membership), so they are an expected extra rather
    than a disagreement. Returns a sorted name list (empty if no sidecar).
    """
    if not reduced_set:
        return []
    from .families.transmutation import DENSITY_THRESHOLD

    seen = set()
    for step in ymc.get("steps", []) or []:
        for name, dens in step.items():
            if abs(dens) >= DENSITY_THRESHOLD and name not in reduced_set:
                seen.add(name)
    return sorted(seen)


def _compare(task, omc, ymc, reduced_set=None):
    """Agreement metrics for one paired (benchmark, backend), or None.

    ``reduced_set`` is OpenMC's reduced-chain nuclide set for this seed (from the
    reduced_chains sidecar), used to annotate transmutation comparisons with the
    nuclides OpenMC's reduced chain cannot produce. ``category`` is the display
    band: the numeric quality normally, or ``EXPECTED`` when the difference is a
    known physics-driven one (YAMC more complete).
    """
    if task.family.startswith("transmutation"):
        from .families.transmutation import compare_transmutation_steps

        try:
            cmp = compare_transmutation_steps(
                ymc, omc, task.extras.get("timesteps", []), task.extras.get("phases", [])
            )
        except Exception:
            return None
        return {
            "kind": "transmutation",
            "quality": cmp["quality"],
            "category": cmp["quality"],
            "weighted_avg_rel_diff_pct": cmp["rms_rel_diff_pct"],
            "max_rel_diff_pct": cmp["max_rel_diff_pct"],
            "products_compared": cmp["n_total"],
            "floats_compared": len(cmp["detail_rows"]),
            "expected_extra_nuclides": _transmutation_extra_nuclides(ymc, reduced_set),
        }

    spectra = _compare_spectra(omc, ymc)
    n_scalars, worst_scalar_pct, worst_scalar = _scalar_rel_diffs(omc, ymc)
    if not spectra and not n_scalars:
        return None
    quality = _worst_quality([s["quality"] for s in spectra])
    out = {
        "kind": "spectrum",
        "quality": quality,
        "category": quality,
        "floats_compared": sum(s["total_bins"] for s in spectra) + n_scalars,
    }
    if spectra:
        weights = [max(s["bins_analyzed"], 1) for s in spectra]
        out["reduced_chi_squared"] = max(s["reduced_chi_squared"] for s in spectra)
        out["weighted_avg_rel_diff_pct"] = float(
            sum(s["weighted_avg_rel_diff"] * w for s, w in zip(spectra, weights)) / sum(weights)
        )
        out["spectra"] = spectra
        ref_total = sum(s.get("ref_total_flux", 0.0) for s in spectra)
        oth_total = sum(s.get("other_total_flux", 0.0) for s in spectra)
        # decay-photon: a diverging comparison is the D1S daughter-gamma difference
        # (YAMC includes the decay products' gammas, OpenMC's D1S omits them). chi2
        # is hypersensitive here (tight per-bin stats make any real difference read
        # as Poor), so a numeric grade is misleading; categorize the divergence as
        # the expected difference instead. Comparisons that agree keep their grade.
        if task.family == "decay-photon" and quality in ("Fair", "Poor"):
            out["category"] = EXPECTED
            out["expected_reason"] = "d1s-daughter-gammas"
            if ref_total > 0:
                out["yamc_over_openmc_flux"] = oth_total / ref_total
    if n_scalars:
        out["max_tally_rel_diff_pct"] = worst_scalar_pct
        out["worst_tally"] = worst_scalar
    return out


def _mismatch_key(cmp):
    """Severity sort key: quality band first, then chi2, then worst rel diff.

    Quality leads so chi2-less comparisons (transmutation) rank with the
    spectrum ones instead of always sorting below them.
    """
    chi2 = cmp.get("reduced_chi_squared")
    rd = cmp.get("max_tally_rel_diff_pct") or cmp.get("max_rel_diff_pct")
    return (
        _QUALITY_ORDER.index(cmp["quality"]) if cmp.get("quality") in _QUALITY_ORDER else -1,
        chi2 if isinstance(chi2, (int, float)) and math.isfinite(chi2) else 0.0,
        rd if isinstance(rd, (int, float)) and math.isfinite(rd) else 0.0,
    )


def _distribution(values):
    """Summary distribution dict for the finite values, or None."""
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return None
    a = np.asarray(finite, dtype=float)
    return {
        "n": len(finite),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def build_index(tasks_list=None, worst_n=15):
    """Walk the cache for every registered task and build the index dict."""
    tasks_list = tasks_list or _tasks.all_tasks()
    try:
        fingerprint = yamc_fingerprint()
    except Exception:
        fingerprint = None  # yamc not installed (e.g. CI); the cache is enough

    records = []
    unreadable = []
    n_simulations = 0
    indexed_lib_dirs = set()

    for task in tasks_list:
        lib_dir = _REPO_ROOT / task.cache_dir / task.library
        indexed_lib_dirs.add(lib_dir)
        if not lib_dir.is_dir():
            continue
        reduced_chains = load_reduced_chains(task) if task.family.startswith("transmutation") else None
        code_files = {}
        for code_dir in sorted(p for p in lib_dir.iterdir() if p.is_dir()):
            files = {p.stem: p for p in code_dir.glob("*.json")}
            if files:
                code_files[code_dir.name] = files
        all_keys = sorted(set().union(*map(set, code_files.values()), set()))
        for key in all_keys:
            loaded = {}
            for code, files in code_files.items():
                if key not in files:
                    continue
                data = _load(files[key])
                if data is None:
                    unreadable.append(str(files[key].relative_to(_REPO_ROOT)))
                else:
                    loaded[code] = data
            if not loaded:
                continue
            n_simulations += len(loaded)
            comparisons = []
            # Pair each OpenMC reference with the YAMC backends that share its
            # electron-treatment suffix (openmc-led vs yamc-cpu-led, ...). Codes
            # without a suffix form the "" group, so every other family is
            # unchanged. The treatment (or None) is carried on each comparison.
            by_treatment: dict = {}
            for code in loaded:
                base, treatment = _split_treatment(code)
                by_treatment.setdefault(treatment, {})[base] = code
            for treatment in sorted(by_treatment):
                group = by_treatment[treatment]
                ref = group.get("openmc")
                if ref is None:
                    continue
                _, omc_res = loaded[ref]
                for base in sorted(group):
                    if base == "openmc":
                        continue
                    code = group[base]
                    y_meta, y_res = loaded[code]
                    reduced_set = reduced_chains.get(key) if reduced_chains else None
                    cmp = _compare(task, omc_res, y_res, reduced_set=reduced_set)
                    if cmp is not None:
                        comparisons.append({
                            "code": code,
                            "treatment": treatment or None,
                            "yamc_version": y_meta.get("yamc_version"),
                            "yamc_core_sha": y_meta.get("yamc_core_sha"),
                            **cmp,
                        })
            records.append({
                "id": f"{task.family}/{task.geometry}/{task.library}/{key}",
                "task": task.key,
                "family": task.family,
                "geometry": task.geometry,
                "library": task.library,
                "item": key,
                "item_kind": task.item_kind,
                "element": key if task.item_kind == "element" else element_from_nuclide(key),
                "codes": sorted(loaded),
                "files": {
                    c: code_files[c][key].relative_to(_REPO_ROOT).as_posix() for c in loaded
                },
                "comparisons": comparisons,
            })

    # Cache library dirs no registered task covers (e.g. data landing before
    # its task exists) -- reported, never silently dropped.
    unindexed = []
    for cache_root in sorted({_REPO_ROOT / t.cache_dir for t in tasks_list}):
        if not cache_root.is_dir():
            continue
        for lib_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
            if lib_dir not in indexed_lib_dirs and any(lib_dir.glob("*/*.json")):
                unindexed.append(lib_dir.relative_to(_REPO_ROOT).as_posix())

    flat = [(r, c) for r in records for c in r["comparisons"]]
    # Expected-differences comparisons are physics-driven (YAMC more complete), so
    # they are graded in their own category and kept out of the deviation
    # distribution and the worst-mismatch list, which are about genuine agreement.
    graded = [(r, c) for r, c in flat if c.get("category") != EXPECTED]
    worst = sorted(graded, key=lambda rc: _mismatch_key(rc[1]), reverse=True)[:worst_n]
    index = {
        "schema_version": 1,
        "generated_unix": int(time.time()),
        "yamc": fingerprint,
        "totals": {
            "simulations": n_simulations,
            "benchmarks": len(records),
            "benchmarks_compared": sum(1 for r in records if r["comparisons"]),
            "comparisons": len(flat),
            "floats_compared": sum(c["floats_compared"] for _, c in flat),
        },
        "facets": {
            "families": sorted({r["family"] for r in records}),
            "geometries": sorted({r["geometry"] for r in records}),
            "libraries": sorted({r["library"] for r in records}),
            "codes": sorted({c for r in records for c in r["codes"]}),
            "elements": sorted({r["element"] for r in records}),
        },
        "agreement": {
            "quality_counts": dict(Counter(c["category"] for _, c in flat if c.get("category"))),
            "expected_differences": sum(1 for _, c in flat if c.get("category") == EXPECTED),
            "weighted_avg_rel_diff_pct": _distribution(
                [c.get("weighted_avg_rel_diff_pct") for _, c in graded]
            ),
            "reduced_chi_squared": _distribution(
                [c.get("reduced_chi_squared") for _, c in graded]
            ),
            "worst": [
                {
                    "id": r["id"],
                    "code": c["code"],
                    "quality": c["quality"],
                    "reduced_chi_squared": c.get("reduced_chi_squared"),
                    "weighted_avg_rel_diff_pct": c.get("weighted_avg_rel_diff_pct"),
                    "max_tally_rel_diff_pct": c.get("max_tally_rel_diff_pct"),
                    "max_rel_diff_pct": c.get("max_rel_diff_pct"),
                }
                for r, c in worst
            ],
        },
        "benchmarks": records,
        "unindexed_cache_dirs": unindexed,
        "unreadable_files": unreadable,
    }
    return _jsonable(index)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build index.json from the results cache.")
    ap.add_argument("--out", default="index.json", help="output path (default: ./index.json)")
    ap.add_argument("--worst", type=int, default=15, help="length of the worst-N list")
    ap.add_argument("--pretty", action="store_true", help="indent the JSON (for debugging)")
    args = ap.parse_args(argv)

    index = build_index(worst_n=args.worst)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        if args.pretty:
            json.dump(index, f, indent=1, allow_nan=False)
        else:
            json.dump(index, f, separators=(",", ":"), allow_nan=False)

    t = index["totals"]
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB)")
    print(f"  simulations:      {t['simulations']}")
    print(f"  benchmarks:       {t['benchmarks']} ({t['benchmarks_compared']} with a comparison)")
    print(f"  comparisons:      {t['comparisons']}")
    print(f"  floats compared:  {t['floats_compared']}")
    dist = index["agreement"]["weighted_avg_rel_diff_pct"]
    if dist:
        print(f"  flux-weighted rel diff %: median {dist['median']:.2f}, "
              f"p90 {dist['p90']:.2f}, max {dist['max']:.2f}")
    print(f"  quality: {index['agreement']['quality_counts']}")
    for msg, items in (("unindexed cache dirs", index["unindexed_cache_dirs"]),
                       ("unreadable cache files", index["unreadable_files"])):
        if items:
            print(f"  WARNING {msg}: {items}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
