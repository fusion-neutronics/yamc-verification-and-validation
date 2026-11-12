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

from cache_utils import _decode, code_identity, iter_records

from . import tasks as _tasks
from .discovery import element_from_nuclide
from .plotting import calculate_spectral_agreement, quality_from_chi2

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
    if isinstance(obj, np.generic):  # np.floating / np.integer / np.bool_ -> Python scalar
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _latest(lib_dir: Path, code: str, key: str):
    """``(path, record)`` of the newest readable record of one item, or ``None``."""
    return next(iter_records(lib_dir.parent, lib_dir.name, code, key), None)


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


def _expected_spec(task, element, spectrum_name, omc, ymc):
    """Per-bin ``{index -> 'over'|'under'}`` expected-difference spec for one spectrum.

    Physics-derived and independent of the observed discrepancy (see
    ``yamc_verify.expected`` and yamc-verification #59). Empty dict when the family /
    spectrum carries no declared expected difference, so the spectrum grades on its
    full residual.
    """
    if task is None:
        return {}
    from . import expected as _exp

    fam = task.family
    edges = omc.get(_SPECTRUM_EDGE_KEY.get(spectrum_name, "energy_bins"))
    if fam in ("photon", "secondary-photon"):
        # fluorescence lives in the photon spectrum only: "flux" for the photon
        # family, "photon" (p_spectrum) for secondary-photon. The neutron spectrum
        # of a secondary-photon run grades normally.
        photon_spectrum = "flux" if fam == "photon" else "photon"
        if spectrum_name == photon_spectrum:
            return _exp.photon_expected_spec(element, edges)
    # decay-photon: the D1S daughter gammas OpenMC omits are YAMC-only lines, which
    # fall in bins where OpenMC has ~zero flux and so never enter the shared-bin
    # chi-squared. They therefore do not affect the grade and need no bin exclusion;
    # they are annotated from ``other_only_flux_pct`` in ``_compare``. Attributing
    # each YAMC-only line to a specific daughter gamma (vs a spurious emission) is a
    # follow-up refinement (#59) that would use ``expected.decay_gamma_bins``.
    return {}


def _compare_spectra(omc, ymc, task=None, element=None):
    """Per-spectrum agreement stats for every spectrum both codes cached.

    When ``task`` is given, each spectrum is graded on its residual after the
    family's declared expected difference (if any) is removed.
    """
    o_map = {name: (m, s) for name, m, s in _spectrum_pairs(omc)}
    y_map = {name: (m, s) for name, m, s in _spectrum_pairs(ymc)}
    spectra = []
    for name, (o_mean, o_std) in o_map.items():
        if name not in y_map or len(y_map[name][0]) != len(o_mean):
            continue
        y_mean, y_std = y_map[name]
        expected = _expected_spec(task, element, name, omc, ymc)
        stats = calculate_spectral_agreement(
            {"flux_spectra_mean": o_mean, "flux_spectra_std_dev": o_std},
            {"flux_spectra_mean": y_mean, "flux_spectra_std_dev": y_std},
            expected=expected or None,
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


# Residual grading (yamc-verification #59). Every comparison gets one of the four
# grades, computed on the residual after a known, physics-derived expected difference
# is removed (see yamc_verify.expected). The expected difference is recorded as an
# annotation, never as a grade band.
_EXPECTED_REASON = {
    "photon": "compton-ionization-fluorescence",
    "secondary-photon": "compton-ionization-fluorescence",
    "decay-photon": "d1s-daughter-gammas",
}

# If the flux removed as an expected directional difference exceeds this fraction of
# the spectrum, the "expected" explanation is implausibly large (fluorescence / decay
# lines are a small part of the flux). Flag the comparison for review and keep its FULL
# grade rather than letting the exclusion rescue it. Tunable.
MAX_EXPECTED_EXCLUDED_FLUX_PCT = 5.0

# spectrum name -> the results key holding that spectrum's energy-bin edges. Names not
# listed (e.g. decay-photon "step1".."stepN") fall back to "energy_bins".
_SPECTRUM_EDGE_KEY = {"flux": "energy_bins", "neutron": "n_energy_bins", "photon": "p_energy_bins"}


def _compare(task, omc, ymc, element=None):
    """Agreement metrics for one paired (benchmark, backend), or None.

    Residual grading (#59): every comparison gets one of the four grades, computed
    on the residual after the family's declared, physics-derived expected difference
    is removed. The expected difference, when present, is recorded in an
    ``expected_difference`` annotation (never a grade band).
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
            "weighted_avg_rel_diff_pct": cmp["rms_rel_diff_pct"],
            "max_rel_diff_pct": cmp["max_rel_diff_pct"],
            "products_compared": cmp["n_total"],
            "floats_compared": len(cmp["detail_rows"]),
        }

    spectra = _compare_spectra(omc, ymc, task=task, element=element)
    n_scalars, worst_scalar_pct, worst_scalar = _scalar_rel_diffs(omc, ymc)
    if not spectra and not n_scalars:
        return None
    # Per-spectrum "quality" is already the residual grade (see plotting).
    quality = _worst_quality([s["quality"] for s in spectra])
    out = {
        "kind": "spectrum",
        "quality": quality,
        "floats_compared": sum(s["total_bins"] for s in spectra) + n_scalars,
    }
    if spectra:
        weights = [max(s["bins_analyzed"], 1) for s in spectra]
        out["reduced_chi_squared"] = max(s["reduced_chi_squared"] for s in spectra)
        out["residual_reduced_chi_squared"] = max(
            s.get("residual_reduced_chi_squared", s["reduced_chi_squared"]) for s in spectra
        )
        out["weighted_avg_rel_diff_pct"] = float(
            sum(s["weighted_avg_rel_diff"] * w for s, w in zip(spectra, weights)) / sum(weights)
        )
        out["spectra"] = spectra
        # Expected-difference annotation: set only when a declared expected difference
        # was actually removed from the residual. Never changes the grade, except that
        # an implausibly large exclusion trips needs_review and reverts to the full grade.
        excluded_pct = max((s.get("excluded_flux_pct", 0.0) for s in spectra), default=0.0)
        other_only_pct = max((s.get("other_only_flux_pct", 0.0) for s in spectra), default=0.0)
        if any(s.get("excluded_bins") for s in spectra):
            needs_review = bool(excluded_pct > MAX_EXPECTED_EXCLUDED_FLUX_PCT)
            out["expected_difference"] = {
                "reason": _EXPECTED_REASON.get(task.family, "expected"),
                "excluded_flux_pct": excluded_pct,
                "full_reduced_chi_squared": out["reduced_chi_squared"],
                "residual_reduced_chi_squared": out["residual_reduced_chi_squared"],
                "needs_review": needs_review,
            }
            if needs_review:
                out["quality"] = quality_from_chi2(out["reduced_chi_squared"])
        elif task.family == "decay-photon" and other_only_pct > 1.0:
            # D1S daughter gammas are YAMC-only flux (not in the shared-bin chi-squared,
            # so the grade is already the residual); annotate them. residual_adjusted
            # stays False until the daughter-line attribution refinement lands (#59).
            out["expected_difference"] = {
                "reason": "d1s-daughter-gammas",
                "other_only_flux_pct": other_only_pct,
                "residual_adjusted": False,
            }
    if n_scalars:
        out["max_tally_rel_diff_pct"] = worst_scalar_pct
        out["worst_tally"] = worst_scalar
    return out


def _mismatch_key(cmp):
    """Severity sort key: quality band first, then chi2, then worst rel diff.

    Quality leads so chi2-less comparisons (transmutation) rank with the
    spectrum ones instead of always sorting below them.
    """
    # Rank on the residual chi-squared (the graded value), not the full one, so a
    # comparison whose only difference is an accounted-for expected one sorts low.
    chi2 = cmp.get("residual_reduced_chi_squared", cmp.get("reduced_chi_squared"))
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
        ident = code_identity("yamc")
        installed = {"version": ident["version"], "build_id": ident["build_id"]}
    except Exception:
        installed = None  # yamc not installed (e.g. CI); the records are enough

    records = []
    unreadable = []
    n_simulations = 0
    indexed_lib_dirs = set()

    for task in tasks_list:
        lib_dir = _REPO_ROOT / task.cache_dir / task.library
        indexed_lib_dirs.add(lib_dir)
        if not lib_dir.is_dir():
            continue
        # <library>/<code>/<item>/<build_id>.json; the newest record of each
        # item is the one shown.
        code_items = {}
        for code_dir in sorted(p for p in lib_dir.iterdir() if p.is_dir()):
            items = {d.name for d in code_dir.iterdir() if d.is_dir() and any(d.glob("*.json"))}
            if items:
                code_items[code_dir.name] = items
        all_keys = sorted(set().union(*code_items.values(), set()))
        for key in all_keys:
            loaded = {}
            paths = {}
            for code, items in code_items.items():
                if key not in items:
                    continue
                newest = _latest(lib_dir, code, key)
                if newest is None:
                    unreadable.append((lib_dir / code / key).relative_to(_REPO_ROOT).as_posix())
                    continue
                path, record = newest
                loaded[code] = (record["code_identity"], _decode(record.get("results") or {}))
                paths[code] = path
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
            element = key if task.item_kind == "element" else element_from_nuclide(key)
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
                    y_ident, y_res = loaded[code]
                    cmp = _compare(task, omc_res, y_res, element=element)
                    if cmp is not None:
                        comparisons.append({
                            "code": code,
                            "treatment": treatment or None,
                            "yamc_version": y_ident.get("version"),
                            "yamc_build": y_ident.get("build_id"),
                            "openmc_build": loaded[ref][0].get("build_id"),
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
                "element": element,
                "codes": sorted(loaded),
                "files": {
                    c: paths[c].relative_to(_REPO_ROOT).as_posix() for c in loaded
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
            if lib_dir not in indexed_lib_dirs and any(lib_dir.glob("*/*/*.json")):
                unindexed.append(lib_dir.relative_to(_REPO_ROOT).as_posix())

    flat = [(r, c) for r in records for c in r["comparisons"]]
    # Residual grading (#59): every comparison carries a genuine grade, so all of
    # them enter the deviation distribution and worst-mismatch list. A comparison
    # whose only difference was an accounted-for expected one grades Excellent and
    # sorts low naturally (its residual chi-squared is small).
    graded = flat
    worst = sorted(graded, key=lambda rc: _mismatch_key(rc[1]), reverse=True)[:worst_n]
    index = {
        "schema_version": 1,
        "generated_unix": int(time.time()),
        "yamc": installed,
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
            "quality_counts": dict(Counter(c["quality"] for _, c in flat if c.get("quality"))),
            "expected_differences": sum(1 for _, c in flat if c.get("expected_difference")),
            "needs_review": sum(
                1 for _, c in flat if (c.get("expected_difference") or {}).get("needs_review")
            ),
            "weighted_avg_rel_diff_pct": _distribution(
                [c.get("weighted_avg_rel_diff_pct") for _, c in graded]
            ),
            "residual_reduced_chi_squared": _distribution(
                [c.get("residual_reduced_chi_squared", c.get("reduced_chi_squared")) for _, c in graded]
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
