"""Offline audit of YAMC-vs-OpenMC agreement from the results cache.

Finds which nuclides / simulation modes disagree with OpenMC, using only the
cached results (no re-running). It compares every ``(task, nuclide, yamc
backend)`` for which both an OpenMC and a YAMC result are cached:

- spectrum families (neutron): reduced chi-squared + flux-weighted relative
  difference of the flux spectrum, and the worst energy-integrated reaction-rate
  (tally) relative difference;
- depletion (transmutation-reaction-rate): the worst nuclide-density relative
  difference across timesteps.

IMPORTANT: only families that cache BOTH codes can be audited offline. Neutron
and transmutation-reaction-rate cache YAMC; photon / secondary-photon /
decay-photon / transmutation-flux cache OpenMC only (YAMC runs live and is never
stored), so they are reported as "not auditable offline" -- run them to compare.

Usage::

    python -m yamc_verify.audit                 # rank all mismatches
    python -m yamc_verify.audit --worst 20       # only the 20 worst
    python -m yamc_verify.audit --current-only    # ignore stale-build YAMC caches
    python -m yamc_verify.audit --chi2 3 --reldiff 5   # thresholds for "mismatch"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from cache_utils import _cache_path, load_cached, yamc_fingerprint

from . import tasks as _tasks
from .plotting import calculate_spectral_agreement


@dataclass
class Row:
    task: str
    nuclide: str
    backend: str
    quality: str
    chi2: float
    wavg_reldiff: float
    max_tally_reldiff: float
    worst_score: str
    stale: bool

    @property
    def mismatch_key(self):
        # rank primarily by reduced chi-squared, then by worst tally rel diff
        return (self.chi2 if np.isfinite(self.chi2) else 0.0,
                self.max_tally_reldiff if np.isfinite(self.max_tally_reldiff) else 0.0)


def _cached_keys(cache_dir, library, code):
    d = os.path.join(cache_dir, library, code)
    return {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(d, "*.json"))}


def _yamc_core_sha(cache_dir, library, code, key):
    try:
        with open(_cache_path(cache_dir, library, code, key)) as f:
            return json.load(f).get("_meta", {}).get("yamc_core_sha")
    except (OSError, json.JSONDecodeError):
        return None


def _max_tally_reldiff(openmc, yamc):
    """Worst relative difference across shared (non-trivial) tally scores."""
    worst, score = 0.0, "-"
    ot, yt = openmc.get("tallies", {}), yamc.get("tallies", {})
    for name in set(ot) & set(yt):
        o, y = ot[name], yt[name]
        if o is None or y is None or not np.isfinite(o) or o == 0:
            continue
        d = abs((y - o) / o) * 100.0
        if np.isfinite(d) and d > worst:
            worst, score = d, name
    return worst, score


def audit(tasks_list=None, *, current_only=False, raw=False):
    """Return (rows, unauditable) where rows are per (task, nuclide, backend).

    By default rows are graded on the residual after known expected differences
    (Compton fluorescence lines, etc.) are removed, matching the dashboard. ``raw``
    grades on the full spectrum, ignoring expected-difference signatures.
    """
    fp_sha = yamc_fingerprint().get("yamc_core_sha")
    tasks_list = tasks_list or _tasks.all_tasks()
    rows: list[Row] = []
    unauditable: list[str] = []

    for t in tasks_list:
        yamc_codes = [c for c in t.codes if c.startswith("yamc")]
        if not t.caches_yamc or not yamc_codes:
            unauditable.append(f"{t.label} [{t.library}] — YAMC not cached (run to compare)")
            continue
        omc_keys = _cached_keys(t.cache_dir, t.library, "openmc")
        for backend in yamc_codes:
            for key in sorted(omc_keys & _cached_keys(t.cache_dir, t.library, backend)):
                stale = _yamc_core_sha(t.cache_dir, t.library, backend, key) != fp_sha
                if current_only and stale:
                    continue
                omc = load_cached(t.cache_dir, t.library, "openmc", key)
                ymc = load_cached(t.cache_dir, t.library, backend, key)
                if omc is None or ymc is None:
                    continue
                if t.family.startswith("transmutation"):
                    row = _transmutation_row(t, key, backend, omc, ymc, stale)
                else:
                    row = _spectrum_row(t, key, backend, omc, ymc, stale, raw=raw)
                if row is not None:
                    rows.append(row)

    rows.sort(key=lambda r: r.mismatch_key, reverse=True)
    return rows, unauditable


def _spectrum_row(t, key, backend, omc, ymc, stale, raw=False):
    if "flux_spectra_mean" not in omc or "flux_spectra_mean" not in ymc:
        return None
    expected = None
    if not raw:
        from .discovery import element_from_nuclide
        from .index_builder import _expected_spec

        element = key if getattr(t, "item_kind", None) == "element" else element_from_nuclide(key)
        expected = _expected_spec(t, element, "flux", omc, ymc) or None
    stats = calculate_spectral_agreement(omc, ymc, expected=expected)
    worst, score = _max_tally_reldiff(omc, ymc)
    # Report (and rank / threshold on) the residual chi-squared, the graded value.
    chi2 = stats.get("residual_reduced_chi_squared", stats["reduced_chi_squared"])
    return Row(task=f"{t.family}/{t.geometry}", nuclide=key, backend=backend,
               quality=stats["quality"], chi2=chi2,
               wavg_reldiff=stats["weighted_avg_rel_diff"], max_tally_reldiff=worst,
               worst_score=score, stale=stale)


def _transmutation_row(t, key, backend, omc, ymc, stale):
    from .families import transmutation as tr
    from .expected import transmutation_expected_products

    timesteps = t.extras.get("timesteps", [])
    phases = t.extras.get("phases", [])
    # Isomer ground/metastable pairs differ by an expected branching amount only
    # when the overlay is enabled; otherwise the set is empty and isomers grade
    # normally (see expected.transmutation_expected_products).
    branching_enabled = bool(t.extras.get("transmutation_branch_ratios"))
    products = set()
    for step in list(ymc.get("steps", [])) + list(omc.get("steps", [])):
        products.update(step.keys())
    expected = transmutation_expected_products(products, branching_enabled=branching_enabled)
    try:
        cmp = tr.compare_transmutation_steps(ymc, omc, timesteps, phases, expected_products=expected)
    except Exception:
        return None
    return Row(task=f"{t.family}/{t.geometry}", nuclide=key, backend=backend,
               quality=cmp.get("quality", "?"), chi2=float("nan"),
               wavg_reldiff=cmp.get("rms_rel_diff_pct", float("nan")),
               max_tally_reldiff=cmp.get("max_rel_diff_pct", float("nan")),
               worst_score="density", stale=stale)


def _is_mismatch(r, chi2_warn, reldiff_warn):
    if r.quality in ("Fair", "Poor"):
        return True
    if np.isfinite(r.chi2) and r.chi2 >= chi2_warn:
        return True
    if np.isfinite(r.max_tally_reldiff) and r.max_tally_reldiff >= reldiff_warn:
        return True
    if np.isfinite(r.wavg_reldiff) and r.wavg_reldiff >= reldiff_warn:
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit YAMC-vs-OpenMC agreement from the cache.")
    ap.add_argument("--worst", type=int, default=0, help="show only the N worst mismatches")
    ap.add_argument("--chi2", type=float, default=3.0, help="reduced chi-squared mismatch threshold")
    ap.add_argument("--reldiff", type=float, default=5.0, help="relative-difference %% mismatch threshold")
    ap.add_argument("--current-only", action="store_true", help="skip YAMC caches from an older build")
    ap.add_argument("--all", action="store_true", help="list every compared benchmark, not just mismatches")
    ap.add_argument("--raw", action="store_true",
                    help="grade on the full spectrum, ignoring expected-difference signatures")
    args = ap.parse_args(argv)

    rows, unauditable = audit(current_only=args.current_only, raw=args.raw)
    mismatches = [r for r in rows if _is_mismatch(r, args.chi2, args.reldiff)]
    shown = (rows if args.all else mismatches)
    if args.worst:
        shown = shown[: args.worst]

    stale_n = sum(1 for r in rows if r.stale)
    print(f"Compared {len(rows)} cached (nuclide, mode) pairs; "
          f"{len(mismatches)} not matching OpenMC "
          f"(reduced-chi2 >= {args.chi2} or rel.diff >= {args.reldiff}% or quality Fair/Poor)."
          + (f" [{stale_n} from an older YAMC build; --current-only to skip]" if stale_n else ""))
    print()
    hdr = (f"{'quality':10} {'chi2':>8} {'wavg%':>7} {'maxRR%':>8} {'worst':<14} "
           f"{'code':9} {'mode':28} nuclide")
    print(hdr)
    print("-" * len(hdr))
    for r in shown:
        mark = " *stale" if r.stale else ""
        chi2 = f"{r.chi2:8.2f}" if np.isfinite(r.chi2) else f"{'-':>8}"
        print(f"{r.quality:10} {chi2} {r.wavg_reldiff:7.1f} {r.max_tally_reldiff:8.1f} "
              f"{r.worst_score:<14} {r.backend:9} {r.task:28} {r.nuclide}{mark}")

    if unauditable:
        print("\nNot auditable offline (YAMC not cached for these modes — run them to compare):")
        for u in unauditable:
            print(f"  - {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
