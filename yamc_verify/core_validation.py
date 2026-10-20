"""Compare the selected core checkout with fresh OpenMC results.

Only this V&V harness removes MT=5 for an OpenMC depletion comparison. Core's
own regression tests retain the full chain and check MT=5 against the data.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess

from .core_references import TYPES, core_config, prepare_openmc_data


def scalar_check(name, expected, actual, reference_std=0.0, actual_std=0.0,
                 relative_tolerance=0.05, absolute_tolerance=1e-10):
    values = (expected, actual, reference_std, actual_std)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name}: non-finite comparison input")
    if reference_std < 0 or actual_std < 0:
        raise ValueError(f"{name}: negative uncertainty")
    tolerance = max(
        absolute_tolerance, relative_tolerance * abs(expected),
        4.0 * math.hypot(reference_std, actual_std),
    )
    return {
        "name": name, "reference": expected, "yamc": actual,
        "reference_std": reference_std, "yamc_std": actual_std,
        "tolerance": tolerance, "passed": abs(expected - actual) <= tolerance,
    }


def spectrum_check(name, expected, actual, reference_std, actual_std, limit=1.5):
    if len({len(values) for values in (expected, actual, reference_std, actual_std)}) != 1:
        raise ValueError(f"{name}: different spectrum lengths")
    values = [value for array in (expected, actual, reference_std, actual_std) for value in array]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name}: non-finite spectrum input")
    if any(value < 0 for array in (reference_std, actual_std) for value in array):
        raise ValueError(f"{name}: negative uncertainty")
    floor = max((abs(value) for array in (expected, actual) for value in array), default=0.0) * 1e-6
    terms = []
    for reference, value, ref_std, std in zip(expected, actual, reference_std, actual_std):
        if max(abs(reference), abs(value)) < floor:
            continue
        uncertainty = math.hypot(ref_std, std)
        if uncertainty == 0.0:
            if reference != value:
                return {"name": name, "passed": False, "reason": "difference with zero uncertainty"}
        else:
            terms.append(((value - reference) / uncertainty) ** 2)
    chi2 = sum(terms) / len(terms) if terms else 0.0
    return {"name": name, "chi2_dof": chi2, "bins": len(terms), "limit": limit, "passed": chi2 < limit}


def _scalar_scores(expected, actual, ref_stds, actual_stds, prefix):
    checks = []
    flux = expected["flux"]
    for score, reference in expected.items():
        value = actual[score]
        floor = flux * 1e-3 if max(abs(reference), abs(value)) < flux * 1e-3 else 1e-10
        checks.append(scalar_check(
            f"{prefix}/{score}", reference, value, ref_stds[score], actual_stds[score],
            relative_tolerance=0.10 if score in ("heating", "photoelectric") else 0.05,
            absolute_tolerance=floor,
        ))
    return checks


def compare_case(sim_type, expected, actual):
    checks = []
    if sim_type == "neutron":
        checks.extend(_scalar_scores(
            expected["scalar_results"], actual["scalar_results"],
            expected["scalar_stds"], actual["scalar_stds"], "neutron",
        ))
        checks.append(spectrum_check(
            "neutron spectrum", expected["spectrum_mean"], actual["spectrum_mean"],
            expected["spectrum_std"], actual["spectrum_std"],
        ))
    elif sim_type == "coupled":
        checks.extend(_scalar_scores(
            expected["n_scalars"], actual["n_scalars"],
            expected["n_stds"], actual["n_stds"], "neutron",
        ))
        photon, deviations = {}, {}
        for score in actual["p_scalars"]:
            reference = expected["photon"]["led" if score == "heating" else "ttb"]
            photon[score] = reference["scalars"][score]
            deviations[score] = reference["stds"][score]
        checks.extend(_scalar_scores(photon, actual["p_scalars"], deviations, actual["p_stds"], "photon"))
        checks.append(spectrum_check(
            "neutron spectrum", expected["n_spectrum_mean"], actual["n_spectrum_mean"],
            expected["n_spectrum_std"], actual["n_spectrum_std"],
        ))
        photon = expected["photon"]["ttb"]
        checks.append(spectrum_check(
            "photon spectrum", photon["spectrum_mean"], actual["p_spectrum_mean"],
            photon["spectrum_std"], actual["p_spectrum_std"], limit=3.0,
        ))
    elif sim_type == "decay_photons":
        peak_step = max(
            range(len(expected["spectra"])),
            key=lambda index: sum(abs(value) for value in expected["spectra"][index]),
            default=None,
        )
        for index, reference in enumerate(expected["spectra"]):
            value = actual["spectra"][index] if actual["spectra"] else [0.0] * len(reference)
            ref_std = expected["spectra_std"][index]
            std = actual["spectra_std"][index] if actual["spectra_std"] else [0.0] * len(reference)
            checks.append(scalar_check(
                f"step {index + 1}/decay flux", sum(reference), sum(value),
                math.sqrt(sum(error**2 for error in ref_std)),
                math.sqrt(sum(error**2 for error in std)), relative_tolerance=0.02,
                absolute_tolerance=0.0,
            ))
            spectrum = spectrum_check(
                f"step {index + 1}/decay spectrum", reference, value, ref_std, std, limit=2.0
            )
            spectrum["required"] = index == peak_step
            checks.append(spectrum)
    elif sim_type == "transmutation":
        if len(expected["steps"]) != len(actual["steps"]):
            raise ValueError("different inventory step counts")
        for index, (reference, inventory) in enumerate(zip(expected["steps"], actual["steps"])):
            for product in sorted(reference.keys() | inventory.keys()):
                ref_value, value = reference.get(product, 0.0), inventory.get(product, 0.0)
                if max(abs(ref_value), abs(value)) >= 1e-14:
                    checks.append(scalar_check(
                        f"step {index + 1}/{product}", ref_value, value,
                        relative_tolerance=0.005, absolute_tolerance=1e-14,
                    ))
    else:
        raise ValueError(f"Unknown comparison type {sim_type}")
    if not checks:
        raise ValueError(f"{sim_type}: no comparisons were made")
    return checks


def checks_pass(checks):
    required = [check for check in checks if check.get("required", True)]
    return bool(required) and all(check["passed"] for check in required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=Path(os.environ.get("YAMC_CORE_ROOT", "../core")))
    parser.add_argument("--cross-sections", default=os.environ.get("OPENMC_CROSS_SECTIONS"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation-results"))
    parser.add_argument("--types", nargs="+", choices=TYPES, default=list(TYPES))
    parser.add_argument("--nuclides", nargs="+", default=["Fe54", "Fe56", "Fe57"])
    args = parser.parse_args()
    config = core_config(args.core_dir)
    if not args.cross_sections:
        args.cross_sections = prepare_openmc_data(
            config, args.nuclides, with_depletion="transmutation" in args.types
        )
    from .core_references import OpenMCReferences
    import yamc

    references = OpenMCReferences(config, args.cross_sections)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.core_dir, text=True
    ).strip()
    report = {
        "core_commit": commit, "yamc_version": yamc.__version__,
        "openmc_version": references.openmc.__version__, "library": "endf-b8.1",
        "cross_sections_sha256": hashlib.sha256(Path(args.cross_sections).read_bytes()).hexdigest(),
        "mt5_policy": "excluded only from this OpenMC comparison; full-chain and data checks remain in core",
        "cases": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sim_type in args.types:
        module = importlib.import_module(f"regression_tests.test_{'neutron_transport' if sim_type == 'neutron' else 'coupled_transport' if sim_type == 'coupled' else sim_type}")
        runner = getattr(module, f"run_yamc_{'transmute' if sim_type == 'transmutation' else sim_type}")
        for nuclide in args.nuclides:
            if nuclide not in config.ALL_NEUTRON_NUCLIDES:
                parser.error(f"No core fixture for {nuclide}")
            if sim_type in ("coupled", "decay_photons") and nuclide not in config.COUPLED_NUCLIDES:
                parser.error(f"No core photon fixture for {nuclide}")
            print(f"[{sim_type}] {nuclide}: OpenMC and core {commit[:12]}", flush=True)
            expected = getattr(references, sim_type)(nuclide)
            if sim_type == "transmutation":
                module.NEUTRON_DATA_DIR = "endf-b8.1"
                yamc.cross_section_data = "endf-b8.1"
                actual = runner(nuclide, with_anything=False)
            else:
                actual = runner(nuclide)
            payload = {"openmc": expected, "yamc": {"steps": actual["steps"]} if sim_type == "transmutation" else actual}
            (args.output_dir / f"{sim_type}_{nuclide}.json").write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n"
            )
            checks = compare_case(sim_type, expected, actual)
            case = {"type": sim_type, "nuclide": nuclide, "checks": checks, "passed": checks_pass(checks)}
            report["cases"].append(case)
            required = [check for check in checks if check.get("required", True)]
            diagnostics = sum(not check["passed"] for check in checks if not check.get("required", True))
            print(f"  {sum(check['passed'] for check in required)}/{len(required)} required checks passed; {diagnostics} flagged diagnostics", flush=True)
    report["passed"] = all(case["passed"] for case in report["cases"])
    report["expected_cases"] = len(args.types) * len(args.nuclides)
    if len(report["cases"]) != report["expected_cases"]:
        raise ValueError("not all requested comparison cases ran")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()