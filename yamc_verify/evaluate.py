"""Test your own ENDF evaluation against the OKTAVIAN measurements.

    python -m yamc_verify.evaluate --sphere Al27 --neutron-file my-Al27.endf --label "my Al27"
    python -m yamc_verify.evaluate --sphere Mn55 --neutron-file my-Mn55.endf --label "new Mn55" \\
        --compare endf-b8.1 tendl-2025 jeff-4.0
    python -m yamc_verify.evaluate --sphere Al27 --compare endf-b8.1 tendl-2025

Runs the OKTAVIAN sphere of ``--sphere`` (see
:mod:`yamc_verify.families.oktavian`) once per evaluation of its sample
nuclide and compares each with the measurement. ``--neutron-file`` is any
ENDF-6 neutron evaluation of that nuclide, processed here with NJOY at 294 K
and cached by the file's sha256, so a changed file is converted again; it is
named by ``--label`` in the plot and the JSON. ``--compare`` adds published
libraries by yamc keyword (``tendl-2025``, ``tendl-2017``, ``endf-b8.1``,
``jeff-4.0``, ``jendl-5.0``, ``fendl-3.2d``).

Only the sample's neutron data changes between the curves. The vessel steel's
neutron data and every element's photon data come from ``--library``
(ENDF/B-VIII.1 unless named) and are the same for every evaluation, so the
differences between the curves come from the sample's evaluation alone.

Writes ``oktavian_<sphere>.html`` (the leakage spectra with the measurement,
C/E below) and ``oktavian_<sphere>.json`` (for each evaluation, where its data
came from, the calculated spectra with their Monte Carlo errors and the
goodness of fit) into ``--output-dir``. None of this is a record: the
verification and validation runs, and the site, are untouched.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

from . import tasks
from .families import oktavian

LIBRARIES = {
    "tendl-2025": "TENDL-2025",
    "tendl-2017": "TENDL-2017",
    "endf-b8.1": "ENDF/B-VIII.1",
    "jeff-4.0": "JEFF-4.0",
    "jendl-5.0": "JENDL-5.0",
    "fendl-3.2d": "FENDL-3.2d",
}
"""The library keywords ``--compare`` and ``--library`` take, and their display names."""

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_of(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_evaluation(endf_path: Path | str, item: str) -> str:
    """Convert an ENDF-6 neutron evaluation of ``item`` with NJOY at 294 K; return its Arrow path.

    The conversion is cached by the file's sha256 under
    ``~/.cache/yamc-verification/oktavian/user/``. An evaluation of another
    nuclide is refused, naming it.
    """
    endf = Path(endf_path).expanduser().resolve()
    if not endf.is_file():
        raise FileNotFoundError(f"--neutron-file {endf} does not exist")
    out_dir = oktavian._download_dir() / "user" / sha256_of(endf)[:16]
    existing = sorted(out_dir.glob("*.arrow"))
    if not existing:
        import yamc

        print(f"converting {endf} (NJOY, 294 K)")
        written = yamc.convert_neutron_transport(str(endf), str(out_dir), njoy_exec=shutil.which("njoy") or "njoy",
                                                 temperatures=[294.0], library=f"user:{endf.name}")
        existing = [Path(written)]
    found = existing[0].name.removesuffix(".arrow")
    if found != item:
        raise ValueError(f"{endf.name} is an evaluation of {found}, but the sphere is {item}: pass an "
                         f"evaluation of {item}, or choose the sphere with --sphere")
    return str(existing[0])


def cases(item: str, neutron_file=None, label=None, compare=()) -> list[tuple[str, dict, str]]:
    """``(label, where the data came from, sample data)`` for each evaluation, the file first."""
    out = []
    if neutron_file is not None:
        if not label:
            raise ValueError("--neutron-file needs --label, the name to show for it")
        path = Path(neutron_file).expanduser().resolve()
        arrow = convert_evaluation(path, item)
        out.append((label, {"file": str(path), "sha256": sha256_of(path), "arrow": arrow}, arrow))
    for keyword in compare:
        if keyword not in LIBRARIES:
            raise ValueError(f"unknown library {keyword!r}; libraries: {list(LIBRARIES)}")
        out.append((LIBRARIES[keyword], {"library": keyword}, keyword))
    if not out:
        raise ValueError("nothing to evaluate: give --neutron-file and --label, or --compare libraries")
    return out


def evaluate(item: str, evaluations, *, library: str = "endf-b8.1", particles: int | None = None) -> dict:
    """Run the sphere once per evaluation and compare each with the measurement."""
    task = next(t for t in tasks.all_tasks() if t.family == "oktavian")
    task = dataclasses.replace(task, library=library, particles=particles or task.particles)
    experiment = oktavian.load_measurement(task, item, REPO_ROOT)
    report = {
        "benchmark": f"OKTAVIAN {item} sphere ({oktavian.REPORT})",
        "sphere": item,
        "vessel_and_photon_data": f"{library}, the same for every evaluation",
        "source_neutrons_per_evaluation": task.particles,
        "measurement": json.loads(oktavian.measurement_path(task, item, REPO_ROOT).read_text())["code_identity"],
        "evaluations": [],
    }
    for label, source, data in evaluations:
        print(f"=== {label}")
        results = oktavian.run_yamc(item, task, experiment, "cpu", sample_data=data)
        fit = oktavian.compare(experiment, results)
        report["evaluations"].append({"label": label, "source": source, "results": results, "fit": fit})
    return report


def figure(report: dict, experiment: dict):
    """The leakage spectra of every evaluation with the measurement, C/E below (a plotly figure)."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2, shared_xaxes="columns", row_heights=[0.72, 0.28], vertical_spacing=0.06,
                        subplot_titles=("Neutron leakage", "Photon leakage", "", ""))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
    for col, (particle, bins_key) in enumerate((("neutron", "n_energy_bins"), ("photon", "p_energy_bins")), start=1):
        bins = experiment[bins_key]
        centers = [(a * b) ** 0.5 if particle == "neutron" else (a + b) / 2 for a, b in zip(bins, bins[1:])]
        for i, case in enumerate(report["evaluations"]):
            r, color = case["results"], colors[i % len(colors)]
            mean, std = r[f"{particle}_leakage_mean"], r[f"{particle}_leakage_std"]
            x = [e for a, b in zip(bins, bins[1:]) for e in (a, b)]
            # The evaluation under test (the first) is drawn thicker, over the references.
            width = 2.5 if i == 0 else 1.2
            fig.add_scatter(x=x, y=[v for v in mean for _ in (0, 1)], mode="lines", name=case["label"],
                            legendgroup=case["label"], showlegend=col == 1, line={"color": color, "width": width},
                            row=1, col=col)
            fig.add_scatter(x=x, y=[v + s for v, s in zip(mean, std) for _ in (0, 1)], mode="lines",
                            line={"width": 0, "color": color}, showlegend=False, hoverinfo="skip", row=1, col=col)
            fig.add_scatter(x=x, y=[max(v - s, 0) for v, s in zip(mean, std) for _ in (0, 1)], mode="lines",
                            line={"width": 0, "color": color}, fill="tonexty", opacity=0.25, showlegend=False,
                            hoverinfo="skip", row=1, col=col)
            ce = case["fit"][particle]["c_over_e"]
            fig.add_scatter(x=x, y=[v for v in ce for _ in (0, 1)], mode="lines", showlegend=False,
                            legendgroup=case["label"], line={"color": color, "width": width}, row=2, col=col)
        fig.add_scatter(x=centers, y=experiment[f"{particle}_leakage"], mode="markers", name="OKTAVIAN measurement",
                        legendgroup="measurement", showlegend=col == 1, marker={"color": "black", "size": 4},
                        error_y={"type": "data", "array": experiment[f"{particle}_leakage_error"], "thickness": 1},
                        row=1, col=col)
        fig.update_yaxes(type="log", exponentformat="e", row=1, col=col,
                         title_text="per unit lethargy per source n" if particle == "neutron" else "per MeV per source n")
        fig.update_yaxes(range=[0, 2], title_text="C/E", row=2, col=col)
        # both rows: shared x-axes still each need their own type
        fig.update_xaxes(type="log" if particle == "neutron" else "linear", exponentformat="e", col=col)
        fig.update_xaxes(title_text="Energy (eV)", row=2, col=col)
        fig.add_hline(y=1, line={"color": "gray", "dash": "dash", "width": 1}, row=2, col=col)
    fig.update_layout(title=report["benchmark"] + ", vessel and photon data: " + report["vessel_and_photon_data"],
                      height=720, template="plotly_white")
    return fig


def summary(report: dict) -> str:
    """The goodness of fit of every evaluation, as a table."""
    lines = [f"{'':32s} {'':8s} {'chi2/N':>9s} {'C/E':>7s} {'rms ln(C/E)':>12s} {'max MC err':>11s}"]
    for case in report["evaluations"]:
        for particle, fit in case["fit"].items():
            if not fit["bins"]:
                lines.append(f"{case['label'][:32]:32s} {particle:8s} no measured bins")
                continue
            lines.append(f"{case['label'][:32]:32s} {particle:8s} {fit['chi2_per_bin']:9.3g} "
                         f"{fit['integral_c_over_e']:7.3f} {fit['bin_deviation_percent']:11.1f}% "
                         f"{fit['mc_error_max_percent']:10.1f}%")
            if fit["mc_error_max_percent"] > 5:
                lines.append(f"  some {particle} bins have a Monte Carlo error above 5%: raise --particles "
                             "for a sharper comparison")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m yamc_verify.evaluate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sphere", choices=sorted(oktavian.SPHERES), required=True, help="the sphere, by its sample nuclide")
    ap.add_argument("--neutron-file", type=Path, help="an ENDF-6 neutron evaluation of the sample nuclide to test")
    ap.add_argument("--label", help="the name of the --neutron-file evaluation in the plot and JSON (required with it)")
    ap.add_argument("--compare", nargs="*", default=[], metavar="LIBRARY",
                    help="published libraries to compare with, by yamc keyword: " + ", ".join(LIBRARIES))
    ap.add_argument("--library", default="endf-b8.1", choices=list(LIBRARIES),
                    help="the vessel's neutron data and the photon data, the same for every evaluation "
                         "(default: endf-b8.1)")
    ap.add_argument("--particles", type=int, help="source neutrons per evaluation (default: the task's, 1e7)")
    ap.add_argument("--output-dir", type=Path, default=Path("."), help="where to write the HTML and JSON")
    args = ap.parse_args(argv)
    try:
        evaluations = cases(args.sphere, args.neutron_file, args.label, args.compare)
    except (ValueError, FileNotFoundError) as e:
        ap.error(str(e))
    report = evaluate(args.sphere, evaluations, library=args.library, particles=args.particles)
    task = next(t for t in tasks.all_tasks() if t.family == "oktavian")
    experiment = oktavian.load_measurement(task, args.sphere, REPO_ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    html = args.output_dir / f"oktavian_{args.sphere}.html"
    out_json = args.output_dir / f"oktavian_{args.sphere}.json"
    figure(report, experiment).write_html(html, include_plotlyjs="cdn")
    out_json.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print()
    print(summary(report))
    print(f"\nwrote {html}\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
