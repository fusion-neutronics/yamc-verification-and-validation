"""Generate the OpenMC cases formerly used as core's regression references.

The models and schedules are shared with the selected core checkout. Results
are validation inputs, not yamc regression goldens. Both electron treatments
are retained so photon heating conventions can be reported explicitly.
"""

import argparse
from contextlib import chdir
import hashlib
import importlib
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from urllib.parse import urljoin
from urllib.request import urlopen
from xml.etree import ElementTree


TYPES = ("neutron", "coupled", "decay_photons", "transmutation")
NEUTRON_SCORES = (
    "flux", "heating", "total", "absorption", "H1-production",
    "H2-production", "H3-production", "He3-production", "He4-production",
)


def core_config(core_dir):
    tests = Path(core_dir).resolve() / "packages" / "yamc-core" / "tests"
    if not (tests / "regression_tests" / "_config.py").is_file():
        raise FileNotFoundError(f"No core regression configuration under {tests}")
    sys.path.insert(0, str(tests))
    return importlib.import_module("regression_tests._config")


def openmc_chain_file():
    import openmc_data

    path = Path(openmc_data.__file__).parent / "depletion" / "chain_endf_b8.1.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Install openmc_data's ENDF/B-VIII.1 chain: {path}")
    return path


def extract_subset(stream, destination, names):
    """Stream selected XZ-compressed HDF5 files into a relocatable library.

    LZMAFile bounds decoded reads; tarfile's direct XZ streaming retains and
    repeatedly copies huge plaintext buffers when skipping compressible files.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    filenames = {f"{name}.h5" for name in names}
    index = None
    with lzma.LZMAFile(stream) as decoded, tarfile.open(
        fileobj=decoded, mode="r|", bufsize=1024 * 1024
    ) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe data archive path: {member.name}")
            if not member.isfile():
                continue
            if path.name == "cross_sections.xml":
                with archive.extractfile(member) as source:
                    index = source.read()
            elif path.name in filenames:
                with archive.extractfile(member) as source, (destination / path.name).open("wb") as output:
                    shutil.copyfileobj(source, output)
    if index is None:
        raise ValueError("OpenMC archive has no cross_sections.xml")
    root = ElementTree.Element("cross_sections")
    materials = set()
    for library in ElementTree.fromstring(index).findall("library"):
        filename = PurePosixPath(library.attrib["path"]).name
        if filename in filenames and (destination / filename).is_file():
            library.set("path", filename)
            root.append(library)
            materials.update(library.attrib["materials"].split())
    if not materials:
        raise ValueError("OpenMC archive contains none of the requested materials")
    ElementTree.ElementTree(root).write(
        destination / "cross_sections.xml", encoding="utf-8", xml_declaration=True
    )
    return materials


def prepare_openmc_data(config, nuclides, with_depletion=True):
    """Cache the official VIII.1 HDF5 subset needed by the selected cases."""
    import openmc.deplete
    from openmc_data.urls_h5 import all_h5_release_details

    unknown = set(nuclides) - set(config.ALL_NEUTRON_NUCLIDES)
    if unknown:
        raise ValueError(f"No core fixtures for {sorted(unknown)}")
    required = set(nuclides)
    required.update(config.NUCLIDE_TO_ELEMENT[name] for name in nuclides if name in config.NUCLIDE_TO_ELEMENT)
    names = set(required)
    chain_file = openmc_chain_file()
    if with_depletion:
        chain = openmc.deplete.Chain.from_xml(chain_file).reduce(nuclides, level=20)
        names.update(nuclide.name for nuclide in chain.nuclides)
    details = all_h5_release_details["endf"]["b8.1"]["neutron-photon"]
    url = urljoin(details["base_url"], details["compressed_files"][0])
    provenance = {
        "library": "endf-b8.1", "url": url, "materials": sorted(names),
        "chain_sha256": hashlib.sha256(chain_file.read_bytes()).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    destination = cache / "yamc-verification" / "core-comparison" / key
    index = destination / "cross_sections.xml"
    if index.is_file():
        libraries = ElementTree.parse(index).getroot().findall("library")
        available = {name for library in libraries for name in library.attrib["materials"].split()}
        if required <= available and all((destination / library.attrib["path"]).is_file() for library in libraries):
            return index
    print(f"Preparing OpenMC VIII.1 data for {len(names)} materials from {url}", flush=True)
    with urlopen(url, timeout=120) as stream:
        available = extract_subset(stream, destination, names)
    if not required <= available:
        raise ValueError(f"Missing OpenMC materials: {sorted(required - available)}")
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return index


class OpenMCReferences:
    def __init__(self, config, cross_sections):
        import openmc
        import openmc.deplete
        from openmc.deplete import d1s

        self.openmc = openmc
        self.d1s = d1s
        self.config = config
        self.chain = openmc_chain_file()
        cross_sections = Path(cross_sections).resolve()
        if not cross_sections.is_file():
            raise FileNotFoundError(f"OpenMC cross sections not found: {cross_sections}")
        openmc.config["cross_sections"] = str(cross_sections)
        openmc.config["chain_file"] = str(self.chain)

    def _model(self, nuclide, photon=False):
        self.openmc.reset_auto_ids()
        material = self.openmc.Material(name=nuclide)
        material.add_nuclide(nuclide, 1.0, percent_type="ao")
        material.set_density("g/cm3", 1.0)
        material.temperature = 294.0
        cylinder = self.openmc.ZCylinder(
            r=self.config.CYLINDER_RADIUS, boundary_type="vacuum"
        )
        bottom = self.openmc.ZPlane(
            z0=-self.config.CYLINDER_HALF_HEIGHT, boundary_type="vacuum"
        )
        top = self.openmc.ZPlane(
            z0=self.config.CYLINDER_HALF_HEIGHT, boundary_type="vacuum"
        )
        cell = self.openmc.Cell(region=-cylinder & +bottom & -top, fill=material)
        settings = self.openmc.Settings()
        settings.run_mode = "fixed source"
        settings.particles = self.config.PARTICLES
        settings.batches = self.config.BATCHES
        settings.seed = self.config.SEED
        settings.source = self.openmc.IndependentSource(
            space=self.openmc.stats.Point((0, 0, 0)),
            angle=self.openmc.stats.Isotropic(),
            energy=self.openmc.stats.Discrete([self.config.NEUTRON_SOURCE_ENERGY], [1.0]),
            particle="neutron",
        )
        settings.photon_transport = photon
        settings.output = {"tallies": False, "summary": False}
        model = self.openmc.Model(
            self.openmc.Geometry([cell]), self.openmc.Materials([material]), settings
        )
        return model, cell

    def _tally(self, name, cell, particle, scores, groups=None):
        tally = self.openmc.Tally(name=name)
        tally.filters = [self.openmc.CellFilter([cell]), self.openmc.ParticleFilter([particle])]
        if groups:
            tally.filters.append(self.openmc.EnergyFilter.from_group_structure(groups))
        tally.scores = list(scores)
        return tally

    @staticmethod
    def _scalar(statepoint, name):
        tally = statepoint.get_tally(name=name)
        return float(tally.mean.ravel()[0]), float(tally.std_dev.ravel()[0])

    @staticmethod
    def _spectrum(statepoint, name):
        tally = statepoint.get_tally(name=name)
        return tally.mean.ravel().tolist(), tally.std_dev.ravel().tolist()

    def neutron(self, nuclide):
        model, cell = self._model(nuclide)
        tallies = [self._tally(score, cell, "neutron", [score]) for score in NEUTRON_SCORES]
        tallies.append(self._tally(
            "spectrum", cell, "neutron", ["flux"], self.config.NEUTRON_GROUP_STRUCTURE
        ))
        model.tallies = self.openmc.Tallies(tallies)
        with tempfile.TemporaryDirectory() as work:
            path = model.run(cwd=work, output=False)
            with self.openmc.StatePoint(path) as statepoint:
                scalars, deviations = {}, {}
                for score in NEUTRON_SCORES:
                    scalars[score], deviations[score] = self._scalar(statepoint, score)
                mean, std = self._spectrum(statepoint, "spectrum")
        return {
            "scalar_results": scalars, "scalar_stds": deviations,
            "spectrum_mean": mean, "spectrum_std": std,
        }

    def _coupled(self, nuclide, treatment):
        import math

        model, cell = self._model(nuclide, photon=True)
        model.settings.electron_treatment = treatment
        tallies = [
            self._tally(f"n_{score}", cell, "neutron", [score])
            for score in NEUTRON_SCORES
        ] + [
            self._tally(f"p_{score}", cell, "photon", [score])
            for score in self.config.PHOTON_SCALAR_SCORES if score != "heating"
        ]
        for particle in ("photon", "electron", "positron"):
            tallies.append(self._tally(f"heating_{particle}", cell, particle, ["heating"]))
        for particle, groups in (
            ("neutron", self.config.NEUTRON_GROUP_STRUCTURE),
            ("photon", self.config.PHOTON_GROUP_STRUCTURE),
        ):
            tallies.append(self._tally(f"{particle}_spectrum", cell, particle, ["flux"], groups))
        model.tallies = self.openmc.Tallies(tallies)
        with tempfile.TemporaryDirectory() as work:
            path = model.run(cwd=work, output=False)
            with self.openmc.StatePoint(path) as statepoint:
                neutron, photon = {"scalars": {}, "stds": {}}, {"scalars": {}, "stds": {}}
                for score in NEUTRON_SCORES:
                    neutron["scalars"][score], neutron["stds"][score] = self._scalar(
                        statepoint, f"n_{score}"
                    )
                for score in self.config.PHOTON_SCALAR_SCORES:
                    if score != "heating":
                        photon["scalars"][score], photon["stds"][score] = self._scalar(
                            statepoint, f"p_{score}"
                        )
                heat = [self._scalar(statepoint, f"heating_{p}") for p in ("photon", "electron", "positron")]
                photon["scalars"]["heating"] = sum(value for value, _ in heat)
                photon["stds"]["heating"] = math.sqrt(sum(std**2 for _, std in heat))
                for particle, result in (("neutron", neutron), ("photon", photon)):
                    result["spectrum_mean"], result["spectrum_std"] = self._spectrum(
                        statepoint, f"{particle}_spectrum"
                    )
        return neutron, photon

    def coupled(self, nuclide):
        photon = {}
        for treatment in ("led", "ttb"):
            neutron, photon[treatment] = self._coupled(nuclide, treatment)
        return {
            "n_scalars": neutron["scalars"], "n_stds": neutron["stds"],
            "n_spectrum_mean": neutron["spectrum_mean"],
            "n_spectrum_std": neutron["spectrum_std"], "photon": photon,
        }

    def decay_photons(self, nuclide):
        model, cell = self._model(nuclide, photon=True)
        model.settings.use_decay_photons = True
        tally = self._tally(
            "d1s_spectrum", cell, "photon", ["flux"], self.config.PHOTON_GROUP_STRUCTURE
        )
        model.tallies = self.openmc.Tallies([tally])
        self.d1s.prepare_tallies(model=model)
        radionuclides = self.d1s.get_radionuclides(model)
        with tempfile.TemporaryDirectory() as work:
            path = model.run(cwd=work, output=False)
            with self.openmc.StatePoint(path) as statepoint:
                result = statepoint.get_tally(name="d1s_spectrum")
                result.mean
                result.std_dev
            factors = self.d1s.time_correction_factors(
                nuclides=radionuclides, timesteps=self.config.DECAY_PHOTON_TIMESTEPS,
                source_rates=self.config.DECAY_PHOTON_SOURCE_RATES, timestep_units="s",
            )
            spectra, deviations = [], []
            for index in range(len(self.config.DECAY_PHOTON_TIMESTEPS)):
                corrected = self.d1s.apply_time_correction(
                    tally=result, time_correction_factors=factors,
                    index=index + 1, sum_nuclides=True,
                )
                spectra.append(corrected.mean.ravel().tolist())
                deviations.append(corrected.std_dev.ravel().tolist())
        return {"spectra": spectra, "spectra_std": deviations}

    def transmutation(self, nuclide):
        chain = self.openmc.deplete.Chain.from_xml(self.chain).reduce([nuclide], level=20)
        material = self.openmc.Material(name=nuclide)
        material.add_nuclide(nuclide, 1.0, percent_type="ao")
        material.set_density("g/cm3", 1.0)
        material.volume = 1.0
        material.temperature = 294.0
        material.depletable = True
        flux = sum(self.config.TRANSMUTE_MULTIGROUP_FLUX)
        with tempfile.TemporaryDirectory() as work, chdir(work):
            result = material.deplete(
                multigroup_flux=self.config.TRANSMUTE_MULTIGROUP_FLUX,
                energy_group_structure=self.config.TRANSMUTE_ENERGY_GROUPS,
                timesteps=self.config.TRANSMUTE_TIMESTEPS,
                source_rates=[rate * flux for rate in self.config.TRANSMUTE_SOURCE_RATES],
                timestep_units="s", chain_file=chain,
            )
            steps = [
                {name: float(value) for name, value in mat.get_nuclide_atom_densities().items()}
                for mat in result[1:]
            ]
        return {"steps": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=Path(os.environ.get("YAMC_CORE_ROOT", "../core")))
    parser.add_argument("--cross-sections", default=os.environ.get("OPENMC_CROSS_SECTIONS"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation-results/openmc"))
    parser.add_argument("--types", nargs="+", choices=TYPES, default=list(TYPES))
    parser.add_argument("--nuclides", nargs="+")
    args = parser.parse_args()
    config = core_config(args.core_dir)
    if not args.cross_sections:
        args.cross_sections = prepare_openmc_data(
            config, args.nuclides or config.ALL_NEUTRON_NUCLIDES,
            with_depletion="transmutation" in args.types,
        )
    references = OpenMCReferences(config, args.cross_sections)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sim_type in args.types:
        nuclides = config.ALL_NEUTRON_NUCLIDES if sim_type in ("neutron", "transmutation") else config.COUPLED_NUCLIDES
        if args.nuclides:
            nuclides = [nuclide for nuclide in nuclides if nuclide in args.nuclides]
        for nuclide in nuclides:
            print(f"[OpenMC/{sim_type}] {nuclide}", flush=True)
            data = getattr(references, sim_type)(nuclide)
            data.update({"generator": "openmc", "version": references.openmc.__version__, "library": "endf-b8.1"})
            path = args.output_dir / f"{sim_type}_{nuclide}.json"
            path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()