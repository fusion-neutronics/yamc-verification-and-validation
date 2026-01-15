# [
#     {
#         "nuclear_data_library": "endfb8.0",
#         "model": "sphere",
#         "spectra": [1,2,3],
#         "n,2n": 1
#     },
# ]


import numpy as np
import glob
import os


def run_verification(
    geometry: str,
    nuclide: str,
    nuclear_data: str,
) -> dict:

    results = []
    for code in ['openmc', 'yamc']:
        if geometry == 'sphere':
            result = sphere_geometry(nuclide, code, nuclear_data)
            results.append(result)
    return results

def sphere_geometry(
    nuclide,
    code,
    nuclear_data
):

    for code in ['openmc', 'yamc']:
        if code == 'openmc':
            import openmc as mc
            mc.config['cross_sections'] = f"{NUCLEAR_DATA_DIR[nuclear_data]}cross_sections.xml"
        elif code == 'yamc':
            import yamc as mc

    sphere1 = mc.Sphere(
        surface_id=2,
        x0=0.0,
        y0=0.0,
        z0=0.0,
        r=35.0,
        boundary_type='vacuum',
    )
    region1 = -sphere1

    material1 = mc.Material()
    material1.material_id = 1
    material1.add_nuclide(nuclide, 1.0)
    material1.set_density("g/cm3", 1.0)

    if code == 'openmc':
        material1.temperature = 294
    else:
        material1.temperature = '294'

    if code == 'yamc':
        print(f"Reading nuclide data for {nuclide}...")
        # Load from OpenMC-format HDF5 file
        material1.read_nuclides_from_h5({nuclide: f"{NUCLEAR_DATA_DIR[nuclear_data]}neutron/{nuclide}.h5"})

    # Create cells
    cell1 = mc.Cell(
        cell_id=1,
        name="inner_sphere",
        region=region1,
        fill=material1
    )

    geometry = mc.Geometry([cell1])

    particles = 50000
    batches = 10

    cell_filter = mc.CellFilter(cell1)

    tally1 = mc.Tally()
    tally1.filters = [cell_filter]
    tally1.scores = ['flux']
    tally1.name = "flux"
    tallies = [tally1]

    tally2 = mc.Tally()
    tally2.filters = [cell_filter]
    tally2.scores = ['heating']
    tally2.name = "heating"

    source = mc.IndependentSource(
        angle=mc.stats.Isotropic(),
        energy=mc.stats.Discrete([14.06e6], [1.0]),
        space=mc.stats.Point([0,0,0])
    )
    settings = mc.Settings(
        particles=particles,
        batches=batches,
        source=source,
        seed=1
    )

    if code == 'openmc':
        settings.run_mode='fixed source'
        settings.output={'tallies': False, 'summary': False}
        in_neu = mc.data.IncidentNeutron.from_hdf5(f"{NUCLEAR_DATA_DIR[nuclear_data]}/{nuclide}.h5")
        mts_int = list(in_neu.reactions.keys())+ ['total']
        mts = [str(mt) for mt in mts_int]

    elif code == 'yamc':
        mts = material1.reaction_mts

    tallies = [tally1, tally2]

    for mt in mts:
        tally_mt = mc.Tally()
        tally_mt.filters = [cell_filter]
        tally_mt.scores = [mt]
        tally_mt.name = str(mt)
        tallies.append(tally_mt)


    # Create energy-binned flux tally
    energy_filter = mc.EnergyFilter.from_group_structure('VITAMIN-J-175')

    tally3 = mc.Tally()
    tally3.filters = [cell_filter, energy_filter]
    tally3.scores = ['flux']
    tally3.name = "flux_energy_binned"
    tallies.append(tally3)

    model = mc.Model(geometry=geometry, settings=settings, tallies=tallies)

    if code == 'openmc':
        model.run(apply_tally_results=True)
    elif code == 'yamc':
        model.run()

    results_data = {'nuclide': nuclide, 'code': code}
    if code == 'openmc':
        sp_filename = model.run(apply_tally_results=True)
        sp = mc.StatePoint(sp_filename)
        transport_time = sp.runtime['transport']
        total_particles = particles * batches
        particles_per_second = total_particles / transport_time
        results_data['particles/s'] =  particles_per_second
        for tally in tallies:
            result = tally.mean.flatten()[0]
            # print(tally.name, result)
            results_data[tally.name] = result
    elif code == 'yamc':
        results_data['particles/s'] = model.particles_per_second
        for tally in tallies:
            result = tally.mean[0]
            # print(tally.name, result)
            results_data[tally.name] = result
    
    mean_flux = np.array(tally3.mean).squeeze()
    std_flux = np.array(tally3.std_dev).squeeze()
    results_data['flux_spectra_mean'] = mean_flux
    results_data['flux_spectra_std_dev'] = std_flux

    return results_data

NUCLEAR_DATA_DIR = {
    "endf-b8.0": "/home/jon/nuclear_data/endf-b8.0-hdf5/endfb-viii.0-hdf5/",
    "tendl-2021": "/home/jon/nuclear_data/tendl-2021-hdf5/tendl-2021-hdf5/",
}



all_results = []
for nuclear_data in ['endf-b8.0']:
    h5_files = sorted(glob.glob(os.path.join(NUCLEAR_DATA_DIR[nuclear_data],'neutron', "*.h5")))

    nuclides = [os.path.splitext(os.path.basename(f))[0] for f in h5_files]

    # nuclides = ['Cm244', 'Cm246', 'Cf251', 'Cf252', 'Np239', 'Pa233', 'Pu239', 'Pu240']

    for geometry in ['sphere']:
        for nuclide in nuclides:
            both_code_results = run_verification(
                geometry=geometry,
                nuclide=nuclide,
                nuclear_data=nuclear_data,
            )
            all_results += both_code_results

print(all_results)