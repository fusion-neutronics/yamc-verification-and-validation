# YAMC Verification

This project provides verification of [YAMC](https://github.com/yamc/yamc) by comparing simulation results against the gold standard Monte Carlo code - [OpenMC](https://openmc.org/).

## Purpose

YAMC is a Monte Carlo particle transport code. To ensure accuracy, we compare its results against OpenMC across various:

- **Geometries**: Starting with simple sphere models
- **Nuclear data libraries**: ENDF/B-VIII.0, TENDL-2021, etc.
- **Nuclides**: All available nuclides in each library

## What's Compared

For each nuclide, we compare:

1. **Flux spectra**: Energy-binned neutron flux using VITAMIN-J-175 group structure
2. **Reaction rates**: All available reaction channels (flux, heating, (n,2n), etc.)
3. **Performance**: Particles simulated per second

## Navigation

Use the sidebar to navigate between different models and nuclear data libraries:

- **Sphere Model**: A simple 35 cm radius sphere filled with a single nuclide
  - ENDF/B-VIII.0: Results using the ENDF/B-VIII.0 nuclear data library

## Methodology

Each notebook:
1. Runs simulations with both YAMC and OpenMC using identical settings
2. Collects flux spectra, tally results, and timing data
3. Displays interactive Plotly plots comparing the results
4. Shows ratio plots with uncertainty bands
5. Lists all tallies sorted by relative difference
