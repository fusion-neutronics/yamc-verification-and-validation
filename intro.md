# YAMC Verification

This project provides verification of [YAMC](https://github.com/yamc/yamc) by comparing simulation results with [OpenMC](https://openmc.org/) across various:

- **Geometries**
- **Nuclear data libraries**
- **Materials**

## What's Compared

For each simulation, we compare:

1. **Flux spectra**: Energy-binned neutron flux using VITAMIN-J-175 group structure
2. **Reaction rates**: All available reaction channels (flux, heating, (n,2n), etc.)
3. **GPU and CPU**: Both compute options are compared
