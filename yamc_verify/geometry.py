"""Shared geometry builders for the verification models.

Within a physics family the sphere and broomstick notebooks differ *only* in
the geometry block, so the region construction is factored here and injected
into the shared runners. Two geometries are supported:

- ``"sphere"``: a single vacuum-bounded sphere, source at the centre.
- ``"broomstick"``: a thin z-aligned cylinder capped by two z-planes (all
  vacuum), source at the origin -- a long thin "broomstick" for leakage spectra.

Region *labels* (surface/cell ids and names) do not affect Monte-Carlo tally
results, so the construction is standardised here; only the physical dimensions
matter and they are passed in.
"""
from __future__ import annotations

from dataclasses import dataclass

SPHERE = "sphere"
BROOMSTICK = "broomstick"


@dataclass(frozen=True)
class GeometryDims:
    """Physical dimensions (cm) for the supported geometries."""

    sphere_radius: float = 35.0
    cylinder_radius: float = 1.0
    cylinder_half_height: float = 50.0


def yamc_region(geometry: str, dims: GeometryDims):
    """Return the yamc ``Region`` for ``geometry`` (source region interior)."""
    import yamc as mc

    if geometry == SPHERE:
        sphere = mc.Sphere(radius=dims.sphere_radius, boundary="vacuum")
        return sphere.below
    if geometry == BROOMSTICK:
        cylinder = mc.Cylinder(axis="z", radius=dims.cylinder_radius, boundary="vacuum")
        z_bot = mc.Plane(axis="z", offset=-dims.cylinder_half_height, boundary="vacuum")
        z_top = mc.Plane(axis="z", offset=dims.cylinder_half_height, boundary="vacuum")
        return cylinder.below & z_bot.above & z_top.below
    raise ValueError(f"unknown geometry {geometry!r}")


def openmc_region(geometry: str, dims: GeometryDims):
    """Return the OpenMC ``Region`` for ``geometry`` (source region interior)."""
    import openmc

    if geometry == SPHERE:
        sphere = openmc.Sphere(r=dims.sphere_radius, boundary_type="vacuum")
        return -sphere
    if geometry == BROOMSTICK:
        cylinder = openmc.ZCylinder(r=dims.cylinder_radius, boundary_type="vacuum")
        z_bot = openmc.ZPlane(z0=-dims.cylinder_half_height, boundary_type="vacuum")
        z_top = openmc.ZPlane(z0=dims.cylinder_half_height, boundary_type="vacuum")
        return -cylinder & +z_bot & -z_top
    raise ValueError(f"unknown geometry {geometry!r}")
