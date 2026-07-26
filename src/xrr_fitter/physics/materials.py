"""Wavelength-dependent scattering-length densities."""

from __future__ import annotations

from math import isfinite

import periodictable

from xrr_fitter.model.structure import MaterialSpec


def _require_positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _formula_sld(
    material: MaterialSpec,
    density_scale: float,
    wavelength_a: float,
) -> complex:
    assert material.formula is not None and material.bulk_density_g_cm3 is not None
    real, absorption = periodictable.xray_sld(
        periodictable.formula(material.formula),
        density=material.bulk_density_g_cm3 * density_scale,
        wavelength=wavelength_a,
    )[:2]
    return complex(real, abs(absorption)) * 1e-6


def material_sld(material: MaterialSpec, density_scale: float, wavelength_a: float) -> complex:
    """Return a passive complex SLD in inverse square angstroms."""
    _require_positive(density_scale, "density scale")
    _require_positive(wavelength_a, "wavelength_a")
    if material.sld_override_a2 is not None:
        return material.sld_override_a2 * density_scale
    return _formula_sld(material, density_scale, wavelength_a)
