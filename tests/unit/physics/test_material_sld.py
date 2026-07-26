from __future__ import annotations

import numpy as np
import periodictable
import pytest

from xrr_fitter.model.structure import MaterialSpec
from xrr_fitter.physics.materials import material_sld


def test_material_sld_preserves_angstrom_inverse_squared_units() -> None:
    material = MaterialSpec("Si", "Si", 2.329)
    real, absorption = periodictable.xray_sld(
        periodictable.formula("Si"), density=2.329, wavelength=1.5406
    )[:2]
    assert material_sld(material, 1.0, 1.5406) == complex(real, abs(absorption)) * 1e-6


def test_material_sld_scales_mass_density() -> None:
    material = MaterialSpec("Si", "Si", 2.329)
    base = material_sld(material, 1.0, 1.5406)
    np.testing.assert_allclose(material_sld(material, 0.73, 1.5406), 0.73 * base)


def test_direct_sld_material_uses_passive_absorption_sign() -> None:
    material = MaterialSpec("custom", None, None, 12e-6 + 0.4e-6j)
    assert material_sld(material, 0.5, 1.5406) == 6e-6 + 0.2e-6j


def test_direct_sld_rejects_active_gain_sign() -> None:
    with pytest.raises(ValueError, match="nonnegative absorption"):
        MaterialSpec("gain", None, None, 12e-6 - 0.4e-6j)


@pytest.mark.parametrize("wavelength", [-1.0, 0.0, float("inf"), float("nan")])
def test_material_sld_rejects_invalid_wavelength(wavelength: float) -> None:
    with pytest.raises(ValueError, match="wavelength_a"):
        material_sld(MaterialSpec("Si", "Si", 2.329), 1.0, wavelength)


def test_material_sld_rejects_nonfinite_density_scale() -> None:
    material = MaterialSpec("Si", "Si", 2.329)
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="density scale"):
            material_sld(material, value, 1.5406)
