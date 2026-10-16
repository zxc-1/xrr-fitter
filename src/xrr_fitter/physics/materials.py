"""Wavelength-dependent scattering-length densities."""

from __future__ import annotations

from functools import lru_cache
from math import isfinite

import periodictable

from xrr_fitter.model.structure import MaterialSpec

CLASSICAL_ELECTRON_RADIUS_A = 2.8179403262e-5


def _require_positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _finite_sld(value: complex) -> complex:
    if not isfinite(value.real) or not isfinite(value.imag):
        raise ValueError("material SLD must be finite")
    if value.imag < 0.0:
        raise ValueError("material SLD must have nonnegative absorption")
    return value


@lru_cache(maxsize=256)
def _parsed_formula(formula: str):
    """Parse each declared formula once while leaving SLD inputs dynamic.

    periodictable 的语法错误是 pyparsing 的 ``ParseException``，不是 ``ValueError``；而服务层
    和 GUI 都按 ``ValueError`` 收拒绝，漏过去的那一种会直接落进 excepthook。pyparsing 不是本
    仓库声明的依赖，所以按异常基类收，原异常挂在 ``__cause__`` 上、原文留在消息里。
    """
    try:
        return periodictable.formula(formula)
    except Exception as error:
        raise ValueError(f"invalid material formula {formula!r}: {error}") from error


def _formula_sld(
    material: MaterialSpec,
    density_scale: float,
    wavelength_a: float,
) -> complex:
    assert material.formula is not None and material.bulk_density_g_cm3 is not None
    real, absorption = periodictable.xray_sld(
        _parsed_formula(material.formula),
        density=material.bulk_density_g_cm3 * density_scale,
        wavelength=wavelength_a,
    )[:2]
    return _finite_sld(complex(real, abs(absorption)) * 1e-6)


def material_sld(material: MaterialSpec, density_scale: float, wavelength_a: float) -> complex:
    """Return a passive complex SLD in inverse square angstroms."""
    _require_positive(density_scale, "density scale")
    _require_positive(wavelength_a, "wavelength_a")
    if material.sld_override_a2 is not None:
        return _finite_sld(material.sld_override_a2 * density_scale)
    return _formula_sld(material, density_scale, wavelength_a)
