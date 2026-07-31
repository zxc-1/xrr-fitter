"""Versioned nominal materials used by filename-driven structure import."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from xrr_fitter.model.structure import LayerSpec, MaterialSpec, StructureSpec


INITIAL_DENSITY_TABLE_VERSION = "initial-density-v1"
INITIAL_DENSITIES_G_CM3: Mapping[str, float] = MappingProxyType(
    {
        "CrSiC": 4.50,
        "Si": 2.329,
        "Si3N4": 3.17,
        "SiCMo": 5.50,
        "TaN": 14.30,
        "Zr": 6.52,
    }
)
DEFAULT_LAYER_THICKNESS_A = 100.0
DEFAULT_LAYER_ROUGHNESS_A = 3.0


def material_from_initial_density(formula: str) -> MaterialSpec:
    """Build one formula material from the auditable nominal-density table."""
    try:
        density = INITIAL_DENSITIES_G_CM3[formula]
    except KeyError as error:
        raise ValueError(
            f"no initial density for filename material {formula!r} "
            f"in {INITIAL_DENSITY_TABLE_VERSION}"
        ) from error
    return MaterialSpec(formula, formula, density)


def initial_structure(formulas: tuple[str, ...]) -> StructureSpec:
    """Create an editable Air/film/Si declaration in surface-to-backing order."""
    if not formulas:
        raise ValueError("filename material stack must not be empty")
    components = tuple(
        LayerSpec(
            formula,
            material_from_initial_density(formula),
            DEFAULT_LAYER_THICKNESS_A,
            roughness_a=DEFAULT_LAYER_ROUGHNESS_A,
        )
        for formula in formulas
    )
    return StructureSpec(
        MaterialSpec("Air", None, None, 0.0j),
        components,
        material_from_initial_density("Si"),
    )
