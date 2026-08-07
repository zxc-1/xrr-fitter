"""Versioned nominal materials used by filename-driven structure import."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, StructureSpec


INITIAL_DENSITY_TABLE_VERSION = "initial-density-v2"
INITIAL_DENSITIES_G_CM3: Mapping[str, float] = MappingProxyType(
    {
        "Si": 2.329,
        "SiO2": 2.20,
        "Si3N4": 3.17,
        "TaN": 14.30,
        "Zr": 6.52,
    }
)
DEFAULT_DIRECT_SLD_A2 = 20e-6 + 0.0j
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


def material_from_token(token: str) -> MaterialSpec:
    name = token.strip()
    if not name:
        raise ValueError("material token must not be empty")
    density = INITIAL_DENSITIES_G_CM3.get(name)
    if density is None:
        return MaterialSpec(name, None, None, DEFAULT_DIRECT_SLD_A2)
    return MaterialSpec(name, name, density)


def initial_structure(formulas: tuple[str, ...]) -> StructureSpec:
    """Create an editable Air/film/Si declaration in surface-to-backing order."""
    if not formulas:
        raise ValueError("filename material stack must not be empty")
    components = tuple(
        LayerSpec(
            token,
            material_from_token(token),
            DEFAULT_LAYER_THICKNESS_A,
            roughness_a=DEFAULT_LAYER_ROUGHNESS_A,
        )
        for token in formulas
    )
    return StructureSpec(
        MaterialSpec("Air", None, None, 0.0j),
        components,
        material_from_initial_density("Si"),
    )


def automatic_structure(
    formulas_surface_to_backing: tuple[str, ...],
    backing_token: str,
) -> tuple[StructureSpec, tuple[ParameterSetting, ...]]:
    if not formulas_surface_to_backing:
        raise ValueError("filename material stack must not be empty")
    components = [
        LayerSpec(
            token,
            material_from_token(token),
            DEFAULT_LAYER_THICKNESS_A,
            roughness_a=DEFAULT_LAYER_ROUGHNESS_A,
        )
        for token in formulas_surface_to_backing
    ]
    backing = material_from_token(backing_token)
    backing_adjacent = components[-1].material.formula if components else None
    if backing.formula == "Si" and backing_adjacent != "SiO2":
        components.append(
            LayerSpec(
                "SiO2 native oxide",
                MaterialSpec("SiO2", "SiO2", 2.20),
                10.0,
                roughness_a=3.0,
            )
        )
    settings: list[ParameterSetting] = []
    for index, layer in enumerate(components):
        prefix = f"component.{index}"
        if layer.material.sld_override_a2 is not None:
            settings.append(
                ParameterSetting(
                    f"{prefix}.density_scale",
                    1.0,
                    1.0,
                    1.0,
                    locked=True,
                )
            )
        if layer.name == "SiO2 native oxide":
            settings.extend(
                (
                    ParameterSetting(
                        f"{prefix}.thickness_a",
                        10.0,
                        2.0,
                        50.0,
                    ),
                    ParameterSetting(
                        f"{prefix}.density_scale",
                        1.0,
                        1.0,
                        1.0,
                        locked=True,
                    ),
                )
            )
    deduplicated = {setting.name: setting for setting in settings}
    structure = StructureSpec(
        MaterialSpec("Air", None, None, 0.0j),
        tuple(components),
        backing,
    )
    return structure, tuple(deduplicated.values())
