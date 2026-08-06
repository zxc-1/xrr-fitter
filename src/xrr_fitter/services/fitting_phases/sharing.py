"""Automatic material and roughness sharing rules."""

from __future__ import annotations

from xrr_fitter.model.parameters import ParameterReference, SharingRule
from xrr_fitter.model.structure import LayerSpec

from .common import PreparedDatasetFit

def _automatic_material_occurrences(prepared: PreparedDatasetFit):
    for index, component in enumerate(prepared.problem.structure.components):
        if not isinstance(component, LayerSpec):
            raise ValueError(
                "automatic joint sharing requires homogeneous layer components"
            )
        yield f"component.{index}", component.material
    yield "backing", prepared.problem.structure.backing


def _sharing_rule(
    fit_group_id: str,
    family: str,
    owner: str,
    members: list[ParameterReference],
) -> SharingRule | None:
    if len(members) < 2:
        return None
    return SharingRule(
        f"automatic:{fit_group_id}:{family}:{owner}",
        tuple(members),
    )


def _collect_material_sharing(
    prepared: PreparedDatasetFit,
    density: dict[str, list[ParameterReference]],
    real_sld: dict[str, list[ParameterReference]],
    imag_sld: dict[str, list[tuple[ParameterReference | None, bool]]],
) -> None:
    free_names = {coordinate.name for coordinate in prepared.problem.variables}
    explicit = {
        setting.name for setting in prepared.updated_dataset.parameter_settings
    }
    for path, material in _automatic_material_occurrences(prepared):
        if material.sld_override_a2 is None:
            name = f"{path}.density_scale"
            if name in free_names:
                density.setdefault(material.name, []).append(
                    ParameterReference(prepared.dataset_id, name)
                )
            continue
        real_name = f"{path}.sld_real_a2"
        if real_name in free_names:
            real_sld.setdefault(material.name, []).append(
                ParameterReference(prepared.dataset_id, real_name)
            )
        imag_name = f"{path}.sld_imag_a2"
        imag_sld.setdefault(material.name, []).append(
            (
                ParameterReference(prepared.dataset_id, imag_name),
                imag_name in explicit,
            )
        )


def _collect_roughness_sharing(
    prepared: PreparedDatasetFit,
    roughness: dict[str, list[ParameterReference]],
) -> None:
    for coordinate in prepared.problem.variables:
        name = coordinate.name
        if name.endswith("roughness_a"):
            path = name.rsplit(".", 1)[0]
            roughness.setdefault(path, []).append(
                ParameterReference(prepared.dataset_id, name)
            )


def _group_sharing_rules(
    fit_group_id: str,
    family: str,
    grouped: dict[str, list[ParameterReference]],
) -> tuple[SharingRule, ...]:
    return tuple(
        rule
        for owner, members in grouped.items()
        if (rule := _sharing_rule(fit_group_id, family, owner, members))
        is not None
    )


def _absorption_sharing_rules(
    fit_group_id: str,
    grouped: dict[str, list[tuple[ParameterReference | None, bool]]],
) -> tuple[SharingRule, ...]:
    rules = []
    for material_name, evidence in grouped.items():
        if not evidence or not all(released for _member, released in evidence):
            continue
        rule = _sharing_rule(
            fit_group_id,
            "sld_imag_a2",
            material_name,
            [member for member, _released in evidence if member is not None],
        )
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


def automatic_sharing_rules(
    prepared: tuple[PreparedDatasetFit, ...],
    fit_group_id: str,
    *,
    share_roughness: bool,
) -> tuple[SharingRule, ...]:
    """Declare automatic material sharing and optional path-local roughness."""
    values = tuple(prepared)
    if not fit_group_id.strip():
        raise ValueError("fit_group_id must not be empty")
    if not isinstance(share_roughness, bool):
        raise TypeError("share_roughness must be bool")

    density: dict[str, list[ParameterReference]] = {}
    real_sld: dict[str, list[ParameterReference]] = {}
    imag_sld: dict[str, list[tuple[ParameterReference | None, bool]]] = {}
    roughness: dict[str, list[ParameterReference]] = {}
    for item in values:
        _collect_material_sharing(item, density, real_sld, imag_sld)
        if share_roughness:
            _collect_roughness_sharing(item, roughness)

    return (
        *_group_sharing_rules(fit_group_id, "density_scale", density),
        *_group_sharing_rules(fit_group_id, "sld_real_a2", real_sld),
        *_absorption_sharing_rules(fit_group_id, imag_sld),
        *_group_sharing_rules(fit_group_id, "roughness_a", roughness),
    )
