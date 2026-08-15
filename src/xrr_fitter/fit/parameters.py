"""Fit parameter declarations, overrides, and stage-specific locking."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

import numpy as np

from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterDefinition, ParameterSetting
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.transitions import transition_width


def _definition(
    name: str,
    display_name: str,
    unit: str,
    category: str,
    initial: float,
    lower: float,
    upper: float,
    transform: str,
    locked: bool,
    *,
    integer: bool = False,
    expert_only: bool = False,
) -> ParameterDefinition:
    return ParameterDefinition(
        name=name,
        display_name=display_name,
        unit=unit,
        category=category,
        initial=float(initial),
        lower=float(lower),
        upper=float(upper),
        transform=transform,
        locked=locked,
        integer=integer,
        expert_only=expert_only,
    )


def thickness_bounds(data: PreparedData) -> tuple[float, float]:
    """Infer broad search bounds from the fitted q span and spacing."""
    qz = data.qz_a_inv[data.fit_mask]
    if qz.size < 2 or np.ptp(qz) <= 0.0:
        return 2.0, 2e5
    observed_min = max(2.0, 2.0 * np.pi / np.ptp(qz))
    observed_max = min(2e5, np.pi / (2.0 * np.median(np.diff(qz))))
    return max(2.0, 0.25 * observed_min), min(2e5, 4.0 * observed_max)


def _material_definitions(prefix: str, material: MaterialSpec) -> list[ParameterDefinition]:
    if material.sld_override_a2 is None:
        return []
    return [
        _definition(
            f"{prefix}.sld_real_a2",
            f"{prefix} SLD 实部",
            "Å⁻²",
            "material",
            material.sld_override_a2.real,
            -150e-6,
            150e-6,
            "linear",
            False,
        ),
        _definition(
            f"{prefix}.sld_imag_a2",
            f"{prefix} SLD 吸收部",
            "Å⁻²",
            "material",
            material.sld_override_a2.imag,
            0.0,
            20e-6,
            "linear",
            True,
            expert_only=True,
        ),
    ]


def _roughness_axis(layer: LayerSpec) -> tuple[float, float, float, bool]:
    """Resolve the incident roughness axis as ``(initial, lower, upper, locked)``.

    A transition already sets the interface width by microslab blending, so
    leaving Névot-Croce free here would broaden the same interface twice.
    """
    if layer.transition is not None:
        return 0.0, 0.0, 0.0, True
    return layer.roughness_a, 0.0, max(50.0, 0.49 * layer.thickness_a), False


def _layer_definitions(
    prefix: str,
    layer: LayerSpec,
    bounds: tuple[float, float],
) -> list[ParameterDefinition]:
    transition_lower = transition_width(layer.transition) if layer.transition is not None else 2.0
    lower = max(2.0, min(bounds[0], layer.thickness_a), transition_lower)
    upper = min(2e5, max(bounds[1], layer.thickness_a))
    direct_sld = layer.material.sld_override_a2 is not None
    density_initial = 1.0 if direct_sld else layer.density_scale
    roughness_initial, roughness_lower, roughness_upper, roughness_locked = _roughness_axis(layer)
    definitions = [
        _definition(
            f"{prefix}.thickness_a",
            f"{layer.name} 厚度",
            "Å",
            "structure",
            layer.thickness_a,
            lower,
            upper,
            "log",
            False,
        ),
        _definition(
            f"{prefix}.density_scale",
            f"{layer.name} 相对密度",
            "",
            "material",
            density_initial,
            density_initial if direct_sld else min(0.5, density_initial),
            density_initial if direct_sld else max(1.1, density_initial),
            "linear",
            direct_sld,
        ),
        _definition(
            f"{prefix}.roughness_a",
            f"{layer.name} 入射侧粗糙度",
            "Å",
            "interface",
            roughness_initial,
            roughness_lower,
            roughness_upper,
            "roughness_fraction",
            roughness_locked,
        ),
    ]
    definitions.extend(_material_definitions(prefix, layer.material))
    return definitions


def _periodic_definitions(
    prefix: str,
    block: PeriodicBlock,
    bounds: tuple[float, float],
) -> list[ParameterDefinition]:
    definitions: list[ParameterDefinition] = []
    for index, layer in enumerate(block.layers):
        definitions.extend(_layer_definitions(f"{prefix}.layer.{index}", layer, bounds))
    top_initial = block.layers[0].roughness_a if block.top_roughness_a is None else block.top_roughness_a
    definitions.extend(
        (
            _definition(
                f"{prefix}.top_roughness_a",
                f"{block.name} 顶界面粗糙度",
                "Å",
                "interface",
                top_initial,
                0.0,
                max(50.0, 0.49 * block.layers[0].thickness_a),
                "roughness_fraction",
                block.top_roughness_a is None,
            ),
            _definition(
                f"{prefix}.repeats",
                f"{block.name} 重复次数",
                "",
                "structure",
                block.repeats,
                block.repeats,
                block.repeats,
                "linear",
                True,
                integer=True,
            ),
        )
    )
    return definitions


def _sld_definition(
    prefix: str,
    suffix: str,
    display_name: str,
    value: float,
    lower: float,
    upper: float,
) -> ParameterDefinition:
    return _definition(
        f"{prefix}.{suffix}",
        display_name,
        "Å⁻²",
        "material",
        value,
        lower,
        upper,
        "linear",
        True,
        expert_only=True,
    )


def _gradient_definitions(
    prefix: str,
    layer: GradientLayerSpec,
    bounds: tuple[float, float],
) -> list[ParameterDefinition]:
    lower = max(2.0, min(bounds[0], layer.thickness_a))
    upper = min(2e5, max(bounds[1], layer.thickness_a))
    return [
        _sld_definition(
            prefix,
            "upper_sld_real_a2",
            f"{layer.name} 上侧 SLD 实部",
            layer.upper_sld_a2.real,
            -150e-6,
            150e-6,
        ),
        _sld_definition(
            prefix,
            "upper_sld_imag_a2",
            f"{layer.name} 上侧 SLD 吸收部",
            layer.upper_sld_a2.imag,
            0.0,
            20e-6,
        ),
        _sld_definition(
            prefix,
            "lower_sld_real_a2",
            f"{layer.name} 下侧 SLD 实部",
            layer.lower_sld_a2.real,
            -150e-6,
            150e-6,
        ),
        _sld_definition(
            prefix,
            "lower_sld_imag_a2",
            f"{layer.name} 下侧 SLD 吸收部",
            layer.lower_sld_a2.imag,
            0.0,
            20e-6,
        ),
        _definition(
            f"{prefix}.thickness_a",
            f"{layer.name} 厚度",
            "Å",
            "structure",
            layer.thickness_a,
            lower,
            upper,
            "log",
            False,
        ),
        _definition(
            f"{prefix}.roughness_a",
            f"{layer.name} 入射侧粗糙度",
            "Å",
            "interface",
            layer.roughness_a,
            0.0,
            max(50.0, 0.49 * layer.thickness_a),
            "roughness_fraction",
            False,
        ),
        _definition(
            f"{prefix}.microslab_max_a",
            f"{layer.name} 微薄片上限",
            "Å",
            "structure",
            layer.microslab_max_a,
            0.1,
            layer.thickness_a,
            "log",
            True,
            expert_only=True,
        ),
    ]


def _component_definitions(
    prefix: str,
    component: LayerSpec | PeriodicBlock | GradientLayerSpec,
    bounds: tuple[float, float],
) -> list[ParameterDefinition]:
    if isinstance(component, LayerSpec):
        return _layer_definitions(prefix, component, bounds)
    if isinstance(component, PeriodicBlock):
        return _periodic_definitions(prefix, component, bounds)
    return _gradient_definitions(prefix, component, bounds)


def _footprint_upper_deg(data: PreparedData) -> float:
    from xrr_fitter.fit.initialization import critical_edge_candidates

    mask = data.fit_mask
    edges = critical_edge_candidates(
        data.qz_a_inv[mask],
        data.intensity_normalized[mask],
    )
    if not edges:
        return 1.0
    argument = np.clip(
        edges[0] * data.beam.effective_wavelength_a / (4.0 * np.pi),
        -1.0,
        1.0,
    )
    theta_c = float(np.rad2deg(np.arcsin(argument)))
    return 2.0 * theta_c if theta_c > 0.0 else 1.0


def _background_upper(data: PreparedData) -> float:
    fitted = data.intensity_normalized[data.fit_mask]
    high_count = max(1, int(np.ceil(0.20 * fitted.size)))
    return max(0.1, 10.0 * max(0.0, float(np.median(fitted[-high_count:]))))


def _instrument_definitions(
    data: PreparedData,
    instrument: InstrumentSpec,
) -> tuple[ParameterDefinition, ...]:
    background_upper = _background_upper(data)
    footprint_locked = instrument.footprint_mode != "fit"
    footprint_initial = instrument.footprint_spill_angle_deg
    footprint_upper = footprint_initial if footprint_locked else _footprint_upper_deg(data)
    q_resolution = instrument.resolution_domain == "q"
    return (
        _definition(
            "instrument.angle_offset_deg",
            "入射角零点偏移",
            "°",
            "instrument",
            data.import_angle_offset_deg,
            -0.1,
            0.1,
            "linear",
            False,
        ),
        _definition(
            "instrument.scale",
            "尺度",
            "",
            "instrument",
            1.0,
            1e-3,
            1e3,
            "log",
            False,
        ),
        _definition(
            "instrument.background",
            "常数背景",
            "",
            "instrument",
            0.0,
            0.0,
            background_upper,
            "linear",
            False,
        ),
        _definition(
            "instrument.linear_background_per_a_inv",
            "线性背景",
            "Å",
            "instrument",
            0.0,
            -background_upper,
            background_upper,
            "linear",
            instrument.background_kind != "linear",
            expert_only=True,
        ),
        _definition(
            "instrument.powerlaw_background_amplitude",
            "幂律背景幅值 B₂",
            "",
            "instrument",
            0.0,
            0.0,
            background_upper,
            "linear",
            instrument.background_kind != "powerlaw",
            expert_only=True,
        ),
        _definition(
            "instrument.powerlaw_background_exponent",
            "幂律背景指数 p",
            "",
            "instrument",
            3.0,
            1.0,
            4.0,
            "linear",
            True,
            expert_only=True,
        ),
        _definition(
            "instrument.relative_sigma",
            "相对分辨率 σq/q",
            "",
            "instrument",
            0.0,
            0.0,
            0.1,
            "linear",
            not q_resolution,
        ),
        _definition(
            "instrument.footprint_spill_angle_deg",
            "足迹满斑角 θ_fp",
            "°",
            "instrument",
            footprint_initial,
            footprint_initial if footprint_locked else 0.0,
            footprint_upper,
            "linear",
            footprint_locked,
        ),
        _definition(
            "instrument.absolute_sigma_a_inv",
            "绝对分辨率 σq,0",
            "Å⁻¹",
            "instrument",
            0.0,
            0.0,
            0.1,
            "linear",
            True,
            expert_only=True,
        ),
        _definition(
            "instrument.sigma_theta_deg",
            "角域分辨率 σθ",
            "°",
            "instrument",
            0.0,
            0.0,
            0.2,
            "linear",
            q_resolution,
            expert_only=True,
        ),
    )


def default_parameter_definitions(
    data: PreparedData,
    structure: StructureSpec,
    instrument: InstrumentSpec,
    config: FitConfig,
) -> tuple[ParameterDefinition, ...]:
    del config
    bounds = thickness_bounds(data)
    definitions: list[ParameterDefinition] = []
    for index, component in enumerate(structure.components):
        definitions.extend(_component_definitions(f"component.{index}", component, bounds))
    definitions.extend(_material_definitions("backing", structure.backing))
    definitions.append(
        _definition(
            "backing.roughness_a",
            "基底连接界面粗糙度",
            "Å",
            "interface",
            structure.backing_roughness_a,
            0.0,
            50.0,
            "roughness_fraction",
            False,
        )
    )
    definitions.extend(_instrument_definitions(data, instrument))
    return tuple(definitions)


def _validate_setting(
    setting: ParameterSetting,
    definition: ParameterDefinition,
) -> None:
    values = (setting.initial, setting.lower, setting.upper)
    if not all(isfinite(value) for value in values):
        raise ValueError(f"nonfinite parameter setting: {setting.name}")
    if not setting.locked and setting.lower >= setting.upper:
        raise ValueError(f"invalid bounds: {setting.name}")
    if not setting.lower <= setting.initial <= setting.upper:
        raise ValueError(f"initial outside bounds: {setting.name}")
    if definition.integer and any(value != int(value) for value in values):
        raise ValueError(f"integer parameter requires integer values: {setting.name}")


def _validate_settings(
    definitions: tuple[ParameterDefinition, ...],
    settings: tuple[ParameterSetting, ...],
) -> None:
    by_name = {definition.name: definition for definition in definitions}
    if len(by_name) != len(definitions):
        raise ValueError("duplicate parameter definition")
    seen: set[str] = set()
    for setting in settings:
        if setting.name in seen:
            raise ValueError(f"duplicate parameter setting: {setting.name}")
        seen.add(setting.name)
        definition = by_name.get(setting.name)
        if definition is None:
            raise ValueError(f"unknown parameter setting: {setting.name}")
        _validate_setting(setting, definition)


def apply_parameter_settings(
    definitions: tuple[ParameterDefinition, ...],
    settings: tuple[ParameterSetting, ...],
) -> tuple[ParameterDefinition, ...]:
    _validate_settings(definitions, settings)
    by_name = {setting.name: setting for setting in settings}
    return tuple(
        replace(
            definition,
            initial=by_name[definition.name].initial,
            lower=by_name[definition.name].lower,
            upper=by_name[definition.name].upper,
            locked=by_name[definition.name].locked,
        )
        if definition.name in by_name
        else definition
        for definition in definitions
    )


def _require_locked_value(
    definitions: dict[str, ParameterDefinition],
    name: str,
    expected: float,
    message: str,
) -> None:
    definition = definitions[name]
    if not definition.locked or definition.initial != expected:
        raise ValueError(message)


def validate_instrument_modes(
    definitions: tuple[ParameterDefinition, ...],
    instrument: InstrumentSpec,
) -> None:
    by_name = {definition.name: definition for definition in definitions}
    if instrument.resolution_domain == "theta":
        for name in ("instrument.relative_sigma", "instrument.absolute_sigma_a_inv"):
            _require_locked_value(
                by_name,
                name,
                0.0,
                "theta-domain mode requires q-domain resolution parameters locked at zero",
            )
    else:
        _require_locked_value(
            by_name,
            "instrument.sigma_theta_deg",
            0.0,
            "q-domain mode requires theta-domain resolution locked at zero",
        )
    if instrument.background_kind != "linear":
        _require_locked_value(
            by_name,
            "instrument.linear_background_per_a_inv",
            0.0,
            "inactive linear background must stay locked at zero",
        )
    if instrument.background_kind != "powerlaw":
        _require_locked_value(
            by_name,
            "instrument.powerlaw_background_amplitude",
            0.0,
            "inactive power-law background must stay locked at zero",
        )
        _require_locked_value(
            by_name,
            "instrument.powerlaw_background_exponent",
            3.0,
            "inactive power-law exponent must stay locked at three",
        )
    if instrument.footprint_mode != "fit":
        _require_locked_value(
            by_name,
            "instrument.footprint_spill_angle_deg",
            instrument.footprint_spill_angle_deg,
            "geometry/none footprint angle must stay locked",
        )


def validate_transition_modes(
    definitions: tuple[ParameterDefinition, ...],
    structure: StructureSpec,
) -> None:
    """Reject settings that violate geometry owned by a transition.

    Periodic blocks need no handling: their layers already refuse transitions at
    construction time.
    """
    by_name = {definition.name: definition for definition in definitions}
    for index, component in enumerate(structure.components):
        if isinstance(component, LayerSpec) and component.transition is not None:
            thickness = by_name[f"component.{index}.thickness_a"]
            width = transition_width(component.transition)
            if thickness.lower < width or thickness.initial < width:
                raise ValueError("带过渡的层厚度初值和下界不得小于过渡宽度")
            _require_locked_value(
                by_name,
                f"component.{index}.roughness_a",
                0.0,
                "带过渡的界面粗糙度必须锁定在 0",
            )


def _validate_compiled_definition(definition: ParameterDefinition) -> None:
    values = (definition.initial, definition.lower, definition.upper)
    if not all(isfinite(value) for value in values):
        raise ValueError(f"nonfinite compiled parameter: {definition.name}")
    if definition.lower > definition.upper:
        raise ValueError(f"invalid compiled bounds: {definition.name}")
    if not definition.lower <= definition.initial <= definition.upper:
        raise ValueError(f"initial outside compiled bounds: {definition.name}")
    if not definition.locked and definition.lower == definition.upper:
        raise ValueError(f"free parameter has zero range: {definition.name}")
    if definition.transform == "log" and definition.lower <= 0.0:
        raise ValueError(f"log parameter must have positive bounds: {definition.name}")


def validate_compiled_definitions(definitions: tuple[ParameterDefinition, ...]) -> None:
    names = [definition.name for definition in definitions]
    if len(names) != len(set(names)):
        raise ValueError("duplicate parameter definition")
    for definition in definitions:
        _validate_compiled_definition(definition)


STAGE_CATEGORIES = {"B": "structure", "C": "material", "D": "interface"}
STAGE_INSTRUMENT_NAMES = {
    "B": {
        "instrument.angle_offset_deg",
        "instrument.scale",
        "instrument.background",
        "instrument.linear_background_per_a_inv",
        "instrument.powerlaw_background_amplitude",
        "instrument.powerlaw_background_exponent",
        "instrument.footprint_spill_angle_deg",
    },
    "C": set(),
    "D": {
        "instrument.relative_sigma",
        "instrument.absolute_sigma_a_inv",
        "instrument.sigma_theta_deg",
    },
}


def stage_parameter_is_free(stage: str, definition: ParameterDefinition) -> bool:
    if stage == "E":
        return True
    if stage not in STAGE_CATEGORIES:
        raise ValueError(f"unsupported fit stage: {stage}")
    return definition.category == STAGE_CATEGORIES[stage] or definition.name in STAGE_INSTRUMENT_NAMES[stage]


def _missing_stage_values(
    definitions: tuple[ParameterDefinition, ...],
    current_values: dict[str, float],
) -> tuple[str, ...]:
    return tuple(
        definition.name
        for definition in definitions
        if not (definition.locked or definition.constrained) and definition.name not in current_values
    )


def _stage_parameter_setting(
    definition: ParameterDefinition,
    stage: str,
    current_values: dict[str, float],
) -> ParameterSetting:
    if definition.constrained:
        return ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.locked,
        )
    if definition.locked:
        return ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=True,
        )
    current = current_values[definition.name]
    free = stage_parameter_is_free(stage, definition)
    return ParameterSetting(
        definition.name,
        current,
        definition.lower if free else current,
        definition.upper if free else current,
        locked=not free,
    )


def stage_parameter_settings(
    definitions: tuple[ParameterDefinition, ...],
    stage: str,
    current_values: dict[str, float],
) -> tuple[ParameterSetting, ...]:
    missing = _missing_stage_values(definitions, current_values)
    if missing:
        raise ValueError("missing current stage values: " + ", ".join(missing))
    return tuple(_stage_parameter_setting(definition, stage, current_values) for definition in definitions)
