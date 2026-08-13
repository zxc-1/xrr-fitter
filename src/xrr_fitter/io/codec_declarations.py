"""Project-codec declarations for data, instruments, fitting, and structures."""

from __future__ import annotations

from collections.abc import Callable

from xrr_fitter.io.codec_common import (
    ProjectSchemaError,
    _complex_from_dict,
    _complex_to_dict,
    _mapping,
    _sequence,
)
from xrr_fitter.model.data import BeamSpec, DataColumnMapping
from xrr_fitter.model.fitting import (
    ConfidenceThresholds,
    FitConfig,
    SearchBudget,
)
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    InterfaceTransition,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
    TransitionBranch,
)


def _beam_to_dict(value: BeamSpec) -> dict[str, object]:
    return {
        "kind": value.kind,
        "wavelength_a": value.wavelength_a,
        "wavelength_1_a": value.wavelength_1_a,
        "wavelength_2_a": value.wavelength_2_a,
        "intensity_ratio_21": value.intensity_ratio_21,
    }


def _beam_from_dict(value: object) -> BeamSpec:
    payload = _mapping(
        value,
        {
            "kind",
            "wavelength_a",
            "wavelength_1_a",
            "wavelength_2_a",
            "intensity_ratio_21",
        },
        "beam",
    )
    return BeamSpec(**payload)


def _column_mapping_to_dict(value: DataColumnMapping) -> dict[str, object]:
    return {
        "two_theta": value.two_theta,
        "intensity": value.intensity,
        "intensity_sigma": value.intensity_sigma,
        "resolution": value.resolution,
        "resolution_kind": value.resolution_kind,
    }


def _column_mapping_from_dict(value: object) -> DataColumnMapping:
    payload = _mapping(
        value,
        {"two_theta", "intensity", "intensity_sigma", "resolution", "resolution_kind"},
        "column mapping",
    )
    return DataColumnMapping(**payload)


def _instrument_to_dict(value: InstrumentSpec) -> dict[str, object]:
    return {
        "instrument_id": value.instrument_id,
        "footprint_mode": value.footprint_mode,
        "footprint_spill_angle_deg": value.footprint_spill_angle_deg,
        "sample_length_mm": value.sample_length_mm,
        "beam_width_mm": value.beam_width_mm,
        "background_kind": value.background_kind,
        "resolution_domain": value.resolution_domain,
    }


def _instrument_from_dict(value: object) -> InstrumentSpec:
    payload = _mapping(
        value,
        {
            "instrument_id",
            "footprint_mode",
            "footprint_spill_angle_deg",
            "sample_length_mm",
            "beam_width_mm",
            "background_kind",
            "resolution_domain",
        },
        "instrument",
    )
    return InstrumentSpec(**payload)


# prior_conflict_sigmas is emitted only when it departs from the default, so
# projects written before the field existed re-encode to byte-identical
# confidence and older readers keep loading them unchanged.
CONFIDENCE_DEFAULT_SIGMAS = ConfidenceThresholds().prior_conflict_sigmas
CONFIDENCE_FIELDS = frozenset(ConfidenceThresholds.__dataclass_fields__) - {"prior_conflict_sigmas"}


def _confidence_to_dict(value: ConfidenceThresholds) -> dict[str, object]:
    payload: dict[str, object] = {field: getattr(value, field) for field in CONFIDENCE_FIELDS}
    if value.prior_conflict_sigmas != CONFIDENCE_DEFAULT_SIGMAS:
        payload["prior_conflict_sigmas"] = value.prior_conflict_sigmas
    return payload


def _fit_config_to_dict(value: FitConfig) -> dict[str, object]:
    return {
        "master_seed": value.master_seed,
        "objective_name": value.objective_name,
        "objective_version": value.objective_version,
        "c_decades": value.c_decades,
        "final_seed_count": value.final_seed_count,
        "budget": {field: getattr(value.budget, field) for field in value.budget.__dataclass_fields__},
        "local_workers": value.local_workers,
        "scale_prior_enabled": value.scale_prior_enabled,
        "scale_prior_tau_decades": value.scale_prior_tau_decades,
        "confidence": _confidence_to_dict(value.confidence),
        "fringe_screen_threshold_version": value.fringe_screen_threshold_version,
        "budget_reclaim_threshold_version": value.budget_reclaim_threshold_version,
        "downsample_rule_version": value.downsample_rule_version,
        "jacobian_version": value.jacobian_version,
    }


def _fit_config_from_dict(value: object) -> FitConfig:
    fields = {
        "master_seed",
        "objective_name",
        "objective_version",
        "c_decades",
        "final_seed_count",
        "budget",
        "local_workers",
        "scale_prior_enabled",
        "scale_prior_tau_decades",
        "confidence",
        "fringe_screen_threshold_version",
        "budget_reclaim_threshold_version",
        "downsample_rule_version",
        "jacobian_version",
    }
    payload = _mapping(value, fields, "fit_config")
    budget_fields = set(SearchBudget.__dataclass_fields__)
    # prior_conflict_sigmas is optional so projects saved before it existed
    # decode to the dataclass default instead of failing the field-set check.
    confidence_fields = set(ConfidenceThresholds.__dataclass_fields__)
    confidence_optional = {"prior_conflict_sigmas"}
    budget = SearchBudget(**_mapping(payload["budget"], budget_fields, "search budget"))
    confidence = ConfidenceThresholds(
        **_mapping(
            payload["confidence"],
            confidence_fields - confidence_optional,
            "confidence thresholds",
            confidence_optional,
        )
    )
    scalar = {key: item for key, item in payload.items() if key not in {"budget", "confidence"}}
    return FitConfig(**scalar, budget=budget, confidence=confidence)


def _material_to_dict(value: MaterialSpec) -> dict[str, object]:
    return {
        "name": value.name,
        "formula": value.formula,
        "bulk_density_g_cm3": value.bulk_density_g_cm3,
        "sld_override_a2": _complex_to_dict(value.sld_override_a2),
    }


def _material_from_dict(value: object) -> MaterialSpec:
    payload = _mapping(
        value,
        {"name", "formula", "bulk_density_g_cm3", "sld_override_a2"},
        "material",
    )
    return MaterialSpec(
        name=payload["name"],
        formula=payload["formula"],
        bulk_density_g_cm3=payload["bulk_density_g_cm3"],
        sld_override_a2=_complex_from_dict(payload["sld_override_a2"], "material SLD"),
    )


def _transition_branch_to_dict(value: TransitionBranch) -> dict[str, object]:
    return {
        "kind": value.kind,
        "weight": value.weight,
        "thickness_a": value.thickness_a,
    }


def _transition_branch_from_dict(value: object) -> TransitionBranch:
    payload = _mapping(value, {"kind", "weight", "thickness_a"}, "transition branch")
    return TransitionBranch(
        kind=payload["kind"],
        weight=payload["weight"],
        thickness_a=payload["thickness_a"],
    )


def _transition_to_dict(value: InterfaceTransition) -> dict[str, object]:
    return {
        "branches": [_transition_branch_to_dict(branch) for branch in value.branches],
        "microslab_max_a": value.microslab_max_a,
    }


def _transition_from_dict(value: object) -> InterfaceTransition:
    payload = _mapping(value, {"branches", "microslab_max_a"}, "interface transition")
    return InterfaceTransition(
        branches=tuple(
            _transition_branch_from_dict(item) for item in _sequence(payload["branches"], "transition branches")
        ),
        microslab_max_a=payload["microslab_max_a"],
    )


def _layer_to_dict(value: LayerSpec) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "layer",
        "name": value.name,
        "material": _material_to_dict(value.material),
        "thickness_a": value.thickness_a,
        "density_scale": value.density_scale,
        "roughness_a": value.roughness_a,
    }
    # Emitting the key only when present keeps files written before transitions
    # existed byte-identical, and lets older readers load them unchanged.
    if value.transition is not None:
        payload["transition"] = _transition_to_dict(value.transition)
    return payload


def _layer_from_dict(value: object) -> LayerSpec:
    payload = _mapping(
        value,
        {"kind", "name", "material", "thickness_a", "density_scale", "roughness_a"},
        "layer",
        optional={"transition"},
    )
    if payload["kind"] != "layer":
        raise ProjectSchemaError(f"unknown structure discriminator: {payload['kind']}")
    transition = payload.get("transition")
    return LayerSpec(
        name=payload["name"],
        material=_material_from_dict(payload["material"]),
        thickness_a=payload["thickness_a"],
        density_scale=payload["density_scale"],
        roughness_a=payload["roughness_a"],
        transition=None if transition is None else _transition_from_dict(transition),
    )


def _periodic_to_dict(value: PeriodicBlock) -> dict[str, object]:
    return {
        "kind": "periodic_block",
        "name": value.name,
        "layers": [_layer_to_dict(layer) for layer in value.layers],
        "repeats": value.repeats,
        "top_roughness_a": value.top_roughness_a,
    }


def _periodic_from_dict(value: object) -> PeriodicBlock:
    payload = _mapping(
        value,
        {"kind", "name", "layers", "repeats", "top_roughness_a"},
        "periodic block",
    )
    return PeriodicBlock(
        name=payload["name"],
        layers=tuple(_layer_from_dict(item) for item in _sequence(payload["layers"], "layers")),
        repeats=payload["repeats"],
        top_roughness_a=payload["top_roughness_a"],
    )


def _required_complex(value: object, label: str) -> complex:
    result = _complex_from_dict(value, label)
    if result is None:
        raise ProjectSchemaError(f"{label} is required")
    return result


def _gradient_to_dict(value: GradientLayerSpec) -> dict[str, object]:
    return {
        "kind": "gradient_layer",
        "name": value.name,
        "upper_sld_a2": _complex_to_dict(value.upper_sld_a2),
        "lower_sld_a2": _complex_to_dict(value.lower_sld_a2),
        "thickness_a": value.thickness_a,
        "roughness_a": value.roughness_a,
        "microslab_max_a": value.microslab_max_a,
    }


def _gradient_from_dict(value: object) -> GradientLayerSpec:
    payload = _mapping(
        value,
        {
            "kind",
            "name",
            "upper_sld_a2",
            "lower_sld_a2",
            "thickness_a",
            "roughness_a",
            "microslab_max_a",
        },
        "gradient layer",
    )
    return GradientLayerSpec(
        name=payload["name"],
        upper_sld_a2=_required_complex(payload["upper_sld_a2"], "upper gradient SLD"),
        lower_sld_a2=_required_complex(payload["lower_sld_a2"], "lower gradient SLD"),
        thickness_a=payload["thickness_a"],
        roughness_a=payload["roughness_a"],
        microslab_max_a=payload["microslab_max_a"],
    )


def _component_to_dict(value: object) -> dict[str, object]:
    if isinstance(value, LayerSpec):
        return _layer_to_dict(value)
    if isinstance(value, PeriodicBlock):
        return _periodic_to_dict(value)
    if isinstance(value, GradientLayerSpec):
        return _gradient_to_dict(value)
    raise ProjectSchemaError(f"unsupported structure component: {type(value).__name__}")


def _component_from_dict(value: object) -> object:
    if not isinstance(value, dict) or "kind" not in value:
        raise ProjectSchemaError("structure component requires kind")
    decoders: dict[str, Callable[[object], object]] = {
        "layer": _layer_from_dict,
        "periodic_block": _periodic_from_dict,
        "gradient_layer": _gradient_from_dict,
    }
    kind = value["kind"]
    try:
        decoder = decoders[kind]
    except (KeyError, TypeError) as error:
        raise ProjectSchemaError(f"unknown structure discriminator: {kind}") from error
    return decoder(value)


def _structure_to_dict(value: StructureSpec | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "fronting": _material_to_dict(value.fronting),
        "components": [_component_to_dict(item) for item in value.components],
        "backing": _material_to_dict(value.backing),
        "backing_roughness_a": value.backing_roughness_a,
    }


def _structure_from_dict(value: object) -> StructureSpec | None:
    if value is None:
        return None
    payload = _mapping(
        value,
        {"fronting", "components", "backing", "backing_roughness_a"},
        "structure",
    )
    return StructureSpec(
        fronting=_material_from_dict(payload["fronting"]),
        components=tuple(
            _component_from_dict(item) for item in _sequence(payload["components"], "structure components")
        ),
        backing=_material_from_dict(payload["backing"]),
        backing_roughness_a=payload["backing_roughness_a"],
    )
