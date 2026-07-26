"""Replay the frozen R22 project-model reference through R23 model values.

Only hash-verified bytes supplied by ``ReplayContext.inputs`` are decoded.
This adapter owns a narrow project-document decoder until the production I/O
codec arrives in Task 4; it does not read files, inspect golden artifacts, or
import test helpers. Every mapping has an exact field set so reference replay
cannot silently accept schema drift.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Any

from xrr_fitter.model.data import BeamSpec, DataColumnMapping
from xrr_fitter.model.fitting import ConfidenceThresholds, FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterReference, ParameterSetting, SharingRule
from xrr_fitter.model.project import (
    DatasetProject,
    OxideDecision,
    ProjectUiState,
    ScalePriorState,
    XrrProject,
    with_active_dataset,
    with_batch_mode,
    with_dataset_fit_mask,
    with_workspace_state,
)
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)


ARTIFACT = "golden/model_project.json"
INPUTS = {
    "mo-si-periodic-project": (
        "bundled-example-project",
        "xrr_fitter/examples/mo-si-periodic.xrrproj.json",
        "mo-si-periodic",
    ),
    "single-layer-project": (
        "bundled-example-project",
        "xrr_fitter/examples/single-layer.xrrproj.json",
        "single-layer",
    ),
}
INPUT_ORDER = tuple(INPUTS)
CONFIGURATION = {
    "cases": ["single-layer", "mo-si-periodic"],
    "operations": [
        "load_project",
        "project_to_dict",
        "with_active_dataset",
        "with_batch_mode",
        "fit_mask_transition",
        "ui_state_transition",
    ],
}


def _mapping(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid {label} field set")
    return value


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate project field: {key}")
        value[key] = item
    return value


def _document(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid project JSON") from error
    return _mapping(
        value,
        {
            "schema_version",
            "algorithm_version",
            "fit_config",
            "input_angle_kind",
            "batch_mode",
            "datasets",
            "sharing_rules",
            "ui_state",
        },
        "project",
    )


def _fit_config(value: object) -> FitConfig:
    payload = _mapping(
        value,
        {
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
        },
        "fit_config",
    )
    budget = _mapping(
        payload["budget"],
        {
            "short_de_maxiter",
            "full_de_maxiter",
            "local_min_nfev",
            "local_nfev_per_parameter",
            "bootstrap_samples",
        },
        "search budget",
    )
    confidence = _mapping(
        payload["confidence"],
        {
            "cluster_join_distance",
            "distinct_cluster_distance",
            "equivalent_cost_fraction",
            "equivalent_cost_floor",
            "boundary_fraction",
            "strong_correlation",
        },
        "confidence thresholds",
    )
    values = dict(payload)
    values["budget"] = SearchBudget(**budget)
    values["confidence"] = ConfidenceThresholds(**confidence)
    return FitConfig(**values)


def _beam(value: object) -> BeamSpec:
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


def _column_mapping(value: object) -> DataColumnMapping:
    payload = _mapping(
        value,
        {"two_theta", "intensity", "intensity_sigma", "resolution", "resolution_kind"},
        "column mapping",
    )
    return DataColumnMapping(**payload)


def _instrument(value: object) -> InstrumentSpec:
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


def _complex(value: object, label: str) -> complex | None:
    if value is None:
        return None
    payload = _mapping(value, {"real", "imag"}, label)
    return complex(payload["real"], payload["imag"])


def _material(value: object) -> MaterialSpec:
    payload = _mapping(
        value,
        {"name", "formula", "bulk_density_g_cm3", "sld_override_a2"},
        "material",
    )
    return MaterialSpec(
        name=payload["name"],
        formula=payload["formula"],
        bulk_density_g_cm3=payload["bulk_density_g_cm3"],
        sld_override_a2=_complex(payload["sld_override_a2"], "material SLD"),
    )


def _layer(payload: dict[str, Any]) -> LayerSpec:
    _mapping(
        payload,
        {"kind", "name", "material", "thickness_a", "density_scale", "roughness_a"},
        "layer",
    )
    return LayerSpec(
        name=payload["name"],
        material=_material(payload["material"]),
        thickness_a=payload["thickness_a"],
        density_scale=payload["density_scale"],
        roughness_a=payload["roughness_a"],
    )


def _gradient(payload: dict[str, Any]) -> GradientLayerSpec:
    _mapping(
        payload,
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
        upper_sld_a2=_complex(payload["upper_sld_a2"], "upper gradient SLD"),
        lower_sld_a2=_complex(payload["lower_sld_a2"], "lower gradient SLD"),
        thickness_a=payload["thickness_a"],
        roughness_a=payload["roughness_a"],
        microslab_max_a=payload["microslab_max_a"],
    )


def _periodic(payload: dict[str, Any]) -> PeriodicBlock:
    _mapping(
        payload,
        {"kind", "name", "layers", "repeats", "top_roughness_a"},
        "periodic block",
    )
    layers = tuple(_layer(_mapping(item, set(item) if isinstance(item, dict) else set(), "layer")) for item in payload["layers"])
    return PeriodicBlock(
        name=payload["name"],
        layers=layers,
        repeats=payload["repeats"],
        top_roughness_a=payload["top_roughness_a"],
    )


def _component(value: object) -> LayerSpec | PeriodicBlock | GradientLayerSpec:
    if not isinstance(value, dict):
        raise ValueError("invalid structure component")
    kind = value.get("kind")
    if kind == "layer":
        return _layer(value)
    if kind == "periodic_block":
        return _periodic(value)
    if kind == "gradient_layer":
        return _gradient(value)
    raise ValueError(f"unsupported structure component: {kind}")


def _structure(value: object) -> StructureSpec | None:
    if value is None:
        return None
    payload = _mapping(
        value,
        {"fronting", "components", "backing", "backing_roughness_a"},
        "structure",
    )
    return StructureSpec(
        fronting=_material(payload["fronting"]),
        components=tuple(_component(item) for item in payload["components"]),
        backing=_material(payload["backing"]),
        backing_roughness_a=payload["backing_roughness_a"],
    )


def _scale_prior(value: object) -> ScalePriorState:
    payload = _mapping(value, {"enabled", "s_hat", "tau_s_decades", "reason"}, "scale prior")
    return ScalePriorState(**payload)


def _oxide_decision(value: object) -> OxideDecision:
    payload = _mapping(
        value,
        {"base_material", "oxide_material", "location", "accepted", "oxide_table_version"},
        "oxide decision",
    )
    return OxideDecision(**payload)


def _parameter_setting(value: object) -> ParameterSetting:
    payload = _mapping(value, {"name", "initial", "lower", "upper", "locked"}, "parameter setting")
    return ParameterSetting(**payload)


def _require_none(value: object, label: str) -> None:
    if value is not None:
        raise ValueError(f"model_project replay requires null {label}")


def _dataset(value: object) -> DatasetProject:
    payload = _mapping(
        value,
        {
            "dataset_id",
            "source_path",
            "source_sha256",
            "beam",
            "import_angle_offset_deg",
            "column_mapping",
            "fit_mask",
            "fit_range_two_theta_deg",
            "structure",
            "instrument",
            "structure_evidence",
            "scale_prior",
            "oxide_decisions",
            "parameter_settings",
            "last_valid_result",
            "checkpoint",
        },
        "dataset",
    )
    for field in ("structure_evidence", "last_valid_result", "checkpoint"):
        _require_none(payload[field], field)
    return DatasetProject(
        dataset_id=payload["dataset_id"],
        source_path=payload["source_path"],
        source_sha256=payload["source_sha256"],
        beam=_beam(payload["beam"]),
        import_angle_offset_deg=payload["import_angle_offset_deg"],
        column_mapping=_column_mapping(payload["column_mapping"]),
        fit_mask=tuple(payload["fit_mask"]),
        fit_range_two_theta_deg=tuple(payload["fit_range_two_theta_deg"]),
        structure=_structure(payload["structure"]),
        instrument=_instrument(payload["instrument"]),
        structure_evidence=None,
        scale_prior=_scale_prior(payload["scale_prior"]),
        oxide_decisions=tuple(_oxide_decision(item) for item in payload["oxide_decisions"]),
        parameter_settings=tuple(
            _parameter_setting(item) for item in payload["parameter_settings"]
        ),
        last_valid_result=None,
        checkpoint=None,
    )


def _sharing_rule(value: object) -> SharingRule:
    payload = _mapping(value, {"sharing_key", "members"}, "sharing rule")
    members = tuple(
        ParameterReference(**_mapping(item, {"dataset_id", "parameter_name"}, "sharing member"))
        for item in payload["members"]
    )
    return SharingRule(payload["sharing_key"], members)


def _ui_state(value: object) -> ProjectUiState:
    payload = _mapping(
        value,
        {
            "active_dataset_id",
            "selected_candidate_ids",
            "expert_mode",
            "workspace_splitter_sizes",
            "left_splitter_sizes",
            "plot_tab_index",
        },
        "project UI state",
    )
    return ProjectUiState(
        active_dataset_id=payload["active_dataset_id"],
        selected_candidate_ids=tuple(tuple(item) for item in payload["selected_candidate_ids"]),
        expert_mode=payload["expert_mode"],
        workspace_splitter_sizes=tuple(payload["workspace_splitter_sizes"]),
        left_splitter_sizes=tuple(payload["left_splitter_sizes"]),
        plot_tab_index=payload["plot_tab_index"],
    )


def _project(content: bytes) -> tuple[XrrProject, dict[str, Any]]:
    payload = _document(content)
    project = XrrProject(
        schema_version=payload["schema_version"],
        algorithm_version=payload["algorithm_version"],
        fit_config=_fit_config(payload["fit_config"]),
        input_angle_kind=payload["input_angle_kind"],
        batch_mode=payload["batch_mode"],
        datasets=tuple(_dataset(item) for item in payload["datasets"]),
        sharing_rules=tuple(_sharing_rule(item) for item in payload["sharing_rules"]),
        ui_state=_ui_state(payload["ui_state"]),
    )
    return project, payload


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclass_fields(value)
        }
    return value


def _component_payload(value: object) -> dict[str, object]:
    payload = _json_value(value)
    assert isinstance(payload, dict)
    kinds = {
        LayerSpec: "layer",
        PeriodicBlock: "periodic_block",
        GradientLayerSpec: "gradient_layer",
    }
    payload["kind"] = kinds[type(value)]
    if isinstance(value, PeriodicBlock):
        payload["layers"] = [_component_payload(layer) for layer in value.layers]
    return payload


def _structure_payload(value: StructureSpec | None) -> object:
    if value is None:
        return None
    return {
        "fronting": _json_value(value.fronting),
        "components": [_component_payload(item) for item in value.components],
        "backing": _json_value(value.backing),
        "backing_roughness_a": value.backing_roughness_a,
    }


def _dataset_payload(value: DatasetProject) -> dict[str, object]:
    payload = {
        field.name: _json_value(getattr(value, field.name))
        for field in dataclass_fields(value)
    }
    payload["structure"] = _structure_payload(value.structure)
    return payload


def _project_payload(value: XrrProject) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "algorithm_version": value.algorithm_version,
        "fit_config": _json_value(value.fit_config),
        "input_angle_kind": value.input_angle_kind,
        "batch_mode": value.batch_mode,
        "datasets": [_dataset_payload(item) for item in value.datasets],
        "sharing_rules": _json_value(value.sharing_rules),
        "ui_state": _json_value(value.ui_state),
    }


def _validate_input(replay_input: object, expected: tuple[str, str, str]) -> str:
    input_class, path, stem = expected
    if replay_input.input_class != input_class or replay_input.path != path:
        raise ValueError("model_project input identity drift")
    content = replay_input.content
    if not isinstance(content, bytes):
        raise ValueError("model_project input content must be bytes")
    digest = hashlib.sha256(content).hexdigest()
    if replay_input.size != len(content) or replay_input.sha256 != digest:
        raise ValueError("model_project input size or hash drift")
    return stem


def _validate_context(context: object) -> dict[str, bytes]:
    if context.group != "model_project":
        raise ValueError("model_project group drift")
    if tuple(context.artifacts) != (ARTIFACT,):
        raise ValueError("model_project artifact drift")
    if context.configuration != CONFIGURATION:
        raise ValueError("model_project configuration drift")
    if tuple(context.seeds):
        raise ValueError("model_project seed drift")
    replay_inputs = tuple(context.inputs)
    if tuple(item.input_id for item in replay_inputs) != INPUT_ORDER:
        raise ValueError("model_project input set or order drift")
    contents: dict[str, bytes] = {}
    for replay_input in replay_inputs:
        stem = _validate_input(replay_input, INPUTS[replay_input.input_id])
        contents[stem] = replay_input.content
    return contents


def _cases(contents: dict[str, bytes]) -> tuple[dict[str, XrrProject], dict[str, object]]:
    projects: dict[str, XrrProject] = {}
    cases: dict[str, object] = {}
    for stem in CONFIGURATION["cases"]:
        project, source = _project(contents[stem])
        serialized = _project_payload(project)
        if serialized != source:
            raise ValueError(f"model_project roundtrip drift: {stem}")
        projects[stem] = project
        cases[stem] = {
            "project": serialized,
            "project_field_order": [field.name for field in dataclass_fields(XrrProject)],
            "dataset_field_order": [field.name for field in dataclass_fields(DatasetProject)],
            "roundtrip_equal": True,
        }
    return projects, cases


def _transitions(project: XrrProject) -> dict[str, object]:
    dataset = project.datasets[0]
    active = with_active_dataset(project, dataset.dataset_id)
    second = replace(dataset, dataset_id="single-layer-copy")
    batch = with_batch_mode(replace(project, datasets=(dataset, second)), "joint")
    mask = tuple(index % 7 != 0 for index in range(len(dataset.fit_mask)))
    masked = with_dataset_fit_mask(project, dataset.dataset_id, mask)
    state = replace(
        project.ui_state,
        active_dataset_id=dataset.dataset_id,
        expert_mode=True,
        workspace_splitter_sizes=(300, 700, 400),
        left_splitter_sizes=(260, 500),
        plot_tab_index=1,
    )
    workspace = with_workspace_state(project, state)
    included = sum(masked.datasets[0].fit_mask)
    return {
        "active_dataset": _json_value(active.ui_state),
        "batch_mode": {
            "before": project.batch_mode,
            "after": batch.batch_mode,
            "dataset_ids": [item.dataset_id for item in batch.datasets],
        },
        "fit_mask": {
            "dataset_id": masked.datasets[0].dataset_id,
            "included": included,
            "excluded": len(masked.datasets[0].fit_mask) - included,
        },
        "ui_state": _json_value(workspace.ui_state),
    }


def replay(context: object) -> dict[str, object]:
    """Build the normalized model-project artifact from declared replay inputs."""
    contents = _validate_context(context)
    projects, cases = _cases(contents)
    artifact = {
        "schema": "xrr-r22-model-project-reference-v1",
        "cases": cases,
        "transitions": _transitions(projects["single-layer"]),
    }
    return {ARTIFACT: artifact}
