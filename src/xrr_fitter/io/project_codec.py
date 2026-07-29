"""The single R22-compatible JSON codec for immutable R23 projects."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from xrr_fitter.io.codec_common import (
    ProjectSchemaError,
    ProjectVersionError,
    _mapping,
    _object,
    _reject_constant,
    _sequence,
    _validate_nulls,
)
from xrr_fitter.io.codec_declarations import (
    _beam_from_dict,
    _beam_to_dict,
    _column_mapping_from_dict,
    _column_mapping_to_dict,
    _fit_config_from_dict,
    _fit_config_to_dict,
    _instrument_from_dict,
    _instrument_to_dict,
    _structure_from_dict,
    _structure_to_dict,
)
from xrr_fitter.io.codec_results import (
    _checkpoint_from_dict,
    _checkpoint_to_dict,
    fit_result_from_dict,
    fit_result_to_dict,
)
from xrr_fitter.model.analysis import StructureEvidence
from xrr_fitter.model.parameters import (
    ParameterReference,
    ParameterSetting,
    SharingRule,
)
from xrr_fitter.model.project import (
    SCHEMA_VERSION,
    DatasetProject,
    OxideDecision,
    ProjectUiState,
    ScalePriorState,
    XrrProject,
)


def _evidence_to_dict(value: StructureEvidence | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "m_data": value.m_data,
        "m_model": value.m_model,
        "warning": value.warning,
        "peak_positions_a": list(value.peak_positions_a),
    }


def _evidence_from_dict(value: object) -> StructureEvidence | None:
    if value is None:
        return None
    payload = _mapping(
        value,
        {"m_data", "m_model", "warning", "peak_positions_a"},
        "structure evidence",
    )
    return StructureEvidence(
        m_data=payload["m_data"],
        m_model=payload["m_model"],
        warning=payload["warning"],
        peak_positions_a=tuple(
            _sequence(payload["peak_positions_a"], "peak positions")
        ),
    )


def _scale_prior_to_dict(value: ScalePriorState) -> dict[str, object]:
    return {
        "enabled": value.enabled,
        "s_hat": value.s_hat,
        "tau_s_decades": value.tau_s_decades,
        "reason": value.reason,
    }


def _scale_prior_from_dict(value: object) -> ScalePriorState:
    fields = {"enabled", "s_hat", "tau_s_decades", "reason"}
    return ScalePriorState(**_mapping(value, fields, "scale prior"))


def _oxide_to_dict(value: OxideDecision) -> dict[str, object]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


def _oxide_from_dict(value: object) -> OxideDecision:
    fields = set(OxideDecision.__dataclass_fields__)
    return OxideDecision(**_mapping(value, fields, "oxide decision"))


def _setting_to_dict(value: ParameterSetting) -> dict[str, object]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


def _setting_from_dict(value: object) -> ParameterSetting:
    fields = set(ParameterSetting.__dataclass_fields__)
    return ParameterSetting(**_mapping(value, fields, "parameter setting"))


def _sharing_to_dict(value: SharingRule) -> dict[str, object]:
    return {
        "sharing_key": value.sharing_key,
        "members": [
            {
                "dataset_id": member.dataset_id,
                "parameter_name": member.parameter_name,
            }
            for member in value.members
        ],
    }


def _sharing_from_dict(value: object) -> SharingRule:
    payload = _mapping(value, {"sharing_key", "members"}, "sharing rule")
    members = tuple(
        ParameterReference(
            **_mapping(item, {"dataset_id", "parameter_name"}, "sharing member")
        )
        for item in _sequence(payload["members"], "sharing members")
    )
    return SharingRule(payload["sharing_key"], members)


def _ui_to_dict(value: ProjectUiState) -> dict[str, object]:
    return {
        "active_dataset_id": value.active_dataset_id,
        "selected_candidate_ids": [list(item) for item in value.selected_candidate_ids],
        "expert_mode": value.expert_mode,
        "workspace_splitter_sizes": list(value.workspace_splitter_sizes),
        "left_splitter_sizes": list(value.left_splitter_sizes),
        "plot_tab_index": value.plot_tab_index,
    }


def _ui_from_dict(value: object) -> ProjectUiState:
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
        selected_candidate_ids=tuple(
            tuple(_sequence(item, "selected candidate"))
            for item in _sequence(
                payload["selected_candidate_ids"],
                "selected candidates",
            )
        ),
        expert_mode=payload["expert_mode"],
        workspace_splitter_sizes=tuple(
            _sequence(
                payload["workspace_splitter_sizes"],
                "workspace splitter sizes",
            )
        ),
        left_splitter_sizes=tuple(
            _sequence(payload["left_splitter_sizes"], "left splitter sizes")
        ),
        plot_tab_index=payload["plot_tab_index"],
    )


def _dataset_to_dict(value: DatasetProject) -> dict[str, object]:
    payload = {
        "dataset_id": value.dataset_id,
        "source_path": value.source_path,
        "source_sha256": value.source_sha256,
        "beam": _beam_to_dict(value.beam),
        "import_angle_offset_deg": value.import_angle_offset_deg,
        "column_mapping": _column_mapping_to_dict(value.column_mapping),
        "fit_mask": list(value.fit_mask),
        "fit_range_two_theta_deg": list(value.fit_range_two_theta_deg),
        "structure": _structure_to_dict(value.structure),
        "instrument": _instrument_to_dict(value.instrument),
        "structure_evidence": _evidence_to_dict(value.structure_evidence),
        "scale_prior": _scale_prior_to_dict(value.scale_prior),
        "oxide_decisions": [
            _oxide_to_dict(item) for item in value.oxide_decisions
        ],
        "parameter_settings": [
            _setting_to_dict(item) for item in value.parameter_settings
        ],
        "last_valid_result": fit_result_to_dict(value.last_valid_result),
        "checkpoint": _checkpoint_to_dict(value.checkpoint),
    }
    if value.display_name != value.dataset_id:
        payload["display_name"] = value.display_name
    return payload


def _dataset_fields() -> set[str]:
    return {
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
    }


def _dataset_from_dict(value: object) -> DatasetProject:
    payload = _mapping(value, _dataset_fields(), "dataset", {"display_name"})
    return DatasetProject(
        dataset_id=payload["dataset_id"],
        source_path=payload["source_path"],
        source_sha256=payload["source_sha256"],
        beam=_beam_from_dict(payload["beam"]),
        import_angle_offset_deg=payload["import_angle_offset_deg"],
        column_mapping=_column_mapping_from_dict(payload["column_mapping"]),
        fit_mask=tuple(_sequence(payload["fit_mask"], "fit mask")),
        fit_range_two_theta_deg=tuple(
            _sequence(payload["fit_range_two_theta_deg"], "fit range")
        ),
        structure=_structure_from_dict(payload["structure"]),
        instrument=_instrument_from_dict(payload["instrument"]),
        structure_evidence=_evidence_from_dict(payload["structure_evidence"]),
        scale_prior=_scale_prior_from_dict(payload["scale_prior"]),
        oxide_decisions=tuple(
            _oxide_from_dict(item)
            for item in _sequence(payload["oxide_decisions"], "oxide decisions")
        ),
        parameter_settings=tuple(
            _setting_from_dict(item)
            for item in _sequence(
                payload["parameter_settings"],
                "parameter settings",
            )
        ),
        last_valid_result=fit_result_from_dict(payload["last_valid_result"]),
        checkpoint=_checkpoint_from_dict(payload["checkpoint"]),
        display_name=payload.get("display_name"),
    )


def _project_fields() -> set[str]:
    return {
        "schema_version",
        "algorithm_version",
        "fit_config",
        "input_angle_kind",
        "batch_mode",
        "datasets",
        "sharing_rules",
        "ui_state",
    }


def project_to_dict(project: XrrProject) -> dict[str, object]:
    """Encode a project without its runtime-only base directory."""
    if not isinstance(project, XrrProject):
        raise TypeError("project must be an XrrProject")
    return {
        "schema_version": project.schema_version,
        "algorithm_version": project.algorithm_version,
        "fit_config": _fit_config_to_dict(project.fit_config),
        "input_angle_kind": project.input_angle_kind,
        "batch_mode": project.batch_mode,
        "datasets": [_dataset_to_dict(item) for item in project.datasets],
        "sharing_rules": [
            _sharing_to_dict(item) for item in project.sharing_rules
        ],
        "ui_state": _ui_to_dict(project.ui_state),
    }


def _validate_version(value: object) -> None:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ProjectVersionError(f"unsupported project schema: {value}")


def _validate_required_dataset_fields(value: object) -> None:
    if not isinstance(value, dict):
        raise ProjectSchemaError("dataset must be a JSON object")
    dataset_id = value.get("dataset_id", "<unknown>")
    for field in ("beam", "instrument"):
        if field not in value:
            raise ProjectSchemaError(f"dataset {dataset_id}: missing {field}")
    if not isinstance(value["beam"], dict) or "kind" not in value["beam"]:
        raise ProjectSchemaError(f"dataset {dataset_id}: missing beam kind")


def _result_payload(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("last_valid_result") is None:
        return None
    result = value["last_valid_result"]
    if not isinstance(result, dict):
        raise ProjectSchemaError("last_valid_result must be a JSON object")
    return result


def _candidate_ids(result: dict[str, Any]) -> tuple[str, ...]:
    candidates = result.get("candidates", ())
    if not isinstance(candidates, list):
        raise ProjectSchemaError("fit candidates must be a JSON array")
    candidate_ids = tuple(
        item.get("candidate_id") if isinstance(item, dict) else None
        for item in candidates
    )
    valid = all(
        isinstance(candidate_id, str) and bool(candidate_id)
        for candidate_id in candidate_ids
    )
    if not valid or len(candidate_ids) != len(set(candidate_ids)):
        raise ProjectSchemaError("candidate_id values must be unique and nonempty")
    return candidate_ids


def _validate_best_index(result: dict[str, Any], candidate_count: int) -> None:
    best_index = result.get("best_index")
    valid = best_index is None or (
        isinstance(best_index, int)
        and not isinstance(best_index, bool)
        and 0 <= best_index < candidate_count
    )
    if not valid:
        raise ProjectSchemaError("best_index does not identify a candidate")


def _validate_result_identity(value: object) -> None:
    result = _result_payload(value)
    if result is None:
        return
    candidate_ids = _candidate_ids(result)
    _validate_best_index(result, len(candidate_ids))


def _validated_document(value: object) -> dict[str, Any]:
    payload = _mapping(value, _project_fields(), "project")
    _validate_nulls(payload)
    _validate_version(payload["schema_version"])
    datasets = _sequence(payload["datasets"], "datasets")
    for dataset in datasets:
        _validate_required_dataset_fields(dataset)
        _validate_result_identity(dataset)
    return payload


def project_from_dict(value: object) -> XrrProject:
    """Decode a complete R22-compatible document with exact field sets."""
    try:
        payload = _validated_document(value)
        return XrrProject(
            schema_version=payload["schema_version"],
            algorithm_version=payload["algorithm_version"],
            fit_config=_fit_config_from_dict(payload["fit_config"]),
            input_angle_kind=payload["input_angle_kind"],
            batch_mode=payload["batch_mode"],
            datasets=tuple(
                _dataset_from_dict(item) for item in payload["datasets"]
            ),
            sharing_rules=tuple(
                _sharing_from_dict(item)
                for item in _sequence(
                    payload["sharing_rules"],
                    "sharing rules",
                )
            ),
            ui_state=_ui_from_dict(payload["ui_state"]),
        )
    except ProjectSchemaError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectSchemaError(str(error)) from error


def project_to_bytes(project: XrrProject) -> bytes:
    """Serialize standards-compliant UTF-8 JSON using R22 formatting."""
    document = project_to_dict(project)
    _validate_nulls(document)
    try:
        payload = json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ProjectSchemaError(str(error)) from error
    return payload.encode("utf-8")


def project_from_bytes(
    content: bytes,
    *,
    base_directory: str | None = None,
) -> XrrProject:
    """Decode hash-bound project bytes without reading any source curve."""
    if not isinstance(content, bytes):
        raise TypeError("project content must be bytes")
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=_reject_constant,
        )
    except ProjectSchemaError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectSchemaError("invalid project JSON") from error
    project = project_from_dict(value)
    return replace(project, base_directory=base_directory)


def _write_all(file_descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            raise OSError("project temporary write made no progress")
        remaining = remaining[written:]


def atomic_replace_bytes(target: Path, payload: bytes) -> None:
    """Fsync a same-directory temporary file before atomic replacement."""
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        closing_descriptor = descriptor
        descriptor = -1
        os.close(closing_descriptor)
        os.replace(temp_path, target)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def save_project(project: XrrProject, path: str | Path) -> None:
    """Atomically save one complete project document."""
    atomic_replace_bytes(Path(path), project_to_bytes(project))


def load_project(path: str | Path) -> XrrProject:
    """Load a project and attach its resolved runtime source base."""
    project_path = Path(path)
    return project_from_bytes(
        project_path.read_bytes(),
        base_directory=str(project_path.resolve().parent),
    )
