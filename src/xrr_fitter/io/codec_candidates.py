"""Project-codec declarations for candidates, stacks, and stage summaries."""

from __future__ import annotations

from math import isfinite
from typing import Any

from xrr_fitter.io.codec_common import (
    ProjectSchemaError,
    _complex_array_from_list,
    _complex_array_to_list,
    _finite_number,
    _mapping,
    _real_array_from_list,
    _real_array_to_list,
    _sequence,
)
from xrr_fitter.model.fitting import FitCandidate, FitStageSummary
from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.parameters import ParameterDefinition, ParameterValue, PriorSpec
from xrr_fitter.model.structure import PeriodicSpan, SlabStack


def _diagnostic_to_dict(value: PhysicsDiagnostic) -> dict[str, object]:
    return {
        "code": value.code,
        "message": value.message,
        "point_indices": list(value.point_indices),
    }


def _diagnostic_from_dict(value: object) -> PhysicsDiagnostic:
    payload = _mapping(value, {"code", "message", "point_indices"}, "diagnostic")
    return PhysicsDiagnostic(
        payload["code"],
        payload["message"],
        tuple(_sequence(payload["point_indices"], "diagnostic point indices")),
    )


def _prior_to_dict(value: PriorSpec) -> dict[str, object]:
    return {"kind": value.kind, "parameters": list(value.parameters)}


def _prior_from_dict(value: object) -> PriorSpec:
    payload = _mapping(value, {"kind", "parameters"}, "parameter prior")
    return PriorSpec(payload["kind"], tuple(_sequence(payload["parameters"], "prior parameters")))


# prior is omitted from the auto-derived field set: it is emitted only when
# present so projects saved before priors existed stay byte-identical, and it
# is read back as an optional key (a bare definition decodes prior to None).
_DEFINITION_FIELDS = frozenset(ParameterDefinition.__dataclass_fields__) - {"prior"}


def _parameter_definition_to_dict(value: ParameterDefinition) -> dict[str, object]:
    payload: dict[str, object] = {field: getattr(value, field) for field in _DEFINITION_FIELDS}
    if value.prior is not None:
        payload["prior"] = _prior_to_dict(value.prior)
    return payload


def _parameter_definition_from_dict(value: object) -> ParameterDefinition:
    payload = dict(_mapping(value, set(_DEFINITION_FIELDS), "parameter definition", optional={"prior"}))
    prior = payload.pop("prior", None)
    return ParameterDefinition(**payload, prior=None if prior is None else _prior_from_dict(prior))


def _parameter_value_to_dict(value: ParameterValue) -> dict[str, object]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


def _parameter_value_from_dict(value: object) -> ParameterValue:
    fields = set(ParameterValue.__dataclass_fields__)
    return ParameterValue(**_mapping(value, fields, "parameter value"))


def _span_to_dict(value: PeriodicSpan) -> dict[str, int]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


def _span_from_dict(value: object) -> PeriodicSpan:
    fields = set(PeriodicSpan.__dataclass_fields__)
    return PeriodicSpan(**_mapping(value, fields, "periodic span"))


def _stack_to_dict(value: SlabStack | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "thickness_a": _real_array_to_list(value.thickness_a),
        "sld_a2": _complex_array_to_list(value.sld_a2),
        "roughness_a": _real_array_to_list(value.roughness_a),
        "periodic_spans": [_span_to_dict(item) for item in value.periodic_spans],
    }


def _stack_from_dict(value: object) -> SlabStack | None:
    if value is None:
        return None
    payload = _mapping(
        value,
        {"thickness_a", "sld_a2", "roughness_a", "periodic_spans"},
        "slab stack",
    )
    return SlabStack(
        thickness_a=_real_array_from_list(payload["thickness_a"]),
        sld_a2=_complex_array_from_list(payload["sld_a2"]),
        roughness_a=_real_array_from_list(payload["roughness_a"]),
        periodic_spans=tuple(_span_from_dict(item) for item in _sequence(payload["periodic_spans"], "periodic spans")),
    )


def _candidate_objective_to_json(value: FitCandidate) -> float | None:
    if _finite_number(value.objective):
        return float(value.objective)
    if value.valid or value.objective != float("inf"):
        raise ProjectSchemaError("candidate objective must be finite when valid")
    return None


def _candidate_objective_from_json(payload: dict[str, Any]) -> float:
    value = payload["objective"]
    if value is None:
        if payload["valid"]:
            raise ProjectSchemaError("candidate objective is null for valid candidate")
        return float("inf")
    if not _finite_number(value):
        raise ProjectSchemaError("candidate objective must be finite JSON number")
    return float(value)


def _ranking_to_json(value: float | None) -> float | None:
    if value is None:
        return None
    if not _finite_number(value):
        raise ProjectSchemaError("candidate ranking_objective must be finite")
    return float(value)


def _candidate_to_dict(value: FitCandidate) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": value.candidate_id,
        "seed_index": value.seed_index,
        "unit_vector": _real_array_to_list(value.unit_vector),
        "parameters": [_parameter_value_to_dict(item) for item in value.parameters],
        "objective": _candidate_objective_to_json(value),
        "valid": value.valid,
        "stop_reason": value.stop_reason,
        "nfev": value.nfev,
        "qz_a_inv": _real_array_to_list(value.qz_a_inv),
        "model_normalized": _real_array_to_list(value.model_normalized),
        "log_residuals_decades": _real_array_to_list(value.log_residuals_decades),
        "weighted_residuals": _real_array_to_list(value.weighted_residuals),
        "expanded_stack": _stack_to_dict(value.expanded_stack),
        "sld_depth_a": _real_array_to_list(value.sld_depth_a),
        "sld_profile_a2": _complex_array_to_list(value.sld_profile_a2),
        "diagnostics": [_diagnostic_to_dict(item) for item in value.diagnostics],
    }
    if value.ranking_objective is not None:
        payload["ranking_objective"] = _ranking_to_json(value.ranking_objective)
    return payload


def _candidate_from_dict(value: object) -> FitCandidate:
    required = {
        "candidate_id",
        "seed_index",
        "unit_vector",
        "parameters",
        "objective",
        "valid",
        "stop_reason",
        "nfev",
        "qz_a_inv",
        "model_normalized",
        "log_residuals_decades",
        "weighted_residuals",
        "expanded_stack",
        "sld_depth_a",
        "sld_profile_a2",
        "diagnostics",
    }
    payload = _mapping(value, required, "fit candidate", {"ranking_objective"})
    ranking = payload.get("ranking_objective")
    if ranking is not None and not _finite_number(ranking):
        raise ProjectSchemaError("candidate ranking_objective must be finite JSON number")
    return FitCandidate(
        candidate_id=payload["candidate_id"],
        seed_index=payload["seed_index"],
        unit_vector=_real_array_from_list(payload["unit_vector"]),
        parameters=tuple(
            _parameter_value_from_dict(item) for item in _sequence(payload["parameters"], "candidate parameters")
        ),
        objective=_candidate_objective_from_json(payload),
        valid=payload["valid"],
        stop_reason=payload["stop_reason"],
        nfev=payload["nfev"],
        qz_a_inv=_real_array_from_list(payload["qz_a_inv"]),
        model_normalized=_real_array_from_list(payload["model_normalized"]),
        log_residuals_decades=_real_array_from_list(payload["log_residuals_decades"]),
        weighted_residuals=_real_array_from_list(payload["weighted_residuals"]),
        expanded_stack=_stack_from_dict(payload["expanded_stack"]),
        sld_depth_a=_real_array_from_list(payload["sld_depth_a"]),
        sld_profile_a2=_complex_array_from_list(payload["sld_profile_a2"]),
        diagnostics=tuple(
            _diagnostic_from_dict(item) for item in _sequence(payload["diagnostics"], "candidate diagnostics")
        ),
        ranking_objective=None if ranking is None else float(ranking),
    )


def _selectable(candidate: FitCandidate) -> bool:
    objective = candidate.objective if candidate.ranking_objective is None else candidate.ranking_objective
    return candidate.valid and isfinite(objective) and candidate.stop_reason != "early_eliminated"


def _stage_selectable(
    stage: str,
    candidate_ids: tuple[str, ...],
    candidates: dict[str, FitCandidate],
) -> bool:
    if stage in {"A", "stage-a"}:
        return bool(candidate_ids)
    try:
        members = tuple(candidates[candidate_id] for candidate_id in candidate_ids)
    except KeyError as error:
        raise ProjectSchemaError(f"stage references missing candidate: {error.args[0]}") from error
    return any(_selectable(candidate) for candidate in members)


def _stage_objective_to_json(value: float, selectable: bool) -> float | None:
    if _finite_number(value) and selectable:
        return float(value)
    if value == float("inf") and not selectable:
        return None
    raise ProjectSchemaError("stage best_objective is invalid for candidate state")


def _stage_objective_from_json(value: object, selectable: bool) -> float:
    if value is None and not selectable:
        return float("inf")
    if _finite_number(value) and selectable:
        return float(value)
    raise ProjectSchemaError("stage best_objective is invalid for candidate state")


def _candidate_map(candidates: tuple[FitCandidate, ...]) -> dict[str, FitCandidate]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ProjectSchemaError("candidate_id values must be unique")
    return by_id


def _stages_to_list(
    values: tuple[FitStageSummary, ...],
    candidates: tuple[FitCandidate, ...],
) -> list[dict[str, object]]:
    by_id = _candidate_map(candidates)
    result = []
    for value in values:
        selectable = _stage_selectable(value.stage, value.candidate_ids, by_id)
        result.append(
            {
                "stage": value.stage,
                "candidate_ids": list(value.candidate_ids),
                "best_objective": _stage_objective_to_json(
                    value.best_objective,
                    selectable,
                ),
                "total_nfev": value.total_nfev,
                "stop_reasons": list(value.stop_reasons),
            }
        )
    return result


def _stages_from_list(
    value: object,
    candidates: tuple[FitCandidate, ...],
) -> tuple[FitStageSummary, ...]:
    by_id = _candidate_map(candidates)
    result = []
    fields = {
        "stage",
        "candidate_ids",
        "best_objective",
        "total_nfev",
        "stop_reasons",
    }
    for item in _sequence(value, "stage summaries"):
        payload = _mapping(item, fields, "stage summary")
        candidate_ids = tuple(_sequence(payload["candidate_ids"], "stage candidate IDs"))
        selectable = _stage_selectable(payload["stage"], candidate_ids, by_id)
        result.append(
            FitStageSummary(
                stage=payload["stage"],
                candidate_ids=candidate_ids,
                best_objective=_stage_objective_from_json(
                    payload["best_objective"],
                    selectable,
                ),
                total_nfev=payload["total_nfev"],
                stop_reasons=tuple(_sequence(payload["stop_reasons"], "stage stop reasons")),
            )
        )
    return tuple(result)
