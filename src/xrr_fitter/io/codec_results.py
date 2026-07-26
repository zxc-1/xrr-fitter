"""Project-codec declarations for fit results, uncertainty, and checkpoints."""

from __future__ import annotations

from typing import Any

from xrr_fitter.io.codec_candidates import (
    _candidate_from_dict,
    _candidate_to_dict,
    _diagnostic_from_dict,
    _diagnostic_to_dict,
    _parameter_definition_from_dict,
    _parameter_definition_to_dict,
    _stages_from_list,
    _stages_to_list,
)
from xrr_fitter.io.codec_common import (
    ProjectSchemaError,
    _mapping,
    _real_array_from_list,
    _real_array_to_list,
    _sequence,
)
from xrr_fitter.model.analysis import (
    ConfidenceClass,
    FitResult,
    McmcConfig,
    McmcReport,
    ParameterProfile,
    UncertaintyReport,
)
from xrr_fitter.model.fitting import FitCheckpoint


def _profile_to_dict(value: ParameterProfile) -> dict[str, object]:
    return {
        "name": value.name,
        "values": _real_array_to_list(value.values),
        "objectives": _real_array_to_list(value.objectives),
        "lower_closed": value.lower_closed,
        "upper_closed": value.upper_closed,
    }


def _profile_from_dict(value: object) -> ParameterProfile:
    payload = _mapping(
        value,
        {"name", "values", "objectives", "lower_closed", "upper_closed"},
        "parameter profile",
    )
    return ParameterProfile(
        name=payload["name"],
        values=_real_array_from_list(payload["values"]),
        objectives=_real_array_from_list(payload["objectives"]),
        lower_closed=payload["lower_closed"],
        upper_closed=payload["upper_closed"],
    )


def _mcmc_to_dict(value: McmcReport | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "config": {
            field: getattr(value.config, field)
            for field in value.config.__dataclass_fields__
        },
        "child_seed": value.child_seed,
        "parameter_names": list(value.parameter_names),
        "samples_physical": _real_array_to_list(value.samples_physical),
        "log_probability": _real_array_to_list(value.log_probability),
        "acceptance_fraction": _real_array_to_list(value.acceptance_fraction),
        "split_rhat": _real_array_to_list(value.split_rhat),
        "effective_sample_size": _real_array_to_list(value.effective_sample_size),
        "boundary_hits": list(value.boundary_hits),
        "label": value.label,
        "warnings": list(value.warnings),
        "candidate_id": value.candidate_id,
    }


def _mcmc_from_dict(value: object) -> McmcReport | None:
    if value is None:
        return None
    required = {
        "config",
        "child_seed",
        "parameter_names",
        "samples_physical",
        "log_probability",
        "acceptance_fraction",
        "split_rhat",
        "effective_sample_size",
        "boundary_hits",
        "label",
        "warnings",
    }
    payload = _mapping(value, required, "MCMC report", {"candidate_id"})
    config = McmcConfig(
        **_mapping(
            payload["config"],
            set(McmcConfig.__dataclass_fields__),
            "MCMC config",
        )
    )
    return McmcReport(
        config=config,
        child_seed=payload["child_seed"],
        parameter_names=tuple(
            _sequence(payload["parameter_names"], "MCMC parameter names")
        ),
        samples_physical=_real_array_from_list(payload["samples_physical"]),
        log_probability=_real_array_from_list(payload["log_probability"]),
        acceptance_fraction=_real_array_from_list(payload["acceptance_fraction"]),
        split_rhat=_real_array_from_list(payload["split_rhat"]),
        effective_sample_size=_real_array_from_list(payload["effective_sample_size"]),
        boundary_hits=tuple(
            _sequence(payload["boundary_hits"], "MCMC boundary hits")
        ),
        label=payload["label"],
        warnings=tuple(_sequence(payload["warnings"], "MCMC warnings")),
        candidate_id=payload.get("candidate_id"),
    )


def _uncertainty_to_dict(
    value: UncertaintyReport | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "correlation_names": list(value.correlation_names),
        "correlation_matrix": _real_array_to_list(value.correlation_matrix),
        "profiles": [_profile_to_dict(item) for item in value.profiles],
        "bootstrap_intervals": [list(item) for item in value.bootstrap_intervals],
        "bootstrap_failure_rate": value.bootstrap_failure_rate,
        "boundary_hits": list(value.boundary_hits),
        "strong_correlations": [list(item) for item in value.strong_correlations],
        "systematic_residual": value.systematic_residual,
        "diagnostics": [_diagnostic_to_dict(item) for item in value.diagnostics],
        "residual_autocorrelation": value.residual_autocorrelation,
        "mcmc": _mcmc_to_dict(value.mcmc),
        "candidate_id": value.candidate_id,
    }


def _rows(value: object, label: str) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(_sequence(item, label)) for item in _sequence(value, label))


def _uncertainty_from_dict(value: object) -> UncertaintyReport | None:
    if value is None:
        return None
    required = {
        "correlation_names",
        "correlation_matrix",
        "profiles",
        "bootstrap_intervals",
        "bootstrap_failure_rate",
        "boundary_hits",
        "strong_correlations",
        "systematic_residual",
        "diagnostics",
        "residual_autocorrelation",
        "mcmc",
    }
    payload = _mapping(value, required, "uncertainty report", {"candidate_id"})
    return UncertaintyReport(
        correlation_names=tuple(
            _sequence(payload["correlation_names"], "correlation names")
        ),
        correlation_matrix=_real_array_from_list(payload["correlation_matrix"]),
        profiles=tuple(
            _profile_from_dict(item)
            for item in _sequence(payload["profiles"], "parameter profiles")
        ),
        bootstrap_intervals=_rows(
            payload["bootstrap_intervals"], "bootstrap intervals"
        ),
        bootstrap_failure_rate=payload["bootstrap_failure_rate"],
        boundary_hits=tuple(
            _sequence(payload["boundary_hits"], "boundary hits")
        ),
        strong_correlations=_rows(
            payload["strong_correlations"], "strong correlations"
        ),
        systematic_residual=payload["systematic_residual"],
        diagnostics=tuple(
            _diagnostic_from_dict(item)
            for item in _sequence(
                payload["diagnostics"],
                "uncertainty diagnostics",
            )
        ),
        residual_autocorrelation=payload["residual_autocorrelation"],
        mcmc=_mcmc_from_dict(payload["mcmc"]),
        candidate_id=payload.get("candidate_id"),
    )


def fit_result_to_dict(value: FitResult | None) -> dict[str, object] | None:
    """Encode one complete final fit graph using R22 field names."""
    if value is None:
        return None
    return {
        "parameter_definitions": [
            _parameter_definition_to_dict(item)
            for item in value.parameter_definitions
        ],
        "candidates": [_candidate_to_dict(item) for item in value.candidates],
        "best_index": value.best_index,
        "confidence": value.confidence.value,
        "warnings": list(value.warnings),
        "child_seeds": list(value.child_seeds),
        "stage_summaries": _stages_to_list(
            value.stage_summaries,
            value.candidates,
        ),
        "region_labels": _real_array_to_list(value.region_labels),
        "region_weights": _real_array_to_list(value.region_weights),
        "uncertainty": _uncertainty_to_dict(value.uncertainty),
        "classification_evidence": list(value.classification_evidence),
    }


def _classification_evidence(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("classification_evidence", [])
    if not isinstance(raw, list) or not all(
        isinstance(reason, str) and bool(reason) for reason in raw
    ):
        raise ProjectSchemaError("classification_evidence must contain strings")
    return tuple(raw)


def fit_result_from_dict(value: object) -> FitResult | None:
    """Decode one complete final fit graph through immutable model validation."""
    if value is None:
        return None
    required = {
        "parameter_definitions",
        "candidates",
        "best_index",
        "confidence",
        "warnings",
        "child_seeds",
        "stage_summaries",
        "region_labels",
        "region_weights",
        "uncertainty",
    }
    payload = _mapping(value, required, "fit result", {"classification_evidence"})
    candidates = tuple(
        _candidate_from_dict(item)
        for item in _sequence(payload["candidates"], "fit candidates")
    )
    return FitResult(
        parameter_definitions=tuple(
            _parameter_definition_from_dict(item)
            for item in _sequence(
                payload["parameter_definitions"],
                "parameter definitions",
            )
        ),
        candidates=candidates,
        best_index=payload["best_index"],
        confidence=ConfidenceClass(payload["confidence"]),
        warnings=tuple(_sequence(payload["warnings"], "fit warnings")),
        child_seeds=tuple(
            _sequence(payload["child_seeds"], "fit child seeds")
        ),
        stage_summaries=_stages_from_list(payload["stage_summaries"], candidates),
        region_labels=_real_array_from_list(payload["region_labels"], int),
        region_weights=_real_array_from_list(payload["region_weights"]),
        uncertainty=_uncertainty_from_dict(payload["uncertainty"]),
        classification_evidence=_classification_evidence(payload),
    )


def _checkpoint_to_dict(value: FitCheckpoint | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "data_sha256": value.data_sha256,
        "structure_fingerprint": value.structure_fingerprint,
        "config_fingerprint": value.config_fingerprint,
        "stage": value.stage,
        "candidates": [_candidate_to_dict(item) for item in value.candidates],
        "child_seeds": list(value.child_seeds),
        "instrument_fingerprint": value.instrument_fingerprint,
        "parameter_settings_fingerprint": value.parameter_settings_fingerprint,
        "runtime_warnings": list(value.runtime_warnings),
        "stage_summaries": _stages_to_list(
            value.stage_summaries,
            value.candidates,
        ),
        "joint_layout_fingerprint": value.joint_layout_fingerprint,
    }


def _checkpoint_from_dict(value: object) -> FitCheckpoint | None:
    if value is None:
        return None
    required = {
        "data_sha256",
        "structure_fingerprint",
        "config_fingerprint",
        "stage",
        "candidates",
        "child_seeds",
        "instrument_fingerprint",
        "parameter_settings_fingerprint",
        "runtime_warnings",
        "stage_summaries",
    }
    payload = _mapping(
        value,
        required,
        "fit checkpoint",
        {"joint_layout_fingerprint"},
    )
    candidates = tuple(
        _candidate_from_dict(item)
        for item in _sequence(payload["candidates"], "checkpoint candidates")
    )
    return FitCheckpoint(
        data_sha256=payload["data_sha256"],
        structure_fingerprint=payload["structure_fingerprint"],
        config_fingerprint=payload["config_fingerprint"],
        stage=payload["stage"],
        candidates=candidates,
        child_seeds=tuple(
            _sequence(payload["child_seeds"], "checkpoint child seeds")
        ),
        instrument_fingerprint=payload["instrument_fingerprint"],
        parameter_settings_fingerprint=payload[
            "parameter_settings_fingerprint"
        ],
        runtime_warnings=tuple(
            _sequence(payload["runtime_warnings"], "runtime warnings")
        ),
        stage_summaries=_stages_from_list(payload["stage_summaries"], candidates),
        joint_layout_fingerprint=payload.get("joint_layout_fingerprint", ""),
    )
