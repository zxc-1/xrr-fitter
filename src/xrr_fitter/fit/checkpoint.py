"""Canonical fitting checkpoint identity and immutable construction."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from hashlib import sha256

import numpy as np

from xrr_fitter.model.fitting import FitCandidate, FitCheckpoint, FitStageSummary


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    data_sha256: str
    structure_fingerprint: str
    instrument_fingerprint: str
    config_fingerprint: str
    parameter_settings_fingerprint: str


PRIOR_OMITTED_DEFAULTS: dict[tuple[str, str], object] = {
    ("ParameterDefinition", "prior"): None,
    ("ConfidenceThresholds", "prior_conflict_sigmas"): 3.0,
}


def _is_unset_prior_field(type_name: str, field_name: str, current: object) -> bool:
    # Prior-related fields joined the schema after checkpoints were frozen. When
    # they hold their pre-prior default we omit them so an unconfigured run
    # reproduces its historical fingerprint bit-for-bit and still resumes; a
    # configured prior keeps the field and therefore earns a distinct identity.
    key = (type_name, field_name)
    return key in PRIOR_OMITTED_DEFAULTS and current == PRIOR_OMITTED_DEFAULTS[key]


def _canonical_dataclass(value: object) -> dict[str, object]:
    type_name = type(value).__name__
    return {
        field.name: _canonical(getattr(value, field.name))
        for field in fields(value)
        if not _is_unset_prior_field(type_name, field.name, getattr(value, field.name))
    }


def _canonical_sequence(value: tuple[object, ...] | list[object]) -> list[object]:
    return [_canonical(item) for item in value]


def _canonical_mapping(value: dict[object, object]) -> dict[str, object]:
    return {str(key): _canonical(value[key]) for key in sorted(value, key=lambda item: str(item))}


def _canonical(value: object) -> object:
    if is_dataclass(value):
        return _canonical_dataclass(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, (tuple, list)):
        return _canonical_sequence(value)
    if isinstance(value, dict):
        return _canonical_mapping(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        _canonical(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def checkpoint_identity(problem: object) -> CheckpointIdentity:
    """Bind every resume-relevant declaration without source-path identity."""
    return CheckpointIdentity(
        data_sha256=problem.data.source_sha256,
        structure_fingerprint=_fingerprint(problem.structure),
        instrument_fingerprint=_fingerprint(problem.instrument),
        config_fingerprint=_fingerprint(problem.config),
        parameter_settings_fingerprint=_fingerprint(problem.parameter_definitions),
    )


def build_checkpoint(
    problem: object,
    *,
    stage: str,
    candidates: tuple[FitCandidate, ...],
    child_seeds: tuple[int, ...],
    runtime_warnings: tuple[str, ...],
    stage_summaries: tuple[FitStageSummary, ...],
    joint_layout_fingerprint: str = "",
) -> FitCheckpoint:
    """Construct one coherent, identity-bound in-memory checkpoint."""
    identity = checkpoint_identity(problem)
    return FitCheckpoint(
        data_sha256=identity.data_sha256,
        structure_fingerprint=identity.structure_fingerprint,
        instrument_fingerprint=identity.instrument_fingerprint,
        config_fingerprint=identity.config_fingerprint,
        parameter_settings_fingerprint=identity.parameter_settings_fingerprint,
        stage=stage,
        candidates=tuple(candidates),
        child_seeds=tuple(child_seeds),
        runtime_warnings=tuple(runtime_warnings),
        stage_summaries=tuple(stage_summaries),
        joint_layout_fingerprint=joint_layout_fingerprint,
    )
