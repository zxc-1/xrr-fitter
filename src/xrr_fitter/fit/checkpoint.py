"""Canonical fitting checkpoint identity and immutable construction."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import json

import numpy as np

from xrr_fitter.model.fitting import FitCandidate, FitCheckpoint, FitStageSummary


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    data_sha256: str
    structure_fingerprint: str
    instrument_fingerprint: str
    config_fingerprint: str
    parameter_settings_fingerprint: str


def _canonical(value: object) -> object:
    if is_dataclass(value):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"imag": value.imag, "real": value.real}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    return value


def _fingerprint(schema: str, value: object) -> str:
    payload = {"schema": schema, "value": _canonical(value)}
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def checkpoint_identity(problem: object) -> CheckpointIdentity:
    """Bind every resume-relevant declaration without source-path identity."""
    parameter_layout = {
        "definitions": problem.parameter_definitions,
        "variables": problem.variables,
        "region_labels": problem.region_labels,
        "weights": problem.weights,
    }
    return CheckpointIdentity(
        data_sha256=problem.data.source_sha256,
        structure_fingerprint=_fingerprint("xrr-fit-structure-v1", problem.structure),
        instrument_fingerprint=_fingerprint("xrr-fit-instrument-v1", problem.instrument),
        config_fingerprint=_fingerprint("xrr-fit-config-v1", problem.config),
        parameter_settings_fingerprint=_fingerprint(
            "xrr-fit-parameter-layout-v1",
            parameter_layout,
        ),
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
