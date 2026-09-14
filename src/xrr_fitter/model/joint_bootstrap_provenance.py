"""Separate numerical ownership and complete-content seals for joint sampling.

This module only hashes immutable values. The fitting service supplies the
compiled layout and actual winner snapshots; persistence reconstructs those
same values without running forward physics, optimization, or resampling.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from xrr_fitter.model.bootstrap import BootstrapResult
from xrr_fitter.model.provenance import _dataclass_payload, _identity_sha256


def _context(problem):
    value = _dataclass_payload(problem, frozenset())
    value["data"] = _dataclass_payload(problem.data, frozenset({"source_path"}))
    return value


def _validate_seed(child_seed):
    valid_seed = not isinstance(child_seed, bool) and isinstance(child_seed, (int, np.integer))
    if not valid_seed or not 0 <= child_seed < 2**64:
        raise ValueError("joint bootstrap child seed must be uint64")


def _winner_axis(problem, candidates, unit_vector, child_seed):
    identifiers = tuple(problem.dataset_ids)
    if len(identifiers) < 2 or len(set(identifiers)) != len(identifiers):
        raise ValueError("joint bootstrap requires unique aligned dataset IDs")
    if len(problem.problems) != len(identifiers) or len(candidates) != len(identifiers):
        raise ValueError("joint bootstrap context and candidate axes must match")
    if len({candidate.candidate_id for candidate in candidates}) != 1:
        raise ValueError("joint bootstrap candidates must have one common owner")
    vector = np.asarray(unit_vector, dtype=float)
    if vector.shape != (len(problem.global_variables),) or np.any(~np.isfinite(vector)):
        raise ValueError("joint bootstrap vector must match the complete global layout")
    _validate_seed(child_seed)
    return vector


def joint_bootstrap_owner_sha256(problem, candidates, unit_vector, child_seed) -> str:
    """Bind all contexts, the full layout, ordered winners, and actual draw seed."""
    candidates = tuple(candidates)
    vector = _winner_axis(problem, candidates, unit_vector, child_seed)
    return _identity_sha256(
        {
            "schema": "xrr-joint-bootstrap-owner-v1",
            "contexts": tuple(_context(local) for local in problem.problems),
            "layout": _dataclass_payload(problem, frozenset({"problems"})),
            "candidates": candidates,
            "unit_vector": vector,
            "child_seed": int(child_seed),
        }
    )


def _content_sha256(payload) -> str:
    return _identity_sha256({"schema": "xrr-joint-bootstrap-content-v1", "bootstrap": payload})


def joint_bootstrap_provenance_sha256(result: BootstrapResult) -> str:
    """Include owner and candidate identity; exclude only the content seal."""
    return _content_sha256(_dataclass_payload(result, frozenset({"provenance_sha256"})))


def seal_joint_bootstrap(result: BootstrapResult, candidate_id: str, owner_sha256: str) -> BootstrapResult:
    """Seal newly generated or explicitly requalified evidence, never a foreign result."""
    payload = _dataclass_payload(result, frozenset({"provenance_sha256"}))
    payload.update(candidate_id=candidate_id, joint_owner_sha256=owner_sha256)
    return replace(
        result, candidate_id=candidate_id, joint_owner_sha256=owner_sha256, provenance_sha256=_content_sha256(payload)
    )


def validate_joint_bootstrap_content(result: BootstrapResult) -> None:
    """Validate nested immutable values and the seal without trusting a summary."""
    if not isinstance(result, BootstrapResult):
        raise TypeError("joint bootstrap evidence must be BootstrapResult")
    rebuilt = replace(result)
    if rebuilt.joint_owner_sha256 is None or rebuilt.candidate_id is None:
        raise ValueError("joint bootstrap requires a complete numerical owner")
    if rebuilt.provenance_sha256 != joint_bootstrap_provenance_sha256(rebuilt):
        raise ValueError("joint bootstrap content provenance does not match its evidence")


def validate_joint_bootstrap(result, candidate_id, owner_sha256, parameter_names) -> None:
    """Require independently reconstructed numerical ownership at consumption."""
    validate_joint_bootstrap_content(result)
    if result.candidate_id != candidate_id or result.joint_owner_sha256 != owner_sha256:
        raise ValueError("joint bootstrap owner does not match the numerical context and winner")
    if result.parameter_names != tuple(parameter_names):
        raise ValueError("joint bootstrap parameter axis does not match its layout")


def validate_joint_bootstrap_report(report) -> None:
    """Validate a global report without loading its numerical source context."""
    sampling = report.bootstrap_evidence
    if sampling is None:
        return
    validate_joint_bootstrap_content(sampling)
    if report.candidate_id != sampling.candidate_id or report.correlation_names != sampling.parameter_names:
        raise ValueError("joint bootstrap owner and parameter axis must match its report")
    if not report.bootstrap_performed or report.bootstrap_intervals != sampling.intervals:
        raise ValueError("joint bootstrap summary must match sampling evidence")
    if report.bootstrap_failure_rate != sampling.failure_rate:
        raise ValueError("joint bootstrap failure summary must match sampling evidence")


def validate_joint_bootstrap_reports(reports) -> None:
    """Joint sampling is atomic; allow only the separate per-member MCMC sidecar."""
    if not any(report is not None and report.bootstrap_evidence is not None for report in reports):
        return
    identities = []
    for report in reports:
        if report is None or report.bootstrap_evidence is None or report.parameter_members is None:
            raise ValueError("joint bootstrap requires complete shared uncertainty for every member")
        validate_joint_bootstrap_report(report)
        identities.append(_identity_sha256(_dataclass_payload(report, frozenset({"mcmc", "sld_bands"}))))
    if len(set(identities)) != 1:
        raise ValueError("joint bootstrap uncertainty evidence differs across members")
