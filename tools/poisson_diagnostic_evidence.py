"""Pure projections of saved product evidence, without a diagnostic estimator.

Raw screens, an unrequested Monte Carlo calculation, a completed calibration,
and an unavailable calibration are distinct observations. Joint projections
carry one family; their copies must not multiply B or the rejection count.
Profile and bootstrap availability are separate quantities, especially for the
joint scenario, where the product currently does not provide a profile.
"""

from __future__ import annotations

from dataclasses import asdict

import interval_coverage_cases as coverage
from interval_coverage_evidence import _interval, bootstrap_interval, profile_interval

STATISTIC_FIELDS = ("dataset_id", "kind", "observed", "center", "scale", "adjusted_p_value")
CALIBRATION_FIELDS = (
    "status",
    "sample_count",
    "child_seed",
    "owner_sha256",
    "method",
    "refit_policy",
    "alpha",
    "attempted_count",
    "successful_count",
    "tail_count",
    "tie_count",
    "observed_score",
    "null_statistics_sha256",
    "failure_reasons",
    "unavailable_reason",
    "refit_nfev",
    "refit_discrepancy",
    "provenance_sha256",
    "p_value",
    "resolution",
    "rejected",
)


def _calibration_record(value) -> dict:
    return {name: getattr(value, name) for name in CALIBRATION_FIELDS} | {
        "statistics": [{name: getattr(statistic, name) for name in STATISTIC_FIELDS} for statistic in value.statistics],
    }


def _member_record(member) -> dict:
    names = (
        "dataset_id",
        "executed",
        "systematic",
        "autocorrelation",
        "raw_systematic",
        "raw_autocorrelation",
        "unavailable_reason",
    )
    return {name: getattr(member, name) for name in names} | {
        "advisories": [asdict(value) for value in member.advisories],
        "diagnostics": [asdict(value) for value in member.diagnostics],
    }


def _raw_triggered(members: tuple) -> bool | None:
    flags = [flag for member in members for flag in (member.raw_systematic, member.raw_autocorrelation)]
    if any(flag is True for flag in flags):
        return True
    return None if not flags or None in flags else False


def _base_state(members: tuple) -> dict:
    return {
        "status": "not_triggered",
        "raw_triggered": _raw_triggered(members),
        "rejected": False,
        "target_detected": False,
        "rejected_columns": [],
        "unavailable_reason": None,
        "calibration": None,
        "members": [_member_record(member) for member in members],
    }


def unavailable_diagnostic(reason: str) -> dict:
    return _base_state(()) | {"status": "unavailable", "unavailable_reason": reason}


def _family(members: tuple):
    calibrations = [member.calibration for member in members if member.calibration is not None]
    if not calibrations:
        return None
    seals = {value.provenance_sha256 for value in calibrations}
    if len(seals) != 1 or None in seals or len(calibrations) != len(members):
        raise ValueError("saved joint diagnostic family is incomplete or incoherent")
    return calibrations[0]


def _residual_reason(members: tuple) -> str | None:
    reasons = [member.unavailable_reason for member in members if not member.executed]
    if reasons:
        return "; ".join(dict.fromkeys(reasons))
    if any(member.raw_systematic is None or member.raw_autocorrelation is None for member in members):
        return "raw_residual_evidence_missing"
    return None


def _available_state(family, target: str | None) -> dict:
    columns = sorted({value.kind for value in family.statistics if value.adjusted_p_value <= family.alpha})
    return {
        "status": "rejected" if family.rejected else "not_rejected",
        "rejected": family.rejected,
        "target_detected": target in columns,
        "rejected_columns": columns,
    }


def diagnostic_state(members: tuple, *, target: str | None) -> dict:
    """Consume only persisted effective flags and calibrated per-column p_j."""
    if not members:
        return unavailable_diagnostic("residual_evidence_missing")
    base = _base_state(members)
    family = _family(members)
    if family is not None:
        base["calibration"] = _calibration_record(family)
    reason = _residual_reason(members)
    if reason is not None:
        return base | {"status": "unavailable", "unavailable_reason": reason}
    if family is None:
        if base["raw_triggered"]:
            return base | {"status": "unavailable", "unavailable_reason": "triggered_calibration_missing"}
        return base
    if family.status != "available":
        return base | {"status": "unavailable", "unavailable_reason": family.unavailable_reason}
    return base | _available_state(family, target)


def saved_intervals(report, *, dataset_id: str, joint: bool) -> dict:
    if report is None:
        return {
            name: _interval(name, coverage.TARGET, "uncertainty_not_performed") for name in ("profile", "bootstrap")
        }
    name = coverage.TARGET
    if joint and report.bootstrap_evidence is not None:
        name = coverage._target_axis(report, dataset_id)
    profile = next((value for value in report.profiles if value.name == name), None)
    reason = "joint_profile_not_provided" if joint else "target_profile_missing"
    return {
        "profile": _interval("profile", name, reason) if profile is None else profile_interval(profile),
        "bootstrap": bootstrap_interval(report.bootstrap_evidence, name),
    }


def saved_result(project, *, target: str | None) -> dict:
    """Read a result only after the current public API save/load roundtrip."""
    results = [coverage._candidate(dataset) for dataset in project.datasets]
    report = results[0][0].uncertainty
    members = () if report is None else report.member_residuals
    diagnostic = diagnostic_state(members, target=target)
    return (
        coverage._fit_summary(project, results)
        | {
            "diagnostic": diagnostic,
            "saved_diagnostics": coverage._diagnostics(report),
            "evidence_source": "api.fit_project -> api.save_project -> api.load_project",
            "failure_stage": None,
            "failure_reason": None,
        }
        | saved_intervals(report, dataset_id=project.datasets[0].dataset_id, joint=project.batch_mode == "joint")
    )


def failed_outcome(error: Exception, *, stage: str = "fit") -> dict:
    reason = f"{type(error).__name__}: {error}"
    return {
        "estimate": None,
        "fit_available": False,
        "failure_reason": reason,
        "failure_stage": stage,
        "fit_warnings": [],
        "diagnostic": unavailable_diagnostic(reason),
        "profile": _interval("profile", coverage.TARGET, reason),
        "bootstrap": _interval("bootstrap", coverage.TARGET, reason),
        "fit_seconds": 0.0,
        "elapsed_seconds": 0.0,
    }
