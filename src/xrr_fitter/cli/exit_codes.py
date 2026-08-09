"""The CLI exit-code contract for unattended orchestration."""

from __future__ import annotations

from collections.abc import Iterable

import xrr_fitter.api as api


SUCCESS = 0
NOT_CONVERGED = 1
INVALID_INPUT = 2
STALE_SOURCE = 3

_CONFIDENCE_CODES = {
    api.ConfidenceClass.TRUSTED: SUCCESS,
    api.ConfidenceClass.CORRELATED: SUCCESS,
    api.ConfidenceClass.MULTIPLE: NOT_CONVERGED,
    api.ConfidenceClass.UNTRUSTED: NOT_CONVERGED,
}


def confidence_exit_code(confidence: api.ConfidenceClass) -> int:
    """Map one published confidence class onto its exit code."""
    return _CONFIDENCE_CODES[confidence]


def worst_exit_code(values: Iterable[api.ConfidenceClass]) -> int:
    """Return the least favourable exit code across every dataset."""
    return max((confidence_exit_code(item) for item in values), default=SUCCESS)


def cancelled_exit_code() -> int:
    """Report a cancelled run as unconverged rather than successful."""
    return NOT_CONVERGED


def fit_exit_code(result: api.ProjectFitResult) -> int:
    """Derive the process exit code from a published project fit result."""
    if result.cancelled:
        return cancelled_exit_code()
    return worst_exit_code(item.fit_result.confidence for item in result.datasets)


def validation_exit_code(validation: api.ProjectValidation) -> int:
    """Map a source validation onto success or the stale-source code."""
    return SUCCESS if validation.valid else STALE_SOURCE
