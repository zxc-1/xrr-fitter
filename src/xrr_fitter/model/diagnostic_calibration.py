"""Immutable, replayable evidence for the Poisson refit-null diagnostic family.

Raw detector statistics are distinct from calibrated conclusions. A completed
family binds every member to the same Monte Carlo sample axis; incomplete work
retains its exact failure accounting but cannot publish a tail probability.
Construction and pickle reconstruction validate values without running physics
or checking a provenance seal against external context. Consumers own that
separate seal check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real

import numpy as np

from xrr_fitter.model.diagnostic_work import DiagnosticRefitWork, calibration_work, refit_work
from xrr_fitter.model.evaluation import ModelEvaluation
from xrr_fitter.model.fitting import _pickle_values, _positive_integer, _readonly, _sha256

RESIDUAL_ADVISORY_CODES = frozenset(
    {"suspected_unmodeled_footprint", "suspected_diffuse_background", "surface_thin_layer_residual"}
)
STATISTIC_KINDS = frozenset({"footprint", "background", "surface", "acf"})


def _nonempty(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _finite(value: object, field: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{field} must be a finite real number")


def _statistic_scale(statistic: DiagnosticStatistic) -> None:
    if statistic.scale < 0.0:
        raise ValueError("statistic RMS scale must be nonnegative")
    if statistic.scale == 0.0 and (statistic.observed != statistic.center or statistic.adjusted_p_value != 1.0):
        raise ValueError("zero statistic scale must describe a neutral constant column")


def _calibrated_statistic(statistic: DiagnosticStatistic) -> None:
    values = (statistic.center, statistic.scale, statistic.adjusted_p_value)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError("statistic calibration fields must be all present or all absent")
    for name in ("center", "scale", "adjusted_p_value"):
        _finite(getattr(statistic, name), name)
    _statistic_scale(statistic)
    if not 0.0 <= statistic.adjusted_p_value <= 1.0:
        raise ValueError("adjusted_p_value must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DiagnosticStatistic:
    """One detector with optional family evidence and a raw-unit RMS scale.

    Scale zero is an explicit constant-column state. Exact score replay uses
    the full matrix and the versioned factorized algorithm, not rounded scale.
    """

    dataset_id: str | None
    kind: str
    observed: float
    center: float | None = None
    scale: float | None = None
    adjusted_p_value: float | None = None

    def __post_init__(self) -> None:
        if self.dataset_id is not None:
            _nonempty(self.dataset_id, "statistic dataset_id")
        if self.kind not in STATISTIC_KINDS:
            raise ValueError("statistic kind must be footprint, background, surface, or acf")
        _finite(self.observed, "observed statistic")
        _calibrated_statistic(self)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)


def _statistics(values: object) -> tuple[DiagnosticStatistic, ...]:
    statistics = tuple(values)
    if any(not isinstance(value, DiagnosticStatistic) for value in statistics):
        raise TypeError("statistics must contain DiagnosticStatistic values")
    statistics = tuple(replace(value) for value in statistics)
    keys = tuple((value.dataset_id, value.kind) for value in statistics)
    if len(set(keys)) != len(keys):
        raise ValueError("statistics must have unique dataset_id and kind pairs")
    return statistics


def _failure_index(index: object, attempted: int) -> None:
    if not isinstance(index, int) or isinstance(index, bool) or index < -1:
        raise ValueError("failed replicate index must be an integer at least -1")
    if index == -1:
        if attempted:
            raise ValueError("observed refit failure cannot follow null attempts")
    elif index >= attempted:
        raise ValueError("failed replicate index must be below attempted_count")


def _failure_reasons(values: object, attempted: int) -> tuple[tuple[int, str], ...]:
    failures = tuple(tuple(row) for row in values)
    indices = []
    for row in failures:
        if len(row) != 2:
            raise ValueError("failure reasons must contain index and reason pairs")
        index, reason = row
        _failure_index(index, attempted)
        _nonempty(reason, "failure reason")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("failed replicate indices must be unique")
    return failures


def _refit_discrepancy(values: object) -> tuple[float, float, float] | None:
    if values is None:
        return None
    discrepancy = tuple(values)
    if len(discrepancy) != 3:
        raise ValueError("refit_discrepancy must contain unit, objective, and mean distances")
    for value in discrepancy:
        _finite(value, "refit_discrepancy")
        if value < 0.0:
            raise ValueError("refit_discrepancy must be nonnegative")
    return discrepancy


def _calibration_identity(evidence: DiagnosticCalibration) -> None:
    if evidence.status not in {"available", "unavailable"}:
        raise ValueError("calibration status must be available or unavailable")
    if evidence.method != "poisson_refit_null_rms_v4":
        raise ValueError("unsupported diagnostic calibration method")
    if evidence.refit_policy != "declared_sobol4_lbfgsb_trf_v3":
        raise ValueError("unsupported diagnostic refit_policy")
    _finite(evidence.alpha, "diagnostic alpha")
    if evidence.alpha != 0.01:
        raise ValueError("diagnostic alpha must be 0.01 for poisson_refit_null_rms_v4")
    _sha256(evidence.owner_sha256, "owner_sha256")
    for name in ("null_statistics_sha256", "provenance_sha256"):
        value = getattr(evidence, name)
        if value is not None:
            _sha256(value, name)


def _calibration_counts(evidence: DiagnosticCalibration) -> None:
    for name in ("sample_count", "child_seed", "attempted_count", "successful_count", "refit_nfev"):
        _positive_integer(getattr(evidence, name), name, allow_zero=True)
    if not evidence.successful_count <= evidence.attempted_count <= evidence.sample_count:
        raise ValueError("calibration counts must satisfy successful <= attempted <= sample_count")


def _available_counts(evidence: DiagnosticCalibration) -> None:
    if evidence.sample_count < 99:
        raise ValueError("available diagnostic calibration requires at least 99 samples")
    if not evidence.sample_count == evidence.attempted_count == evidence.successful_count:
        raise ValueError("available diagnostic calibration requires all samples to succeed")
    if evidence.failure_reasons or evidence.unavailable_reason is not None:
        raise ValueError("available diagnostic calibration cannot contain failure evidence")
    _positive_integer(evidence.tail_count, "tail_count")
    _positive_integer(evidence.tie_count, "tie_count")
    if not evidence.tie_count <= evidence.tail_count <= evidence.sample_count + 1:
        raise ValueError("calibration must satisfy 1 <= tie_count <= tail_count <= B+1")


def _constant_family(evidence: DiagnosticCalibration) -> None:
    if any(value.scale != 0.0 for value in evidence.statistics):
        return
    total = evidence.sample_count + 1
    if evidence.observed_score != 0.0 or evidence.tail_count != total or evidence.tie_count != total:
        raise ValueError("constant diagnostic family requires zero score and complete inclusive ties")


def _available_statistics(evidence: DiagnosticCalibration) -> None:
    if not evidence.statistics:
        raise ValueError("available calibration requires nonempty statistics")
    probabilities = tuple(value.adjusted_p_value for value in evidence.statistics)
    if any(value is None for value in probabilities):
        raise ValueError("available calibration requires calibrated statistics")
    if min(probabilities) != evidence.p_value:
        raise ValueError("minimum adjusted statistic p-value must equal the family p-value")
    _finite(evidence.observed_score, "observed_score")
    if evidence.observed_score < 0.0:
        raise ValueError("observed_score must be nonnegative")
    _sha256(evidence.null_statistics_sha256, "null_statistics_sha256")
    _constant_family(evidence)


def _unavailable_statistics(evidence: DiagnosticCalibration) -> None:
    _nonempty(evidence.unavailable_reason, "unavailable calibration reason")
    if any(value is not None for value in (evidence.tail_count, evidence.tie_count, evidence.observed_score)):
        raise ValueError("unavailable calibration cannot publish tail, tie, or score evidence")
    if any(value.adjusted_p_value is not None for value in evidence.statistics):
        raise ValueError("unavailable calibration cannot publish calibrated statistics")


@dataclass(frozen=True, slots=True)
class DiagnosticCalibration:
    """A complete family test or explicit unavailable work, never a partial p."""

    status: str
    sample_count: int
    child_seed: int
    owner_sha256: str
    method: str = "poisson_refit_null_rms_v4"
    refit_policy: str = "declared_sobol4_lbfgsb_trf_v3"
    alpha: float = 0.01
    attempted_count: int = 0
    successful_count: int = 0
    statistics: tuple[DiagnosticStatistic, ...] = ()
    tail_count: int | None = None
    tie_count: int | None = None
    observed_score: float | None = None
    null_statistics_sha256: str | None = None
    failure_reasons: tuple[tuple[int, str], ...] = ()
    unavailable_reason: str | None = None
    refit_nfev: int = 0
    refit_discrepancy: tuple[float, float, float] | None = None
    provenance_sha256: str | None = None
    observed_work: DiagnosticRefitWork = DiagnosticRefitWork()
    null_work: DiagnosticRefitWork = DiagnosticRefitWork()

    def __post_init__(self) -> None:
        _calibration_identity(self)
        _calibration_counts(self)
        observed, null = calibration_work(self)
        object.__setattr__(self, "observed_work", observed)
        object.__setattr__(self, "null_work", null)
        statistics = _statistics(self.statistics)
        failures = _failure_reasons(self.failure_reasons, self.attempted_count)
        null_failures = sum(index >= 0 for index, _reason in failures)
        if self.successful_count + null_failures != self.attempted_count:
            raise ValueError("calibration attempts must account for successes and failures")
        object.__setattr__(self, "statistics", statistics)
        object.__setattr__(self, "failure_reasons", failures)
        object.__setattr__(self, "refit_discrepancy", _refit_discrepancy(self.refit_discrepancy))
        if self.status == "available":
            _available_counts(self)
            _available_statistics(self)
        else:
            _unavailable_statistics(self)

    @property
    def p_value(self) -> float | None:
        return None if self.status != "available" else self.tail_count / (self.sample_count + 1)

    @property
    def rejected(self) -> bool | None:
        return None if self.p_value is None else self.p_value <= self.alpha

    @property
    def resolution(self) -> float:
        """Rank-grid spacing, not an independent binomial error bar."""
        return 1.0 / (self.sample_count + 1)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)


def _refit_evaluation(evaluation: object) -> ModelEvaluation:
    if not isinstance(evaluation, ModelEvaluation):
        raise TypeError("refit evaluations must contain ModelEvaluation values")
    if not evaluation.valid or not isfinite(evaluation.objective):
        raise ValueError("successful refit requires valid finite evaluations")
    fitted = (evaluation.fit_residuals, evaluation.fit_weighted_residuals)
    if any(np.any(~np.isfinite(array)) for array in fitted):
        raise ValueError("successful refit requires finite fitted residuals")
    qz, mean = evaluation.qz_a_inv, evaluation.model_normalized
    # Full source axes intentionally retain NaN outside the model/fit mask.
    # These missing reporting rows are not failed fitted observations.
    if np.any(np.isinf(qz)) or np.any(np.isinf(mean)) or not np.array_equal(np.isnan(qz), np.isnan(mean)):
        raise ValueError("refit reporting axes require matching missing rows and no infinity")
    return replace(evaluation)


def _refit_unit_vector(values: object, work: DiagnosticRefitWork) -> np.ndarray:
    unit = _readonly(values, float, "refit unit_vector")
    if np.any(~np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)):
        raise ValueError("refit unit_vector must be finite and in [0, 1]")
    if (unit.size == 0) != (work.declared_paths == 1):
        raise ValueError("refit unit_vector dimension must match diagnostic work")
    return unit


@dataclass(frozen=True, slots=True)
class DiagnosticRefit:
    """Runtime-only numerical outcome of the declared bounded estimator."""

    unit_vector: np.ndarray | None
    evaluations: tuple[ModelEvaluation, ...]
    nfev: int
    attempted_paths: int
    failure_reason: str | None = None
    work: DiagnosticRefitWork = DiagnosticRefitWork()

    def __post_init__(self) -> None:
        _positive_integer(self.nfev, "refit nfev", allow_zero=True)
        _positive_integer(self.attempted_paths, "attempted_paths", allow_zero=True)
        if self.attempted_paths > 4:
            raise ValueError("diagnostic refit cannot attempt more than four paths")
        object.__setattr__(self, "work", refit_work(self))
        evaluations = tuple(self.evaluations)
        if self.failure_reason is not None:
            _nonempty(self.failure_reason, "refit failure_reason")
            if self.unit_vector is not None or evaluations:
                raise ValueError("failed refit cannot publish partial unit or evaluation results")
            object.__setattr__(self, "evaluations", evaluations)
            return
        if not self.attempted_paths or self.unit_vector is None or not evaluations:
            raise ValueError("successful refit requires a path, unit vector, and evaluations")
        object.__setattr__(self, "unit_vector", _refit_unit_vector(self.unit_vector, self.work))
        object.__setattr__(self, "evaluations", tuple(_refit_evaluation(value) for value in evaluations))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)
