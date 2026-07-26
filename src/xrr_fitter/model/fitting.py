"""Immutable fitting configuration, evaluation, candidate, and result values.

This module owns search-only state. Confidence and uncertainty remain in the
higher-level analysis model so fitting never imports analysis. Every published
array is copied and made read-only at construction; callers may therefore pass
temporary optimizer buffers without leaking later mutation into checkpoints or
project results. Configuration records preserve the named algorithm versions
and deterministic seed lineage needed to replay a search.

The values here deliberately do not run optimizers or evaluate reflectivity.
They define the stable handoff between compilation, search, persistence, and
analysis while keeping those higher layers independent of one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.parameters import ParameterDefinition, ParameterValue
from xrr_fitter.model.structure import SlabStack


def _positive_integer(value: int, field: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field} must be a {qualifier} integer")


def _nonempty(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")


def _readonly(value: object, dtype: type, field: str, *, ndim: int = 1) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


def _sha256(value: str, field: str) -> None:
    valid = len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    if not valid:
        raise ValueError(f"{field} must be a lowercase SHA-256")


def _checkpoint_candidates(values: object) -> tuple[FitCandidate, ...]:
    candidates = tuple(values)
    if any(not isinstance(value, FitCandidate) for value in candidates):
        raise TypeError("checkpoint candidates must be FitCandidate values")
    return candidates


def _checkpoint_seeds(values: object) -> tuple[int, ...]:
    seeds = tuple(values)
    invalid = any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in seeds)
    if invalid:
        raise ValueError("checkpoint child_seeds must be nonnegative integers")
    return seeds


def _validate_candidate_objective(candidate: FitCandidate) -> None:
    # Invalid evaluations use positive infinity as the only nonfinite sentinel.
    # Keeping NaN and negative infinity out makes persisted ordering decidable.
    if candidate.valid and not isfinite(candidate.objective):
        raise ValueError("valid candidate objective must be finite")
    if not candidate.valid and not (
        isfinite(candidate.objective) or candidate.objective == float("inf")
    ):
        raise ValueError("invalid candidate objective must be finite or positive infinity")
    ranking = candidate.ranking_objective
    if ranking is not None and not isfinite(ranking):
        raise ValueError("ranking_objective must be finite or None")


def _archived_seed(value: int) -> None:
    # Stage-B archives deliberately replace their original seed with -1.
    valid = isinstance(value, int) and not isinstance(value, bool) and value >= -1
    if not valid:
        raise ValueError("seed_index must be an integer greater than or equal to -1")


def _candidate_members(
    parameters: object,
    diagnostics: object,
) -> tuple[tuple[ParameterValue, ...], tuple[PhysicsDiagnostic, ...]]:
    parameter_values = tuple(parameters)
    diagnostic_values = tuple(diagnostics)
    if any(not isinstance(value, ParameterValue) for value in parameter_values):
        raise TypeError("candidate parameters must be ParameterValue values")
    if any(not isinstance(value, PhysicsDiagnostic) for value in diagnostic_values):
        raise TypeError("candidate diagnostics must be PhysicsDiagnostic values")
    return parameter_values, diagnostic_values


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Versioned work limits for global, local, and bootstrap stages."""

    short_de_maxiter: int
    full_de_maxiter: int
    local_min_nfev: int
    local_nfev_per_parameter: int
    bootstrap_samples: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            allow_zero = field in {"short_de_maxiter", "full_de_maxiter"}
            _positive_integer(getattr(self, field), field, allow_zero=allow_zero)


@dataclass(frozen=True, slots=True)
class ConfidenceThresholds:
    """Distance, cost, boundary, and correlation classification thresholds."""

    cluster_join_distance: float = 0.05
    distinct_cluster_distance: float = 0.10
    equivalent_cost_fraction: float = 0.02
    equivalent_cost_floor: float = 1e-5
    boundary_fraction: float = 0.005
    strong_correlation: float = 0.95

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("confidence thresholds must be finite and nonnegative")
        if not 0.0 <= self.strong_correlation <= 1.0:
            raise ValueError("strong_correlation must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Complete deterministic fitting configuration persisted by projects."""

    master_seed: int
    objective_name: str
    objective_version: str
    c_decades: float
    final_seed_count: int
    budget: SearchBudget
    local_workers: int = 4
    scale_prior_enabled: bool = True
    scale_prior_tau_decades: float = 0.1
    confidence: ConfidenceThresholds = ConfidenceThresholds()
    fringe_screen_threshold_version: str = "fringe-count-v1"
    budget_reclaim_threshold_version: str = "stage-b-reclaim-v1"
    downsample_rule_version: str = "feature-grid-v1"
    jacobian_version: str = "analytic-v1"

    def __post_init__(self) -> None:
        _positive_integer(self.master_seed, "master_seed", allow_zero=True)
        _positive_integer(self.final_seed_count, "final_seed_count")
        _positive_integer(self.local_workers, "local_workers")
        for field in (
            "objective_name",
            "objective_version",
            "fringe_screen_threshold_version",
            "budget_reclaim_threshold_version",
            "downsample_rule_version",
            "jacobian_version",
        ):
            _nonempty(getattr(self, field), field)
        if not isfinite(self.c_decades) or self.c_decades <= 0.0:
            raise ValueError("c_decades must be positive and finite")
        if not isinstance(self.scale_prior_enabled, bool):
            raise TypeError("scale_prior_enabled must be bool")
        if not isfinite(self.scale_prior_tau_decades) or self.scale_prior_tau_decades <= 0.0:
            raise ValueError("scale_prior_tau_decades must be positive and finite")
        if not isinstance(self.budget, SearchBudget):
            raise TypeError("budget must be a SearchBudget")
        if not isinstance(self.confidence, ConfidenceThresholds):
            raise TypeError("confidence must be ConfidenceThresholds")

    @classmethod
    def standard(cls, master_seed: int) -> FitConfig:
        return cls(
            master_seed=master_seed,
            objective_name="robust_log_soft_l1",
            objective_version="1",
            c_decades=0.05,
            final_seed_count=4,
            budget=SearchBudget(60, 200, 2000, 300, 100),
        )

    @classmethod
    def fast(cls, master_seed: int) -> FitConfig:
        return cls(
            master_seed=master_seed,
            objective_name="robust_log_soft_l1",
            objective_version="1",
            c_decades=0.05,
            final_seed_count=4,
            budget=SearchBudget(4, 8, 200, 30, 8),
        )


@dataclass(frozen=True, slots=True)
class FitProgress:
    """Serializable monotonic progress for one dataset and fitting stage."""

    dataset_id: str | None
    stage: str
    completed: int
    total: int
    best_objective: float
    message: str

    def __post_init__(self) -> None:
        if self.dataset_id is not None:
            _nonempty(self.dataset_id, "dataset_id")
        _nonempty(self.stage, "stage")
        _positive_integer(self.completed, "completed", allow_zero=True)
        _positive_integer(self.total, "total", allow_zero=True)
        if self.completed > self.total:
            raise ValueError("completed must not exceed total")
        if np.isnan(self.best_objective):
            raise ValueError("best_objective must not be NaN")


@dataclass(frozen=True, slots=True)
class FitCandidate:
    """Published candidate with physical values and owned reporting arrays."""

    candidate_id: str
    seed_index: int
    unit_vector: np.ndarray
    parameters: tuple[ParameterValue, ...]
    objective: float
    valid: bool
    stop_reason: str
    nfev: int
    qz_a_inv: np.ndarray
    model_normalized: np.ndarray
    log_residuals_decades: np.ndarray
    weighted_residuals: np.ndarray
    expanded_stack: SlabStack | None
    sld_depth_a: np.ndarray
    sld_profile_a2: np.ndarray
    diagnostics: tuple[PhysicsDiagnostic, ...]
    ranking_objective: float | None = None

    def __post_init__(self) -> None:
        _nonempty(self.candidate_id, "candidate_id")
        _archived_seed(self.seed_index)
        _positive_integer(self.nfev, "nfev", allow_zero=True)
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        _nonempty(self.stop_reason, "stop_reason")
        _validate_candidate_objective(self)
        parameters, diagnostics = _candidate_members(self.parameters, self.diagnostics)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "diagnostics", diagnostics)
        self._freeze_arrays()

    def _freeze_arrays(self) -> None:
        arrays = {
            "unit_vector": _readonly(self.unit_vector, float, "unit_vector"),
            "qz_a_inv": _readonly(self.qz_a_inv, float, "qz_a_inv"),
            "model_normalized": _readonly(self.model_normalized, float, "model_normalized"),
            "log_residuals_decades": _readonly(
                self.log_residuals_decades,
                float,
                "log_residuals_decades",
            ),
            "weighted_residuals": _readonly(self.weighted_residuals, float, "weighted_residuals"),
            "sld_depth_a": _readonly(self.sld_depth_a, float, "sld_depth_a"),
            "sld_profile_a2": _readonly(self.sld_profile_a2, complex, "sld_profile_a2"),
        }
        q_size = arrays["qz_a_inv"].size
        q_fields = ("model_normalized", "log_residuals_decades", "weighted_residuals")
        if any(arrays[field].size != q_size for field in q_fields):
            raise ValueError("candidate q-grid arrays must have equal length")
        if arrays["sld_depth_a"].shape != arrays["sld_profile_a2"].shape:
            raise ValueError("candidate SLD arrays must have equal length")
        for field, value in arrays.items():
            object.__setattr__(self, field, value)


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Internal model evaluation before candidate identity is assigned."""

    valid: bool
    reason: str
    parameters: tuple[ParameterValue, ...]
    qz_a_inv: np.ndarray
    model_normalized: np.ndarray
    fit_log_residuals_decades: np.ndarray
    fit_weighted_residuals: np.ndarray
    objective: float
    expanded_stack: SlabStack | None
    diagnostics: tuple[PhysicsDiagnostic, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        _nonempty(self.reason, "reason")
        if self.valid and not isfinite(self.objective):
            raise ValueError("valid evaluation objective must be finite")
        parameters = tuple(self.parameters)
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(value, ParameterValue) for value in parameters):
            raise TypeError("evaluation parameters must be ParameterValue values")
        if any(not isinstance(value, PhysicsDiagnostic) for value in diagnostics):
            raise TypeError("evaluation diagnostics must be PhysicsDiagnostic values")
        arrays = self._freeze_arrays()
        if arrays[0].shape != arrays[1].shape or arrays[2].shape != arrays[3].shape:
            raise ValueError("evaluation array axes are inconsistent")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "diagnostics", diagnostics)

    def _freeze_arrays(self) -> tuple[np.ndarray, ...]:
        fields = (
            ("qz_a_inv", self.qz_a_inv),
            ("model_normalized", self.model_normalized),
            ("fit_log_residuals_decades", self.fit_log_residuals_decades),
            ("fit_weighted_residuals", self.fit_weighted_residuals),
        )
        arrays = tuple(_readonly(value, float, name) for name, value in fields)
        for (name, _value), array in zip(fields, arrays, strict=True):
            object.__setattr__(self, name, array)
        return arrays


@dataclass(frozen=True, slots=True)
class FitStageSummary:
    """Stable candidate IDs, objective, work, and stop reasons for one stage."""

    stage: str
    candidate_ids: tuple[str, ...]
    best_objective: float
    total_nfev: int
    stop_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.stage, "stage")
        candidates = tuple(self.candidate_ids)
        reasons = tuple(self.stop_reasons)
        if any(not value.strip() for value in candidates):
            raise ValueError("stage candidate_ids must not contain empty values")
        if len(candidates) != len(set(candidates)):
            raise ValueError("stage candidate_ids must be unique")
        # A stage with no selectable candidate records +inf, later encoded as null.
        # Negative infinity cannot describe either a completed or empty stage.
        valid_objective = isfinite(self.best_objective) or self.best_objective == float("inf")
        if not valid_objective:
            raise ValueError("stage best_objective must be finite or positive infinity")
        _positive_integer(self.total_nfev, "total_nfev", allow_zero=True)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "stop_reasons", reasons)


@dataclass(frozen=True, slots=True)
class FitCheckpoint:
    """Resume state bound to exact data, structure, config, and seed lineage."""

    data_sha256: str
    structure_fingerprint: str
    config_fingerprint: str
    stage: str
    candidates: tuple[FitCandidate, ...]
    child_seeds: tuple[int, ...]
    instrument_fingerprint: str = ""
    parameter_settings_fingerprint: str = ""
    runtime_warnings: tuple[str, ...] = ()
    stage_summaries: tuple[FitStageSummary, ...] = ()
    joint_layout_fingerprint: str = ""

    def __post_init__(self) -> None:
        for field in ("data_sha256", "structure_fingerprint", "config_fingerprint"):
            _sha256(getattr(self, field), field)
        _nonempty(self.stage, "stage")
        candidates = _checkpoint_candidates(self.candidates)
        seeds = _checkpoint_seeds(self.child_seeds)
        summaries = tuple(self.stage_summaries)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "child_seeds", seeds)
        object.__setattr__(self, "runtime_warnings", tuple(self.runtime_warnings))
        object.__setattr__(self, "stage_summaries", summaries)


def _ranking_objective(candidate: FitCandidate) -> float:
    return candidate.objective if candidate.ranking_objective is None else candidate.ranking_objective


def _final_candidate_ids(summaries: tuple[FitStageSummary, ...]) -> frozenset[str] | None:
    # Earlier candidates remain diagnostic evidence, but the last Stage E scope
    # is the only scope allowed to select the published winner.
    for summary in reversed(summaries):
        if summary.stage in {"E", "stage-e"}:
            return frozenset(summary.candidate_ids) if summary.candidate_ids else None
    return None


def _selectable_indices(
    candidates: tuple[FitCandidate, ...],
    summaries: tuple[FitStageSummary, ...],
) -> tuple[int, ...]:
    # Invalid and explicitly eliminated rows stay addressable while being
    # excluded from winner selection and minimum-objective comparison.
    final_ids = _final_candidate_ids(summaries)
    return tuple(
        index
        for index, candidate in enumerate(candidates)
        if (final_ids is None or candidate.candidate_id in final_ids)
        and candidate.valid
        and candidate.stop_reason != "early_eliminated"
        and isfinite(_ranking_objective(candidate))
    )


def _validate_stage_references(
    candidates: tuple[FitCandidate, ...],
    summaries: tuple[FitStageSummary, ...],
) -> None:
    # Stage A stores feature keys rather than published candidate IDs. Every
    # later stage must resolve its graph edges against the retained candidates.
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    for summary in summaries:
        if summary.stage in {"A", "stage-a"}:
            continue
        missing = next((value for value in summary.candidate_ids if value not in candidate_ids), None)
        if missing is not None:
            raise ValueError(f"stage references missing candidate: {missing}")


def _validate_best_index(
    candidates: tuple[FitCandidate, ...],
    summaries: tuple[FitStageSummary, ...],
    best_index: int | None,
) -> None:
    # None is meaningful only for an entirely unselectable graph. Otherwise the
    # persisted index must choose a selectable global minimum; exact ties remain
    # valid because the stored index is the deterministic tie breaker.
    selectable = _selectable_indices(candidates, summaries)
    if best_index is None:
        if selectable:
            raise ValueError("best_index is required for selectable candidates")
        return
    valid_index = (
        isinstance(best_index, int)
        and not isinstance(best_index, bool)
        and 0 <= best_index < len(candidates)
    )
    if not valid_index:
        raise ValueError("best_index must identify a candidate")
    if best_index not in selectable:
        raise ValueError("best_index must identify a selectable candidate")
    minimum = min(_ranking_objective(candidates[index]) for index in selectable)
    if _ranking_objective(candidates[best_index]) != minimum:
        raise ValueError("best_index must identify a minimum-objective candidate")


@dataclass(frozen=True, slots=True)
class FitSearchResult:
    """Fitting-only candidate graph, best index, stage history, and region mass."""

    parameter_definitions: tuple[ParameterDefinition, ...]
    candidates: tuple[FitCandidate, ...]
    best_index: int | None
    warnings: tuple[str, ...]
    child_seeds: tuple[int, ...]
    stage_summaries: tuple[FitStageSummary, ...]
    region_labels: np.ndarray
    region_weights: np.ndarray

    def __post_init__(self) -> None:
        # Normalize every owned sequence before validating cross-record edges.
        definitions = tuple(self.parameter_definitions)
        candidates = tuple(self.candidates)
        seeds = _checkpoint_seeds(self.child_seeds)
        summaries = tuple(self.stage_summaries)
        if any(not isinstance(value, FitCandidate) for value in candidates):
            raise TypeError("candidates must contain FitCandidate values")
        if any(not isinstance(value, FitStageSummary) for value in summaries):
            raise TypeError("stage_summaries must contain FitStageSummary values")
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id values must be unique")
        _validate_stage_references(candidates, summaries)
        _validate_best_index(candidates, summaries, self.best_index)
        labels = _readonly(self.region_labels, int, "region_labels")
        weights = _readonly(self.region_weights, float, "region_weights")
        if labels.shape != weights.shape:
            raise ValueError("region arrays must have equal length")
        object.__setattr__(self, "parameter_definitions", definitions)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "child_seeds", seeds)
        object.__setattr__(self, "stage_summaries", summaries)
        object.__setattr__(self, "region_labels", labels)
        object.__setattr__(self, "region_weights", weights)

    @property
    def best_candidate(self) -> FitCandidate | None:
        return None if self.best_index is None else self.candidates[self.best_index]
