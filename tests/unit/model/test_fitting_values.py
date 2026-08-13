from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, fields, replace
from importlib import import_module

import numpy as np
import pytest
from tests.support.model_cases import fit_candidate, fit_result

from xrr_fitter.model.fitting import (
    ConfidenceThresholds,
    FitCheckpoint,
    FitConfig,
    FitProgress,
    FitSearchResult,
    FitStageSummary,
    ModelEvaluation,
    SearchBudget,
)


def test_fit_config_standard_is_versioned_finite_and_immutable() -> None:
    config = FitConfig.standard(1201)

    assert config.objective_name == "robust_log_soft_l1"
    assert config.jacobian_version == "analytic-v1"
    assert config.budget == SearchBudget(60, 200, 2000, 300, 100)
    with pytest.raises(FrozenInstanceError):
        config.master_seed = 3
    with pytest.raises(ValueError, match="master_seed"):
        FitConfig.standard(True)


def test_fit_candidate_copies_and_freezes_every_array() -> None:
    candidate = fit_candidate()

    for field in (
        "unit_vector",
        "qz_a_inv",
        "model_normalized",
        "log_residuals_decades",
        "weighted_residuals",
        "sld_depth_a",
        "sld_profile_a2",
    ):
        value = getattr(candidate, field)
        assert value.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            value[0] = value[0]


def test_fit_candidate_preserves_complex_sld_profile() -> None:
    source = np.array([0.0j, 2e-5 + 3e-7j])
    candidate = replace(fit_candidate(), sld_profile_a2=source)

    source[1] = 0.0j

    assert candidate.sld_profile_a2[1] == 2e-5 + 3e-7j
    assert candidate.sld_profile_a2.flags.writeable is False


def test_model_evaluation_copies_reporting_arrays() -> None:
    source = np.linspace(0.01, 0.2, 4)
    evaluation = ModelEvaluation(
        valid=True,
        reason="evaluated",
        parameters=fit_candidate().parameters,
        qz_a_inv=source,
        model_normalized=np.ones(4),
        fit_log_residuals_decades=np.zeros(3),
        fit_weighted_residuals=np.zeros(3),
        objective=1.0,
        expanded_stack=None,
        diagnostics=(),
    )

    source[0] = 99.0

    assert evaluation.qz_a_inv[0] == 0.01
    assert evaluation.qz_a_inv.flags.writeable is False


def test_published_fitting_arrays_remain_read_only_after_pickle() -> None:
    candidate = fit_candidate()
    evaluation = ModelEvaluation(
        valid=True,
        reason="evaluated",
        parameters=candidate.parameters,
        qz_a_inv=candidate.qz_a_inv,
        model_normalized=candidate.model_normalized,
        fit_log_residuals_decades=candidate.log_residuals_decades,
        fit_weighted_residuals=candidate.weighted_residuals,
        objective=candidate.objective,
        expanded_stack=None,
        diagnostics=(),
    )
    search = fit_result(candidate)

    restored_candidate, restored_evaluation, restored_search = pickle.loads(
        pickle.dumps((candidate, evaluation, search))
    )

    candidate_fields = (
        "unit_vector",
        "qz_a_inv",
        "model_normalized",
        "log_residuals_decades",
        "weighted_residuals",
        "sld_depth_a",
        "sld_profile_a2",
    )
    evaluation_fields = (
        "qz_a_inv",
        "model_normalized",
        "fit_log_residuals_decades",
        "fit_weighted_residuals",
    )
    assert all(not getattr(restored_candidate, field).flags.writeable for field in candidate_fields)
    assert all(not getattr(restored_evaluation, field).flags.writeable for field in evaluation_fields)
    assert restored_search.region_labels.flags.writeable is False
    assert restored_search.region_weights.flags.writeable is False
    assert restored_search.candidates[0].unit_vector.flags.writeable is False


def test_search_budget_allows_disabled_de_stages() -> None:
    assert SearchBudget(0, 0, 2000, 300, 100).short_de_maxiter == 0


def test_fit_result_validates_candidate_identity_and_best_index() -> None:
    first = fit_candidate("first", 1.0)
    second = fit_candidate("second", 2.0)
    result = fit_result(first, second)

    assert result.best_candidate is first
    with pytest.raises(ValueError, match="candidate_id"):
        fit_result(first, fit_candidate("first", 2.0))
    values = {field: getattr(result, field) for field in result.__dataclass_fields__}
    values["best_index"] = 3
    with pytest.raises(ValueError, match="best_index"):
        FitSearchResult(**values)


def test_result_type_names_preserve_public_final_result_schema() -> None:
    fitting = import_module("xrr_fitter.model.fitting")
    analysis = import_module("xrr_fitter.model.analysis")

    assert hasattr(fitting, "FitSearchResult")
    assert not hasattr(fitting, "FitResult")
    assert hasattr(analysis, "FitResult")
    assert not hasattr(analysis, "AnalysisResult")
    assert [field.name for field in fields(analysis.FitResult)] == [
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
        "classification_evidence",
    ]


@pytest.mark.parametrize(
    "candidate",
    [
        replace(fit_candidate(), valid=False),
        replace(fit_candidate(), stop_reason="early_eliminated"),
    ],
)
def test_fit_result_rejects_unselectable_best_candidate(candidate: object) -> None:
    with pytest.raises(ValueError, match="selectable candidate"):
        fit_result(candidate)


def test_fit_result_requires_minimum_best_candidate_and_complete_selection() -> None:
    first = fit_candidate("first", 1.0)
    second = fit_candidate("second", 2.0)
    result = fit_result(first, second)

    with pytest.raises(ValueError, match="minimum-objective"):
        replace(result, best_index=1)
    with pytest.raises(ValueError, match="required for selectable"):
        replace(result, best_index=None)


def test_fit_result_rejects_missing_stage_candidate_reference() -> None:
    result = fit_result()
    summary = FitStageSummary("E", ("missing",), 1.0, 1, ("converged",))

    with pytest.raises(ValueError, match="stage references missing candidate"):
        replace(result, stage_summaries=(summary,))


def test_fit_candidate_accepts_archived_seed_sentinel() -> None:
    assert replace(fit_candidate(), seed_index=-1).seed_index == -1


@pytest.mark.parametrize("objective", [float("nan"), float("-inf")])
def test_invalid_fit_candidate_rejects_unsupported_objective_sentinel(objective: float) -> None:
    with pytest.raises(ValueError, match="invalid candidate objective"):
        replace(fit_candidate(), valid=False, objective=objective)


def test_invalid_fit_candidate_accepts_positive_infinity_sentinel() -> None:
    candidate = replace(fit_candidate(), valid=False, objective=float("inf"))

    assert candidate.objective == float("inf")


def test_stage_summary_rejects_negative_infinity() -> None:
    with pytest.raises(ValueError, match="best_objective"):
        FitStageSummary("E", (), float("-inf"), 0, ())


def test_fit_progress_stage_summary_and_checkpoint_validate_schema() -> None:
    progress = FitProgress("curve", "stage-a", 2, 10, 1.5, "running")
    summary = FitStageSummary("stage-a", ("candidate-0",), 1.0, 12, ("converged",))
    checkpoint = FitCheckpoint(
        data_sha256="a" * 64,
        structure_fingerprint="b" * 64,
        config_fingerprint="c" * 64,
        stage="stage-a",
        candidates=(fit_candidate(),),
        child_seeds=(101,),
        stage_summaries=(summary,),
    )

    assert progress.completed == 2
    assert checkpoint.stage_summaries == (summary,)
    with pytest.raises(ValueError, match="completed"):
        FitProgress("curve", "stage-a", 11, 10, 1.5, "bad")


def test_confidence_thresholds_default_prior_conflict_sigmas_is_three() -> None:
    assert ConfidenceThresholds().prior_conflict_sigmas == 3.0


def test_confidence_thresholds_rejects_nonpositive_sigmas() -> None:
    with pytest.raises(ValueError, match="prior_conflict_sigmas"):
        ConfidenceThresholds(prior_conflict_sigmas=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        ConfidenceThresholds(prior_conflict_sigmas=-1.0)
    # A zero boundary_fraction remains a legal, differently-meaning threshold.
    assert ConfidenceThresholds(boundary_fraction=0.0).boundary_fraction == 0.0


def test_confidence_thresholds_rejects_nonfinite_sigmas() -> None:
    with pytest.raises(ValueError, match="finite"):
        ConfidenceThresholds(prior_conflict_sigmas=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        ConfidenceThresholds(prior_conflict_sigmas=float("inf"))
