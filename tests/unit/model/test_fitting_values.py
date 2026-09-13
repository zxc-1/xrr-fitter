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
from xrr_fitter.model.slab_stack import PeriodicSpan, SlabStack


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


def test_nested_expanded_stack_remains_read_only_after_pickle() -> None:
    stack = SlabStack(
        np.array([0.0, 20.0, 20.0, 0.0]),
        np.array([0.0j, 2e-5, 2e-5, 3e-5]),
        np.array([1.0, 1.0, 2.0]),
        (PeriodicSpan(1, 1, 2),),
    )
    candidate = replace(fit_candidate(), expanded_stack=stack)
    evaluation = ModelEvaluation(
        valid=True,
        reason="evaluated",
        parameters=candidate.parameters,
        qz_a_inv=candidate.qz_a_inv,
        model_normalized=candidate.model_normalized,
        fit_log_residuals_decades=candidate.log_residuals_decades,
        fit_weighted_residuals=candidate.weighted_residuals,
        objective=candidate.objective,
        expanded_stack=stack,
        diagnostics=(),
    )

    restored_candidate, restored_evaluation = pickle.loads(pickle.dumps((candidate, evaluation)))

    for restored in (restored_candidate.expanded_stack, restored_evaluation.expanded_stack):
        assert restored is not None
        assert restored.thickness_a.flags.writeable is False
        assert restored.sld_a2.flags.writeable is False
        assert restored.roughness_a.flags.writeable is False


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


@pytest.mark.parametrize(
    "field",
    ("instrument_fingerprint", "parameter_settings_fingerprint", "joint_layout_fingerprint"),
)
def test_checkpoint_rejects_invalid_optional_fingerprint(field: str) -> None:
    values = {
        "data_sha256": "a" * 64,
        "structure_fingerprint": "b" * 64,
        "config_fingerprint": "c" * 64,
        "stage": "stage-a",
        "candidates": (fit_candidate(),),
        "child_seeds": (101,),
        field: "not-a-sha256",
    }

    with pytest.raises(ValueError, match=field):
        FitCheckpoint(**values)


@pytest.mark.parametrize(
    "field",
    ("instrument_fingerprint", "parameter_settings_fingerprint", "joint_layout_fingerprint"),
)
@pytest.mark.parametrize("value", [0, False, None])
def test_checkpoint_rejects_falsy_non_string_optional_fingerprint(field: str, value: object) -> None:
    values = {
        "data_sha256": "a" * 64,
        "structure_fingerprint": "b" * 64,
        "config_fingerprint": "c" * 64,
        "stage": "stage-a",
        "candidates": (fit_candidate(),),
        "child_seeds": (101,),
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        FitCheckpoint(**values)


def test_fit_progress_solver_telemetry_defaults_to_absent() -> None:
    """求解器没发布的量必须是 ``None``，不能是 0。

    GUI 那四行指标（迭代次数/函数评估/接受率/步长）只能照抄 ``FitProgress``。若缺量默认
    成 0，界面会把"没有这个读数"显示成"读数是零"——这正是 ``metrics.py`` 拒绝渲染这四行
    的理由。默认 ``None`` 才让"不适用"可辨认。
    """
    progress = FitProgress("curve", "A", 2, 10, 1.5, "running")

    assert progress.iteration is None
    assert progress.nfev is None
    assert progress.acceptance_rate is None
    assert progress.step_size is None
    assert progress.dataset_objectives is None


def test_fit_progress_carries_solver_telemetry_through_pickle() -> None:
    progress = FitProgress(
        None,
        "MCMC",
        3,
        8,
        2.5,
        "MCMC sampling",
        iteration=3,
        nfev=41,
        acceptance_rate=0.25,
        step_size=0.031,
        dataset_objectives=(("a", 1.5), ("b", 3.5)),
    )

    restored = pickle.loads(pickle.dumps(progress))

    assert restored == progress
    assert restored.dataset_objectives == (("a", 1.5), ("b", 3.5))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("iteration", -1, "iteration"),
        ("iteration", True, "iteration"),
        ("nfev", -1, "nfev"),
        ("nfev", 1.5, "nfev"),
        ("acceptance_rate", -0.1, "acceptance_rate"),
        ("acceptance_rate", 1.1, "acceptance_rate"),
        ("acceptance_rate", float("nan"), "acceptance_rate"),
        ("step_size", -1e-9, "step_size"),
        ("step_size", float("nan"), "step_size"),
        ("step_size", float("inf"), "step_size"),
    ),
)
def test_fit_progress_rejects_impossible_solver_telemetry(field: str, value: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FitProgress("curve", "A", 2, 10, 1.5, "running", **{field: value})


@pytest.mark.parametrize(
    "value",
    (
        (("a", 1.0), ("", 2.0)),
        (("a", float("nan")),),
        ((("a", 1.0), ("a", 2.0))),
        (("a",),),
    ),
)
def test_fit_progress_rejects_malformed_dataset_objectives(value: object) -> None:
    with pytest.raises(ValueError, match="dataset_objectives"):
        FitProgress(None, "joint A", 2, 10, 1.5, "running", dataset_objectives=value)


def test_fit_progress_owns_its_dataset_objectives_as_a_tuple() -> None:
    """联合面板按这份名单画表，所以它不能是调用方还握着的那个 list。"""
    pairs = [("a", 1.0), ("b", 2.0)]

    progress = FitProgress(None, "joint A", 2, 10, 1.5, "running", dataset_objectives=pairs)
    pairs.append(("c", 3.0))

    assert progress.dataset_objectives == (("a", 1.0), ("b", 2.0))


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
