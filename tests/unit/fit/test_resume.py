from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitStageSummary, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec


def _checkpoint_api():
    return import_module("xrr_fitter.fit.checkpoint")


def _pipeline_api():
    return import_module("xrr_fitter.fit.pipeline")


def _resume_api():
    return import_module("xrr_fitter.fit.resume")


def _problem(*, seed: int = 769):
    config = replace(
        FitConfig.fast(seed),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    return compile_fit_problem(
        prepared_data(size=40),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )


def _candidate(problem, candidate_id: str, seed_index: int, unit_value: float):
    unit = np.full(len(problem.variables), unit_value)
    evaluation = evaluate_vector(problem, unit)
    return candidate_from_evaluation(
        problem,
        unit,
        evaluation,
        candidate_id,
        seed_index,
        "converged",
        4,
    )


def _stage_b_checkpoint(problem):
    first = _candidate(problem, "B-0", 0, 0.45)
    second = _candidate(problem, "B-1", 1, 0.55)
    stage_a = FitStageSummary(
        "A",
        ("declared-baseline",),
        min(first.objective, second.objective),
        1,
        ("evaluated",),
    )
    stage_b = FitStageSummary(
        "B",
        (first.candidate_id, second.candidate_id),
        min(first.objective, second.objective),
        first.nfev + second.nfev,
        (first.stop_reason, second.stop_reason),
    )
    return _checkpoint_api().build_checkpoint(
        problem,
        stage="B",
        candidates=(first, second),
        child_seeds=(101, 102),
        runtime_warnings=(),
        stage_summaries=(stage_a, stage_b),
    )


def _assert_equivalent_results(fresh, resumed) -> None:
    assert fresh.best_index == resumed.best_index
    assert fresh.warnings == resumed.warnings
    assert fresh.child_seeds == resumed.child_seeds
    assert fresh.stage_summaries == resumed.stage_summaries
    assert tuple(candidate.candidate_id for candidate in fresh.candidates) == tuple(
        candidate.candidate_id for candidate in resumed.candidates
    )
    for left, right in zip(fresh.candidates, resumed.candidates, strict=True):
        assert left.objective == right.objective
        assert left.ranking_objective == right.ranking_objective
        np.testing.assert_array_equal(left.unit_vector, right.unit_vector)
        np.testing.assert_array_equal(left.model_normalized, right.model_normalized)


def test_resume_plan_accepts_an_exact_seed_prefix_and_starts_after_checkpoint() -> None:
    api = _resume_api()
    problem = _problem()
    checkpoint = _stage_b_checkpoint(problem)

    plan = api.validate_resume_checkpoint(
        problem,
        checkpoint,
        reserved_child_seeds=(101, 102, 201, 202),
    )

    assert plan.completed_stage == "B"
    assert plan.remaining_stages == ("C", "D", "E")
    assert plan.consumed_child_seeds == (101, 102)
    assert tuple(candidate.candidate_id for candidate in plan.candidates) == ("B-0", "B-1")


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda value: replace(value, data_sha256="0" * 64), id="data-fingerprint"),
        pytest.param(
            lambda value: replace(value, structure_fingerprint="0" * 64),
            id="structure-fingerprint",
        ),
        pytest.param(
            lambda value: replace(value, instrument_fingerprint="0" * 64),
            id="instrument-fingerprint",
        ),
        pytest.param(
            lambda value: replace(value, config_fingerprint="0" * 64),
            id="config-fingerprint",
        ),
        pytest.param(
            lambda value: replace(value, parameter_settings_fingerprint="0" * 64),
            id="parameter-fingerprint",
        ),
        pytest.param(lambda value: replace(value, stage="A"), id="unsupported-stage"),
        pytest.param(
            lambda value: replace(value, child_seeds=tuple(reversed(value.child_seeds))),
            id="consumed-seed-order",
        ),
        pytest.param(
            lambda value: replace(value, child_seeds=()),
            id="consumed-seed-count",
        ),
        pytest.param(
            lambda value: replace(value, candidates=tuple(reversed(value.candidates))),
            id="candidate-order",
        ),
        pytest.param(
            lambda value: replace(value, stage_summaries=tuple(reversed(value.stage_summaries))),
            id="stage-summary-order",
        ),
        pytest.param(
            lambda value: replace(value, joint_layout_fingerprint="0" * 64),
            id="joint-layout-on-single-fit",
        ),
        pytest.param(
            lambda value: replace(
                value,
                candidates=(
                    replace(value.candidates[0], unit_vector=np.asarray([0.5])),
                    value.candidates[1],
                ),
            ),
            id="candidate-unit-width",
        ),
        pytest.param(
            lambda value: replace(
                value,
                candidates=(
                    replace(value.candidates[0], qz_a_inv=value.candidates[0].qz_a_inv + 1e-6),
                    value.candidates[1],
                ),
            ),
            id="candidate-q-grid",
        ),
    ],
)
def test_resume_rejects_fingerprint_order_and_seed_mismatch(mutation) -> None:
    api = _resume_api()
    problem = _problem(seed=773)
    checkpoint = mutation(_stage_b_checkpoint(problem))

    with pytest.raises(
        ValueError,
        match="resume|checkpoint|fingerprint|stage|candidate|seed|q|joint|unit|history",
    ):
        api.validate_resume_checkpoint(
            problem,
            checkpoint,
            reserved_child_seeds=(101, 102, 201, 202),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda value: replace(
                value,
                candidates=(
                    replace(value.candidates[0], objective=value.candidates[0].objective + 1.0),
                    value.candidates[1],
                ),
            ),
            id="objective",
        ),
        pytest.param(
            lambda value: replace(
                value,
                candidates=(replace(value.candidates[0], valid=False), value.candidates[1]),
            ),
            id="validity",
        ),
        pytest.param(
            lambda value: replace(
                value,
                candidates=(replace(value.candidates[0], parameters=()), value.candidates[1]),
            ),
            id="parameters",
        ),
        pytest.param(
            lambda value: replace(
                value,
                candidates=(
                    replace(
                        value.candidates[0],
                        model_normalized=value.candidates[0].model_normalized + 1e-6,
                    ),
                    value.candidates[1],
                ),
            ),
            id="model",
        ),
        pytest.param(
            lambda value: replace(
                value,
                candidates=(
                    replace(
                        value.candidates[0],
                        weighted_residuals=value.candidates[0].weighted_residuals + 1e-6,
                    ),
                    value.candidates[1],
                ),
            ),
            id="residual",
        ),
        pytest.param(
            lambda value: replace(
                value,
                candidates=(
                    replace(value.candidates[0], ranking_objective=value.candidates[0].objective),
                    value.candidates[1],
                ),
            ),
            id="single-ranking",
        ),
        pytest.param(
            lambda value: replace(
                value,
                stage_summaries=(
                    value.stage_summaries[0],
                    replace(
                        value.stage_summaries[1],
                        best_objective=value.stage_summaries[1].best_objective + 1.0,
                    ),
                ),
            ),
            id="summary",
        ),
    ],
)
def test_resume_rejects_recomputed_candidate_and_summary_drift(mutation) -> None:
    api = _resume_api()
    problem = _problem(seed=779)

    with pytest.raises(ValueError, match="resume|checkpoint|candidate|summary|ranking"):
        api.validate_resume_checkpoint(
            problem,
            mutation(_stage_b_checkpoint(problem)),
            reserved_child_seeds=(101, 102, 201, 202),
        )


def test_resume_mismatch_is_rejected_before_any_callback() -> None:
    api = _pipeline_api()
    problem = _problem(seed=787)
    checkpoint = replace(_stage_b_checkpoint(problem), config_fingerprint="0" * 64)
    progress: list[object] = []
    checkpoints: list[object] = []

    with pytest.raises(ValueError, match="resume|checkpoint|fingerprint|config"):
        api.run_fit_search(
            api.FitSearchRequest("curve", problem, checkpoint),
            progress=progress.append,
            checkpoint=checkpoints.append,
        )

    assert progress == []
    assert checkpoints == []


@pytest.mark.parametrize(
    ("completed_stage", "expected_suffix"),
    [
        pytest.param("B", ("C", "D", "E"), id="from-b"),
        pytest.param("C", ("D", "E"), id="from-c"),
        pytest.param("D", ("E",), id="from-d"),
        pytest.param("E", (), id="from-e"),
    ],
)
def test_resume_runs_only_the_remaining_suffix_and_matches_fresh_search(
    completed_stage: str,
    expected_suffix: tuple[str, ...],
) -> None:
    api = _pipeline_api()
    problem = _problem(seed=797)
    fresh_checkpoints: list[object] = []
    fresh = api.run_fit_search(
        api.FitSearchRequest("curve", problem),
        checkpoint=fresh_checkpoints.append,
    )
    resume_from = next(value for value in fresh_checkpoints if value.stage == completed_stage)
    resumed_checkpoints: list[object] = []
    resumed = api.run_fit_search(
        api.FitSearchRequest("curve", problem, resume_from),
        checkpoint=resumed_checkpoints.append,
    )

    assert tuple(value.stage for value in resumed_checkpoints) == expected_suffix
    _assert_equivalent_results(fresh, resumed)
