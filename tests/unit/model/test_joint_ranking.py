"""Persisted joint ranks use fitted counts and the saved scale-prior decision."""

from dataclasses import replace
from math import log10

import pytest
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

from xrr_fitter.model.parameters import ParameterValue
from xrr_fitter.model.project import ScalePriorState


def _dataset(name, mode, objective, ranking, *, mask=(True,) * 4, prior=None, scale=1.0, valid=True):
    candidate = replace(
        fit_candidate("shared"),
        objective=objective,
        noise_model=mode,
        ranking_objective=ranking,
        parameters=(ParameterValue("instrument.scale", scale, 0.0, 1000.0),),
        valid=valid,
    )
    result = replace(final_fit_result(), candidates=(candidate,), best_index=0 if valid else None)
    return replace(
        dataset_project(name, result=result),
        fit_mask=mask,
        scale_prior=ScalePriorState(False) if prior is None else prior,
    )


def _joint(*datasets):
    value = project(*datasets)
    mode = datasets[0].last_valid_result.candidates[0].noise_model
    return replace(value, fit_config=replace(value.fit_config, noise_model=mode), batch_mode="joint")


@pytest.mark.parametrize("mode", ("gaussian", "poisson"))
def test_likelihood_rank_uses_fitted_mask_counts_not_source_lengths(mode) -> None:
    joint = _joint(
        _dataset("first", mode, 1.0, 2.0),
        _dataset("second", mode, 4.0, 2.0, mask=(True, False, True, False)),
    )

    assert joint.datasets[0].last_valid_result.best_candidate.ranking_objective == 2.0


@pytest.mark.parametrize("mode", ("gaussian", "poisson"))
def test_likelihood_rejects_the_old_unweighted_rank(mode) -> None:
    with pytest.raises(ValueError, match="global ranking"):
        _joint(
            _dataset("first", mode, 1.0, 2.5),
            _dataset("second", mode, 4.0, 2.5, mask=(True, False, True, False)),
        )


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
@pytest.mark.parametrize("active", ((True, False), (False, True), (True, True)))
def test_joint_rank_counts_each_active_scale_prior_once(mode, active) -> None:
    priors = tuple(ScalePriorState(True, 1.0, 1.0) if enabled else ScalePriorState(False) for enabled in active)
    penalties = tuple(
        log10(scale) ** 2 if enabled else 0.0 for scale, enabled in zip((10.0, 100.0), active, strict=True)
    )
    data_rank = 2.5 if mode == "robust_log" else 2.0
    ranking = data_rank + sum(penalties) / 6

    joint = _joint(
        _dataset("first", mode, 1.0 + penalties[0] / 4, ranking, prior=priors[0], scale=10.0),
        _dataset("second", mode, 4.0 + penalties[1] / 2, ranking, mask=(True, False) * 2, prior=priors[1], scale=100.0),
    )

    assert joint.datasets[0].last_valid_result.best_candidate.ranking_objective == ranking


def test_robust_rank_without_prior_still_balances_members_not_points() -> None:
    joint = _joint(
        _dataset("first", "robust_log", 1.0, 2.5),
        _dataset("second", "robust_log", 4.0, 2.5, mask=(True, False) * 2),
    )

    assert joint.batch_mode == "joint"


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
def test_unequal_point_joint_rank_avoids_overflowing_the_local_sum(mode) -> None:
    ranking = 1.6e308 / 2 + 1.2e308 / 2 if mode == "robust_log" else 1.6e308 * (4 / 6) + 1.2e308 * (2 / 6)

    joint = _joint(
        _dataset("first", mode, 1.6e308, ranking),
        _dataset("second", mode, 1.2e308, ranking, mask=(True, False) * 2),
    )

    assert joint.datasets[0].last_valid_result.best_candidate.ranking_objective == ranking


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
def test_valid_joint_rank_requires_fitted_points_in_each_member(mode) -> None:
    with pytest.raises(ValueError, match="fitted points"):
        _joint(
            _dataset("first", mode, 1.0, 1.0),
            _dataset("second", mode, 1.0, 1.0, mask=(False,) * 4),
        )


def test_invalid_joint_candidates_do_not_acquire_a_denominator_or_rank() -> None:
    joint = _joint(
        _dataset("first", "gaussian", float("inf"), None, valid=False),
        _dataset("second", "gaussian", float("inf"), None, mask=(False,) * 4, valid=False),
    )

    assert joint.datasets[0].last_valid_result.candidates[0].ranking_objective is None


@pytest.mark.parametrize("scale", (None, 0.0))
def test_robust_prior_adjustment_requires_the_saved_positive_scale(scale) -> None:
    first = _dataset("first", "robust_log", 1.0, 1.0, prior=ScalePriorState(True, 1.0, 1.0), scale=0.0)
    if scale is None:
        result = first.last_valid_result
        candidate = replace(result.candidates[0], parameters=())
        first = replace(first, last_valid_result=replace(result, candidates=(candidate,)))

    with pytest.raises(ValueError, match="instrument.scale"):
        _joint(first, _dataset("second", "robust_log", 1.0, 1.0, mask=(True, False) * 2))
