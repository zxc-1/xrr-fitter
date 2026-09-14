"""A valid declared launch must reach full review before rank elimination."""

from __future__ import annotations

import numpy as np
import pytest
from tests.support.synthetic_recovery_layer_cases import _double_layer_cases
from tests.support.synthetic_recovery_runtime import (
    _fit_config,
    _generate_case_intensity,
    _prepared_case_data,
)

from xrr_fitter.fit.adaptive_review import review_on_grids
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.fit.stages import compile_coarse_problem, run_stage_a


def test_progressive_stage_a_preserves_valid_declared_launch_outside_top_eight():
    case = next(value for value in _double_layer_cases() if value.case_id == "double-12019")
    data = _prepared_case_data(case, _generate_case_intensity(case))
    problem = compile_fit_problem(data, case.fit_structure, case.fit_instrument, _fit_config(case))
    coarse = compile_coarse_problem(problem)
    assert (np.count_nonzero(coarse.data.fit_mask), np.count_nonzero(data.fit_mask)) == (256, 518)

    starts, summary, _warnings = run_stage_a(
        problem,
        case.case_id,
        coarse_problem=coarse,
        progress=None,
        cancelled=None,
    )

    assert "declared-baseline" in tuple(start.feature_key for start in starts)
    assert "declared-baseline" in summary.candidate_ids
    first_review = summary.search_evidence[0].reviews[0]
    assert "A-0" in first_review.candidate_ids
    # The protected launch supplements the existing cost and geometry quota.
    assert len(first_review.candidate_ids) >= 9
    assert first_review.full_evaluations >= 9


def _review_fixture():
    units = np.r_[np.linspace(0.1, 0.101, 8), [0.3, 0.5, 0.7, 0.9]][:, None]
    costs = np.arange(1.0, 13.0)
    names = tuple(f"candidate-{index:02d}" for index in range(len(units)))
    return units, costs, names


def test_mandatory_review_participates_in_promotion_without_replacing_diverse_candidates():
    units, costs, names = _review_fixture()
    full = costs.copy()
    full[6] = 0.5
    calls = []

    def evaluate(level, unit):
        index = int(np.flatnonzero(np.all(units == unit, axis=1))[0])
        calls.append((level, index))
        return full[index]

    result = review_on_grids(
        units,
        costs,
        names,
        ((128,), (256,)),
        evaluate,
        initial_evaluations=12,
        mandatory_indices=(6,),
    )

    first, final = result.reviews
    assert set(first.candidate_ids) == {names[index] for index in (0, 1, 2, 3, 6, 8, 9, 10, 11)}
    assert first.promoted is True
    assert result.indices[0] == 6
    assert first.full_evaluations == 9 and final.full_evaluations == 3
    assert sorted(calls) == [(1, index) for index in range(12)]


@pytest.mark.parametrize("full_only", [False, True])
def test_already_selected_mandatory_candidate_is_evaluated_once(full_only):
    units, costs, names = _review_fixture()
    calls = []

    def evaluate(_level, unit):
        index = int(np.flatnonzero(np.all(units == unit, axis=1))[0])
        calls.append(index)
        return costs[index]

    result = review_on_grids(
        units,
        costs,
        names,
        ((256,),) if full_only else ((128,), (256,)),
        evaluate,
        mandatory_indices=(0,),
    )

    assert calls.count(0) == 1
    assert len(calls) == len(set(calls)) == (12 if full_only else 8)
    assert sum(review.full_evaluations for review in result.reviews) == len(calls)


@pytest.mark.parametrize("indices", [(-1,), (12,), (True,), (0.5,), (0, 0)])
def test_mandatory_review_rejects_invalid_candidate_indices(indices):
    units, costs, names = _review_fixture()
    with pytest.raises(ValueError, match="mandatory"):
        review_on_grids(units, costs, names, ((128,),), lambda _level, _unit: 1.0, mandatory_indices=indices)
