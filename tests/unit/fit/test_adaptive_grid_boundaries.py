"""Boundary regressions for real row density, joint geometry, and frozen evidence.

Localized fringes deliberately occupy only a small part of the measured range:
an average points-per-fringe ratio cannot establish adequate local sampling.
The joint probes use compiled sharing and expression-constraint layouts while
controlling only the objective, so nuisance distances cannot hide a structural
full-cost winner. Evidence tests distinguish initial grid work from additional
full-review cache misses, including an invalid initial physical evaluation.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.fit import adaptive_review, joint_evaluation
from xrr_fitter.fit.adaptive_grid import initial_grid_points
from xrr_fitter.fit.candidates import CandidateStart, candidate_from_evaluation
from xrr_fitter.fit.feature_grid import feature_grid_indices
from xrr_fitter.fit.joint_problem import compile_joint_problem
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.fit.screening import fringe_extrema_qz
from xrr_fitter.io.codec_common import ProjectSchemaError
from xrr_fitter.io.codec_search import search_evidence_from_list, search_evidence_to_list
from xrr_fitter.model.fitting import FitConfig, GridReview, SearchEvidence
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ConstraintNode,
    ConstraintRule,
    ParameterFreedom,
    ParameterReference,
    ParameterSetting,
    SharingRule,
)


def _localized_fringes(start, width):
    qz = np.linspace(0.01, 0.30, 1200)
    phase = np.linspace(0.0, 1.0, 1200)
    curve = np.ones(1200)
    active = (phase >= start) & (phase <= start + width)
    curve[active] = 1.0 + 0.6 * np.cos(2 * np.pi * 10 * (phase[active] - start) / width)
    return SimpleNamespace(qz_a_inv=qz, intensity_normalized=curve, r_floor=1e-12, fit_mask=np.ones(1200, dtype=bool))


def _fringe_counts(qz, extrema):
    return [
        int(np.count_nonzero((qz >= extrema[index]) & (qz <= extrema[index + 2]))) for index in range(len(extrema) - 2)
    ]


@pytest.mark.parametrize(("start", "width", "expected"), [(0.1, 0.4, 256), (0.3, 0.3, 512), (0.3, 0.1, 1200)])
def test_initial_grid_counts_real_selected_rows_in_every_detected_fringe(start, width, expected):
    data = _localized_fringes(start, width)
    extrema = fringe_extrema_qz(data.qz_a_inv, data.intensity_normalized, data.r_floor)
    assert len(extrema) >= 3
    assert min(_fringe_counts(data.qz_a_inv, extrema)) >= 8

    level = initial_grid_points(data)
    selected = feature_grid_indices(data, level)

    assert level == expected
    assert min(_fringe_counts(data.qz_a_inv[selected], extrema)) >= 8
    assert len(selected) == len(set(selected)) == level
    assert selected[0] == 0 and selected[-1] == 1199


def test_localized_fringe_probe_has_thirteen_real_rows_but_only_four_at_128():
    data = _localized_fringes(0.3, 0.1)
    extrema = fringe_extrema_qz(data.qz_a_inv, data.intensity_normalized, data.r_floor)
    selected = feature_grid_indices(data, 128)
    assert len(extrema) == 23
    assert _fringe_counts(data.qz_a_inv, extrema)[::2] == [8, *([13] * 9), 223]
    assert _fringe_counts(data.qz_a_inv[selected], extrema)[::2] == [3, *([4] * 9), 16]
    assert initial_grid_points(data) == 1200


def _geometry_member(size):
    base = compile_fit_problem(prepared_data(size=size), simple_structure(), InstrumentSpec(), FitConfig.fast(17))
    free = {"component.0.thickness_a", "instrument.scale", "instrument.background"}
    settings = tuple(
        ParameterSetting(
            value.name,
            value.initial,
            value.lower if value.name in free else value.initial,
            value.upper if value.name in free else value.initial,
            freedom=ParameterFreedom.from_locked(value.name not in free),
        )
        for value in base.parameter_definitions
    )
    settings = tuple(
        replace(value, lower=2.0, upper=500.0) if value.name == "component.0.thickness_a" else value
        for value in settings
    )
    return compile_fit_problem(base.data, base.structure, base.instrument, base.config, settings)


def _geometry_problem(cross_constraint):
    members = tuple(_geometry_member(size) for size in (160, 192))
    refs = tuple(ParameterReference(dataset_id, "component.0.thickness_a") for dataset_id in ("left", "right"))
    if cross_constraint:
        rule = ConstraintRule(refs[0], ConstraintNode("ref", reference=refs[1]))
        return compile_joint_problem(("left", "right"), members, (), (rule,)), 2
    return compile_joint_problem(("left", "right"), members, (SharingRule("thickness", refs),)), 0


def _geometry_population(structure_axis):
    units = np.zeros((12, 5))
    units[:4, 1] = np.arange(4) * 0.01
    units[4:8, 1:] = np.eye(4)
    units[8:, 0] = (0.2, 0.4, 0.6, 0.8)
    units[:, [0, structure_axis]] = units[:, [structure_axis, 0]]
    return units


@pytest.mark.parametrize("cross_constraint", [False, True])
def test_joint_review_excludes_nuisance_distances_and_keeps_structural_full_winner(monkeypatch, cross_constraint):
    problem, structure_axis = _geometry_problem(cross_constraint)
    units = _geometry_population(structure_axis)
    costs = np.arange(1.0, 13.0)
    full_costs = costs.copy()
    full_costs[-1] = 0.5
    contexts = adaptive_review.joint_grid_contexts(problem)
    solved = SimpleNamespace(population=units, population_energies=costs, nfev=12, stop_reason="budget_exhausted")
    calls = []

    def evaluate(context, unit, **_kwargs):
        index = int(np.flatnonzero(np.all(units == unit, axis=1))[0])
        calls.append(index)
        return SimpleNamespace(objective=full_costs[index] if context is contexts[-1] else costs[index])

    monkeypatch.setattr(adaptive_review, "joint_grid_contexts", lambda _problem: contexts)
    monkeypatch.setattr(joint_evaluation, "evaluate_joint_vector", evaluate)
    starts, evidence = adaptive_review.review_joint_population(problem, solved, "E", 17, 20)

    expected_ids = tuple(f"E-population-{index:04d}" for index in (0, 1, 2, 3, 8, 9, 10, 11))
    assert evidence.reviews[0].candidate_ids == expected_ids
    assert evidence.reviews[0].promoted is True
    assert evidence.reviews[0].full_objectives[-1] == 0.5
    np.testing.assert_array_equal(starts[0], units[-1])
    assert 11 in calls


def test_full_only_review_separates_initial_grid_work_from_extra_full_cache_misses(monkeypatch):
    problem = compile_fit_problem(prepared_data(size=80), simple_structure(), InstrumentSpec(), FitConfig.fast(17))
    real_evaluate = adaptive_review.evaluate_vector
    calls = []

    def evaluate(context, unit, **kwargs):
        calls.append(int(np.count_nonzero(context.data.fit_mask)))
        return real_evaluate(context, unit, **kwargs)

    evaluated = []
    validity = []
    for index, value in enumerate((0.2, 0.5, 0.8)):
        unit = np.full(len(problem.variables), value)
        observed = evaluate(problem, unit, fit_only=True)
        validity.append(observed.valid)
        if observed.valid:
            candidate = candidate_from_evaluation(problem, unit, observed, f"a-{index}", -1, "initial", 1)
            start = CandidateStart(tuple((value.name, value.value) for value in observed.parameters), f"a-{index}")
            evaluated.append((start, candidate))
    monkeypatch.setattr(adaptive_review, "evaluate_vector", evaluate)
    _candidates, evidence = adaptive_review.review_stage_a(problem, evaluated, 3, None)

    assert validity == [False, True, True]
    assert calls == [80] * 5
    review = evidence.reviews[0]
    assert getattr(review, "grid_evaluations", None) == 3
    assert review.full_evaluations == 2
    assert evidence.grid_review_evaluations + evidence.full_review_evaluations == len(calls)
    with pytest.raises(ValueError, match="full_evaluations"):
        replace(review, full_evaluations=5)


def _review(coarse=(1.0, 2.0), full=(1.0, float("inf"))):
    return GridReview((128,), ("a", "b"), coarse, full, False, 3, 2)


def _evidence(review):
    return SearchEvidence("boundary", 17, (review,), (), "full_grid_reviewed")


def test_grid_review_has_one_unambiguous_work_count_field_and_strict_codec():
    names = {field.name for field in fields(GridReview)}
    assert "grid_evaluations" in names
    assert "coarse_evaluations" not in names
    evidence = _evidence(_review())
    raw = json.loads(json.dumps(search_evidence_to_list((evidence,)), allow_nan=False))
    assert raw[0]["reviews"][0]["grid_evaluations"] == 3
    assert search_evidence_from_list(raw) == (evidence,)
    raw[0]["reviews"][0]["coarse_evaluations"] = raw[0]["reviews"][0].pop("grid_evaluations")
    with pytest.raises(ProjectSchemaError, match="grid_evaluations|coarse_evaluations"):
        search_evidence_from_list(raw)


@pytest.mark.parametrize("cost", [np.array(1.0), np.array(float("inf")), np.array(True), np.bool_(True)])
def test_grid_review_rejects_mutable_zero_dimensional_costs_and_boolean_scalars(cost):
    with pytest.raises(ValueError, match="review objectives"):
        _review(coarse=(cost, 2.0))


def test_review_costs_detach_external_arrays_and_round_trip_as_python_scalars():
    external_coarse = np.array([1.0, 2.0])
    external_full = [np.float32(1.0), np.float64("inf")]
    review = _review(external_coarse, external_full)
    external_coarse[:] = 99.0
    external_full[:] = [99.0, 99.0]

    assert review.coarse_objectives == (1.0, 2.0)
    assert review.full_objectives == (1.0, float("inf"))
    assert all(type(value) is float for value in (*review.coarse_objectives, *review.full_objectives))
    evidence = _evidence(review)
    raw = json.loads(json.dumps(search_evidence_to_list((evidence,)), allow_nan=False))
    assert raw[0]["reviews"][0]["full_objectives"] == [1.0, None]
    assert search_evidence_from_list(raw) == (evidence,)
    assert pickle.loads(pickle.dumps(evidence)) == evidence
