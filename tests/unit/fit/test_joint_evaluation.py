from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, ModelEvaluation, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterReference, ParameterSetting, SharingRule
from xrr_fitter.model.structure import LayerSpec

SHARED_NAME = "component.0.density_scale"
TIE_THICKNESS_NAME = "component.0.thickness_a"
TIE_ROUGHNESS_NAME = "component.1.roughness_a"


def _problem(*, seed: int, size: int, scale_prior: bool = False):
    config = replace(
        FitConfig.fast(seed),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=scale_prior,
    )
    base = compile_fit_problem(
        prepared_data(size=size),
        simple_structure(),
        InstrumentSpec(footprint_mode="none", instrument_id="shared-lab"),
        config,
    )
    free_names = {SHARED_NAME, "instrument.scale"} if scale_prior else {SHARED_NAME}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name in free_names else definition.initial,
            definition.upper if definition.name in free_names else definition.initial,
            locked=definition.name not in free_names,
        )
        for definition in base.parameter_definitions
    )
    compiled = compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        settings,
    )
    return replace(compiled, scale_prior_center=1.0) if scale_prior else compiled


def _joint(*, scale_prior: bool = False):
    api = import_module("xrr_fitter.fit.joint_problem")
    rule = SharingRule(
        "film-thickness",
        (
            ParameterReference("left", SHARED_NAME),
            ParameterReference("right", SHARED_NAME),
        ),
    )
    return api.compile_joint_problem(
        ("left", "right"),
        (
            _problem(seed=853, size=40, scale_prior=scale_prior),
            _problem(seed=853, size=52, scale_prior=scale_prior),
        ),
        (rule,),
    )


def _tie_problem(*, seed: int, size: int):
    base_structure = simple_structure()
    film = base_structure.components[0]
    assert isinstance(film, LayerSpec)
    structure = replace(
        base_structure,
        components=(
            replace(film, name="upper", thickness_a=20.0, roughness_a=1.0),
            replace(film, name="lower", thickness_a=20.0, roughness_a=2.0),
        ),
    )
    config = replace(FitConfig.fast(seed), scale_prior_enabled=False)
    base = compile_fit_problem(
        prepared_data(size=size),
        structure,
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    settings = []
    for definition in base.parameter_definitions:
        if definition.name == TIE_THICKNESS_NAME:
            settings.append(ParameterSetting(definition.name, 20.0, 10.0, 40.0))
        elif definition.name == TIE_ROUGHNESS_NAME:
            settings.append(ParameterSetting(definition.name, 2.0, 0.0, 20.0))
        else:
            settings.append(
                ParameterSetting(
                    definition.name,
                    definition.initial,
                    definition.initial,
                    definition.initial,
                    locked=True,
                )
            )
    return compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        tuple(settings),
    )


def _tie_joint():
    api = import_module("xrr_fitter.fit.joint_problem")
    rules = tuple(
        SharingRule(
            sharing_key,
            (
                ParameterReference("left", parameter_name),
                ParameterReference("right", parameter_name),
            ),
        )
        for sharing_key, parameter_name in (
            ("shared-thickness", TIE_THICKNESS_NAME),
            ("shared-roughness", TIE_ROUGHNESS_NAME),
        )
    )
    return api.compile_joint_problem(
        ("left", "right"),
        (
            _tie_problem(seed=859, size=40),
            _tie_problem(seed=859, size=52),
        ),
        rules,
    )


def _unequal_roughness_problem(*, thickness_a: float, seed: int, size: int):
    structure = simple_structure()
    film = structure.components[0]
    assert isinstance(film, LayerSpec)
    structure = replace(
        structure,
        components=(replace(film, thickness_a=thickness_a, roughness_a=3.0),),
    )
    base = compile_fit_problem(
        prepared_data(size=size),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(seed), scale_prior_enabled=False),
    )
    free_names = {"component.0.thickness_a", "component.0.roughness_a"}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            (
                thickness_a * 0.5
                if definition.name == "component.0.thickness_a"
                else definition.lower
            )
            if definition.name in free_names
            else definition.initial,
            (
                thickness_a * 1.5
                if definition.name == "component.0.thickness_a"
                else definition.upper
            )
            if definition.name in free_names
            else definition.initial,
            locked=definition.name not in free_names,
        )
        for definition in base.parameter_definitions
    )
    return compile_fit_problem(
        base.data,
        base.structure,
        base.instrument,
        base.config,
        settings,
    )


def _unequal_roughness_joint():
    api = import_module("xrr_fitter.fit.joint_problem")
    rule = SharingRule(
        "shared-physical-roughness",
        (
            ParameterReference("left", "component.0.roughness_a"),
            ParameterReference("right", "component.0.roughness_a"),
        ),
    )
    return api.compile_joint_problem(
        ("left", "right"),
        (
            _unequal_roughness_problem(thickness_a=100.0, seed=863, size=40),
            _unequal_roughness_problem(thickness_a=20.0, seed=863, size=52),
        ),
        (rule,),
    )


def _evaluation(problem, *, objective: float, residual: float, valid: bool = True):
    fit_count = int(np.count_nonzero(problem.data.fit_mask))
    qz = problem.data.qz_a_inv
    return ModelEvaluation(
        valid=valid,
        reason="evaluated" if valid else "physical_constraint",
        parameters=(),
        qz_a_inv=qz,
        model_normalized=np.ones_like(qz),
        fit_log_residuals_decades=np.full(fit_count, residual),
        fit_weighted_residuals=np.full(fit_count, residual),
        objective=objective,
        expanded_stack=None,
        diagnostics=(),
    )


def test_joint_objective_is_the_arithmetic_mean_of_local_objectives() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()

    result = api.evaluate_joint_vector(joint, np.asarray([0.5]))

    assert result.valid
    assert len(result.local_evaluations) == 2
    assert result.objective == pytest.approx(
        np.mean([value.objective for value in result.local_evaluations])
    )
    assert result.local_unit_vectors[0][0] == result.local_unit_vectors[1][0] == 0.5
    assert not result.residuals.flags.writeable


def test_joint_residual_gives_each_dataset_equal_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()
    c_decades = joint.problems[0].config.c_decades
    objective = 2.0 * c_decades**2 * (np.sqrt(1.0 + 1.0 / c_decades**2) - 1.0)
    by_identity = {
        id(problem): _evaluation(problem, objective=objective, residual=1.0)
        for problem in joint.problems
    }
    monkeypatch.setattr(api, "evaluate_vector", lambda problem, _unit: by_identity[id(problem)])

    result = api.evaluate_joint_vector(joint, np.asarray([0.5]))

    first_size = len(by_identity[id(joint.problems[0])].fit_log_residuals_decades)
    first = result.residuals[:first_size]
    second = result.residuals[first_size:]
    np.testing.assert_array_equal(first, np.ones(first.size))
    np.testing.assert_array_equal(second, np.ones(second.size))
    rho = api.joint_least_squares_loss(joint)(result.residuals**2)
    assert 0.5 * np.sum(rho[0]) / result.residuals.size == pytest.approx(objective)
    assert result.objective == pytest.approx(objective)


def test_joint_loss_scales_data_mass_and_each_active_scale_prior_row() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint(scale_prior=True)
    data_sizes = tuple(int(np.count_nonzero(problem.data.fit_mask)) for problem in joint.problems)
    row_count = sum(size + 1 for size in data_sizes)
    squared = np.linspace(0.001, 0.02, row_count)

    rho = api.joint_least_squares_loss(joint)(squared)

    total_data = sum(data_sizes)
    offset = 0
    for problem, size in zip(joint.problems, data_sizes, strict=True):
        alpha = total_data / (len(joint.problems) * size)
        weights = problem.weights[problem.data.fit_mask]
        data_squared = squared[offset : offset + size]
        scaled = 1.0 + data_squared / problem.config.c_decades**2
        np.testing.assert_allclose(
            rho[0, offset : offset + size],
            4.0
            * alpha
            * weights**2
            * problem.config.c_decades**2
            * (np.sqrt(scaled) - 1.0),
        )
        np.testing.assert_allclose(
            rho[1, offset : offset + size],
            2.0 * alpha * weights**2 / np.sqrt(scaled),
        )
        np.testing.assert_allclose(
            rho[2, offset : offset + size],
            -(alpha * weights**2 / problem.config.c_decades**2) * scaled ** (-1.5),
        )
        prior_index = offset + size
        np.testing.assert_allclose(
            rho[:, prior_index],
            (2.0 * alpha * squared[prior_index], 2.0 * alpha, 0.0),
        )
        offset = prior_index + 1

    unit = np.full(len(joint.global_variables), 0.55)
    evaluation = api.evaluate_joint_vector(joint, unit)
    optimizer_rho = api.joint_least_squares_loss(joint)(evaluation.residuals**2)
    optimizer_objective = 0.5 * float(np.sum(optimizer_rho[0])) / total_data
    assert optimizer_objective == pytest.approx(
        evaluation.objective,
        rel=1e-12,
        abs=1e-14,
    )
    jacobian = api.evaluate_joint_jacobian(joint, unit)
    assert jacobian.shape == (row_count, len(joint.global_variables))
    assert np.any(np.abs(jacobian[np.cumsum(np.asarray(data_sizes) + 1) - 1]) > 0.0)


@pytest.mark.parametrize(
    "invalid_rows",
    [
        pytest.param(lambda count: np.zeros(count - 1), id="missing-row"),
        pytest.param(
            lambda count: np.concatenate((np.asarray([-1e-6]), np.zeros(count - 1))),
            id="negative",
        ),
        pytest.param(
            lambda count: np.concatenate((np.asarray([np.nan]), np.zeros(count - 1))),
            id="nonfinite",
        ),
    ],
)
def test_joint_loss_rejects_invalid_squared_residual_rows(invalid_rows) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint(scale_prior=True)
    row_count = sum(
        int(np.count_nonzero(problem.data.fit_mask)) + 1 for problem in joint.problems
    )

    with pytest.raises(ValueError, match="joint loss|squared|row|finite|nonnegative"):
        api.joint_least_squares_loss(joint)(invalid_rows(row_count))


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda joint: replace(joint, problems=()), id="empty-layout"),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(replace(joint.problems[0], weights=np.empty(0)), *joint.problems[1:]),
            ),
            id="empty-weights",
        ),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(
                    replace(joint.problems[0], weights=joint.problems[0].weights[None, :]),
                    *joint.problems[1:],
                ),
            ),
            id="two-dimensional-weights",
        ),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(
                    replace(joint.problems[0], weights=np.zeros_like(joint.problems[0].weights)),
                    *joint.problems[1:],
                ),
            ),
            id="zero-weights",
        ),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(
                    replace(
                        joint.problems[0],
                        weights=np.full_like(joint.problems[0].weights, np.nan),
                    ),
                    *joint.problems[1:],
                ),
            ),
            id="nonfinite-weights",
        ),
        pytest.param(
            lambda joint: replace(
                joint,
                problems=(
                    replace(joint.problems[0], config=SimpleNamespace(c_decades=0.0)),
                    *joint.problems[1:],
                ),
            ),
            id="nonpositive-c-decades",
        ),
    ],
)
def test_joint_loss_rejects_invalid_compiled_layout(mutation) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")

    # The typed evaluation context now rejects several malformed dataset
    # layouts before the joint-loss boundary is reached. Both boundaries are
    # valid owners of this invariant and must keep the rejection explicit.
    with pytest.raises(
        (TypeError, ValueError),
        match="joint loss|layout|weight|c_decades|dataset|region arrays|config",
    ):
        api.joint_least_squares_loss(mutation(_joint(scale_prior=True)))


def test_joint_analytic_jacobian_matches_global_finite_difference() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()
    unit = np.asarray([0.5])

    analytic = api.evaluate_joint_jacobian(joint, unit)
    step = 1e-5
    plus = api.evaluate_joint_vector(joint, unit + step).residuals
    minus = api.evaluate_joint_vector(joint, unit - step).residuals
    finite = ((plus - minus) / (2.0 * step))[:, None]

    assert analytic.shape == finite.shape
    np.testing.assert_allclose(analytic, finite, rtol=2e-5, atol=2e-8)
    assert not analytic.flags.writeable


def test_joint_analytic_jacobian_matches_finite_difference_with_scale_prior() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint(scale_prior=True)
    unit = np.full(len(joint.global_variables), 0.55)

    analytic = api.evaluate_joint_jacobian(joint, unit)
    step = 1e-5
    finite = np.column_stack(
        [
            (
                api.evaluate_joint_vector(joint, unit + np.eye(unit.size)[index] * step).residuals
                - api.evaluate_joint_vector(
                    joint,
                    unit - np.eye(unit.size)[index] * step,
                ).residuals
            )
            / (2.0 * step)
            for index in range(unit.size)
        ]
    )
    prior_rows = np.cumsum(
        [int(np.count_nonzero(problem.data.fit_mask)) + 1 for problem in joint.problems]
    ) - 1

    assert np.all(np.any(np.abs(analytic[prior_rows]) > 0.0, axis=1))
    np.testing.assert_allclose(analytic, finite, rtol=2e-5, atol=2e-8)


def test_dynamic_roughness_tie_jacobian_matches_central_difference() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = _tie_joint()
    unit = sharing.initial_joint_vector(joint)
    thickness_index = next(
        index
        for index, variable in enumerate(joint.global_variables)
        if variable.name == "shared-thickness"
    )

    analytic = api.evaluate_joint_jacobian(joint, unit)[:, thickness_index]
    step = 1e-5
    plus = unit.copy()
    minus = unit.copy()
    plus[thickness_index] += step
    minus[thickness_index] -= step
    finite = (
        api.evaluate_joint_vector(joint, plus).residuals
        - api.evaluate_joint_vector(joint, minus).residuals
    ) / (2.0 * step)

    np.testing.assert_allclose(analytic, finite, rtol=3e-5, atol=3e-8)


def test_shared_roughness_uses_one_physical_value_with_local_thicknesses() -> None:
    evaluation = import_module("xrr_fitter.fit.joint_evaluation")
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = _unequal_roughness_joint()
    unit = sharing.initial_joint_vector(joint)

    result = evaluation.evaluate_joint_vector(joint, unit)

    roughness = tuple(
        next(
            value.value
            for value in local.parameters
            if value.name == "component.0.roughness_a"
        )
        for local in result.local_evaluations
    )
    assert roughness == pytest.approx((3.0, 3.0))
    roughness_index = next(
        index
        for index, variable in enumerate(joint.global_variables)
        if variable.name == "shared-physical-roughness"
    )
    assert joint.global_variables[roughness_index].transform == (
        "shared_roughness_physical"
    )


def test_shared_roughness_consensus_uses_candidate_physical_values() -> None:
    evaluation = import_module("xrr_fitter.fit.joint_evaluation")
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = _unequal_roughness_joint()
    candidates = {}
    for dataset_id, problem in zip(
        joint.dataset_ids,
        joint.problems,
        strict=True,
    ):
        unit = encode_physical_vector(
            problem,
            {"component.0.roughness_a": 4.0},
        )
        candidates[dataset_id] = SimpleNamespace(
            valid=True,
            parameters=evaluate_vector(problem, unit).parameters,
        )

    consensus = sharing.consensus_joint_vector(joint, candidates)
    result = evaluation.evaluate_joint_vector(joint, consensus)

    roughness = tuple(
        next(
            value.value
            for value in local.parameters
            if value.name == "component.0.roughness_a"
        )
        for local in result.local_evaluations
    )
    assert roughness == pytest.approx((4.0, 4.0))


def test_shared_roughness_joint_candidate_rebuilds_from_physical_projection() -> None:
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = _unequal_roughness_joint()
    global_unit = sharing.initial_joint_vector(joint)
    local_units = sharing.scatter_joint_vector(joint, global_unit)
    candidates = tuple(
        (
            SimpleNamespace(
                candidate_id="joint-a",
                unit_vector=unit,
                objective=1.0,
                ranking_objective=1.0,
            ),
        )
        for unit in local_units
    )

    rebuilt = sharing.joint_candidate_vectors(
        joint,
        candidates,
        ("joint-a",),
    )

    np.testing.assert_allclose(rebuilt[0], global_unit)


def test_shared_physical_roughness_jacobian_matches_central_difference() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    sharing = import_module("xrr_fitter.fit.joint_sharing")
    joint = _unequal_roughness_joint()
    unit = sharing.initial_joint_vector(joint)

    analytic = api.evaluate_joint_jacobian(joint, unit)
    step = 1e-5
    finite = np.column_stack(
        [
            (
                api.evaluate_joint_vector(
                    joint,
                    unit + np.eye(unit.size)[index] * step,
                ).residuals
                - api.evaluate_joint_vector(
                    joint,
                    unit - np.eye(unit.size)[index] * step,
                ).residuals
            )
            / (2.0 * step)
            for index in range(unit.size)
        ]
    )

    np.testing.assert_allclose(analytic, finite, rtol=5e-5, atol=5e-8)


def test_shared_global_jacobian_column_contains_both_dataset_blocks() -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()

    jacobian = api.evaluate_joint_jacobian(joint, np.asarray([0.55]))

    first_size = int(np.count_nonzero(joint.problems[0].data.fit_mask))
    assert np.any(np.abs(jacobian[:first_size, 0]) > 0.0)
    assert np.any(np.abs(jacobian[first_size:, 0]) > 0.0)


def test_one_invalid_local_evaluation_invalidates_the_joint_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint()
    values = {
        id(joint.problems[0]): _evaluation(joint.problems[0], objective=1.0, residual=0.2),
        id(joint.problems[1]): _evaluation(
            joint.problems[1],
            objective=float("inf"),
            residual=1e6,
            valid=False,
        ),
    }
    monkeypatch.setattr(api, "evaluate_vector", lambda problem, _unit: values[id(problem)])

    result = api.evaluate_joint_vector(joint, np.asarray([0.5]))

    assert not result.valid
    assert result.objective == float("inf")
    assert tuple(value.valid for value in result.local_evaluations) == (True, False)


def test_invalid_local_evaluation_with_scale_prior_remains_a_joint_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    joint = _joint(scale_prior=True)
    unit = np.full(len(joint.global_variables), 0.55)
    baseline = api.evaluate_joint_vector(joint, unit)
    values = {
        id(joint.problems[0]): baseline.local_evaluations[0],
        id(joint.problems[1]): replace(
            baseline.local_evaluations[1],
            valid=False,
            objective=float("inf"),
            parameters=(),
        ),
    }
    monkeypatch.setattr(api, "evaluate_vector", lambda problem, _unit: values[id(problem)])

    result = api.evaluate_joint_vector(joint, unit)

    assert not result.valid
    assert result.objective == float("inf")
    assert result.residuals.size == sum(
        np.count_nonzero(problem.data.fit_mask) + 1 for problem in joint.problems
    )


def test_joint_evaluation_propagates_unexpected_local_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = import_module("xrr_fitter.fit.joint_evaluation")
    sentinel = RuntimeError("unexpected local evaluation failure")

    def fail(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(api, "evaluate_vector", fail)

    with pytest.raises(RuntimeError, match="unexpected local evaluation failure") as captured:
        api.evaluate_joint_vector(_joint(), np.asarray([0.5]))

    assert captured.value is sentinel
