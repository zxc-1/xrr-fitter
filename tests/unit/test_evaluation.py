from __future__ import annotations

import warnings
from dataclasses import replace
from typing import get_type_hints

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock, StructureSpec
from xrr_fitter.physics.resolution import GaussHermiteConvergenceWarning


def test_model_evaluation_recomputes_qz_and_shared_periodic_layers() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="first", thickness_a=24.0),
            replace(film, name="second", thickness_a=36.0),
        ),
        repeats=4,
        top_roughness_a=1.5,
    )
    problem = compile_fit_problem(
        prepared_data(size=80),
        StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=2.0),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=17), scale_prior_enabled=False),
    )
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical.update(
        {
            "component.0.layer.0.thickness_a": 27.0,
            "component.0.layer.1.thickness_a": 43.0,
            "instrument.angle_offset_deg": 0.025,
        }
    )
    unit = encode_physical_vector(problem, physical)

    evaluation = evaluate_vector(problem, unit)

    expected_qz = (
        4.0
        * np.pi
        * np.sin(np.deg2rad(problem.data.two_theta_deg / 2.0 + physical["instrument.angle_offset_deg"]))
        / problem.data.beam.effective_wavelength_a
    )
    assert evaluation.valid
    np.testing.assert_allclose(evaluation.qz_a_inv, expected_qz)
    assert evaluation.expanded_stack is not None
    np.testing.assert_allclose(
        evaluation.expanded_stack.thickness_a[1:-1],
        np.tile([27.0, 43.0], 4),
    )


def test_fit_evaluation_keeps_unconverged_resolution_as_structured_diagnostic() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1802), scale_prior_enabled=False),
    )
    physical = {definition.name: definition.initial for definition in problem.parameter_definitions}
    physical["instrument.relative_sigma"] = 0.08
    unit = encode_physical_vector(problem, physical)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", GaussHermiteConvergenceWarning)
        result = evaluation.evaluate_model(problem, unit)

    assert result.valid
    assert not any(item.category is GaussHermiteConvergenceWarning for item in caught)
    assert any(diagnostic.code == "gauss_hermite_unconverged" for diagnostic in result.diagnostics)


@pytest.mark.parametrize(
    "name",
    (
        "values_and_jacobians",
        "values_by_name",
        "encode_physical_vector",
        "evaluate_model",
        "expanded_structure_jacobian",
        "evaluate_model_jacobian",
        "least_squares_residual",
        "least_squares_residual_jacobian",
        "least_squares_loss",
        "problem_log_probability",
    ),
)
def test_public_evaluation_entries_require_the_typed_context(name: str) -> None:
    hints = get_type_hints(getattr(evaluation, name))

    assert hints["problem"] is FitEvaluationContext
