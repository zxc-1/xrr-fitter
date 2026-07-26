from __future__ import annotations

from dataclasses import replace

import numpy as np

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock, StructureSpec


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
    physical = {
        definition.name: definition.initial for definition in problem.parameter_definitions
    }
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
        * np.sin(
            np.deg2rad(problem.data.two_theta_deg / 2.0 + physical["instrument.angle_offset_deg"])
        )
        / problem.data.beam.effective_wavelength_a
    )
    assert evaluation.valid
    np.testing.assert_allclose(evaluation.qz_a_inv, expected_qz)
    assert evaluation.expanded_stack is not None
    np.testing.assert_allclose(
        evaluation.expanded_stack.thickness_a[1:-1],
        np.tile([27.0, 43.0], 4),
    )
