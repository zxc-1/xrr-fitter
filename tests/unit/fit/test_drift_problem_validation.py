from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support.drift_cases import (
    AIR,
    SILICON,
    two_layer_block,
    two_layer_block_with_thickness_drift,
)
from tests.support.model_cases import prepared_data

from xrr_fitter.fit.objective import evaluate_declared_initial
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.structure import DriftSpec, StructureSpec


def _runtime_invalid_drift_problem():
    block = replace(
        two_layer_block(3),
        drift=DriftSpec(kind="linear", target="thickness", amount=1.0),
    )
    structure = StructureSpec(
        AIR,
        (block,),
        SILICON,
        backing_roughness_a=3.0,
    )

    return compile_fit_problem(
        prepared_data(size=40),
        structure,
        InstrumentSpec(footprint_mode="none"),
        FitConfig.fast(1),
    )


def test_compile_defers_invalid_initial_drift_to_runtime_evaluation() -> None:
    problem = _runtime_invalid_drift_problem()

    assert problem.constraint_rules


def test_declared_initial_rejects_runtime_invalid_drift() -> None:
    evaluation = evaluate_declared_initial(_runtime_invalid_drift_problem())

    assert evaluation.valid is False
    assert evaluation.reason.startswith("constraint_out_of_bounds:component.0.repeat.")


@pytest.mark.parametrize(
    "block",
    (two_layer_block(3), two_layer_block_with_thickness_drift(3)),
)
def test_compile_rejects_parameter_setting_that_changes_periodic_topology(block) -> None:
    structure = StructureSpec(
        AIR,
        (block,),
        SILICON,
        backing_roughness_a=3.0,
    )
    setting = ParameterSetting(
        "component.0.repeats",
        2.0,
        2.0,
        2.0,
        locked=True,
    )

    with pytest.raises(ValueError, match="locked integer topology"):
        compile_fit_problem(
            prepared_data(size=40),
            structure,
            InstrumentSpec(footprint_mode="none"),
            FitConfig.fast(1),
            (setting,),
        )
