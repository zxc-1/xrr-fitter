from dataclasses import replace

import numpy as np
import pytest
from tests.support.drift_cases import (
    AIR,
    SILICON,
    two_layer_block,
    two_layer_block_with_thickness_drift,
)
from tests.support.model_cases import prepared_data

from xrr_fitter.evaluation import encode_physical_vector, values_by_name
from xrr_fitter.fit.parameters import _periodic_definitions, thickness_bounds
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.structure import DriftSpec, StructureSpec
from xrr_fitter.physics.stack import expand_structure, rebuild_structure


def _names(defs):
    return [d.name for d in defs]


def test_thickness_bounds_stay_finite_for_a_subnormal_q_span() -> None:
    smallest = np.nextafter(0.0, 1.0)
    qz = smallest * np.arange(2048.0, 2080.0)
    data = replace(prepared_data(size=qz.size), qz_a_inv=qz)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        lower, upper = thickness_bounds(data)

    assert np.isfinite(lower) and np.isfinite(upper)
    assert 2.0 <= lower <= upper <= 2e5


def test_no_drift_definitions_unchanged():
    defs = _periodic_definitions("component.0", two_layer_block(), (2.0, 500.0))
    assert not any(".drift_scale" in n or ".repeat." in n for n in _names(defs))


def test_drift_adds_scale_and_percopy():  # repeats=3, 2 layers
    block = two_layer_block_with_thickness_drift()
    defs = _periodic_definitions("component.0", block, (2.0, 500.0))
    names = _names(defs)
    assert "component.0.drift_scale" in names
    for k in (1, 2):
        for i in (0, 1):
            assert f"component.0.repeat.{k}.layer.{i}.thickness_a" in names
    # 非目标族（roughness）不发逐副本
    assert not any(".repeat." in n and n.endswith("roughness_a") for n in names)


def test_linear_drift_bounds_respect_the_largest_repeat_coefficient() -> None:
    block = replace(
        two_layer_block(50),
        drift=DriftSpec(kind="linear", target="thickness", amount=0.0),
    )
    structure = StructureSpec(AIR, (block,), SILICON, backing_roughness_a=3.0)
    problem = compile_fit_problem(
        prepared_data(),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(301), scale_prior_enabled=False),
    )
    definition = next(item for item in problem.parameter_definitions if item.name.endswith(".drift_scale"))

    assert definition.lower > -1.0 / 49.0
    assert definition.upper == pytest.approx(0.5)

    physical = {item.name: item.initial for item in problem.parameter_definitions}
    values = values_by_name(problem, encode_physical_vector(problem, physical))
    assert all(values[f"component.0.repeat.49.layer.{index}.thickness_a"] > 0.0 for index in (0, 1))


def test_linear_drift_amount_outside_repeat_domain_is_rejected_at_compile() -> None:
    block = replace(
        two_layer_block(50),
        drift=DriftSpec(kind="linear", target="thickness", amount=-0.5),
    )
    structure = StructureSpec(AIR, (block,), SILICON, backing_roughness_a=3.0)

    with pytest.raises(ValueError, match="drift_scale"):
        compile_fit_problem(
            prepared_data(),
            structure,
            InstrumentSpec(footprint_mode="none"),
            replace(FitConfig.fast(302), scale_prior_enabled=False),
        )


def test_roughness_drift_initial_value_respects_repeated_interface_cap() -> None:
    block = replace(
        two_layer_block(5),
        drift=DriftSpec(kind="linear", target="roughness", amount=1.0),
    )
    structure = StructureSpec(AIR, (block,), SILICON, backing_roughness_a=3.0)

    with pytest.raises(ValueError, match="drift_scale"):
        compile_fit_problem(
            prepared_data(),
            structure,
            InstrumentSpec(footprint_mode="none"),
            replace(FitConfig.fast(303), scale_prior_enabled=False),
        )


def test_thickness_drift_initial_value_respects_repeated_interface_cap() -> None:
    base = two_layer_block(2)
    layers = (
        replace(base.layers[0], thickness_a=100.0, roughness_a=2.0),
        replace(base.layers[1], thickness_a=2.0, roughness_a=0.1),
    )
    block = replace(
        base,
        layers=layers,
        drift=DriftSpec(kind="linear", target="thickness", amount=0.1),
    )
    structure = StructureSpec(AIR, (block,), SILICON, backing_roughness_a=0.0)

    with pytest.raises(ValueError, match="drift_scale"):
        compile_fit_problem(
            prepared_data(),
            structure,
            InstrumentSpec(footprint_mode="none"),
            replace(FitConfig.fast(304), scale_prior_enabled=False),
        )


def test_roughness_drift_upper_endpoint_remains_constructible_with_exact_bases() -> None:
    block = replace(
        two_layer_block(7),
        drift=DriftSpec(kind="linear", target="roughness", amount=0.1),
    )
    structure = StructureSpec(AIR, (block,), SILICON, backing_roughness_a=3.0)
    config = replace(FitConfig.fast(305), scale_prior_enabled=False)
    initial = compile_fit_problem(
        prepared_data(),
        structure,
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    exact_bases = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.initial,
            definition.initial,
            locked=True,
        )
        for definition in initial.parameter_definitions
        if definition.name.startswith("component.0.layer.")
        and (definition.name.endswith(".thickness_a") or definition.name.endswith(".roughness_a"))
    )
    problem = compile_fit_problem(
        initial.data,
        structure,
        initial.instrument,
        config,
        exact_bases,
    )
    drift = next(
        definition for definition in problem.parameter_definitions if definition.name == "component.0.drift_scale"
    )

    unit = encode_physical_vector(problem, {drift.name: drift.upper})
    values = values_by_name(problem, unit)
    rebuilt = rebuild_structure(structure, values)
    expand_structure(rebuilt, 1.5406)
