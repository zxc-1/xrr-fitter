from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.structure import DriftSpec, LayerSpec, PeriodicBlock, StructureSpec


def test_dynamic_roughness_upper_below_declared_lower_is_an_invalid_candidate() -> None:
    initial = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=183), scale_prior_enabled=False),
    )
    roughness_name = "component.0.roughness_a"
    roughness = next(definition for definition in initial.parameter_definitions if definition.name == roughness_name)
    problem = compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        (
            ParameterSetting(
                roughness_name,
                4.5,
                4.0,
                roughness.upper,
            ),
        ),
    )
    indices = {coordinate.name: index for index, coordinate in enumerate(problem.variables)}
    unit = np.full(len(problem.variables), 0.5)
    unit[indices["component.0.thickness_a"]] = 0.0
    unit[indices[roughness_name]] = 1.0

    result = evaluate_vector(problem, unit)
    residual, jacobian = evaluation.least_squares_system(problem, unit)

    assert result.valid is False
    assert result.reason == "constraint_violation:PhysicalValueError"
    np.testing.assert_array_equal(residual, np.full(residual.shape, 1e6))
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))


def test_encode_physical_vector_handles_a_singleton_dynamic_roughness_domain() -> None:
    initial = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=184), scale_prior_enabled=False),
    )
    roughness_name = "component.0.roughness_a"
    roughness = next(definition for definition in initial.parameter_definitions if definition.name == roughness_name)
    problem = compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        (ParameterSetting(roughness_name, 4.0, 4.0, roughness.upper),),
    )
    thickness = np.nextafter(4.0, np.inf) / 0.49

    unit = encode_physical_vector(
        problem,
        {
            "component.0.thickness_a": thickness,
            roughness_name: 4.0,
        },
    )
    indices = {coordinate.name: index for index, coordinate in enumerate(problem.variables)}
    values = evaluation.values_by_name(problem, unit)

    assert unit[indices[roughness_name]] == 0.0
    assert values[roughness_name] == 4.0


def test_dynamic_roughness_decode_does_not_expand_material_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=21), scale_prior_enabled=False),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dynamic roughness decoding expanded a material stack")

    monkeypatch.setattr(evaluation, "expand_structure", forbidden)

    values = evaluation.values_by_name(problem, np.full(len(problem.variables), 0.5))

    assert values["component.0.roughness_a"] > 0.0


def test_single_repeat_periodic_latent_layer_roughness_decodes_for_primal_and_jacobian() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "single",
        (replace(film, name="only", roughness_a=3.0),),
        repeats=1,
        top_roughness_a=2.0,
    )
    problem = compile_fit_problem(
        prepared_data(size=64),
        StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=4.0),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=211), scale_prior_enabled=False),
    )
    unit = np.full(len(problem.variables), 0.5)
    latent = "component.0.layer.0.roughness_a"

    values = evaluation.values_by_name(problem, unit)
    jacobian_values, value_jacobians = evaluation.values_and_jacobians(problem, unit)
    residual_jacobian = evaluation.evaluate_model_jacobian(problem, unit)

    assert values[latent] == pytest.approx(jacobian_values[latent])
    latent_index = next(index for index, coordinate in enumerate(problem.variables) if coordinate.name == latent)
    assert value_jacobians[latent][latent_index] > 0.0
    assert np.all(np.isfinite(residual_jacobian))
    np.testing.assert_allclose(residual_jacobian[:, latent_index], 0.0, atol=1e-12)


def test_missing_roughness_cap_mapping_is_rejected_for_public_coordinates() -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=22), scale_prior_enabled=False),
    )

    with pytest.raises(ValueError, match="roughness coordinate"):
        evaluation._fill_missing_roughness_caps(problem, {})


def test_roughness_drift_without_explicit_top_does_not_allow_missing_base_coordinate() -> None:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (replace(film, name="film", roughness_a=1.0),),
        repeats=3,
        drift=DriftSpec(kind="linear", target="roughness", amount=0.1),
    )
    problem = compile_fit_problem(
        prepared_data(size=48),
        StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=2.0),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=2201), scale_prior_enabled=False),
    )
    definitions = {
        definition.name: definition
        for definition in problem.parameter_definitions
        if definition.transform == "roughness_fraction"
    }
    missing = "component.0.layer.0.roughness_a"
    dynamic = {name: definition.upper for name, definition in definitions.items() if name != missing}

    with pytest.raises(ValueError, match="roughness coordinate mapping missing"):
        evaluation._fill_missing_roughness_caps(problem, dynamic)
