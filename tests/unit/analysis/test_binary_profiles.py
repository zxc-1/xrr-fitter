from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from tests.support.model_cases import prepared_data, simple_structure
from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    encode_physical_vector,
    values_by_name,
)
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock


def _api():
    return import_module("xrr_fitter.analysis.binary_profiles")


def binary_derived_profiles(*args, **kwargs):
    return _api().binary_derived_profiles(*args, **kwargs)


def build_binary_profile(*args, **kwargs):
    profile_builder = import_module("xrr_fitter.analysis.profiles").profile_parameter
    return _api().build_binary_profile(*args, profile_builder=profile_builder, **kwargs)


def decode_binary_coordinate(*args, **kwargs):
    return _api().decode_binary_coordinate(*args, **kwargs)


def _periodic_problem():
    base = simple_structure()
    layer = base.components[0]
    assert isinstance(layer, LayerSpec)
    block = PeriodicBlock(
        "binary",
        (
            replace(layer, name="a", thickness_a=28.0),
            replace(layer, name="b", thickness_a=42.0),
        ),
        repeats=5,
        top_roughness_a=2.0,
    )
    structure = replace(base, components=(block,))
    initial = compile_fit_problem(
        prepared_data(size=64),
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(907), scale_prior_enabled=False),
    )
    targets = {
        "component.0.layer.0.thickness_a",
        "component.0.layer.1.thickness_a",
    }
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name not in targets,
        )
        for definition in initial.parameter_definitions
    )
    return compile_fit_problem(
        initial.data, structure, initial.instrument, initial.config, settings
    )


def test_profile_selection_adds_binary_period_and_fraction_profiles() -> None:
    specifications = binary_derived_profiles(_periodic_problem())

    assert tuple(value.name for value in specifications) == (
        "component.0.period_a",
        "component.0.layer.0.fraction",
    )


@pytest.mark.parametrize(
    "name",
    (
        pytest.param("component.0.period_a", id="period"),
        pytest.param("component.0.layer.0.fraction", id="fraction"),
    ),
)
def test_binary_derived_profile_holds_the_reported_quantity_while_nuisance_moves(
    name: str,
) -> None:
    problem = _periodic_problem()
    center = encode_physical_vector(problem, {})
    observed: list[tuple[float, float, float]] = []

    profile = build_binary_profile(
        problem,
        center,
        name,
        observer=lambda derived, first, second: observed.append(
            (derived, first, second)
        ),
    )

    assert profile.name == name
    assert profile.values.size >= 9
    assert len(observed) >= profile.values.size
    for derived, first, second in observed:
        expected = first + second if name.endswith("period_a") else first / (first + second)
        assert derived == pytest.approx(expected)
    assert np.ptp(np.asarray([(first, second) for _, first, second in observed]), axis=0).max() > 0.0


def test_binary_coordinate_decode_preserves_period_and_fraction_constraints() -> None:
    problem = _periodic_problem()
    center = values_by_name(problem, encode_physical_vector(problem, {}))

    period = decode_binary_coordinate(problem, "component.0.period_a", 70.0, 0.4)
    fraction = decode_binary_coordinate(
        problem, "component.0.layer.0.fraction", 0.4, 70.0
    )

    assert sum(period) == pytest.approx(70.0)
    assert fraction[0] / sum(fraction) == pytest.approx(0.4)
    assert center["component.0.layer.0.thickness_a"] == pytest.approx(28.0)


def test_binary_profile_treats_physical_constraint_failures_as_unsupported_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _api()
    problem = _periodic_problem()
    center = encode_physical_vector(problem, {})
    first_index = next(
        index
        for index, variable in enumerate(problem.variables)
        if variable.name == "component.0.layer.0.thickness_a"
    )
    original = module.evaluate_model

    def constrained(problem_value, unit):
        if np.asarray(unit, dtype=float)[first_index] < 0.05:
            raise EvaluationConstraintError("constraint_violation:ValueError")
        return original(problem_value, unit)

    monkeypatch.setattr(module, "evaluate_model", constrained)

    profile = module.build_binary_profile(
        problem,
        center,
        "component.0.period_a",
        profile_builder=import_module("xrr_fitter.analysis.profiles").profile_parameter,
    )

    assert profile.values.size >= 9
    assert np.any(np.isfinite(profile.objectives))
    assert np.any(~np.isfinite(profile.objectives))


def test_binary_profile_propagates_unsupported_jacobian_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _api()
    problem = _periodic_problem()
    center = encode_physical_vector(problem, {})

    def unsupported(*_args):
        raise ValueError("unsupported derivative layout")

    def exercise(_objective, transformed, **options):
        options["residual_jacobian"](transformed)
        return None

    monkeypatch.setattr(module, "least_squares_system", unsupported)

    with pytest.raises(ValueError, match="unsupported derivative layout"):
        module.build_binary_profile(
            problem,
            center,
            "component.0.period_a",
            profile_builder=exercise,
        )


def test_binary_profile_includes_active_scale_prior_in_solver_rows() -> None:
    module = _api()
    problem = replace(
        _periodic_problem(),
        scale_prior_center=1.0,
        scale_prior_reason=None,
        warnings=(),
    )
    center = encode_physical_vector(problem, {})
    observed: dict[str, np.ndarray] = {}

    def capture(_objective, transformed, **options):
        observed["residual"] = options["residual"](transformed)
        observed["jacobian"] = options["residual_jacobian"](transformed)
        return None

    module.build_binary_profile(
        problem,
        center,
        "component.0.period_a",
        profile_builder=capture,
    )

    expected_rows = int(np.count_nonzero(problem.data.fit_mask)) + 1
    assert observed["residual"].shape == (expected_rows,)
    assert observed["jacobian"].shape == (expected_rows, len(problem.variables))
