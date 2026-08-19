"""Derived period and composition profiles for binary periodic stacks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from xrr_fitter.analysis.binary_coordinates import (
    BinaryDerivedProfile,
    binary_coordinate_jacobian,
    binary_coordinate_state,
    binary_reported_value,
    decode_binary_pair,
    decode_binary_unit,
    encode_binary_unit,
)
from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    cached_least_squares_callbacks,
    evaluate_model,
    least_squares_loss,
    least_squares_system,
    values_by_name,
)
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.structure import PeriodicBlock


def _component_profiles(
    component_index: int,
    indices: dict[str, int],
) -> tuple[BinaryDerivedProfile, ...]:
    prefix = f"component.{component_index}.layer"
    first, second = f"{prefix}.0.thickness_a", f"{prefix}.1.thickness_a"
    if first not in indices or second not in indices:
        return ()
    common = (first, second, indices[first], indices[second])
    return (
        BinaryDerivedProfile(f"component.{component_index}.period_a", "period", *common),
        BinaryDerivedProfile(f"component.{component_index}.layer.0.fraction", "fraction", *common),
    )


def binary_derived_profiles(
    problem: FitEvaluationContext,
) -> tuple[BinaryDerivedProfile, ...]:
    names = tuple(variable.name for variable in problem.variables)
    indices = {name: index for index, name in enumerate(names)}
    structure = getattr(problem, "structure", None)
    if structure is None:
        return ()
    profiles: list[BinaryDerivedProfile] = []
    for index, component in enumerate(structure.components):
        if isinstance(component, PeriodicBlock) and len(component.layers) == 2:
            profiles.extend(_component_profiles(index, indices))
    return tuple(profiles)


def decode_binary_coordinate(
    problem: FitEvaluationContext,
    name: str,
    primary_value: float,
    nuisance_value: float,
) -> tuple[float, float]:
    specifications = {item.name: item for item in binary_derived_profiles(problem)}
    if name not in specifications:
        raise ValueError(f"unknown binary profile: {name}")
    state = binary_coordinate_state(problem, specifications[name])
    return decode_binary_pair(state, primary_value, nuisance_value)


def build_binary_profile[T](
    problem: FitEvaluationContext,
    unit_vector: np.ndarray,
    name: str,
    *,
    profile_builder: Callable[..., T],
    observer: Callable[[float, float, float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> T:
    specifications = {item.name: item for item in binary_derived_profiles(problem)}
    if name not in specifications:
        raise ValueError(f"unknown binary profile: {name}")
    specification = specifications[name]
    state = binary_coordinate_state(problem, specification)
    center = encode_binary_unit(state, np.asarray(unit_vector, dtype=float))

    def observe(transformed: np.ndarray) -> np.ndarray:
        original = decode_binary_unit(state, transformed)
        if observer is not None:
            values = values_by_name(problem, original)
            first = values[specification.first_name]
            second = values[specification.second_name]
            derived = first + second if specification.kind == "period" else first / (first + second)
            observer(float(derived), float(first), float(second))
        return original

    def objective(transformed: np.ndarray) -> float:
        try:
            evaluation = evaluate_model(problem, observe(transformed))
        except EvaluationConstraintError:
            return np.inf
        return evaluation.objective if evaluation.valid else np.inf

    def system(transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        original = observe(transformed)
        residual, jacobian = least_squares_system(problem, original)
        return residual, jacobian @ binary_coordinate_jacobian(state, transformed)

    residual, jacobian = cached_least_squares_callbacks(system)

    def reported(transformed: np.ndarray) -> float:
        return binary_reported_value(state, transformed)

    maximum = max(
        problem.config.budget.local_min_nfev,
        problem.config.budget.local_nfev_per_parameter * max(1, len(problem.variables)),
    )
    steps = 11 if problem.config.budget.bootstrap_samples < 100 else 41
    return profile_builder(
        objective,
        center,
        parameter_index=specification.first_index,
        name=name,
        value_mapper=reported,
        residual=residual,
        residual_jacobian=jacobian,
        residual_loss=least_squares_loss(problem),
        least_squares_max_nfev=maximum,
        steps=steps,
        cancelled=cancelled,
    )
