"""Derived period and composition profiles for binary periodic stacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from xrr_fitter.evaluation import (
    EvaluationConstraintError,
    cached_least_squares_callbacks,
    evaluate_model,
    least_squares_loss,
    least_squares_system,
    physical_to_unit,
    values_by_name,
)
from xrr_fitter.model.fitting import FitEvaluationContext
from xrr_fitter.model.structure import PeriodicBlock


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class BinaryDerivedProfile:
    name: str
    kind: str
    first_name: str
    second_name: str
    first_index: int
    second_index: int


@dataclass(frozen=True, slots=True)
class _Bounds:
    first_definition: object
    second_definition: object
    first_lower: float
    first_upper: float
    second_lower: float
    second_upper: float
    period_lower: float
    period_upper: float
    fraction_lower: float
    fraction_upper: float


@dataclass(frozen=True, slots=True)
class _State:
    problem: FitEvaluationContext
    specification: BinaryDerivedProfile
    bounds: _Bounds


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
        BinaryDerivedProfile(
            f"component.{component_index}.layer.0.fraction", "fraction", *common
        ),
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


def _bounds(problem: object, specification: BinaryDerivedProfile) -> _Bounds:
    first = problem.parameter_definitions[
        problem.variables[specification.first_index].parameter_index
    ]
    second = problem.parameter_definitions[
        problem.variables[specification.second_index].parameter_index
    ]
    if min(first.lower, second.lower) <= 0.0:
        raise ValueError("derived periodic profiles require positive thickness bounds")
    return _Bounds(
        first,
        second,
        first.lower,
        first.upper,
        second.lower,
        second.upper,
        first.lower + second.lower,
        first.upper + second.upper,
        first.lower / (first.lower + second.upper),
        first.upper / (first.upper + second.lower),
    )


def _scaled_value(unit: float, lower: float, upper: float) -> float:
    return float(lower + unit * (upper - lower))


def _scaled_unit(value: float, lower: float, upper: float) -> float:
    span = upper - lower
    if span <= np.finfo(float).eps * max(1.0, abs(lower), abs(upper)):
        return 0.5
    return float(np.clip((value - lower) / span, 0.0, 1.0))


def _fraction_bounds(bounds: _Bounds, period: float) -> tuple[float, float]:
    lower = max(bounds.first_lower / period, 1.0 - bounds.second_upper / period)
    upper = min(bounds.first_upper / period, 1.0 - bounds.second_lower / period)
    if lower > upper:
        if lower - upper > 1e-12:
            raise ValueError("period has no feasible binary-layer fraction")
        lower = upper = 0.5 * (lower + upper)
    return float(lower), float(upper)


def _period_bounds(bounds: _Bounds, fraction: float) -> tuple[float, float]:
    complement = 1.0 - fraction
    lower = max(bounds.first_lower / fraction, bounds.second_lower / complement)
    upper = min(bounds.first_upper / fraction, bounds.second_upper / complement)
    if lower > upper:
        tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
        if lower - upper > tolerance:
            raise ValueError("fraction has no feasible binary-layer period")
        lower = upper = 0.5 * (lower + upper)
    return float(lower), float(upper)


def _physical_pair(state: _State, transformed: np.ndarray) -> tuple[float, float]:
    primary = transformed[state.specification.first_index]
    nuisance = transformed[state.specification.second_index]
    if state.specification.kind == "period":
        period = _scaled_value(primary, state.bounds.period_lower, state.bounds.period_upper)
        fraction = _scaled_value(nuisance, *_fraction_bounds(state.bounds, period))
    else:
        fraction = _scaled_value(
            primary, state.bounds.fraction_lower, state.bounds.fraction_upper
        )
        period = _scaled_value(nuisance, *_period_bounds(state.bounds, fraction))
    return period, fraction


def _decode(state: _State, transformed_unit: np.ndarray) -> np.ndarray:
    transformed = np.asarray(transformed_unit, dtype=float)
    valid = (
        transformed.shape == (len(state.problem.variables),)
        and np.all(np.isfinite(transformed))
        and np.all((transformed >= 0.0) & (transformed <= 1.0))
    )
    if not valid:
        raise ValueError("derived profile coordinate is outside unit bounds")
    period, fraction = _physical_pair(state, transformed)
    first = float(np.clip(period * fraction, state.bounds.first_lower, state.bounds.first_upper))
    second = float(
        np.clip(
            period * (1.0 - fraction),
            state.bounds.second_lower,
            state.bounds.second_upper,
        )
    )
    original = np.array(transformed, copy=True)
    original[state.specification.first_index] = physical_to_unit(
        state.bounds.first_definition, first
    )
    original[state.specification.second_index] = physical_to_unit(
        state.bounds.second_definition, second
    )
    return original


def _encode(state: _State, original_unit: np.ndarray) -> np.ndarray:
    original = np.asarray(original_unit, dtype=float)
    values = values_by_name(state.problem, original)
    first = values[state.specification.first_name]
    second = values[state.specification.second_name]
    period, fraction = first + second, first / (first + second)
    transformed = np.array(original, copy=True)
    if state.specification.kind == "period":
        primary = _scaled_unit(period, state.bounds.period_lower, state.bounds.period_upper)
        nuisance = _scaled_unit(fraction, *_fraction_bounds(state.bounds, period))
    else:
        primary = _scaled_unit(
            fraction, state.bounds.fraction_lower, state.bounds.fraction_upper
        )
        nuisance = _scaled_unit(period, *_period_bounds(state.bounds, fraction))
    transformed[state.specification.first_index] = primary
    transformed[state.specification.second_index] = nuisance
    return transformed


def _coordinate_jacobian(state: _State, transformed: np.ndarray) -> np.ndarray:
    values = np.asarray(transformed, dtype=float)
    jacobian = np.eye(values.size)
    rows = (state.specification.first_index, state.specification.second_index)
    jacobian[list(rows), :] = 0.0
    for coordinate in rows:
        lower_step = min(1e-6, float(values[coordinate]))
        upper_step = min(1e-6, 1.0 - float(values[coordinate]))
        step = lower_step + upper_step
        if step <= 0.0:
            continue
        lower, upper = values.copy(), values.copy()
        lower[coordinate] -= lower_step
        upper[coordinate] += upper_step
        jacobian[:, coordinate] = (_decode(state, upper) - _decode(state, lower)) / step
    return jacobian


def decode_binary_coordinate(
    problem: FitEvaluationContext,
    name: str,
    primary_value: float,
    nuisance_value: float,
) -> tuple[float, float]:
    specifications = {item.name: item for item in binary_derived_profiles(problem)}
    if name not in specifications:
        raise ValueError(f"unknown binary profile: {name}")
    specification = specifications[name]
    if specification.kind == "period":
        period, fraction = float(primary_value), float(nuisance_value)
    else:
        fraction, period = float(primary_value), float(nuisance_value)
    bounds = _bounds(problem, specification)
    if not bounds.period_lower <= period <= bounds.period_upper:
        raise ValueError("binary period is outside feasible bounds")
    if not bounds.fraction_lower <= fraction <= bounds.fraction_upper:
        raise ValueError("binary fraction is outside feasible bounds")
    first, second = period * fraction, period * (1.0 - fraction)
    if not (
        bounds.first_lower <= first <= bounds.first_upper
        and bounds.second_lower <= second <= bounds.second_upper
    ):
        raise ValueError("binary coordinate has no feasible thickness pair")
    return float(first), float(second)


def build_binary_profile(
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
    state = _State(problem, specification, _bounds(problem, specification))
    center = _encode(state, np.asarray(unit_vector, dtype=float))

    def observe(transformed: np.ndarray) -> np.ndarray:
        original = _decode(state, transformed)
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
        return residual, jacobian @ _coordinate_jacobian(state, transformed)

    residual, jacobian = cached_least_squares_callbacks(system)

    def reported(transformed: np.ndarray) -> float:
        primary = transformed[specification.first_index]
        if specification.kind == "period":
            return _scaled_value(primary, state.bounds.period_lower, state.bounds.period_upper)
        return _scaled_value(primary, state.bounds.fraction_lower, state.bounds.fraction_upper)

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
