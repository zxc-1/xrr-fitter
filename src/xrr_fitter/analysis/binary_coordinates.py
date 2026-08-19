"""Feasible coordinate transforms for binary periodic-layer profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from xrr_fitter.evaluation import physical_to_unit, values_by_name
from xrr_fitter.model.fitting import FitEvaluationContext


@dataclass(frozen=True, slots=True)
class BinaryDerivedProfile:
    name: str
    kind: str
    first_name: str
    second_name: str
    first_index: int
    second_index: int


@dataclass(frozen=True, slots=True)
class BinaryCoordinateBounds:
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
class BinaryCoordinateState:
    problem: FitEvaluationContext
    specification: BinaryDerivedProfile
    bounds: BinaryCoordinateBounds


def _positive_sum(first: float, second: float) -> float:
    result = float(first) + float(second)
    if not isfinite(result):
        raise ValueError("derived periodic period bounds are not representable")
    return result


def _positive_ratio(numerator: float, denominator_part: float) -> float:
    numerator = float(numerator)
    denominator_part = float(denominator_part)
    if not (isfinite(numerator) and isfinite(denominator_part) and numerator > 0.0 and denominator_part > 0.0):
        raise ValueError("derived periodic fraction bounds are not representable")
    if numerator >= denominator_part:
        result = 1.0 / (1.0 + denominator_part / numerator)
    else:
        scaled = numerator / denominator_part
        result = scaled / (1.0 + scaled)
    if not 0.0 < result < 1.0:
        raise ValueError("derived periodic fraction bounds are not representable")
    return float(result)


def _coordinate_bounds(
    problem: object,
    specification: BinaryDerivedProfile,
) -> BinaryCoordinateBounds:
    first = problem.parameter_definitions[problem.variables[specification.first_index].parameter_index]
    second = problem.parameter_definitions[problem.variables[specification.second_index].parameter_index]
    thickness_bounds = (
        float(first.lower),
        float(first.upper),
        float(second.lower),
        float(second.upper),
    )
    if not all(isfinite(value) and value > 0.0 for value in thickness_bounds):
        raise ValueError("derived periodic profiles require positive thickness bounds")
    first_lower, first_upper, second_lower, second_upper = thickness_bounds
    period_lower = _positive_sum(first_lower, second_lower)
    period_upper = _positive_sum(first_upper, second_upper)
    fraction_lower = _positive_ratio(first_lower, second_upper)
    fraction_upper = _positive_ratio(first_upper, second_lower)
    if period_lower > period_upper or fraction_lower > fraction_upper:
        raise ValueError("derived periodic bounds are empty")
    return BinaryCoordinateBounds(
        first,
        second,
        first_lower,
        first_upper,
        second_lower,
        second_upper,
        period_lower,
        period_upper,
        fraction_lower,
        fraction_upper,
    )


def binary_coordinate_state(
    problem: FitEvaluationContext,
    specification: BinaryDerivedProfile,
) -> BinaryCoordinateState:
    return BinaryCoordinateState(problem, specification, _coordinate_bounds(problem, specification))


def _relative_difference(denominator: float, numerator: float) -> float:
    denominator = float(denominator)
    numerator = float(numerator)
    if numerator <= denominator:
        return float((denominator - numerator) / denominator)
    return float(-(numerator - denominator) / denominator)


def _scaled_value(unit: float, lower: float, upper: float) -> float:
    return float(lower + unit * (upper - lower))


def _scaled_unit(value: float, lower: float, upper: float) -> float:
    span = upper - lower
    if span <= np.finfo(float).eps * max(1.0, abs(lower), abs(upper)):
        return 0.5
    return float(np.clip((value - lower) / span, 0.0, 1.0))


def _fraction_bounds(
    bounds: BinaryCoordinateBounds,
    period: float,
) -> tuple[float, float]:
    period = float(period)
    if not isfinite(period) or period <= 0.0:
        raise ValueError("period is not representable for binary-layer fraction")
    lower = max(
        bounds.first_lower / period,
        _relative_difference(period, bounds.second_upper),
    )
    upper = min(
        bounds.first_upper / period,
        _relative_difference(period, bounds.second_lower),
    )
    if not (isfinite(lower) and isfinite(upper)):
        raise ValueError("period is not representable for binary-layer fraction")
    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, 0.0, 1.0))
    if lower > upper:
        if lower - upper > 1e-12:
            raise ValueError("period has no feasible binary-layer fraction")
        lower = upper = 0.5 * (lower + upper)
    return float(lower), float(upper)


def _period_bounds(
    bounds: BinaryCoordinateBounds,
    fraction: float,
) -> tuple[float, float]:
    fraction = float(fraction)
    if not isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("fraction is not representable for binary-layer period")
    complement = 1.0 - fraction
    lower = max(bounds.first_lower / fraction, bounds.second_lower / complement)
    upper = min(bounds.first_upper / fraction, bounds.second_upper / complement)
    if not (isfinite(lower) and isfinite(upper)):
        raise ValueError("fraction is not representable for binary-layer period")
    if lower > upper:
        tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
        if lower - upper > tolerance:
            raise ValueError("fraction has no feasible binary-layer period")
        lower = upper = 0.5 * (lower + upper)
    return float(lower), float(upper)


def _physical_pair(
    state: BinaryCoordinateState,
    transformed: np.ndarray,
) -> tuple[float, float]:
    primary = transformed[state.specification.first_index]
    nuisance = transformed[state.specification.second_index]
    if state.specification.kind == "period":
        period = _scaled_value(primary, state.bounds.period_lower, state.bounds.period_upper)
        fraction = _scaled_value(nuisance, *_fraction_bounds(state.bounds, period))
    else:
        fraction = _scaled_value(
            primary,
            state.bounds.fraction_lower,
            state.bounds.fraction_upper,
        )
        period = _scaled_value(nuisance, *_period_bounds(state.bounds, fraction))
    return period, fraction


def decode_binary_unit(
    state: BinaryCoordinateState,
    transformed_unit: np.ndarray,
) -> np.ndarray:
    transformed = np.asarray(transformed_unit, dtype=float)
    valid = (
        transformed.shape == (len(state.problem.variables),)
        and np.all(np.isfinite(transformed))
        and np.all((transformed >= 0.0) & (transformed <= 1.0))
    )
    if not valid:
        raise ValueError("derived profile coordinate is outside unit bounds")
    period, fraction = _physical_pair(state, transformed)
    first = float(
        np.clip(
            period * fraction,
            state.bounds.first_lower,
            state.bounds.first_upper,
        )
    )
    second = float(
        np.clip(
            period * (1.0 - fraction),
            state.bounds.second_lower,
            state.bounds.second_upper,
        )
    )
    original = np.array(transformed, copy=True)
    original[state.specification.first_index] = physical_to_unit(
        state.bounds.first_definition,
        first,
    )
    original[state.specification.second_index] = physical_to_unit(
        state.bounds.second_definition,
        second,
    )
    return original


def encode_binary_unit(
    state: BinaryCoordinateState,
    original_unit: np.ndarray,
) -> np.ndarray:
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
            fraction,
            state.bounds.fraction_lower,
            state.bounds.fraction_upper,
        )
        nuisance = _scaled_unit(period, *_period_bounds(state.bounds, fraction))
    transformed[state.specification.first_index] = primary
    transformed[state.specification.second_index] = nuisance
    return transformed


def binary_coordinate_jacobian(
    state: BinaryCoordinateState,
    transformed: np.ndarray,
) -> np.ndarray:
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
        jacobian[:, coordinate] = (decode_binary_unit(state, upper) - decode_binary_unit(state, lower)) / step
    return jacobian


def decode_binary_pair(
    state: BinaryCoordinateState,
    primary_value: float,
    nuisance_value: float,
) -> tuple[float, float]:
    if state.specification.kind == "period":
        period, fraction = float(primary_value), float(nuisance_value)
    else:
        fraction, period = float(primary_value), float(nuisance_value)
    bounds = state.bounds
    if not bounds.period_lower <= period <= bounds.period_upper:
        raise ValueError("binary period is outside feasible bounds")
    if not bounds.fraction_lower <= fraction <= bounds.fraction_upper:
        raise ValueError("binary fraction is outside feasible bounds")
    first, second = period * fraction, period * (1.0 - fraction)
    if not (bounds.first_lower <= first <= bounds.first_upper and bounds.second_lower <= second <= bounds.second_upper):
        raise ValueError("binary coordinate has no feasible thickness pair")
    return float(first), float(second)


def binary_reported_value(
    state: BinaryCoordinateState,
    transformed: np.ndarray,
) -> float:
    primary = transformed[state.specification.first_index]
    bounds = state.bounds
    if state.specification.kind == "period":
        return _scaled_value(primary, bounds.period_lower, bounds.period_upper)
    return _scaled_value(primary, bounds.fraction_lower, bounds.fraction_upper)


__all__ = [
    "BinaryCoordinateState",
    "BinaryDerivedProfile",
    "binary_coordinate_jacobian",
    "binary_coordinate_state",
    "binary_reported_value",
    "decode_binary_pair",
    "decode_binary_unit",
    "encode_binary_unit",
]
