"""Physical-coordinate projection for shared joint roughness variables."""

from __future__ import annotations

from statistics import median

import numpy as np

from xrr_fitter.evaluation import EvaluationConstraintError, roughness_dynamic_uppers, values_by_name
from xrr_fitter.model.parameters import PhysicalValueError, physical_to_unit, unit_to_physical

SHARED_ROUGHNESS_TRANSFORM = "shared_roughness_physical"


def _shared_unit_to_physical(definition: object, unit: float, dynamic_upper: float) -> float:
    try:
        return unit_to_physical(definition, unit, dynamic_upper=dynamic_upper)
    except PhysicalValueError as error:
        raise EvaluationConstraintError("constraint_violation:PhysicalValueError") from error


def _shared_physical_to_unit(definition: object, value: float, dynamic_upper: float) -> float:
    try:
        return physical_to_unit(definition, value, dynamic_upper=dynamic_upper)
    except PhysicalValueError as error:
        raise EvaluationConstraintError("constraint_violation:PhysicalValueError") from error


def _definition(problem: object, parameter_name: str) -> object:
    return next(definition for definition in problem.parameter_definitions if definition.name == parameter_name)


def _coordinate_index(problem: object, parameter_name: str) -> int:
    return next(index for index, coordinate in enumerate(problem.variables) if coordinate.name == parameter_name)


def _member_layouts(
    problem: object,
    local: list[np.ndarray],
    variable: object,
) -> tuple[tuple[int, int, object, float], ...]:
    dataset_indices = {dataset_id: index for index, dataset_id in enumerate(problem.dataset_ids)}
    layouts = []
    for member in variable.members:
        dataset_index = dataset_indices[member.dataset_id]
        local_problem = problem.problems[dataset_index]
        definition = _definition(local_problem, member.parameter_name)
        dynamic_upper = roughness_dynamic_uppers(
            local_problem,
            local[dataset_index],
        )[member.parameter_name]
        layouts.append(
            (
                dataset_index,
                _coordinate_index(local_problem, member.parameter_name),
                definition,
                dynamic_upper,
            )
        )
    return tuple(layouts)


def _common_upper(
    layouts: tuple[tuple[int, int, object, float], ...],
) -> float:
    return min(min(definition.upper, dynamic) for _, _, definition, dynamic in layouts)


def _encode_common_physical(
    definition: object,
    physical: float,
    common_upper: float,
) -> float:
    # A median outside the shared feasible geometry starts on its exact boundary.
    feasible = min(physical, common_upper)
    return _shared_physical_to_unit(
        definition,
        feasible,
        dynamic_upper=common_upper,
    )


def apply_shared_roughness(
    problem: object,
    unit: np.ndarray,
    local: list[np.ndarray],
) -> None:
    """Mutate local unit vectors so shared roughness decodes to one Å value."""
    for global_index, variable in enumerate(problem.global_variables):
        if variable.transform != SHARED_ROUGHNESS_TRANSFORM:
            continue
        layouts = _member_layouts(problem, local, variable)
        common_upper = _common_upper(layouts)
        physical = _shared_unit_to_physical(
            layouts[0][2],
            unit[global_index],
            dynamic_upper=common_upper,
        )
        for dataset_index, local_index, definition, dynamic in layouts:
            local[dataset_index][local_index] = _shared_physical_to_unit(
                definition,
                physical,
                dynamic_upper=dynamic,
            )


def initialize_shared_roughness(
    problem: object,
    global_unit: np.ndarray,
    local: list[np.ndarray],
) -> None:
    """Replace raw first-owner roughness starts with physical medians."""
    for global_index, variable in enumerate(problem.global_variables):
        if variable.transform != SHARED_ROUGHNESS_TRANSFORM:
            continue
        layouts = _member_layouts(problem, local, variable)
        physical = float(median(definition.initial for _, _, definition, _ in layouts))
        global_unit[global_index] = _encode_common_physical(
            layouts[0][2],
            physical,
            _common_upper(layouts),
        )


def apply_consensus_roughness(
    problem: object,
    consensus: np.ndarray,
    local: list[np.ndarray],
    physical_by_dataset: dict[str, dict[str, float]],
) -> None:
    """Replace raw unit medians with candidate physical roughness medians."""
    for global_index, variable in enumerate(problem.global_variables):
        if variable.transform != SHARED_ROUGHNESS_TRANSFORM:
            continue
        layouts = _member_layouts(problem, local, variable)
        physical = float(
            median(physical_by_dataset[member.dataset_id][member.parameter_name] for member in variable.members)
        )
        consensus[global_index] = _encode_common_physical(
            layouts[0][2],
            physical,
            _common_upper(layouts),
        )


def _physical_maps(
    problem: object,
    local_units: list[np.ndarray],
) -> tuple[dict[str, float], ...]:
    return tuple(
        values_by_name(local_problem, local)
        for local_problem, local in zip(
            problem.problems,
            local_units,
            strict=True,
        )
    )


def _member_physical_values(
    problem: object,
    variable: object,
    physical_maps: tuple[dict[str, float], ...],
) -> tuple[float, ...]:
    dataset_indices = {dataset_id: index for index, dataset_id in enumerate(problem.dataset_ids)}
    return tuple(
        physical_maps[dataset_indices[member.dataset_id]][member.parameter_name] for member in variable.members
    )


def rebuild_candidate_roughness(
    problem: object,
    global_unit: np.ndarray,
    local_units: list[np.ndarray],
) -> None:
    """Rebuild shared global roughness from projected local candidates."""
    physical_maps = _physical_maps(problem, local_units)
    for global_index, variable in enumerate(problem.global_variables):
        if variable.transform != SHARED_ROUGHNESS_TRANSFORM:
            continue
        layouts = _member_layouts(problem, local_units, variable)
        physical = _member_physical_values(problem, variable, physical_maps)
        if not np.allclose(physical, physical[0], rtol=1e-10, atol=1e-12):
            raise ValueError("joint candidate shared physical roughness projection mismatch")
        global_unit[global_index] = _encode_common_physical(
            layouts[0][2],
            float(median(physical)),
            _common_upper(layouts),
        )
