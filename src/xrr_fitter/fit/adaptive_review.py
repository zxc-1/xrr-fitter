"""Progressive candidate review over frozen single or joint numerical contexts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from xrr_fitter.fit.adaptive_grid import grid_levels, initial_grid_points, review_candidate_indices, should_promote
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.feature_grid import feature_grid_indices
from xrr_fitter.fit.global_search import downsample_prepared_data
from xrr_fitter.fit.joint_constraint_compilation import definitions_by_reference
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.model.fitting import GridReview, SearchAllocation, SearchEvidence


def compile_grid_problem(problem: object, max_points: int) -> object:
    """Subset observations while preserving full-data priors, regions, and mass."""
    indices = feature_grid_indices(problem.data, max_points)
    if len(indices) == np.count_nonzero(problem.data.fit_mask):
        return problem
    data = downsample_prepared_data(problem.data, indices)
    mass = problem.sampling_multipliers[indices].copy()
    selected = data.fit_mask
    mass[selected] *= np.sum(problem.sampling_multipliers[problem.data.fit_mask]) / np.sum(mass[selected])
    return replace(
        problem,
        data=data,
        region_labels=problem.region_labels[indices],
        weights=problem.weights[indices],
        sampling_multipliers=mass,
    )


def single_grid_contexts(problem: object) -> tuple[object, ...]:
    minimum = initial_grid_points(problem.data)
    return tuple(
        compile_grid_problem(problem, level)
        for level in grid_levels(int(np.count_nonzero(problem.data.fit_mask)))
        if level >= minimum
    )


def joint_grid_contexts(problem: object) -> tuple[object, ...]:
    """Build aligned joint contexts without inventing rows for short members."""
    local_contexts = tuple(single_grid_contexts(member) for member in problem.problems)
    level_count = max(len(contexts) for contexts in local_contexts)
    result = []
    for level in range(level_count):
        members = tuple(contexts[min(level, len(contexts) - 1)] for contexts in local_contexts)
        result.append(replace(problem, problems=members))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ReviewedPopulation:
    indices: tuple[int, ...]
    objectives: np.ndarray
    reviews: tuple[GridReview, ...]

    def __post_init__(self) -> None:
        values = np.array(self.objectives, dtype=float, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "objectives", values)


def _declared_baseline_indices(evaluated) -> tuple[int, ...]:
    return tuple(
        index for index, (start, _candidate) in enumerate(evaluated) if start.feature_key == "declared-baseline"
    )


def review_stage_a(problem, evaluated, initial_evaluations, cancelled):
    if not evaluated:
        raise ValueError("stage A produced no valid fitting candidates")
    contexts = single_grid_contexts(problem)
    full_evaluations = {}

    def evaluate(level, unit):
        observed = evaluate_vector(contexts[level], unit, fit_only=True)
        if level == len(contexts) - 1:
            full_evaluations[tuple(unit)] = observed
        return observed.objective

    units = np.asarray([candidate.unit_vector for _start, candidate in evaluated])
    geometry = [
        index
        for index, coordinate in enumerate(problem.variables)
        if problem.parameter_definitions[coordinate.parameter_index].category == "structure"
    ]
    reviewed = review_on_grids(
        units,
        np.asarray([candidate.objective for _start, candidate in evaluated]),
        tuple(candidate.candidate_id for _start, candidate in evaluated),
        tuple((int(np.count_nonzero(context.data.fit_mask)),) for context in contexts),
        evaluate,
        initial_evaluations=initial_evaluations,
        geometry=units[:, list(geometry)],
        cancelled=cancelled,
        mandatory_indices=_declared_baseline_indices(evaluated),
    )
    candidates = tuple(
        (
            evaluated[index][0],
            candidate_from_evaluation(
                problem,
                units[index],
                full_evaluations[tuple(units[index])],
                evaluated[index][1].candidate_id,
                -1,
                "full_grid_reviewed",
                1,
            ),
        )
        for index in reviewed.indices
    )
    evidence = SearchEvidence(
        "initial-candidate-pool", problem.config.master_seed, reviewed.reviews, (), "full_grid_reviewed"
    )
    return candidates, evidence


def _poll(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SearchCancelled("search cancelled")


def _cached_full_cost(
    index: int,
    units: np.ndarray,
    complete: dict[tuple[float, ...], float],
    full_costs: dict[int, float],
    evaluate: Callable[[int, np.ndarray], float],
    full_index: int,
    cancelled: Callable[[], bool] | None,
) -> float:
    _poll(cancelled)
    key = tuple(units[index])
    if key not in complete:
        complete[key] = float(evaluate(full_index, units[index]))
    full_costs[index] = complete[key]
    return complete[key]


def _grid_level(
    level: int,
    full_index: int,
    units: np.ndarray,
    coordinates: np.ndarray,
    costs: np.ndarray,
    candidate_ids: tuple[str, ...],
    initial_evaluations: int,
    complete: dict[tuple[float, ...], float],
    full_costs: dict[int, float],
    evaluate: Callable[[int, np.ndarray], float],
    cancelled: Callable[[], bool] | None,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    grid_count = initial_evaluations if level == 0 else 0
    if level == full_index:
        values = np.asarray(
            [
                _cached_full_cost(index, units, complete, full_costs, evaluate, full_index, cancelled)
                for index in range(len(units))
            ]
        )
        return values, tuple(range(len(units))), grid_count
    if level:
        costs = np.asarray([_review_cost(evaluate, level, row, cancelled) for row in units])
        grid_count = len(units)
    selected = review_candidate_indices(coordinates, costs, candidate_ids)
    return costs, selected, grid_count


def _grid_review(
    points: tuple[int, ...],
    selected: tuple[int, ...],
    costs: np.ndarray,
    checked: np.ndarray,
    promoted: bool,
    grid_count: int,
    full_count: int,
    candidate_ids: tuple[str, ...],
) -> GridReview:
    return GridReview(
        points,
        tuple(candidate_ids[index] for index in selected),
        tuple(float(value) for value in costs[list(selected)]),
        tuple(float(value) for value in checked),
        promoted,
        grid_count,
        full_count,
    )


def _validated_mandatory_indices(values, candidate_count: int) -> tuple[int, ...]:
    indices = tuple(values)
    valid = all(
        isinstance(index, (int, np.integer))
        and not isinstance(index, (bool, np.bool_))
        and 0 <= index < candidate_count
        for index in indices
    )
    if not valid or len(set(indices)) != len(indices):
        raise ValueError("mandatory review indices must be unique candidate indices")
    return tuple(int(index) for index in indices)


def review_on_grids(
    unit_vectors: np.ndarray,
    objectives: np.ndarray,
    candidate_ids: tuple[str, ...],
    grid_points: tuple[tuple[int, ...], ...],
    evaluate: Callable[[int, np.ndarray], float],
    *,
    initial_evaluations: int = 0,
    geometry: np.ndarray | None = None,
    cancelled: Callable[[], bool] | None = None,
    mandatory_indices: tuple[int, ...] = (),
) -> ReviewedPopulation:
    """Review before elimination; retain every reviewed full-cost winner."""
    units = np.asarray(unit_vectors, dtype=float)
    coordinates = units if geometry is None else np.asarray(geometry, dtype=float)
    costs = np.array(objectives, dtype=float, copy=True)
    review_candidate_indices(coordinates, costs, candidate_ids)
    mandatory = _validated_mandatory_indices(mandatory_indices, len(units))
    if not grid_points:
        raise ValueError("review requires at least one numerical grid")
    full_index = len(grid_points) - 1
    complete: dict[tuple[float, ...], float] = {}
    full_costs: dict[int, float] = {}
    reviews = []

    for level, points in enumerate(grid_points):
        _poll(cancelled)
        before = len(complete)
        costs, selected, grid_count = _grid_level(
            level,
            full_index,
            units,
            coordinates,
            costs,
            candidate_ids,
            initial_evaluations,
            complete,
            full_costs,
            evaluate,
            cancelled,
        )
        selected = tuple(dict.fromkeys((*selected, *mandatory)))
        checked = np.asarray(
            [
                _cached_full_cost(index, units, complete, full_costs, evaluate, full_index, cancelled)
                for index in selected
            ]
        )
        original = costs[list(selected)]
        promote = level < full_index and should_promote(original, checked)
        reviews.append(
            _grid_review(points, selected, costs, checked, promote, grid_count, len(complete) - before, candidate_ids)
        )
        if not promote:
            break
    _poll(cancelled)
    ranked = tuple(sorted(full_costs, key=lambda index: (full_costs[index], candidate_ids[index])))
    return ReviewedPopulation(ranked, np.asarray([full_costs[index] for index in ranked]), tuple(reviews))


def _review_cost(evaluate, level: int, unit: np.ndarray, cancelled) -> float:
    _poll(cancelled)
    return float(evaluate(level, unit))


def review_single_population(problem, solved, origin: str, seed: int, max_nfev: int, *, cancelled=None):
    """Rank a DE trace by full objective before selecting local-refinement starts."""
    contexts = single_grid_contexts(problem)
    population = np.asarray(solved.population, dtype=float)
    energies = getattr(solved, "population_energies", None)
    initial_count = 0
    if energies is None:
        energies = np.asarray([evaluate_vector(contexts[0], row, fit_only=True).objective for row in population])
        initial_count = len(population)
    ids = tuple(f"{origin}-population-{index:04d}" for index in range(len(population)))
    geometry = [
        index
        for index, coordinate in enumerate(problem.variables)
        if problem.parameter_definitions[coordinate.parameter_index].category == "structure"
    ]

    def evaluate(level, row):
        return evaluate_vector(contexts[level], row, fit_only=True).objective

    reviewed = review_on_grids(
        population,
        energies,
        ids,
        tuple((int(np.count_nonzero(context.data.fit_mask)),) for context in contexts),
        evaluate,
        initial_evaluations=initial_count,
        geometry=population[:, geometry],
        cancelled=cancelled,
    )
    allocation = SearchAllocation(origin, origin, 0, max_nfev, solved.nfev)
    evidence = SearchEvidence(origin, seed, reviewed.reviews, (allocation,), solved.stop_reason)
    return tuple(np.array(population[index], copy=True) for index in reviewed.indices), evidence


def _joint_structure_axes(problem) -> list[int]:
    definitions = definitions_by_reference(problem.dataset_ids, problem.problems)
    return [
        index
        for index, variable in enumerate(problem.global_variables)
        if any(definitions[member].category == "structure" for member in variable.members)
    ]


def review_joint_population(problem, solved, origin: str, seed: int, max_nfev: int, *, cancelled=None):
    """Rank a joint DE population on aligned progressively denser grids."""
    population = np.asarray(solved.population, dtype=float)
    energies = np.asarray(solved.population_energies, dtype=float)
    if population.ndim != 2 or population.shape[0] == 0 or energies.shape != (population.shape[0],):
        raise ValueError("joint DE trace has an invalid population layout")
    contexts = joint_grid_contexts(problem)
    ids = tuple(f"{origin}-population-{index:04d}" for index in range(len(population)))

    def evaluate(level, row):
        from xrr_fitter.fit.joint_evaluation import evaluate_joint_vector

        return evaluate_joint_vector(contexts[level], row, fit_only=True).objective

    reviewed = review_on_grids(
        population,
        energies,
        ids,
        tuple(
            tuple(int(np.count_nonzero(member.data.fit_mask)) for member in context.problems) for context in contexts
        ),
        evaluate,
        geometry=population[:, _joint_structure_axes(problem)],
        cancelled=cancelled,
    )
    allocation = SearchAllocation(origin, origin, 0, max_nfev, solved.nfev)
    evidence = SearchEvidence(origin, seed, reviewed.reviews, (allocation,), solved.stop_reason)
    return tuple(np.array(population[index], copy=True) for index in reviewed.indices), evidence
