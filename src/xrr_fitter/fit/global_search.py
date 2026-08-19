"""Deterministic coarse preparation and differential-evolution search.

This module owns the bounded global-search state before stage orchestration.
It does not choose stage winners, publish candidates, or emit progress.

Coarse preparation preserves the semantic rows needed by global search:

- fitted endpoints are always retained;
- sharp log-reflectivity features replace only unprotected grid rows;
- every optional prepared-data array follows the same selected indices;
- candidate geometry products protect declared and one-axis hypotheses;
- large discrete products use one replayable Latin-hypercube stream;
- the two-angstrom thickness floor is enforced before solver evaluation.

Population construction then preserves explicit replay evidence:

- the supplied incumbent remains the first differential-evolution row;
- local Gaussian rows and global Latin-hypercube rows share one seed;
- Stage E represents every selected basin before adding global coverage;
- the final population and its aligned energies remain immutable;
- cancellation is polled inside the objective, not only between stages.

All stochastic choices therefore come from caller-owned seeds. Stage code can
combine these primitives without depending on SciPy defaults, global RNG state,
or data-row layouts that differ between a population and its objective.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from math import prod

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from xrr_fitter.fit.initialization import InitialCandidates
from xrr_fitter.fit.local_search import SearchCancelled
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.model.data import fit_ready
from xrr_fitter.model.fitting import ModelEvaluation
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    PeriodicBlock,
    StructureSpec,
)


def _readonly(value: object) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _ranked_feature_positions(data: object, by_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qz = data.qz_a_inv[by_q]
    intensity = np.log10(np.clip(data.intensity_normalized[by_q], data.r_floor, np.inf))
    span = qz[2:] - qz[:-2]
    fraction = np.divide(
        qz[1:-1] - qz[:-2],
        span,
        out=np.full(span.shape, 0.5),
        where=span > 0.0,
    )
    baseline = intensity[:-2] + fraction * (intensity[2:] - intensity[:-2])
    scores = np.abs(intensity[1:-1] - baseline)
    median = float(np.median(scores))
    deviation = float(np.median(np.abs(scores - median)))
    numerical_floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(intensity))))
    reliable = np.flatnonzero(scores > median + max(5.0 * deviation, numerical_floor)) + 1
    return qz, reliable[np.lexsort((reliable, -scores[reliable - 1]))]


def _spaced_features(ranked: np.ndarray, budget: int) -> tuple[int, ...]:
    selected: list[int] = []
    for value in ranked:
        position = int(value)
        if any(abs(position - chosen) <= 1 for chosen in selected):
            continue
        selected.append(position)
        if len(selected) == budget:
            break
    return tuple(selected)


def _insert_features(
    selected: set[int],
    protected: set[int],
    features: tuple[int, ...],
    qz: np.ndarray,
) -> None:
    for position in features:
        if position in selected:
            protected.add(position)
            continue
        replaceable = selected - protected
        if not replaceable:
            return
        nearest = min(
            replaceable,
            key=lambda current: (abs(qz[current] - qz[position]), current),
        )
        selected.remove(nearest)
        selected.add(position)
        protected.add(position)


def feature_grid_indices(data: object, max_points: int = 128) -> np.ndarray:
    """Select a bounded q grid while retaining endpoints and sharp features."""
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points must be an integer of at least two")
    if max_points < 2:
        raise ValueError("max_points must be an integer of at least two")
    fit_indices = np.flatnonzero(data.fit_mask)
    if fit_indices.size == 0:
        raise ValueError("fit data contain no enabled points")
    if fit_indices.size <= max_points:
        return np.sort(fit_indices)
    by_q = fit_indices[np.argsort(data.qz_a_inv[fit_indices], kind="stable")]
    positions = np.rint(np.linspace(0, by_q.size - 1, max_points)).astype(int)
    selected = {int(position) for position in positions}
    protected = {0, by_q.size - 1}
    qz, ranked = _ranked_feature_positions(data, by_q)
    budget = min(max(1, max_points // 4), max_points - 2)
    _insert_features(selected, protected, _spaced_features(ranked, budget), qz)
    return np.sort(by_q[np.asarray(sorted(selected), dtype=int)])


def _optional_rows(values: np.ndarray | None, selected: np.ndarray) -> np.ndarray | None:
    return None if values is None else values[selected]


def downsample_prepared_data(data: object, indices: np.ndarray) -> object:
    """Return immutable prepared data with every derived row field aligned."""
    selected = np.asarray(indices, dtype=int)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("downsample indices must be a nonempty vector")
    if np.any((selected < 0) | (selected >= data.qz_a_inv.size)):
        raise ValueError("downsample index out of range")
    angles = data.two_theta_deg[selected]
    qz = data.qz_a_inv[selected]
    mask = data.fit_mask[selected]
    return replace(
        data,
        source_row_groups=tuple(data.source_row_groups[int(index)] for index in selected),
        two_theta_deg=angles,
        intensity_raw=data.intensity_raw[selected],
        intensity_sigma_raw=_optional_rows(data.intensity_sigma_raw, selected),
        resolution_raw=_optional_rows(data.resolution_raw, selected),
        qz_a_inv=qz,
        intensity_normalized=data.intensity_normalized[selected],
        intensity_sigma_normalized=_optional_rows(
            data.intensity_sigma_normalized,
            selected,
        ),
        sigma_q_a_inv=_optional_rows(data.sigma_q_a_inv, selected),
        validation_mask=data.validation_mask[selected],
        fit_mask=mask,
        fit_ready=fit_ready(angles, qz, mask),
    )


def _ratio_variants(
    declared_thickness: np.ndarray,
    rng: np.random.Generator,
    binary_defaults: tuple[float, ...],
) -> tuple[np.ndarray, ...]:
    """Build deterministic layer fractions before geometry products."""
    base = declared_thickness / declared_thickness.sum()
    variants = [base, np.full(base.size, 1.0 / base.size)]
    if base.size == 2:
        variants.extend(np.array([fraction, 1.0 - fraction], dtype=float) for fraction in binary_defaults)
    for index in range(base.size):
        for change in (-0.15, 0.15):
            candidate = base.copy()
            candidate[index] = max(0.02, candidate[index] + change)
            candidate /= candidate.sum()
            variants.append(candidate)
    if base.size > 2:
        seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        unit = qmc.LatinHypercube(d=base.size, seed=seed).random(8)
        positive = -np.log(np.clip(unit, 1e-12, 1.0))
        variants.extend(positive / positive.sum(axis=1, keepdims=True))
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for variant in variants:
        array = np.asarray(variant, dtype=float)
        unique.setdefault(tuple(np.round(array, 12)), array)
    return tuple(unique.values())


def _ordinary_components(
    structure: StructureSpec,
) -> list[tuple[int, LayerSpec | GradientLayerSpec]]:
    return [
        (index, component)
        for index, component in enumerate(structure.components)
        if isinstance(component, (LayerSpec, GradientLayerSpec))
    ]


def _ordinary_geometry_group(
    ordinary: list[tuple[int, LayerSpec | GradientLayerSpec]],
    initial: InitialCandidates,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Build ordinary-layer geometries from totals and layer ratios."""
    declared = np.array([component.thickness_a for _, component in ordinary], dtype=float)
    ratios = _ratio_variants(declared, rng, initial.layer_fractions)
    totals = initial.thickness_a or (float(declared.sum()),)
    declared_geometry = tuple(
        (f"component.{index}.thickness_a", float(component.thickness_a)) for index, component in ordinary
    )
    generated = tuple(
        tuple(
            (f"component.{index}.thickness_a", float(total * ratio))
            for (index, _component), ratio in zip(ordinary, ratio_vector, strict=True)
        )
        for total in totals
        for ratio_vector in ratios
        if np.all(total * ratio_vector >= 2.0)
    )
    return tuple(dict.fromkeys((declared_geometry, *generated)))


def _periodic_geometry_group(
    component_index: int,
    block: PeriodicBlock,
    initial: InitialCandidates,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Build one periodic block's shared-cell thickness hypotheses."""
    declared = np.array([layer.thickness_a for layer in block.layers], dtype=float)
    ratios = _ratio_variants(declared, rng, initial.layer_fractions)
    periods = initial.period_a or (float(declared.sum()),)
    declared_geometry = tuple(
        (
            f"component.{component_index}.layer.{index}.thickness_a",
            float(layer.thickness_a),
        )
        for index, layer in enumerate(block.layers)
    )
    generated = tuple(
        tuple(
            (
                f"component.{component_index}.layer.{index}.thickness_a",
                float(period * ratio),
            )
            for index, ratio in enumerate(ratio_vector)
        )
        for period in periods
        for ratio_vector in ratios
        if np.all(period * ratio_vector >= 2.0)
    )
    return tuple(dict.fromkeys((declared_geometry, *generated)))


def _axis_cover(
    sizes: tuple[int, ...],
    center: tuple[int, ...],
    selected: set[tuple[int, ...]],
    limit: int,
) -> bool:
    for dimension, size in enumerate(sizes):
        for option in range(size):
            candidate = list(center)
            candidate[dimension] = option
            selected.add(tuple(candidate))
            if len(selected) >= limit:
                return True
    return False


def _latin_hypercube_fill(
    sizes: tuple[int, ...],
    selected: set[tuple[int, ...]],
    rng: np.random.Generator,
    limit: int,
) -> None:
    seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    sampler = qmc.LatinHypercube(d=len(sizes), seed=seed)
    stagnant = 0
    while len(selected) < limit and stagnant < 32:
        before = len(selected)
        for row in sampler.random(max(32, limit - len(selected))):
            selected.add(tuple(min(int(value * size), size - 1) for value, size in zip(row, sizes, strict=True)))
            if len(selected) == limit:
                return
        stagnant = stagnant + 1 if len(selected) == before else 0


def _cartesian_fill(
    sizes: tuple[int, ...],
    selected: set[tuple[int, ...]],
    limit: int,
) -> None:
    for indices in product(*(range(size) for size in sizes)):
        selected.add(indices)
        if len(selected) == limit:
            return


def _validate_product_dimensions(sizes: tuple[int, ...]) -> None:
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("index-product dimensions must be nonempty")


def bounded_index_product(
    sizes: tuple[int, ...],
    rng: np.random.Generator,
    limit: int,
) -> tuple[tuple[int, ...], ...]:
    """Select a capped product while protecting corners and axis coverage."""
    _validate_product_dimensions(sizes)
    if prod(sizes) <= limit:
        return tuple(product(*(range(size) for size in sizes)))
    center = tuple(size // 2 for size in sizes)
    selected = {center, tuple(0 for _ in sizes), tuple(size - 1 for size in sizes)}
    if _axis_cover(sizes, center, selected, limit):
        return tuple(sorted(selected)[:limit])
    _latin_hypercube_fill(sizes, selected, rng, limit)
    if len(selected) < limit:
        _cartesian_fill(sizes, selected, limit)
    return tuple(sorted(selected))


def _geometry_groups(
    structure: StructureSpec,
    initial: InitialCandidates,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[tuple[str, float], ...], ...], ...]:
    groups: list[tuple[tuple[tuple[str, float], ...], ...]] = []
    ordinary = _ordinary_components(structure)
    if ordinary:
        groups.append(_ordinary_geometry_group(ordinary, initial, rng))
    for index, component in enumerate(structure.components):
        if isinstance(component, PeriodicBlock):
            groups.append(_periodic_geometry_group(index, component, initial, rng))
    return tuple(groups)


def geometry_variants(
    structure: StructureSpec,
    initial: InitialCandidates,
    rng: np.random.Generator,
    limit: int = 128,
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    """Combine independently derived ordinary and periodic geometries."""
    groups = _geometry_groups(structure, initial, rng)
    if not groups:
        return (("declared-geometry", ()),)
    if any(not group for group in groups):
        raise ValueError("no legal geometry variants within the 2 Å hard bound")
    indices = bounded_index_product(tuple(len(group) for group in groups), rng, limit)
    return tuple(
        (
            "geometry-" + "-".join(str(index) for index in option_indices),
            tuple(
                value
                for group_index, option_index in enumerate(option_indices)
                for value in groups[group_index][option_index]
            ),
        )
        for option_indices in indices
    )


def _validated_center(value: np.ndarray, field: str) -> np.ndarray:
    center = np.asarray(value, dtype=float)
    valid = (
        center.ndim == 1
        and center.size > 0
        and np.all(np.isfinite(center))
        and np.all((center >= 0.0) & (center <= 1.0))
    )
    if not valid:
        raise ValueError(f"{field} must be a nonempty finite unit vector")
    return center


def _validated_population_size(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 5:
        raise ValueError(f"{field} must be an integer of at least five")
    return int(value)


def build_de_population(
    start: np.ndarray,
    *,
    seed: int,
    population_size: int,
) -> np.ndarray:
    """Build one centered, replayable Gaussian/LHS population."""
    center = _validated_center(start, "DE start")
    size = _validated_population_size(population_size, "DE population_size")
    rng = np.random.default_rng(seed)
    gaussian_count = round(0.75 * (size - 1))
    gaussian = np.clip(
        center + rng.normal(0.0, 0.1, size=(gaussian_count, center.size)),
        0.0,
        1.0,
    )
    latin = qmc.LatinHypercube(d=center.size, seed=rng).random(size - 1 - gaussian_count)
    return _readonly(np.vstack((center, gaussian, latin)))


def _validated_centers(centers: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    if not centers:
        raise ValueError("Stage-E population requires at least one center")
    values = tuple(_validated_center(center, "Stage-E center") for center in centers)
    if len({center.shape for center in values}) != 1:
        raise ValueError("Stage-E centers must share one unit-vector layout")
    return values


def build_stage_e_population(
    centers: tuple[np.ndarray, ...],
    *,
    seed: int,
    population_size: int,
    perturbations_per_center: int = 2,
) -> np.ndarray:
    """Seed every Stage-E basin and retain global Latin-hypercube rows."""
    values = _validated_centers(centers)
    size = _validated_population_size(population_size, "Stage-E population_size")
    valid_count = (
        not isinstance(perturbations_per_center, bool)
        and isinstance(perturbations_per_center, (int, np.integer))
        and perturbations_per_center >= 0
    )
    if not valid_count:
        raise ValueError("perturbations_per_center must be a nonnegative integer")
    seeded_count = len(values) * (int(perturbations_per_center) + 1)
    if seeded_count >= size:
        raise ValueError("Stage-E population must retain at least one LHS point")
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for center in values:
        rows.append(center)
        rows.extend(
            np.clip(
                center
                + rng.normal(
                    0.0,
                    0.1,
                    size=(int(perturbations_per_center), center.size),
                ),
                0.0,
                1.0,
            )
        )
    latin = qmc.LatinHypercube(d=values[0].size, seed=rng).random(size - seeded_count)
    return _readonly(np.vstack((*rows, latin)))


@dataclass(frozen=True, slots=True)
class GlobalSearchResult:
    unit_vector: np.ndarray
    evaluation: ModelEvaluation
    population: np.ndarray
    population_energies: np.ndarray
    trace: tuple[float, ...]
    stop_reason: str
    nfev: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_vector", _readonly(self.unit_vector))
        object.__setattr__(self, "population", _readonly(self.population))
        energies = _readonly(self.population_energies)
        if energies.shape != (self.population.shape[0],):
            raise ValueError("population energies must align with population rows")
        object.__setattr__(self, "population_energies", energies)
        object.__setattr__(self, "trace", tuple(float(value) for value in self.trace))


def _validate_layout(problem: object, start: np.ndarray, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = np.asarray(start, dtype=float)
    members = np.asarray(population, dtype=float)
    width = len(problem.variables)
    valid_start = (
        unit.ndim == 1
        and unit.shape == (width,)
        and np.all(np.isfinite(unit))
        and np.all((unit >= 0.0) & (unit <= 1.0))
    )
    valid_population = (
        members.ndim == 2
        and members.shape[1:] == (width,)
        and members.shape[0] >= 5
        and np.all(np.isfinite(members))
        and np.all((members >= 0.0) & (members <= 1.0))
    )
    if not valid_start:
        raise ValueError("start must be a finite unit vector with the compiled shape and bounds")
    if not valid_population:
        raise ValueError("population must be a finite unit matrix with the compiled shape and bounds")
    return np.array(unit, copy=True), np.array(members, copy=True)


def _validated_nonnegative_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _scipy_seed(seed: int) -> int | np.random.Generator:
    value = _validated_nonnegative_int(seed, "seed")
    if value <= np.iinfo(np.uint32).max:
        return value
    return np.random.default_rng(value)


def solve_global(
    problem: object,
    start: np.ndarray,
    *,
    population: np.ndarray,
    seed: int,
    maxiter: int,
    cancelled: Callable[[], bool] | None = None,
) -> GlobalSearchResult:
    """Run SciPy DE with a caller-supplied, replayable initial population."""
    unit, members = _validate_layout(problem, start, population)
    scipy_seed = _scipy_seed(seed)
    iteration_limit = _validated_nonnegative_int(maxiter, "maxiter")
    trace: list[float] = []

    def objective(value: np.ndarray) -> float:
        if cancelled is not None and cancelled():
            raise SearchCancelled("search cancelled")
        result = evaluate_vector(problem, value)
        trace.append(result.objective)
        return result.objective

    optimized = differential_evolution(
        objective,
        tuple((0.0, 1.0) for _ in problem.variables),
        x0=unit,
        init=members,
        seed=scipy_seed,
        maxiter=iteration_limit,
        updating="deferred",
        polish=False,
        workers=1,
    )
    result_unit = np.asarray(optimized.x, dtype=float)
    evaluation = evaluate_vector(problem, result_unit)
    final_population = np.asarray(getattr(optimized, "population", members), dtype=float)
    population_energies = np.asarray(optimized.population_energies, dtype=float)
    return GlobalSearchResult(
        result_unit,
        evaluation,
        final_population,
        population_energies,
        tuple(trace),
        str(optimized.message),
        int(optimized.nfev),
    )
