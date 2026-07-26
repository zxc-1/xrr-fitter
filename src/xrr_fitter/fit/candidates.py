"""Deterministic candidate construction, deduplication, and publication."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from math import isfinite, prod

import numpy as np
from scipy.stats import qmc

from xrr_fitter.fit.initialization import InitialCandidates, estimate_initial_candidates
from xrr_fitter.fit.problem import CompiledFitProblem
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.fitting import FitCandidate, ModelEvaluation
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    PeriodicBlock,
    StructureSpec,
)
from xrr_fitter.physics.sld_profile import sld_depth_profile


CURVE_MERGE_DECADES = 0.02
PARAMETER_PRECLUSTER_DISTANCE = 0.25


@dataclass(frozen=True, slots=True)
class CandidateStart:
    values: tuple[tuple[str, float], ...]
    feature_key: str

    def value(self, name: str) -> float:
        return dict(self.values)[name]


def _readonly(value: object, dtype: type | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_perturbation_center(center: np.ndarray) -> None:
    if center.ndim != 1 or not np.all(np.isfinite(center)):
        raise ValueError("perturbation center must be a finite unit vector")
    if np.any((center < 0.0) | (center > 1.0)):
        raise ValueError("perturbation center is outside unit bounds")


def _validate_perturbation_count(count: object) -> None:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("perturbation count must be a nonnegative integer")


def _validate_perturbation_sigma(sigma: float) -> None:
    if not isfinite(sigma) or sigma < 0.0:
        raise ValueError("perturbation sigma must be finite and nonnegative")


def bounded_perturbations(
    center: np.ndarray,
    count: int,
    *,
    seed: int,
    sigma: float = 0.1,
) -> tuple[np.ndarray, ...]:
    """Draw clipped perturbations without mutating the carried center.

    An empty coordinate layout has one logical center but no useful perturbed
    vectors, so it returns an empty tuple before touching the RNG stream.
    """
    unit_center = np.asarray(center, dtype=float)
    _validate_perturbation_center(unit_center)
    _validate_perturbation_count(count)
    _validate_perturbation_sigma(sigma)
    if unit_center.size == 0:
        return ()
    perturbed = np.clip(
        unit_center
        + np.random.default_rng(seed).normal(
            0.0,
            sigma,
            size=(int(count), unit_center.size),
        ),
        0.0,
        1.0,
    )
    return tuple(_readonly(vector, float) for vector in perturbed)


def _ratio_variants(
    declared_thickness: np.ndarray,
    rng: np.random.Generator,
    binary_defaults: tuple[float, ...],
) -> tuple[np.ndarray, ...]:
    """Build deterministic layer fractions before geometry products.

    Declared and equal fractions are protected. Binary defaults and one-axis
    offsets retain interpretable starts; larger stacks add seeded simplex LHS.
    """
    base = declared_thickness / declared_thickness.sum()
    variants = [base, np.full(base.size, 1.0 / base.size)]
    if base.size == 2:
        variants.extend(
            np.array([fraction, 1.0 - fraction], dtype=float)
            for fraction in binary_defaults
        )
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
    """Build ordinary-layer geometries from total thickness and layer ratios.

    The exact declared geometry is always first and survives generated dedup.
    """
    declared = np.array([component.thickness_a for _, component in ordinary], dtype=float)
    ratios = _ratio_variants(declared, rng, initial.layer_fractions)
    totals = initial.thickness_a or (float(declared.sum()),)
    declared_geometry = tuple(
        (f"component.{index}.thickness_a", float(component.thickness_a))
        for index, component in ordinary
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
    """Build one periodic block's shared-cell thickness hypotheses.

    Every returned geometry preserves a legal two-angstrom minimum per layer.
    """
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
    """Add every one-axis option around the center in stable axis order."""
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
    """Fill a bounded discrete product with the versioned LHS retry policy.

    The sampler is seeded from the caller's stream exactly once. A batch stops
    as soon as the cap is reached, preserving the historical selected set.
    """
    seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    sampler = qmc.LatinHypercube(d=len(sizes), seed=seed)
    stagnant = 0
    while len(selected) < limit and stagnant < 32:
        before = len(selected)
        for row in sampler.random(max(32, limit - len(selected))):
            selected.add(
                tuple(
                    min(int(value * size), size - 1)
                    for value, size in zip(row, sizes, strict=True)
                )
            )
            if len(selected) == limit:
                return
        stagnant = stagnant + 1 if len(selected) == before else 0


def _cartesian_fill(
    sizes: tuple[int, ...],
    selected: set[tuple[int, ...]],
    limit: int,
) -> None:
    """Deterministically complete any points LHS could not discover."""
    for indices in product(*(range(size) for size in sizes)):
        selected.add(indices)
        if len(selected) == limit:
            return


def _validate_index_sizes(sizes: tuple[int, ...]) -> None:
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("index-product dimensions must be nonempty")


def _bounded_index_product(
    sizes: tuple[int, ...],
    rng: np.random.Generator,
    limit: int,
) -> tuple[tuple[int, ...], ...]:
    """Select a capped product while protecting corners and axis coverage.

    Products below the cap retain canonical itertools order. Larger products
    protect center/corners, then axis probes, LHS points, and a stable fallback.
    """
    _validate_index_sizes(sizes)
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


def _geometry_variants(
    structure: StructureSpec,
    initial: InitialCandidates,
    rng: np.random.Generator,
    limit: int = 128,
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    """Combine independently derived ordinary and periodic geometries.

    One source group maps to one index dimension. The bounded product therefore
    cannot accidentally couple periods belonging to separate periodic blocks.
    """
    groups = _geometry_groups(structure, initial, rng)
    if not groups:
        return (("declared-geometry", ()),)
    if any(not group for group in groups):
        raise ValueError("no legal geometry variants within the 2 Å hard bound")
    indices = _bounded_index_product(tuple(len(group) for group in groups), rng, limit)
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


def _ordinary_interface_values(
    component_index: int,
    component: LayerSpec | GradientLayerSpec,
    previous_thickness: float | None,
    density_scale: float,
    roughness_fraction: float,
    geometry: dict[str, float],
) -> tuple[list[tuple[str, float]], float]:
    prefix = f"component.{component_index}"
    thickness = geometry.get(f"{prefix}.thickness_a", component.thickness_a)
    effective = thickness if previous_thickness is None else min(previous_thickness, thickness)
    values: list[tuple[str, float]] = []
    if isinstance(component, LayerSpec):
        values.append((f"{prefix}.density_scale", density_scale))
    values.append((f"{prefix}.roughness_a", roughness_fraction * effective))
    return values, thickness


def _periodic_interface_values(
    component_index: int,
    component: PeriodicBlock,
    previous_thickness: float | None,
    density_scale: float,
    roughness_fraction: float,
    geometry: dict[str, float],
) -> tuple[list[tuple[str, float]], float]:
    """Map shared periodic starts onto their tight adjacent interfaces.

    The first interface may see the preceding component and last layer of the
    repeated cell; an explicit block-top roughness is handled separately.
    """
    thicknesses = tuple(
        geometry.get(
            f"component.{component_index}.layer.{index}.thickness_a",
            layer.thickness_a,
        )
        for index, layer in enumerate(component.layers)
    )
    values: list[tuple[str, float]] = []
    for index, _layer in enumerate(component.layers):
        prefix = f"component.{component_index}.layer.{index}"
        thickness = thicknesses[index]
        if index:
            effective = min(thicknesses[index - 1], thickness)
        else:
            neighbors = [thickness]
            if component.top_roughness_a is None and previous_thickness is not None:
                neighbors.append(previous_thickness)
            if component.repeats > 1:
                neighbors.append(thicknesses[-1])
            effective = min(neighbors)
        values.extend(
            (
                (f"{prefix}.density_scale", density_scale),
                (f"{prefix}.roughness_a", roughness_fraction * effective),
            )
        )
    if component.top_roughness_a is not None:
        effective = (
            thicknesses[0]
            if previous_thickness is None
            else min(previous_thickness, thicknesses[0])
        )
        values.append(
            (
                f"component.{component_index}.top_roughness_a",
                roughness_fraction * effective,
            )
        )
    return values, thicknesses[-1]


def _material_and_interface_values(
    structure: StructureSpec,
    geometry_values: tuple[tuple[str, float], ...],
    density_scale: float,
    roughness_fraction: float,
) -> tuple[tuple[str, float], ...]:
    """Apply one material/roughness hypothesis across declared components.

    Component traversal carries the preceding finite thickness between groups.
    """
    geometry = dict(geometry_values)
    values: list[tuple[str, float]] = []
    previous: float | None = None
    for index, component in enumerate(structure.components):
        if isinstance(component, (LayerSpec, GradientLayerSpec)):
            component_values, previous = _ordinary_interface_values(
                index,
                component,
                previous,
                density_scale,
                roughness_fraction,
                geometry,
            )
        elif isinstance(component, PeriodicBlock):
            component_values, previous = _periodic_interface_values(
                index,
                component,
                previous,
                density_scale,
                roughness_fraction,
                geometry,
            )
        else:
            continue
        values.extend(component_values)
    return tuple(values)


def _make_start(
    structure: StructureSpec,
    geometry: tuple[str, tuple[tuple[str, float], ...]],
    density_scale: float,
    roughness_fraction: float,
    angle_offset: float,
    scale: float,
    background: float,
    relative_sigma: float,
    footprint_angle_deg: float,
) -> CandidateStart:
    """Combine one geometry with material, interface, and instrument values.

    Sorting makes the immutable start independent of intermediate append order.
    """
    feature_key, geometry_values = geometry
    values = list(geometry_values)
    values.extend(
        _material_and_interface_values(
            structure,
            geometry_values,
            density_scale,
            roughness_fraction,
        )
    )
    values.extend(
        (
            ("instrument.angle_offset_deg", angle_offset),
            ("instrument.scale", scale),
            ("instrument.background", background),
            ("instrument.relative_sigma", relative_sigma),
            ("instrument.footprint_spill_angle_deg", footprint_angle_deg),
        )
    )
    return CandidateStart(tuple(sorted(values)), feature_key)


def _baseline_component_values(
    component_index: int,
    component: LayerSpec | GradientLayerSpec | PeriodicBlock,
) -> list[tuple[str, float]]:
    """Flatten one declared component without inventing inactive parameters.

    Periodic cells retain their shared source names and optional top interface.
    """
    prefix = f"component.{component_index}"
    values: list[tuple[str, float]] = []
    if isinstance(component, (LayerSpec, GradientLayerSpec)):
        values.append((f"{prefix}.thickness_a", component.thickness_a))
        if isinstance(component, LayerSpec):
            values.append((f"{prefix}.density_scale", component.density_scale))
        values.append((f"{prefix}.roughness_a", component.roughness_a))
    elif isinstance(component, PeriodicBlock):
        for index, layer in enumerate(component.layers):
            layer_prefix = f"{prefix}.layer.{index}"
            values.extend(
                (
                    (f"{layer_prefix}.thickness_a", layer.thickness_a),
                    (f"{layer_prefix}.density_scale", layer.density_scale),
                    (f"{layer_prefix}.roughness_a", layer.roughness_a),
                )
            )
        if component.top_roughness_a is not None:
            values.append((f"{prefix}.top_roughness_a", component.top_roughness_a))
    return values


def _declared_baseline_start(
    data: PreparedData,
    structure: StructureSpec,
    instrument: InstrumentSpec,
) -> CandidateStart:
    """Publish the complete declared state as the protected first start."""
    values: list[tuple[str, float]] = []
    for index, component in enumerate(structure.components):
        values.extend(_baseline_component_values(index, component))
    values.extend(
        (
            ("backing.roughness_a", structure.backing_roughness_a),
            ("instrument.angle_offset_deg", data.import_angle_offset_deg),
            ("instrument.scale", 1.0),
            ("instrument.background", 0.0),
            ("instrument.relative_sigma", 0.0),
            ("instrument.footprint_spill_angle_deg", instrument.footprint_spill_angle_deg),
        )
    )
    return CandidateStart(tuple(sorted(values)), "declared-baseline")


def _validate_candidate_limit(limit: object) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("candidate limit must be a positive integer")


def _selected_combinations(
    dimensions: tuple[tuple[object, ...], ...],
    rng: np.random.Generator,
    limit: int,
) -> Iterable[tuple[object, ...]]:
    if prod(len(values) for values in dimensions) <= limit:
        return product(*dimensions)
    indices = _bounded_index_product(tuple(len(values) for values in dimensions), rng, limit)
    return (
        tuple(dimensions[dimension][option] for dimension, option in enumerate(index_tuple))
        for index_tuple in indices
    )


def build_candidate_pool(
    data: PreparedData,
    structure: StructureSpec,
    instrument: InstrumentSpec,
    rng: np.random.Generator,
    limit: int = 512,
) -> tuple[CandidateStart, ...]:
    """Build a deterministic capped pool with a protected declared baseline.

    Feature grids are combined only after each geometry family is bounded. The
    caller's RNG controls every stochastic choice without global state.
    """
    _validate_candidate_limit(limit)
    baseline = _declared_baseline_start(data, structure, instrument)
    if limit == 1:
        return (baseline,)
    initial = estimate_initial_candidates(data, structure, instrument, rng)
    geometry = _geometry_variants(structure, initial, rng)
    dimensions = (
        initial.density_scales,
        initial.roughness_fractions,
        initial.angle_offsets_deg,
        initial.scales,
        initial.backgrounds,
        initial.relative_resolutions,
        initial.footprint_angles_deg,
    )
    all_dimensions: tuple[tuple[object, ...], ...] = (geometry, *dimensions)
    generated_limit = limit - 1
    combinations = _selected_combinations(all_dimensions, rng, generated_limit)
    generated = tuple(
        _make_start(structure, *combination)
        for combination in combinations
    )
    return (baseline, *generated[:generated_limit])


def _start_distance(first: CandidateStart, second: CandidateStart) -> float:
    first_values = dict(first.values)
    second_values = dict(second.values)
    shared = sorted(set(first_values) & set(second_values))
    if not shared:
        return float("inf")
    left = np.array([first_values[name] for name in shared], dtype=float)
    right = np.array([second_values[name] for name in shared], dtype=float)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-12)
    return float(np.sqrt(np.mean(((left - right) / scale) ** 2)))


def _curve_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _connected_components(
    indices: tuple[int, ...],
    linked: object,
) -> tuple[tuple[int, ...], ...]:
    """Return stable graph components for a symmetric candidate relation.

    Minimum-index roots and sorted frontiers make membership order reproducible.
    """
    remaining = set(indices)
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        frontier = [root]
        component = {root}
        while frontier:
            current = frontier.pop()
            neighbors = {index for index in remaining if linked(current, index)}
            remaining -= neighbors
            component |= neighbors
            frontier.extend(sorted(neighbors))
        components.append(tuple(sorted(component)))
    return tuple(components)


def _validated_scored_starts(
    scored: tuple[tuple[float, CandidateStart], ...],
) -> tuple[CandidateStart, ...]:
    starts = tuple(start for _cost, start in scored)
    if len(set(starts)) != len(starts):
        raise ValueError("scored starts must be unique")
    if any(not np.isfinite(cost) for cost, _start in scored):
        raise ValueError("candidate costs must be finite")
    return starts


def _coerced_curves(
    starts: tuple[CandidateStart, ...],
    curves: dict[CandidateStart, np.ndarray],
) -> dict[CandidateStart, np.ndarray]:
    if set(starts) != set(curves):
        raise ValueError("coarse_log_curves must contain exactly the scored starts")
    arrays = {start: np.asarray(curves[start], dtype=float) for start in starts}
    if any(curve.ndim != 1 or curve.size == 0 for curve in arrays.values()):
        raise ValueError("coarse log curves must be nonempty vectors")
    if len({curve.shape for curve in arrays.values()}) > 1:
        raise ValueError("coarse log curves must have the same shape")
    if any(np.any(~np.isfinite(curve)) for curve in arrays.values()):
        raise ValueError("coarse log curves must be finite")
    return arrays


def _validated_curves(
    scored: tuple[tuple[float, CandidateStart], ...],
    curves: dict[CandidateStart, np.ndarray],
) -> dict[CandidateStart, np.ndarray]:
    """Validate cost/curve ownership before clustering any candidates.

    Every scored start must own one finite, nonempty curve of the same shape;
    accepting a partial mapping would make deduplication order-dependent.
    """
    return _coerced_curves(_validated_scored_starts(scored), curves)


def _representative(
    scored: tuple[tuple[float, CandidateStart], ...],
    indices: tuple[int, ...],
) -> tuple[float, CandidateStart]:
    return min(
        (scored[index] for index in indices),
        key=lambda item: (item[0], item[1].feature_key, item[1].values),
    )


def _deduplicated(
    scored: tuple[tuple[float, CandidateStart], ...],
    coarse_log_curves: dict[CandidateStart, np.ndarray],
) -> tuple[tuple[float, CandidateStart], ...]:
    """Merge locally preclustered curves, then merge global degeneracies.

    Parameter preclustering bounds pairwise work but cannot hide far-apart
    parameter vectors that produce the same reflectivity curve.
    """
    curves = _validated_curves(scored, coarse_log_curves)
    ordered = tuple(
        sorted(scored, key=lambda item: (item[0], item[1].feature_key, item[1].values))
    )
    parameter_clusters = _connected_components(
        tuple(range(len(ordered))),
        lambda first, second: _start_distance(ordered[first][1], ordered[second][1])
        <= PARAMETER_PRECLUSTER_DISTANCE,
    )
    first_level: list[tuple[float, CandidateStart]] = []
    for cluster in parameter_clusters:
        families = _connected_components(
            cluster,
            lambda first, second: _curve_distance(
                curves[ordered[first][1]], curves[ordered[second][1]]
            )
            < CURVE_MERGE_DECADES,
        )
        first_level.extend(_representative(ordered, family) for family in families)
    provisional = tuple(
        sorted(first_level, key=lambda item: (item[0], item[1].feature_key, item[1].values))
    )
    global_families = _connected_components(
        tuple(range(len(provisional))),
        lambda first, second: _curve_distance(
            curves[provisional[first][1]], curves[provisional[second][1]]
        )
        < CURVE_MERGE_DECADES,
    )
    return tuple(
        sorted(
            (_representative(provisional, family) for family in global_families),
            key=lambda item: (item[0], item[1].feature_key, item[1].values),
        )
    )


def select_coarse_candidates(
    scored: tuple[tuple[float, CandidateStart], ...],
    coarse_log_curves: dict[CandidateStart, np.ndarray],
    limit: int = 24,
) -> tuple[CandidateStart, ...]:
    """Reserve the best representative of each feature before filling by cost.

    Curve deduplication runs first so feature protection never retains clones.
    """
    if limit < 1:
        return ()
    deduplicated = _deduplicated(scored, coarse_log_curves)
    best_by_feature: dict[str, tuple[float, CandidateStart]] = {}
    for item in deduplicated:
        best_by_feature.setdefault(item[1].feature_key, item)
    representatives = sorted(
        best_by_feature.values(),
        key=lambda item: (item[0], item[1].feature_key, item[1].values),
    )
    selected = [start for _cost, start in representatives[:limit]]
    for _cost, start in deduplicated:
        if start not in selected:
            selected.append(start)
        if len(selected) == limit:
            break
    return tuple(selected)


def select_full_search_candidates(
    scored: tuple[tuple[float, CandidateStart], ...],
    coarse_log_curves: dict[CandidateStart, np.ndarray],
    limit: int = 8,
) -> tuple[CandidateStart, ...]:
    """Protect the declared baseline while selecting full-resolution starts.

    Remaining slots follow deduplicated objective order with no random tie break.
    """
    if limit < 1:
        return ()
    deduplicated = _deduplicated(scored, coarse_log_curves)
    baseline_items = tuple(
        item for item in scored if item[1].feature_key == "declared-baseline"
    )
    if not baseline_items:
        return tuple(start for _cost, start in deduplicated[:limit])
    baseline = min(baseline_items, key=lambda item: (item[0], item[1].values))[1]
    remaining = tuple(
        start
        for _cost, start in deduplicated
        if start != baseline
        and _curve_distance(coarse_log_curves[start], coarse_log_curves[baseline])
        >= CURVE_MERGE_DECADES
    )
    return (baseline, *remaining[: limit - 1])


def candidate_from_evaluation(
    problem: CompiledFitProblem,
    unit_vector: np.ndarray,
    evaluation: ModelEvaluation,
    candidate_id: str,
    seed_index: int,
    stop_reason: str,
    nfev: int,
) -> FitCandidate:
    full_log = np.full(problem.data.qz_a_inv.shape, np.nan, dtype=float)
    full_weighted = np.full(problem.data.qz_a_inv.shape, np.nan, dtype=float)
    full_log[problem.data.fit_mask] = evaluation.fit_log_residuals_decades
    full_weighted[problem.data.fit_mask] = evaluation.fit_weighted_residuals
    if evaluation.expanded_stack is None:
        depth = np.empty(0, dtype=float)
        profile = np.empty(0, dtype=np.complex128)
    else:
        depth, profile = sld_depth_profile(evaluation.expanded_stack)
    return FitCandidate(
        candidate_id=candidate_id,
        seed_index=seed_index,
        unit_vector=_readonly(unit_vector, float),
        parameters=evaluation.parameters,
        objective=evaluation.objective,
        valid=evaluation.valid,
        stop_reason=stop_reason if evaluation.valid else evaluation.reason,
        nfev=nfev,
        qz_a_inv=evaluation.qz_a_inv,
        model_normalized=evaluation.model_normalized,
        log_residuals_decades=_readonly(full_log, float),
        weighted_residuals=_readonly(full_weighted, float),
        expanded_stack=evaluation.expanded_stack,
        sld_depth_a=_readonly(depth, float),
        sld_profile_a2=_readonly(profile, complex),
        diagnostics=evaluation.diagnostics,
    )
