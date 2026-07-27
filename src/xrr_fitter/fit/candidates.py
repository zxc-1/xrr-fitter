"""Deterministic candidate construction, deduplication, and publication."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import product
from math import isfinite, prod

import numpy as np

from xrr_fitter.fit.global_search import bounded_index_product, geometry_variants
from xrr_fitter.fit.initialization import estimate_initial_candidates
from xrr_fitter.model.data import PreparedData
from xrr_fitter.model.fitting import FitCandidate, FitEvaluationContext, ModelEvaluation
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


@dataclass(frozen=True, slots=True)
class StageBArchive:
    active: tuple[FitCandidate, ...]
    archived: tuple[FitCandidate, ...]
    perturbation_counts: tuple[int, ...]


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
    indices = bounded_index_product(tuple(len(values) for values in dimensions), rng, limit)
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
    geometry = geometry_variants(structure, initial, rng)
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


def _effective_objective(candidate: FitCandidate) -> float:
    return (
        candidate.objective
        if candidate.ranking_objective is None
        else candidate.ranking_objective
    )


def rank_candidate_indices(
    candidates: tuple[FitCandidate, ...],
    *,
    eligible_ids: tuple[str, ...] | None = None,
) -> tuple[int, ...]:
    """Return selectable candidate indices in stable effective-cost order."""
    values = tuple(candidates)
    eligible = None if eligible_ids is None else frozenset(eligible_ids)
    selected = (
        index
        for index, candidate in enumerate(values)
        if candidate.valid
        and candidate.stop_reason != "early_eliminated"
        and isfinite(_effective_objective(candidate))
        and (eligible is None or candidate.candidate_id in eligible)
    )
    return tuple(
        sorted(selected, key=lambda index: (_effective_objective(values[index]), index))
    )


def best_candidate_index(
    candidates: tuple[FitCandidate, ...],
    *,
    eligible_ids: tuple[str, ...] | None = None,
) -> int | None:
    """Return the deterministic winner, or ``None`` for an empty scope."""
    ranked = rank_candidate_indices(candidates, eligible_ids=eligible_ids)
    return ranked[0] if ranked else None


def cluster_candidate_indices(
    candidates: tuple[FitCandidate, ...],
    *,
    distance: float,
) -> tuple[tuple[int, ...], ...]:
    """Build stable connected components in normalized parameter space."""
    values = tuple(candidates)
    if not isfinite(distance) or distance < 0.0:
        raise ValueError("cluster distance must be finite and nonnegative")
    if values and any(
        candidate.unit_vector.size != values[0].unit_vector.size
        for candidate in values
    ):
        raise ValueError("candidate unit vectors must have equal width")
    remaining = set(range(len(values)))
    clusters: list[tuple[int, ...]] = []
    while remaining:
        pending = [min(remaining)]
        component: list[int] = []
        remaining.remove(pending[0])
        while pending:
            current = pending.pop(0)
            component.append(current)
            neighbors = tuple(
                index
                for index in sorted(remaining)
                if np.linalg.norm(values[current].unit_vector - values[index].unit_vector)
                <= distance
            )
            pending.extend(neighbors)
            remaining.difference_update(neighbors)
        clusters.append(tuple(sorted(component)))
    return tuple(clusters)


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
    problem: FitEvaluationContext,
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


def _archived_candidate(candidate: FitCandidate) -> FitCandidate:
    return replace(candidate, seed_index=-1, stop_reason="early_eliminated")


def _validate_archive_policy(threshold_ratio: float, base_perturbations: int) -> None:
    if not isfinite(threshold_ratio) or threshold_ratio <= 1.0:
        raise ValueError("stage-B archive threshold must exceed one")
    if (
        isinstance(base_perturbations, bool)
        or not isinstance(base_perturbations, int)
        or base_perturbations < 0
    ):
        raise ValueError("base perturbation count must be a nonnegative integer")


def _ordered_archive_candidates(
    candidates: tuple[FitCandidate, ...],
) -> tuple[FitCandidate, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                not candidate.valid,
                candidate.objective,
                candidate.candidate_id,
            ),
        )
    )


def _is_selectable_archive_candidate(candidate: FitCandidate) -> bool:
    return candidate.valid and isfinite(candidate.objective)


def _partition_stage_b_archive(
    ordered: tuple[FitCandidate, ...],
    threshold_ratio: float,
) -> tuple[tuple[FitCandidate, ...], tuple[FitCandidate, ...]]:
    selectable = tuple(
        candidate
        for candidate in ordered
        if _is_selectable_archive_candidate(candidate)
    )
    if not selectable:
        return (), tuple(_archived_candidate(candidate) for candidate in ordered)
    cutoff = threshold_ratio * selectable[0].objective
    active = tuple(candidate for candidate in selectable if candidate.objective <= cutoff)
    active_ids = {candidate.candidate_id for candidate in active}
    archived = tuple(
        _archived_candidate(candidate)
        for candidate in ordered
        if candidate.candidate_id not in active_ids
    )
    return active, archived


def _reclaimed_perturbation_counts(
    active_count: int,
    archived_count: int,
    base_perturbations: int,
) -> tuple[int, ...]:
    counts = [base_perturbations] * active_count
    for index in range((base_perturbations + 1) * archived_count):
        counts[index % active_count] += 1
    return tuple(counts)


def archive_stage_b_candidates(
    candidates: tuple[FitCandidate, ...],
    *,
    threshold_ratio: float = 10.0,
    base_perturbations: int = 2,
) -> StageBArchive:
    """Archive hopeless Stage-B evidence and retain the local-start budget."""
    _validate_archive_policy(threshold_ratio, base_perturbations)
    ordered = _ordered_archive_candidates(candidates)
    active, archived = _partition_stage_b_archive(ordered, threshold_ratio)
    if not active:
        return StageBArchive((), archived, ())
    counts = _reclaimed_perturbation_counts(
        len(active),
        len(archived),
        base_perturbations,
    )
    return StageBArchive(active, archived, counts)
