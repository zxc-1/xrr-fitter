"""Validation and aggregation for deterministic bootstrap samples."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

import numpy as np

from xrr_fitter.model.analysis import BootstrapResult

BootstrapFit = Callable[[np.random.Generator, int], np.ndarray | None]
BootstrapProgress = Callable[[int, int], None]


class TaskRunner(Protocol):
    def __call__[T](
        self,
        tasks: tuple[Callable[[], T], ...],
        /,
    ) -> tuple[T, ...]: ...


def run_tasks[T](
    tasks: tuple[Callable[[], T], ...],
    task_runner: TaskRunner | None,
) -> tuple[T, ...]:
    results = tuple(task() for task in tasks) if task_runner is None else tuple(task_runner(tasks))
    if len(results) != len(tasks):
        raise RuntimeError("task runner returned an unexpected result count")
    return results


def validated_bootstrap_names(
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    names = tuple(parameter_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("bootstrap parameter_names must be nonempty strings")
    return names


def validated_sample_count(sample_count: int) -> int:
    valid = not isinstance(sample_count, bool) and isinstance(sample_count, (int, np.integer)) and sample_count >= 1
    if not valid:
        raise ValueError("bootstrap sample_count must be a positive integer")
    return int(sample_count)


def _collect_bootstrap_samples(
    names: tuple[str, ...],
    fitted_values: Iterable[np.ndarray | None],
    count: int,
    progress: BootstrapProgress | None,
) -> tuple[np.ndarray, int]:
    samples: list[np.ndarray] = []
    failures = 0
    observed = 0
    for sample_index, fitted in enumerate(fitted_values):
        if sample_index >= count:
            raise RuntimeError("bootstrap produced too many fitted samples")
        observed += 1
        if fitted is None:
            failures += 1
        else:
            vector = np.asarray(fitted, dtype=float)
            if vector.shape != (len(names),) or np.any(~np.isfinite(vector)):
                raise ValueError("bootstrap fit returned an invalid parameter vector")
            samples.append(vector)
        if progress is not None:
            progress(sample_index + 1, count)
    if observed != count:
        raise RuntimeError("bootstrap produced an unexpected fitted sample count")
    matrix = np.vstack(samples) if samples else np.empty((0, len(names)), dtype=float)
    return matrix, failures


def _stable_percentiles(matrix: np.ndarray) -> np.ndarray:
    probabilities = (0.025, 0.975)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        result = np.percentile(
            matrix,
            tuple(value * 100.0 for value in probabilities),
            axis=0,
        )
    if np.all(np.isfinite(result)):
        return result
    ordered = np.sort(matrix, axis=0)
    for row, probability in enumerate(probabilities):
        position = probability * (ordered.shape[0] - 1)
        lower_index = int(position)
        upper_index = min(lower_index + 1, ordered.shape[0] - 1)
        fraction = position - lower_index
        unstable = ~np.isfinite(result[row])
        left = ordered[lower_index, unstable]
        right = ordered[upper_index, unstable]
        scale = np.maximum(np.abs(left), np.abs(right))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            left_scaled = np.divide(
                left,
                scale,
                out=np.zeros_like(left),
                where=scale > 0.0,
            )
            right_scaled = np.divide(
                right,
                scale,
                out=np.zeros_like(right),
                where=scale > 0.0,
            )
            weighted = (1.0 - fraction) * left_scaled + fraction * right_scaled
            result[row, unstable] = scale * np.clip(weighted, -1.0, 1.0)
    if np.any(~np.isfinite(result)):
        raise ValueError("bootstrap percentile bounds are not representable")
    return result


def _bootstrap_intervals(
    names: tuple[str, ...],
    matrix: np.ndarray,
    failure_rate: float,
) -> tuple[tuple[str, float, float], ...]:
    if failure_rate > 0.20 or matrix.shape[0] == 0:
        return ()
    lower, upper = _stable_percentiles(matrix)
    return tuple((name, float(lower[index]), float(upper[index])) for index, name in enumerate(names))


def bootstrap_result_from_fits(
    names: tuple[str, ...],
    fitted_values: Iterable[np.ndarray | None],
    count: int,
    progress: BootstrapProgress | None,
) -> BootstrapResult:
    matrix, failures = _collect_bootstrap_samples(
        names,
        fitted_values,
        count,
        progress,
    )
    failure_rate = failures / count
    intervals = _bootstrap_intervals(names, matrix, failure_rate)
    return BootstrapResult(names, matrix, intervals, float(failure_rate))


def bootstrap_local(
    fit_sample: BootstrapFit,
    parameter_names: tuple[str, ...],
    *,
    sample_count: int,
    child_seed: int,
    progress: BootstrapProgress | None = None,
) -> BootstrapResult:
    """Aggregate callback fits in index order using one deterministic stream."""
    names = validated_bootstrap_names(parameter_names)
    count = validated_sample_count(sample_count)
    rng = np.random.default_rng(child_seed)

    def fitted_values():
        for sample_index in range(count):
            yield fit_sample(rng, sample_index)

    return bootstrap_result_from_fits(names, fitted_values(), count, progress)


__all__ = [
    "BootstrapFit",
    "BootstrapProgress",
    "TaskRunner",
    "bootstrap_local",
    "bootstrap_result_from_fits",
    "run_tasks",
    "validated_bootstrap_names",
    "validated_sample_count",
]
