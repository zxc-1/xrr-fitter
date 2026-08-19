"""Residual block selection and moving-block resampling."""

from __future__ import annotations

import numpy as np


def _first_nonpositive_lag(centered: np.ndarray, denominator: float) -> int | None:
    for lag in range(1, min(25, centered.size - 1) + 1):
        autocorrelation = float(centered[:-lag] @ centered[lag:]) / denominator
        if autocorrelation <= 0.0:
            return lag
    return None


def residual_block_length(residuals: np.ndarray) -> int:
    values = np.asarray(residuals, dtype=float)
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 3
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        centered = values / scale
        centered -= np.mean(centered)
    amplitude = float(np.max(np.abs(centered)))
    if amplitude == 0.0:
        return 3
    centered /= amplitude
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        return 3
    crossing = _first_nonpositive_lag(centered, denominator)
    return int(np.clip(25 if crossing is None else crossing, 3, 25))


def moving_block_draw(
    residuals: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(residuals, dtype=float)
    blocks: list[np.ndarray] = []
    size = 0
    while size < values.size:
        start = int(rng.integers(0, values.size))
        block = values[(start + np.arange(block_length)) % values.size]
        blocks.append(block)
        size += block.size
    return np.concatenate(blocks)[: values.size]


__all__ = ["moving_block_draw", "residual_block_length"]
