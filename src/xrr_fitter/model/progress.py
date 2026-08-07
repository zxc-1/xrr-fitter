"""Immutable preview axes carried by fitting progress values."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


def _positive_integer(value: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


def _nonempty(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")


def _readonly_axis(value: object, field: str) -> np.ndarray:
    axis = np.array(value, dtype=float, copy=True)
    if axis.ndim != 1:
        raise ValueError(f"{field} must be 1-dimensional")
    axis.setflags(write=False)
    return axis


def freeze_preview_axes(
    qz_a_inv: object | None,
    model_normalized: object | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate and own an optional pair of preview axes."""
    if qz_a_inv is None and model_normalized is None:
        return None, None
    if qz_a_inv is None or model_normalized is None:
        raise ValueError("preview axes must be provided together")
    qz = _readonly_axis(qz_a_inv, "preview_qz_a_inv")
    model = _readonly_axis(model_normalized, "preview_model_normalized")
    if qz.size != model.size:
        raise ValueError("preview axes must have equal lengths")
    if qz.size == 0:
        raise ValueError("preview axes must not be empty")
    return qz, model


def downsampled_preview(
    qz_a_inv: np.ndarray,
    model_normalized: np.ndarray,
    *,
    max_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded, endpoint-preserving preview curve."""
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 2:
        raise ValueError("max_points must be an integer of at least two")
    qz, model = freeze_preview_axes(qz_a_inv, model_normalized)
    assert qz is not None and model is not None
    if qz.size <= max_points:
        return qz, model
    selected = np.unique(np.rint(np.linspace(0, qz.size - 1, max_points)).astype(int))
    return _readonly_axis(qz[selected], "preview_qz_a_inv"), _readonly_axis(
        model[selected],
        "preview_model_normalized",
    )


@dataclass(frozen=True, slots=True)
class FitProgress:
    """Serializable monotonic progress with an optional preview curve."""

    dataset_id: str | None
    stage: str
    completed: int
    total: int
    best_objective: float
    message: str
    preview_qz_a_inv: np.ndarray | None = None
    preview_model_normalized: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.dataset_id is not None:
            _nonempty(self.dataset_id, "dataset_id")
        _nonempty(self.stage, "stage")
        _positive_integer(self.completed, "completed")
        _positive_integer(self.total, "total")
        if self.completed > self.total:
            raise ValueError("completed must not exceed total")
        if np.isnan(self.best_objective):
            raise ValueError("best_objective must not be NaN")
        qz, model = freeze_preview_axes(
            self.preview_qz_a_inv,
            self.preview_model_normalized,
        )
        object.__setattr__(self, "preview_qz_a_inv", qz)
        object.__setattr__(self, "preview_model_normalized", model)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), tuple(getattr(self, field.name) for field in fields(self))
