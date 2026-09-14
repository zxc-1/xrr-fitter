"""Immutable calibration and per-dataset residual evidence."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from xrr_fitter.model.diagnostic_calibration import DiagnosticCalibration
from xrr_fitter.model.fitting import _sha256
from xrr_fitter.model.instrument import PhysicsDiagnostic


def _nonempty(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _count(value: int, field: str, maximum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} exceeds its axis")


def _names(value: object) -> tuple[str, ...]:
    names = tuple(value)
    for name in names:
        _nonempty(name, "covariance name")
    if len(set(names)) != len(names):
        raise ValueError("covariance names must be unique nonempty strings")
    return names


def _covariance_matrix(value: object, dimension: int) -> np.ndarray:
    matrix = np.array(value, dtype=float, copy=True)
    if matrix.shape != (dimension, dimension) or np.any(~np.isfinite(matrix)):
        raise ValueError("covariance matrix must be finite and match names")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=0.0):
        raise ValueError("covariance matrix must be symmetric")
    if dimension:
        scale = np.max(np.abs(matrix))
        if np.any(np.diag(matrix) < 0.0) or (scale > 0 and np.min(np.linalg.eigvalsh(matrix / scale)) < -1e-10):
            raise ValueError("covariance matrix must be positive semidefinite")
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class CovarianceEvidence:
    """Physical covariance, or an explicit reason no full matrix is estimable."""

    names: tuple[str, ...]
    matrix: np.ndarray | None
    method: str
    rank: int
    unidentifiable_names: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        names = _names(self.names)
        unknown = _names(self.unidentifiable_names)
        if not set(unknown) <= set(names):
            raise ValueError("unidentifiable names must belong to covariance names")
        _count(self.rank, "covariance rank", len(names))
        _nonempty(self.method, "covariance method")
        self._validate_availability(names, unknown)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "unidentifiable_names", unknown)

    def _validate_availability(self, names: tuple[str, ...], unknown: tuple[str, ...]) -> None:
        if self.matrix is None:
            _nonempty(self.unavailable_reason, "unavailable covariance reason")
        else:
            if self.unavailable_reason is not None or self.rank != len(names) or unknown:
                raise ValueError("available covariance requires full rank and no unavailable evidence")
            object.__setattr__(self, "matrix", _covariance_matrix(self.matrix, len(names)))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), tuple(getattr(self, field.name) for field in fields(self))


def _calibrated_member_flags(calibration: DiagnosticCalibration, dataset_id: str | None) -> tuple[bool, bool]:
    statistics = tuple(value for value in calibration.statistics if value.dataset_id == dataset_id)
    if not statistics:
        raise ValueError("calibration statistics must represent the residual dataset member")
    systematic = any(value.adjusted_p_value <= calibration.alpha for value in statistics)
    autocorrelation = any(value.kind == "acf" and value.adjusted_p_value <= calibration.alpha for value in statistics)
    return systematic, autocorrelation


@dataclass(frozen=True, slots=True)
class ResidualEvidence:
    """Executed diagnostics are distinct from unavailable or unrequested work."""

    dataset_id: str | None
    executed: bool
    systematic: bool | None
    autocorrelation: bool | None
    point_count: int
    diagnostics: tuple[PhysicsDiagnostic, ...] = ()
    unavailable_reason: str | None = None
    raw_systematic: bool | None = None
    raw_autocorrelation: bool | None = None
    advisories: tuple[PhysicsDiagnostic, ...] = ()
    calibration: DiagnosticCalibration | None = None
    owner_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.dataset_id is not None:
            _nonempty(self.dataset_id, "residual dataset_id")
        _count(self.point_count, "residual point_count")
        self._validate_execution()
        for name in ("diagnostics", "advisories"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, PhysicsDiagnostic) for value in values):
                raise TypeError(f"residual {name} must contain PhysicsDiagnostic values")
            object.__setattr__(self, name, values)
        self._validate_raw_evidence()
        self._validate_calibration()

    def _validate_raw_evidence(self) -> None:
        for name in ("raw_systematic", "raw_autocorrelation"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"residual {name} must be bool or None")
        if self.owner_sha256 is not None:
            _sha256(self.owner_sha256, "residual owner_sha256")

    def _validate_calibration(self) -> None:
        calibration = self.calibration
        if calibration is None:
            return
        if not isinstance(calibration, DiagnosticCalibration):
            raise TypeError("residual calibration must be DiagnosticCalibration or None")
        if calibration.status == "unavailable":
            if self.executed or self.unavailable_reason != calibration.unavailable_reason:
                raise ValueError("unavailable calibration requires unexecuted matching residual evidence")
            return
        expected = _calibrated_member_flags(calibration, self.dataset_id)
        if not self.executed or (self.systematic, self.autocorrelation) != expected:
            raise ValueError("residual effective flags must match calibrated member statistics")

    def _validate_execution(self) -> None:
        if not isinstance(self.executed, bool):
            raise TypeError("residual executed must be bool")
        if self.executed:
            if not isinstance(self.systematic, bool) or not isinstance(self.autocorrelation, bool):
                raise TypeError("executed residual diagnostics must contain boolean conclusions")
            if self.unavailable_reason is not None:
                raise ValueError("executed residual diagnostics cannot have an unavailable reason")
        else:
            if self.systematic is not None or self.autocorrelation is not None:
                raise ValueError("unexecuted residual diagnostics require unknown conclusions")
            _nonempty(self.unavailable_reason, "unexecuted residual reason")

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), tuple(getattr(self, field.name) for field in fields(self))
