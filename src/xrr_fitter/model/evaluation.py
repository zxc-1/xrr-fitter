"""Immutable numerical evaluations before search candidate publication."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from xrr_fitter.model.data import ResidualMetadata, validate_noise_model
from xrr_fitter.model.fitting import _nonempty, _pickle_values, _readonly
from xrr_fitter.model.instrument import PhysicsDiagnostic
from xrr_fitter.model.parameters import ParameterValue
from xrr_fitter.model.slab_stack import SlabStack


@dataclass(frozen=True, slots=True)
class ModelEvaluation(ResidualMetadata):
    """Full model axes and fitted mode residuals from one numerical traversal."""

    valid: bool
    reason: str
    parameters: tuple[ParameterValue, ...]
    qz_a_inv: np.ndarray
    model_normalized: np.ndarray
    fit_residuals: np.ndarray
    fit_weighted_residuals: np.ndarray
    objective: float
    expanded_stack: SlabStack | None
    diagnostics: tuple[PhysicsDiagnostic, ...]
    noise_model: str = "robust_log"

    def __post_init__(self) -> None:
        validate_noise_model(self.noise_model)
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be bool")
        _nonempty(self.reason, "reason")
        if self.valid and not isfinite(self.objective):
            raise ValueError("valid evaluation objective must be finite")
        parameters = tuple(self.parameters)
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(value, ParameterValue) for value in parameters):
            raise TypeError("evaluation parameters must be ParameterValue values")
        if any(not isinstance(value, PhysicsDiagnostic) for value in diagnostics):
            raise TypeError("evaluation diagnostics must be PhysicsDiagnostic values")
        arrays = self._freeze_arrays()
        if arrays[0].shape != arrays[1].shape or arrays[2].shape != arrays[3].shape:
            raise ValueError("evaluation array axes are inconsistent")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "diagnostics", diagnostics)

    def _freeze_arrays(self) -> tuple[np.ndarray, ...]:
        fields = (
            ("qz_a_inv", self.qz_a_inv),
            ("model_normalized", self.model_normalized),
            ("fit_residuals", self.fit_residuals),
            ("fit_weighted_residuals", self.fit_weighted_residuals),
        )
        arrays = tuple(_readonly(value, float, name) for name, value in fields)
        for (name, _value), array in zip(fields, arrays, strict=True):
            object.__setattr__(self, name, array)
        return arrays

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), _pickle_values(self)
