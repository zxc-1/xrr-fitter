"""Analytic forward tangents for Parratt and resolution physics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from xrr_fitter.model.structure import PeriodicSpan, SlabStack
from xrr_fitter.physics.parratt import BRANCH_EPSILON_A2, parratt_reflectivity
from xrr_fitter.physics.resolution import (
    MAX_QUERY_VALUES,
    RULES,
    gauss_hermite_values,
    gh_converged,
)


DifferentiableFunction = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
ComplexTangent = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class _Inputs:
    qz: np.ndarray
    qz_jacobian: np.ndarray
    thickness_jacobian: np.ndarray
    sld_jacobian: np.ndarray
    roughness_jacobian: np.ndarray
    parameter_count: int


def _inputs(
    qz_a_inv: np.ndarray,
    stack: SlabStack,
    qz_jacobian: np.ndarray,
    thickness_jacobian: np.ndarray,
    sld_jacobian: np.ndarray,
    roughness_jacobian: np.ndarray,
) -> _Inputs:
    # Normalize every tangent to a common trailing parameter axis before any
    # vectorized complex arithmetic begins.
    qz = np.asarray(qz_a_inv, dtype=float)
    if np.any(~np.isfinite(qz)) or np.any(qz < 0.0):
        raise ValueError("qz_a_inv must be finite and nonnegative")
    # Every derivative array shares one trailing real-parameter axis.
    q_tangent = np.asarray(qz_jacobian, dtype=float)
    if q_tangent.ndim != qz.ndim + 1 or q_tangent.shape[:-1] != qz.shape:
        raise ValueError("qz_jacobian must append a parameter axis")
    count = q_tangent.shape[-1]
    thickness = np.asarray(thickness_jacobian, dtype=float)
    sld = np.asarray(sld_jacobian, dtype=np.complex128)
    roughness = np.asarray(roughness_jacobian, dtype=float)
    if thickness.shape != (stack.thickness_a.size, count):
        raise ValueError("thickness_jacobian has invalid shape")
    if sld.shape != (stack.sld_a2.size, count):
        raise ValueError("sld_jacobian has invalid shape")
    if roughness.shape != (stack.roughness_a.size, count):
        raise ValueError("roughness_jacobian has invalid shape")
    arrays = (q_tangent, thickness, sld.real, sld.imag, roughness)
    if any(np.any(~np.isfinite(value)) for value in arrays):
        raise ValueError("Parratt Jacobian inputs must be finite")
    return _Inputs(qz, q_tangent, thickness, sld, roughness, count)


def _all_kz(stack: SlabStack, inputs: _Inputs) -> ComplexTangent:
    # The discrete root sign is inherited from the primal branch; its tangent
    # is flipped only after differentiating the analytic square root.
    qz = inputs.qz.ravel()
    q_tangent = inputs.qz_jacobian.reshape(qz.size, inputs.parameter_count)
    # Pin real radicands before applying the same decaying-root branch as the primal.
    relative = (stack.sld_a2 - stack.sld_a2[0]).copy()
    relative.imag[relative.imag == 0.0] = BRANCH_EPSILON_A2
    relative_tangent = inputs.sld_jacobian - inputs.sld_jacobian[0]
    radicand = (qz[:, None] / 2.0) ** 2 - 4.0 * np.pi * relative
    radicand_tangent = (qz[:, None, None] / 2.0) * q_tangent[:, None, :] - 4.0 * np.pi * relative_tangent[None, :, :]
    raw = np.sqrt(radicand.astype(np.complex128))
    if np.any(raw == 0.0):
        raise FloatingPointError("zero layer wavevector in Parratt Jacobian")
    # Differentiate the complex square root before applying its discrete sign choice.
    tangent = radicand_tangent / (2.0 * raw[:, :, None])
    flip = (raw.imag > 0.0) | ((raw.imag == 0.0) & (raw.real < 0.0))
    kz = raw.copy()
    kz[flip] *= -1.0
    tangent[flip] *= -1.0
    return kz, tangent


def _all_interfaces(stack: SlabStack, inputs: _Inputs, kz: np.ndarray, kz_tangent: np.ndarray) -> ComplexTangent:
    # Fresnel, roughness, and exponential terms are differentiated together so
    # no interface parameter is evaluated through a second primal traversal.
    upper, lower = kz[:, :-1], kz[:, 1:]
    upper_tangent, lower_tangent = kz_tangent[:, :-1], kz_tangent[:, 1:]
    denominator = upper + lower
    if np.any(denominator == 0.0):
        raise FloatingPointError("zero Fresnel denominator")
    numerator = upper - lower
    bare = numerator / denominator
    # The bare-interface tangent is the exact quotient rule for each parameter.
    bare_tangent = (
        (upper_tangent - lower_tangent) * denominator[:, :, None]
        - numerator[:, :, None] * (upper_tangent + lower_tangent)
    ) / denominator[:, :, None] ** 2
    sigma = stack.roughness_a
    sigma_tangent = inputs.roughness_jacobian
    # Nevot-Croce roughness contributes through both wavevectors and sigma.
    exponent = -2.0 * upper * lower * sigma[None, :] ** 2
    exponent_tangent = -2.0 * (
        (upper_tangent * lower[:, :, None] + upper[:, :, None] * lower_tangent) * sigma[None, :, None] ** 2
        + upper[:, :, None] * lower[:, :, None] * 2.0 * sigma[None, :, None] * sigma_tangent[None, :, :]
    )
    factor = np.exp(exponent)
    reflection = bare * factor
    factor_tangent = factor[:, :, None] * exponent_tangent
    tangent = bare_tangent * factor[:, :, None] + bare[:, :, None] * factor_tangent
    return reflection, tangent


def _standard_jacobian(stack: SlabStack, inputs: _Inputs) -> tuple[np.ndarray, np.ndarray]:
    # Keep the backward Parratt recurrence identical to the primal amplitude
    # path, carrying one quotient-rule tangent alongside each amplitude.
    kz, kz_tangent = _all_kz(stack, inputs)
    reflection, reflection_tangent = _all_interfaces(stack, inputs, kz, kz_tangent)
    amplitude = reflection[:, -1]
    amplitude_tangent = reflection_tangent[:, -1]
    # Preserve the primal backward recurrence and propagate its quotient tangent.
    for interface in range(reflection.shape[1] - 2, -1, -1):
        lower = interface + 1
        thickness = stack.thickness_a[lower]
        exponent = -2j * kz[:, lower] * thickness
        exponent_tangent = -2j * (kz_tangent[:, lower] * thickness + kz[:, lower, None] * inputs.thickness_jacobian[lower])
        phase = np.exp(exponent)
        phase_tangent = phase[:, None] * exponent_tangent
        propagated = amplitude * phase
        propagated_tangent = amplitude_tangent * phase[:, None] + amplitude[:, None] * phase_tangent
        numerator = reflection[:, interface] + propagated
        numerator_tangent = reflection_tangent[:, interface] + propagated_tangent
        denominator = 1.0 + reflection[:, interface] * propagated
        if np.any(denominator == 0.0):
            raise FloatingPointError("zero Parratt denominator")
        denominator_tangent = reflection_tangent[:, interface] * propagated[:, None] + reflection[:, interface, None] * propagated_tangent
        amplitude = numerator / denominator
        amplitude_tangent = (numerator_tangent * denominator[:, None] - numerator[:, None] * denominator_tangent) / denominator[:, None] ** 2
    intensity = np.abs(amplitude) ** 2
    jacobian = 2.0 * np.real(np.conjugate(amplitude)[:, None] * amplitude_tangent)
    return intensity.reshape(inputs.qz.shape), jacobian.reshape(inputs.qz.shape + (inputs.parameter_count,))


def _normalize(matrix: np.ndarray, tangent: np.ndarray) -> ComplexTangent:
    scale = np.max(np.abs(matrix), axis=(1, 2))
    if np.any(~np.isfinite(scale)) or np.any(scale == 0.0):
        raise FloatingPointError("invalid periodic Parratt transform")
    # A common scalar does not change the represented Mobius map or its tangent.
    return matrix / scale[:, None, None], tangent / scale[:, None, None, None]


def _compose(left: np.ndarray, left_tangent: np.ndarray, right: np.ndarray, right_tangent: np.ndarray) -> ComplexTangent:
    matrix = np.empty_like(left)
    # Preserve the primal element order while batching the parameter-axis work.
    for row in range(2):
        for column in range(2):
            matrix[:, row, column] = left[:, row, 0] * right[:, 0, column] + left[:, row, 1] * right[:, 1, column]
    tangent = np.einsum(
        "qijp,qjk->qikp",
        left_tangent,
        right,
        optimize=True,
    ) + np.einsum(
        "qij,qjkp->qikp",
        left,
        right_tangent,
        optimize=True,
    )
    return _normalize(matrix, tangent)


def _power(matrix: np.ndarray, tangent: np.ndarray, exponent: int) -> ComplexTangent:
    result: np.ndarray | None = None
    result_tangent: np.ndarray | None = None
    base, base_tangent = matrix, tangent
    remaining = exponent
    # Binary exponentiation keeps periodic derivative work logarithmic in repeats.
    while remaining:
        if remaining & 1:
            if result is None:
                result, result_tangent = base.copy(), base_tangent.copy()
            else:
                assert result_tangent is not None
                result, result_tangent = _compose(result, result_tangent, base, base_tangent)
        remaining >>= 1
        if remaining:
            base, base_tangent = _compose(base, base_tangent, base, base_tangent)
    if result is not None:
        assert result_tangent is not None
        return result, result_tangent
    identity = np.zeros_like(matrix)
    identity[:, 0, 0] = identity[:, 1, 1] = 1.0
    return identity, np.zeros_like(tangent)


@dataclass(slots=True)
class _PeriodicTangents:
    stack: SlabStack
    inputs: _Inputs
    qz: np.ndarray = field(init=False)
    qz_tangent: np.ndarray = field(init=False)
    kz_cache: dict[tuple[complex, bytes], ComplexTangent] = field(default_factory=dict)
    interface_cache: dict[tuple[object, ...], ComplexTangent] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.qz = self.inputs.qz.ravel()
        self.qz_tangent = self.inputs.qz_jacobian.reshape(
            self.qz.size,
            self.inputs.parameter_count,
        )

    def _kz_key(self, medium: int) -> tuple[complex, bytes]:
        return complex(self.stack.sld_a2[medium]), self.inputs.sld_jacobian[medium].tobytes()

    def _interface_key(self, interface: int) -> tuple[object, ...]:
        return (
            self._kz_key(interface),
            self._kz_key(interface + 1),
            float(self.stack.roughness_a[interface]),
            self.inputs.roughness_jacobian[interface].tobytes(),
        )

    def kz_at(self, medium: int) -> ComplexTangent:
        key = self._kz_key(medium)
        cached = self.kz_cache.get(key)
        if cached is not None:
            return cached
        # Periodic evaluation computes only media touched by the compressed traversal.
        relative = complex(self.stack.sld_a2[medium] - self.stack.sld_a2[0])
        if relative.imag == 0.0:
            relative = complex(relative.real, BRANCH_EPSILON_A2)
        relative_tangent = self.inputs.sld_jacobian[medium] - self.inputs.sld_jacobian[0]
        radicand = (self.qz / 2.0) ** 2 - 4.0 * np.pi * relative
        radicand_tangent = (self.qz[:, None] / 2.0) * self.qz_tangent - 4.0 * np.pi * relative_tangent
        raw = np.sqrt(radicand.astype(np.complex128))
        if np.any(raw == 0.0):
            raise FloatingPointError("zero layer wavevector in Parratt Jacobian")
        tangent = radicand_tangent / (2.0 * raw[:, None])
        flip = (raw.imag > 0.0) | ((raw.imag == 0.0) & (raw.real < 0.0))
        kz = raw.copy()
        kz[flip] *= -1.0
        tangent[flip] *= -1.0
        self.kz_cache[key] = (kz, tangent)
        return kz, tangent

    def interface_at(self, interface: int) -> ComplexTangent:
        key = self._interface_key(interface)
        cached = self.interface_cache.get(key)
        if cached is not None:
            return cached
        upper, upper_tangent = self.kz_at(interface)
        lower, lower_tangent = self.kz_at(interface + 1)
        denominator = upper + lower
        if np.any(denominator == 0.0):
            raise FloatingPointError("zero Fresnel denominator")
        numerator = upper - lower
        bare = numerator / denominator
        bare_tangent = ((upper_tangent - lower_tangent) * denominator[:, None] - numerator[:, None] * (upper_tangent + lower_tangent)) / denominator[:, None] ** 2
        sigma = self.stack.roughness_a[interface]
        sigma_tangent = self.inputs.roughness_jacobian[interface]
        exponent = -2.0 * upper * lower * sigma**2
        exponent_tangent = -2.0 * ((upper_tangent * lower[:, None] + upper[:, None] * lower_tangent) * sigma**2 + upper[:, None] * lower[:, None] * 2.0 * sigma * sigma_tangent)
        factor = np.exp(exponent)
        reflection = bare * factor
        factor_tangent = factor[:, None] * exponent_tangent
        tangent = bare_tangent * factor[:, None] + bare[:, None] * factor_tangent
        self.interface_cache[key] = (reflection, tangent)
        return reflection, tangent

    def phase_at(self, medium: int) -> ComplexTangent:
        kz, kz_tangent = self.kz_at(medium)
        thickness = self.stack.thickness_a[medium]
        exponent = -2j * kz * thickness
        exponent_tangent = -2j * (kz_tangent * thickness + kz[:, None] * self.inputs.thickness_jacobian[medium])
        phase = np.exp(exponent)
        return phase, phase[:, None] * exponent_tangent


def _layer_transform_tangent(optics: _PeriodicTangents, medium: int) -> ComplexTangent:
    reflection, reflection_tangent = optics.interface_at(medium - 1)
    phase, phase_tangent = optics.phase_at(medium)
    # Encode one lower-medium propagation and its upper interface as a Mobius matrix.
    matrix = np.empty((optics.qz.size, 2, 2), dtype=np.complex128)
    tangent = np.zeros((optics.qz.size, 2, 2, optics.inputs.parameter_count), dtype=np.complex128)
    matrix[:, 0, 0], matrix[:, 0, 1] = phase, reflection
    matrix[:, 1, 0], matrix[:, 1, 1] = reflection * phase, 1.0
    tangent[:, 0, 0], tangent[:, 0, 1] = phase_tangent, reflection_tangent
    tangent[:, 1, 0] = reflection_tangent * phase[:, None] + reflection[:, None] * phase_tangent
    return matrix, tangent


def _layer_product_tangent(optics: _PeriodicTangents, start: int, count: int) -> ComplexTangent:
    result, tangent = _normalize(*_layer_transform_tangent(optics, start))
    for medium in range(start + 1, start + count):
        layer, layer_tangent = _layer_transform_tangent(optics, medium)
        result, tangent = _compose(result, tangent, layer, layer_tangent)
    return result, tangent


def _apply_transform_tangent(matrix: np.ndarray, matrix_tangent: np.ndarray, amplitude: np.ndarray, amplitude_tangent: np.ndarray) -> ComplexTangent:
    numerator = matrix[:, 0, 0] * amplitude + matrix[:, 0, 1]
    numerator_tangent = matrix_tangent[:, 0, 0] * amplitude[:, None] + matrix[:, 0, 0, None] * amplitude_tangent + matrix_tangent[:, 0, 1]
    denominator = matrix[:, 1, 0] * amplitude + matrix[:, 1, 1]
    if np.any(denominator == 0.0):
        raise FloatingPointError("zero periodic Parratt denominator")
    denominator_tangent = matrix_tangent[:, 1, 0] * amplitude[:, None] + matrix[:, 1, 0, None] * amplitude_tangent + matrix_tangent[:, 1, 1]
    return numerator / denominator, (numerator_tangent * denominator[:, None] - numerator[:, None] * denominator_tangent) / denominator[:, None] ** 2


def _apply_layer_tangent(optics: _PeriodicTangents, amplitude: np.ndarray, amplitude_tangent: np.ndarray, medium: int) -> ComplexTangent:
    reflection, reflection_tangent = optics.interface_at(medium - 1)
    phase, phase_tangent = optics.phase_at(medium)
    propagated = amplitude * phase
    propagated_tangent = amplitude_tangent * phase[:, None] + amplitude[:, None] * phase_tangent
    numerator = reflection + propagated
    numerator_tangent = reflection_tangent + propagated_tangent
    denominator = 1.0 + reflection * propagated
    if np.any(denominator == 0.0):
        raise FloatingPointError("zero periodic Parratt denominator")
    denominator_tangent = reflection_tangent * propagated[:, None] + reflection[:, None] * propagated_tangent
    return numerator / denominator, (numerator_tangent * denominator[:, None] - numerator[:, None] * denominator_tangent) / denominator[:, None] ** 2


def _apply_range_tangent(optics: _PeriodicTangents, amplitude: np.ndarray, tangent: np.ndarray, indices: range) -> ComplexTangent:
    for medium in indices:
        amplitude, tangent = _apply_layer_tangent(optics, amplitude, tangent, medium)
    return amplitude, tangent


def _apply_span_tangent(optics: _PeriodicTangents, amplitude: np.ndarray, tangent: np.ndarray, cursor: int, span: PeriodicSpan) -> tuple[np.ndarray, np.ndarray, int]:
    stop = span.start_medium + span.layer_count * span.repeats
    amplitude, tangent = _apply_range_tangent(optics, amplitude, tangent, range(cursor, stop - 1, -1))
    # Power only the normal cells; the first cell retains its declared top roughness.
    normal, normal_tangent = _layer_product_tangent(optics, span.start_medium + span.layer_count, span.layer_count)
    powered, powered_tangent = _power(normal, normal_tangent, span.repeats - 1)
    amplitude, tangent = _apply_transform_tangent(powered, powered_tangent, amplitude, tangent)
    first_stop = span.start_medium + span.layer_count - 1
    amplitude, tangent = _apply_range_tangent(optics, amplitude, tangent, range(first_stop, span.start_medium - 1, -1))
    return amplitude, tangent, span.start_medium - 1


def _periodic_jacobian(stack: SlabStack, inputs: _Inputs) -> tuple[np.ndarray, np.ndarray]:
    optics = _PeriodicTangents(stack, inputs)
    amplitude, tangent = optics.interface_at(stack.roughness_a.size - 1)
    cursor = stack.thickness_a.size - 2
    for span in reversed(stack.periodic_spans):
        amplitude, tangent, cursor = _apply_span_tangent(optics, amplitude, tangent, cursor, span)
    amplitude, tangent = _apply_range_tangent(optics, amplitude, tangent, range(cursor, 0, -1))
    intensity = np.abs(amplitude) ** 2
    jacobian = 2.0 * np.real(np.conj(amplitude)[:, None] * tangent)
    return intensity.reshape(inputs.qz.shape), jacobian.reshape(inputs.qz.shape + (inputs.parameter_count,))


def _normal_cells_repeat(
    values: np.ndarray,
    start: int,
    span: PeriodicSpan,
) -> bool:
    count = span.repeats - 1
    stop = start + count * span.layer_count
    cells = values[start:stop].reshape(count, span.layer_count, values.shape[1])
    return np.array_equal(cells, np.broadcast_to(cells[0], cells.shape))


def _span_normal_tangents_repeat(span: PeriodicSpan, inputs: _Inputs) -> bool:
    # The powered transform represents cells two through N, whose tangents
    # must be exact replicas; the explicit first cell may remain distinct.
    start = span.start_medium + span.layer_count
    return all(
        _normal_cells_repeat(values, start - offset, span)
        for values, offset in (
            (inputs.thickness_jacobian, 0),
            (inputs.sld_jacobian, 0),
            (inputs.roughness_jacobian, 1),
        )
    )


def _active_parameter_mask(inputs: _Inputs) -> np.ndarray:
    """Return columns that can affect the optical recurrence."""
    q_axes = tuple(range(inputs.qz_jacobian.ndim - 1))
    active = np.any(inputs.qz_jacobian != 0.0, axis=q_axes)
    for tangent in (
        inputs.thickness_jacobian,
        inputs.sld_jacobian,
        inputs.roughness_jacobian,
    ):
        active |= np.any(tangent != 0.0, axis=0)
    return active


def _selected_parameter_inputs(inputs: _Inputs, selected: np.ndarray) -> _Inputs:
    return _Inputs(
        inputs.qz,
        inputs.qz_jacobian[..., selected],
        inputs.thickness_jacobian[:, selected],
        inputs.sld_jacobian[:, selected],
        inputs.roughness_jacobian[:, selected],
        int(np.count_nonzero(selected)),
    )


def _jacobian_for_inputs(
    stack: SlabStack,
    inputs: _Inputs,
) -> tuple[np.ndarray, np.ndarray]:
    if stack.periodic_spans:
        if all(
            _span_normal_tangents_repeat(span, inputs)
            for span in stack.periodic_spans
        ):
            return _periodic_jacobian(stack, inputs)
        # Keep the public periodic primal grouping while expanding only the
        # analytic tangent recurrence for independently moving cells.
        _, jacobian = _standard_jacobian(stack, inputs)
        return parratt_reflectivity(inputs.qz, stack), jacobian
    return _standard_jacobian(stack, inputs)


def parratt_reflectivity_jacobian(
    qz_a_inv: np.ndarray,
    stack: SlabStack,
    qz_jacobian: np.ndarray,
    thickness_jacobian: np.ndarray,
    sld_jacobian: np.ndarray,
    roughness_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Parratt reflectivity and its analytic real forward Jacobian."""
    inputs = _inputs(qz_a_inv, stack, qz_jacobian, thickness_jacobian, sld_jacobian, roughness_jacobian)
    if not stack.periodic_spans:
        return _standard_jacobian(stack, inputs)
    active = _active_parameter_mask(inputs)
    if np.all(active):
        return _jacobian_for_inputs(stack, inputs)
    values, reduced = _jacobian_for_inputs(
        stack,
        _selected_parameter_inputs(inputs, active),
    )
    jacobian = np.zeros(values.shape + (inputs.parameter_count,), dtype=float)
    jacobian[..., active] = reduced
    return values, jacobian


def _quadrature_tangent(
    samples: np.ndarray,
    sample_tangent: np.ndarray,
    widths: np.ndarray,
    width_tangent: np.ndarray,
    function: DifferentiableFunction,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Chunking bounds the temporary query tensor while preserving the exact
    # node order used by the scalar resolution implementation.
    nodes, weights = RULES[order]
    count = sample_tangent.shape[1]
    values_out = np.empty(samples.size)
    tangent_out = np.empty((samples.size, count))
    chunk = max(1, MAX_QUERY_VALUES // order)
    for start in range(0, samples.size, chunk):
        stop = min(start + chunk, samples.size)
        query = samples[start:stop, None] + np.sqrt(2.0) * widths[start:stop, None] * nodes
        query_tangent = sample_tangent[start:stop, None] + np.sqrt(2.0) * width_tangent[start:stop, None] * nodes[None, :, None]
        keep = query >= 0.0
        # Negative-q nodes carry zero weight, followed by retained-weight normalization.
        safe_query = np.where(keep, query, 0.0)
        safe_tangent = np.where(keep[:, :, None], query_tangent, 0.0)
        values, tangent = _validated_function_values(
            function(safe_query, safe_tangent),
            safe_query.shape,
            count,
        )
        retained = weights * keep
        normalizer = retained.sum(axis=1)
        values_out[start:stop] = np.sum(values * retained, axis=1) / normalizer
        tangent_out[start:stop] = np.sum(tangent * retained[:, :, None], axis=1) / normalizer[:, None]
    return values_out, tangent_out


def _validated_function_values(
    result: tuple[np.ndarray, np.ndarray],
    query_shape: tuple[int, ...],
    parameter_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(result[0], dtype=float)
    tangent = np.asarray(result[1], dtype=float)
    if values.shape != query_shape or tangent.shape != query_shape + (parameter_count,):
        raise ValueError("differentiable function returned invalid values")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(tangent)):
        raise ValueError("differentiable function returned invalid values")
    return values, tangent


def _adaptive_tangent(
    samples: np.ndarray,
    sample_tangent: np.ndarray,
    widths: np.ndarray,
    width_tangent: np.ndarray,
    function: DifferentiableFunction,
    primal_function: Callable[[np.ndarray], np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    # Adaptive order is selected from primal values first; tangent evaluation
    # then follows that frozen mask without changing convergence decisions.
    if primal_function is None:
        # Without a separate primal callback, reuse differentiable values for selection.
        values_17, _ = _quadrature_tangent(samples, sample_tangent, widths, width_tangent, function, 17)
        values, tangent = _quadrature_tangent(samples, sample_tangent, widths, width_tangent, function, 33)
        needs_65 = ~gh_converged(values_17, values)
    else:
        # The 33-point tangent pass already returns the canonical primal value.
        # Reusing it removes a complete duplicate 33-point physics traversal while
        # retaining the exact 17-to-33 convergence comparison.
        primal_17 = gauss_hermite_values(samples, widths, primal_function, 17)
        values, tangent = _quadrature_tangent(
            samples,
            sample_tangent,
            widths,
            width_tangent,
            function,
            33,
        )
        needs_65 = ~gh_converged(primal_17, values)
    if np.any(needs_65):
        values[needs_65], tangent[needs_65] = _quadrature_tangent(samples[needs_65], sample_tangent[needs_65], widths[needs_65], width_tangent[needs_65], function, 65)
    return values, tangent


def _validate_smearing_shapes(
    samples: np.ndarray,
    widths: np.ndarray,
    sample_tangent: np.ndarray,
    width_tangent: np.ndarray,
) -> None:
    if samples.ndim != 1 or widths.shape != samples.shape:
        raise ValueError("differentiable smearing requires equal one-dimensional samples")
    if (
        sample_tangent.ndim != 2
        or sample_tangent.shape[0] != samples.size
        or width_tangent.shape != sample_tangent.shape
    ):
        raise ValueError("smearing Jacobians must have shape (points, parameters)")


def _validate_smearing_values(
    samples: np.ndarray,
    widths: np.ndarray,
    sample_tangent: np.ndarray,
    width_tangent: np.ndarray,
) -> None:
    if (
        np.any(~np.isfinite(samples))
        or np.any(samples < 0.0)
        or np.any(~np.isfinite(widths))
        or np.any(widths < 0.0)
    ):
        raise ValueError("differentiable smearing inputs must be finite and nonnegative")
    if np.any(~np.isfinite(sample_tangent)) or np.any(~np.isfinite(width_tangent)):
        raise ValueError("differentiable smearing inputs must be finite")


def _smearing_inputs(
    samples: np.ndarray,
    sample_jacobian: np.ndarray,
    widths: np.ndarray,
    width_jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_values = np.asarray(samples, dtype=float)
    width_values = np.asarray(widths, dtype=float)
    sample_tangent = np.asarray(sample_jacobian, dtype=float)
    width_tangent = np.asarray(width_jacobian, dtype=float)
    _validate_smearing_shapes(
        sample_values,
        width_values,
        sample_tangent,
        width_tangent,
    )
    _validate_smearing_values(
        sample_values,
        width_values,
        sample_tangent,
        width_tangent,
    )
    return sample_values, sample_tangent, width_values, width_tangent


def smear_with_widths_jacobian(
    samples: np.ndarray,
    sample_jacobian: np.ndarray,
    widths: np.ndarray,
    width_jacobian: np.ndarray,
    function: DifferentiableFunction,
    primal_function: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate tangents through a primal-selected adaptive convolution."""
    # Zero-width rows are exact point evaluations and never enter quadrature;
    # mixed inputs are split once so each row follows one deterministic branch.
    samples, sample_tangent, widths, width_tangent = _smearing_inputs(
        samples,
        sample_jacobian,
        widths,
        width_jacobian,
    )
    if np.all(widths == 0.0):
        values, tangent = _validated_function_values(
            function(samples, sample_tangent),
            samples.shape,
            sample_tangent.shape[1],
        )
        return np.array(values, copy=True), np.array(tangent, copy=True)
    values = np.empty(samples.size)
    tangent = np.empty_like(sample_tangent)
    zero = widths == 0.0
    if np.any(zero):
        exact = _validated_function_values(
            function(samples[zero], sample_tangent[zero]),
            samples[zero].shape,
            sample_tangent.shape[1],
        )
        values[zero], tangent[zero] = exact
    if np.any(~zero):
        values[~zero], tangent[~zero] = _adaptive_tangent(samples[~zero], sample_tangent[~zero], widths[~zero], width_tangent[~zero], function, primal_function)
    return values, tangent
