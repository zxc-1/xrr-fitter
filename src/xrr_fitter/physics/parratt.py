"""Stable ordinary and periodic Parratt reflectivity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xrr_fitter.model.slab_stack import PeriodicSpan, SlabStack

BRANCH_EPSILON_A2 = 1e-36


def _validated_qz(qz_a_inv: np.ndarray) -> np.ndarray:
    qz = np.asarray(qz_a_inv, dtype=float)
    if np.any(~np.isfinite(qz)) or np.any(qz < 0.0):
        raise ValueError("qz_a_inv must be finite and nonnegative")
    return qz


def select_decaying_branch(wavevectors: np.ndarray) -> np.ndarray:
    """Select roots with negative imaginary part and nonnegative real ties."""
    selected = np.array(wavevectors, dtype=np.complex128, copy=True)
    flip = (selected.imag > 0.0) | ((selected.imag == 0.0) & (selected.real < 0.0))
    selected[flip] *= -1.0
    return selected


def layer_kz(qz_a_inv: np.ndarray, sld_a2: np.ndarray) -> np.ndarray:
    """Return branch-stable wave vectors relative to the fronting SLD."""
    qz = _validated_qz(qz_a_inv).ravel()
    sld = np.asarray(sld_a2, dtype=np.complex128)
    if sld.ndim != 1 or sld.size == 0 or np.any(~np.isfinite(sld.real)) or np.any(~np.isfinite(sld.imag)):
        raise ValueError("SLD must be a nonempty finite vector")
    relative = (sld - sld[0]).copy()
    relative.imag[relative.imag == 0.0] = BRANCH_EPSILON_A2
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        radicand = (qz[:, None] / 2.0) ** 2 - 4.0 * np.pi * relative[None, :]
        wavevectors = select_decaying_branch(np.sqrt(radicand.astype(np.complex128)))
    if np.any(~np.isfinite(wavevectors.real)) or np.any(~np.isfinite(wavevectors.imag)):
        raise FloatingPointError("nonfinite layer wavevectors")
    return wavevectors


def _validated_wavevectors(kz: np.ndarray) -> np.ndarray:
    wavevectors = np.asarray(kz, dtype=np.complex128)
    if (
        wavevectors.ndim != 2
        or wavevectors.shape[1] < 2
        or np.any(~np.isfinite(wavevectors.real))
        or np.any(~np.isfinite(wavevectors.imag))
    ):
        raise ValueError("wavevectors must be a finite two-dimensional array with at least two media")
    return wavevectors


def _validated_roughness(roughness_a: np.ndarray, medium_count: int) -> np.ndarray:
    roughness = np.asarray(roughness_a, dtype=float)
    if roughness.shape != (medium_count - 1,) or np.any(~np.isfinite(roughness)) or np.any(roughness < 0.0):
        raise ValueError("roughness must be a finite nonnegative vector matching the interfaces")
    return roughness


def fresnel_interfaces(kz: np.ndarray, roughness_a: np.ndarray) -> np.ndarray:
    """Return exact Nevot-Croce corrected Fresnel amplitudes."""
    wavevectors = _validated_wavevectors(kz)
    roughness = _validated_roughness(roughness_a, wavevectors.shape[1])
    upper = wavevectors[:, :-1]
    lower = wavevectors[:, 1:]
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        denominator = upper + lower
        if np.any(denominator == 0.0):
            raise FloatingPointError("zero Fresnel denominator")
        reflection = (upper - lower) / denominator
        reflection *= np.exp(-2.0 * upper * lower * roughness[None, :] ** 2)
    if np.any(~np.isfinite(reflection.real)) or np.any(~np.isfinite(reflection.imag)):
        raise FloatingPointError("nonfinite Fresnel interfaces")
    return reflection


def _standard_reflectivity(qz: np.ndarray, stack: SlabStack) -> np.ndarray:
    kz = layer_kz(qz, stack.sld_a2)
    reflection = fresnel_interfaces(kz, stack.roughness_a)
    amplitude = reflection[:, -1]
    for interface in range(reflection.shape[1] - 2, -1, -1):
        lower = interface + 1
        propagated = amplitude * np.exp(-2j * kz[:, lower] * stack.thickness_a[lower])
        denominator = 1.0 + reflection[:, interface] * propagated
        if np.any(denominator == 0.0):
            raise FloatingPointError("zero Parratt denominator")
        amplitude = (reflection[:, interface] + propagated) / denominator
    return (np.abs(amplitude) ** 2).reshape(qz.shape)


def normalize_mobius(matrix: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(matrix), axis=(1, 2))
    if np.any(~np.isfinite(scale)) or np.any(scale == 0.0):
        raise FloatingPointError("invalid periodic Parratt transform: nonfinite matrix")
    return matrix / scale[:, None, None]


def _compose_mobius(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    product = np.empty_like(left)
    product[:, 0, 0] = left[:, 0, 0] * right[:, 0, 0] + left[:, 0, 1] * right[:, 1, 0]
    product[:, 0, 1] = left[:, 0, 0] * right[:, 0, 1] + left[:, 0, 1] * right[:, 1, 1]
    product[:, 1, 0] = left[:, 1, 0] * right[:, 0, 0] + left[:, 1, 1] * right[:, 1, 0]
    product[:, 1, 1] = left[:, 1, 0] * right[:, 0, 1] + left[:, 1, 1] * right[:, 1, 1]
    return normalize_mobius(product)


def _mobius_power(matrix: np.ndarray, exponent: int) -> np.ndarray:
    result: np.ndarray | None = None
    base = matrix
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = base.copy() if result is None else _compose_mobius(result, base)
        remaining >>= 1
        if remaining:
            base = _compose_mobius(base, base)
    if result is not None:
        return result
    identity = np.zeros_like(matrix)
    identity[:, 0, 0] = 1.0
    identity[:, 1, 1] = 1.0
    return identity


@dataclass(slots=True)
class _PeriodicOptics:
    qz: np.ndarray
    stack: SlabStack
    flat_qz: np.ndarray = field(init=False)
    kz_cache: dict[int, np.ndarray] = field(default_factory=dict)
    interface_cache: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.flat_qz = self.qz.ravel()

    def kz_at(self, medium: int) -> np.ndarray:
        cached = self.kz_cache.get(medium)
        if cached is not None:
            return cached
        relative = complex(self.stack.sld_a2[medium] - self.stack.sld_a2[0])
        if relative.imag == 0.0:
            relative = complex(relative.real, BRANCH_EPSILON_A2)
        kz = select_decaying_branch(np.sqrt(((self.flat_qz / 2.0) ** 2 - 4.0 * np.pi * relative).astype(np.complex128)))
        self.kz_cache[medium] = kz
        return kz

    def interface_at(self, interface: int) -> np.ndarray:
        cached = self.interface_cache.get(interface)
        if cached is not None:
            return cached
        upper = self.kz_at(interface)
        lower = self.kz_at(interface + 1)
        denominator = upper + lower
        if np.any(denominator == 0.0):
            raise FloatingPointError("zero Fresnel denominator")
        reflection = (upper - lower) / denominator
        reflection *= np.exp(-2.0 * upper * lower * self.stack.roughness_a[interface] ** 2)
        self.interface_cache[interface] = reflection
        return reflection


def _layer_transform(optics: _PeriodicOptics, medium: int) -> np.ndarray:
    reflection = optics.interface_at(medium - 1)
    phase = np.exp(-2j * optics.kz_at(medium) * optics.stack.thickness_a[medium])
    matrix = np.empty((optics.flat_qz.size, 2, 2), dtype=np.complex128)
    matrix[:, 0, 0] = phase
    matrix[:, 0, 1] = reflection
    matrix[:, 1, 0] = reflection * phase
    matrix[:, 1, 1] = 1.0
    return matrix


def _layer_product(optics: _PeriodicOptics, start: int, count: int) -> np.ndarray:
    result = normalize_mobius(_layer_transform(optics, start))
    for medium in range(start + 1, start + count):
        result = _compose_mobius(result, _layer_transform(optics, medium))
    return result


def _apply_transform(matrix: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    numerator = matrix[:, 0, 0] * amplitude + matrix[:, 0, 1]
    denominator = matrix[:, 1, 0] * amplitude + matrix[:, 1, 1]
    if np.any(denominator == 0.0):
        raise FloatingPointError("zero periodic Parratt denominator")
    return numerator / denominator


def _apply_layer(optics: _PeriodicOptics, amplitude: np.ndarray, medium: int) -> np.ndarray:
    reflection = optics.interface_at(medium - 1)
    propagated = amplitude * np.exp(-2j * optics.kz_at(medium) * optics.stack.thickness_a[medium])
    denominator = 1.0 + reflection * propagated
    if np.any(denominator == 0.0):
        raise FloatingPointError("zero periodic Parratt denominator")
    return (reflection + propagated) / denominator


def _apply_range(optics: _PeriodicOptics, amplitude: np.ndarray, indices: range) -> np.ndarray:
    for medium in indices:
        amplitude = _apply_layer(optics, amplitude, medium)
    return amplitude


def _apply_span(
    optics: _PeriodicOptics, amplitude: np.ndarray, cursor: int, span: PeriodicSpan
) -> tuple[np.ndarray, int]:
    stop = span.start_medium + span.layer_count * span.repeats
    amplitude = _apply_range(optics, amplitude, range(cursor, stop - 1, -1))
    normal = _layer_product(optics, span.start_medium + span.layer_count, span.layer_count)
    amplitude = _apply_transform(_mobius_power(normal, span.repeats - 1), amplitude)
    first_stop = span.start_medium + span.layer_count - 1
    amplitude = _apply_range(optics, amplitude, range(first_stop, span.start_medium - 1, -1))
    return amplitude, span.start_medium - 1


def _periodic_reflectivity(qz: np.ndarray, stack: SlabStack) -> np.ndarray:
    optics = _PeriodicOptics(qz, stack)
    amplitude = optics.interface_at(stack.roughness_a.size - 1)
    cursor = stack.thickness_a.size - 2
    for span in reversed(stack.periodic_spans):
        amplitude, cursor = _apply_span(optics, amplitude, cursor, span)
    amplitude = _apply_range(optics, amplitude, range(cursor, 0, -1))
    return (np.abs(amplitude) ** 2).reshape(qz.shape)


def parratt_reflectivity(qz_a_inv: np.ndarray, stack: SlabStack) -> np.ndarray:
    """Evaluate reflectivity through ordinary recurrence or periodic Mobius powers."""
    qz = _validated_qz(qz_a_inv)
    try:
        # Treat overflow/invalid/divide as a candidate-domain failure instead of
        # allowing NumPy warnings to turn into a published NaN curve. Underflow
        # is benign for exponentially decaying propagation and remains ignored.
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            result = _periodic_reflectivity(qz, stack) if stack.periodic_spans else _standard_reflectivity(qz, stack)
    except FloatingPointError as error:
        # Preserve explicit denominator/transform diagnostics while normalizing
        # NumPy's arithmetic errors to the public finite-value contract.
        if str(error).startswith(("zero ", "invalid periodic Parratt transform")):
            raise
        raise FloatingPointError(f"nonfinite Parratt arithmetic: {error}") from error
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("nonfinite Parratt reflectivity")
    return result
