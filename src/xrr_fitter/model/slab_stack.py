"""Immutable expanded slab arrays and periodic fast-path metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PeriodicSpan:
    """Coordinates of exactly repeated finite media in an expanded stack."""

    start_medium: int
    layer_count: int
    repeats: int


def _stack_arrays(
    thickness_a: object,
    sld_a2: object,
    roughness_a: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(thickness_a, dtype=float, copy=True),
        np.array(sld_a2, dtype=np.complex128, copy=True),
        np.array(roughness_a, dtype=float, copy=True),
    )


def _validate_stack_shapes(
    thickness: np.ndarray,
    sld: np.ndarray,
    roughness: np.ndarray,
) -> None:
    if thickness.ndim != 1 or sld.ndim != 1 or roughness.ndim != 1:
        raise ValueError("slab stack arrays must be one-dimensional")
    expected = thickness.size - 1
    if thickness.size < 2 or sld.shape != thickness.shape or roughness.size != expected:
        raise ValueError("slab stack array lengths are inconsistent")


def _validate_thickness(thickness: np.ndarray) -> None:
    if np.any(~np.isfinite(thickness)) or np.any(thickness < 0.0):
        raise ValueError("slab thickness must be finite and nonnegative")


def _validate_sld(sld: np.ndarray) -> None:
    invalid = np.any(~np.isfinite(sld.real)) or np.any(~np.isfinite(sld.imag))
    if invalid or np.any(sld.imag < 0.0):
        raise ValueError("slab SLD must be finite with nonnegative absorption")


def _validate_stack_roughness(roughness: np.ndarray) -> None:
    if np.any(~np.isfinite(roughness)) or np.any(roughness < 0.0):
        raise ValueError("interface roughness must be finite and nonnegative")


def _validate_span_coordinates(span: PeriodicSpan) -> None:
    values = (span.start_medium, span.layer_count, span.repeats)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("periodic span coordinates must be integers")
    if span.start_medium < 1 or span.layer_count < 1 or span.repeats < 2:
        raise ValueError("periodic span is outside finite media")


def _span_end(span: PeriodicSpan, finite_end: int, previous_end: int) -> int:
    end = span.start_medium + span.layer_count * span.repeats
    if end > finite_end:
        raise ValueError("periodic span is outside finite media")
    if span.start_medium < previous_end:
        raise ValueError("periodic spans overlap")
    return end


def _validate_media_repetition(
    thickness: np.ndarray,
    sld: np.ndarray,
    span: PeriodicSpan,
) -> None:
    start = span.start_medium
    stop = start + span.layer_count
    base_thickness = thickness[start:stop]
    base_sld = sld[start:stop]
    for repeat in range(1, span.repeats):
        offset = start + repeat * span.layer_count
        repeated = slice(offset, offset + span.layer_count)
        if not np.array_equal(thickness[repeated], base_thickness):
            raise ValueError("periodic span thickness does not repeat")
        if not np.array_equal(sld[repeated], base_sld):
            raise ValueError("periodic span SLD does not repeat")


def _validate_roughness_repetition(roughness: np.ndarray, span: PeriodicSpan) -> None:
    start = span.start_medium
    stop = start + span.layer_count
    first = roughness[start - 1 : stop - 1]
    normal_start = stop - 1
    normal = roughness[normal_start : normal_start + span.layer_count]
    if not np.array_equal(first[1:], normal[1:]):
        raise ValueError("periodic span roughness does not repeat")
    for repeat in range(2, span.repeats):
        offset = start + repeat * span.layer_count - 1
        if not np.array_equal(roughness[offset : offset + span.layer_count], normal):
            raise ValueError("periodic span roughness does not repeat")


def _validate_spans(
    thickness: np.ndarray,
    sld: np.ndarray,
    roughness: np.ndarray,
    spans: tuple[PeriodicSpan, ...],
) -> None:
    previous_end = 1
    finite_end = thickness.size - 1
    for span in spans:
        if not isinstance(span, PeriodicSpan):
            raise TypeError("periodic spans must be PeriodicSpan values")
        _validate_span_coordinates(span)
        previous_end = _span_end(span, finite_end, previous_end)
        _validate_media_repetition(thickness, sld, span)
        _validate_roughness_repetition(roughness, span)


@dataclass(frozen=True, slots=True)
class SlabStack:
    """Owned read-only slab arrays plus validated periodic fast-path metadata."""

    thickness_a: np.ndarray
    sld_a2: np.ndarray
    roughness_a: np.ndarray
    periodic_spans: tuple[PeriodicSpan, ...] = ()

    def __post_init__(self) -> None:
        thickness, sld, roughness = _stack_arrays(
            self.thickness_a,
            self.sld_a2,
            self.roughness_a,
        )
        spans = tuple(self.periodic_spans)
        _validate_stack_shapes(thickness, sld, roughness)
        _validate_thickness(thickness)
        _validate_sld(sld)
        _validate_stack_roughness(roughness)
        _validate_spans(thickness, sld, roughness, spans)
        for value in (thickness, sld, roughness):
            value.setflags(write=False)
        object.__setattr__(self, "thickness_a", thickness)
        object.__setattr__(self, "sld_a2", sld)
        object.__setattr__(self, "roughness_a", roughness)
        object.__setattr__(self, "periodic_spans", spans)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), (
            self.thickness_a,
            self.sld_a2,
            self.roughness_a,
            self.periodic_spans,
        )
