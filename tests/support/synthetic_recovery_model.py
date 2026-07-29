from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, PeriodicBlock, StructureSpec


AIR = MaterialSpec("Air", None, None, 0.0j)
SILICON = MaterialSpec("Si", "Si", 2.329)
MOLYBDENUM = MaterialSpec("Mo", "Mo", 10.28)
SILICA = MaterialSpec("SiO2", "SiO2", 2.20)


@dataclass(frozen=True, slots=True)
class MetricThreshold:
    error_kind: str
    median: float
    p95: float | None
    minimum_truth: float | None = None


DESIGN_THRESHOLDS = {
    "thickness_period": MetricThreshold("relative", 0.02, 0.05),
    "fraction": MetricThreshold("absolute", 0.02, 0.05),
    "density": MetricThreshold("relative", 0.03, 0.08),
    "roughness": MetricThreshold("absolute", 1.0, 3.0, minimum_truth=2.0),
    "angle_offset": MetricThreshold("absolute", 0.002, 0.005),
    "scale": MetricThreshold("relative", 0.05, None),
    "background": MetricThreshold("relative", 0.05, None),
    "resolution": MetricThreshold("absolute", 0.001, None, minimum_truth=0.002),
}

EXTRA_THRESHOLDS = {
    "footprint_angle": MetricThreshold("absolute", 0.02, 0.05),
}

NOISE_KINDS = (
    "none",
    "lognormal_1pct",
    "lognormal_5pct",
    "high_angle_background",
)


@dataclass(frozen=True, slots=True)
class RecoveryMetric:
    label: str
    family: str
    truth: float
    parameter_truths: tuple[tuple[str, float], ...]
    value: Callable[[dict[str, float]], float]


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    case_id: str
    category: str
    seed: int
    fit_seed: int
    theta_deg: np.ndarray
    generating_structure: StructureSpec
    fit_structure: StructureSpec
    generation_beam: BeamSpec
    fit_beam: BeamSpec
    generation_options: tuple[tuple[str, float], ...]
    fit_instrument: InstrumentSpec
    noise_kind: str
    metrics: tuple[RecoveryMetric, ...]
    expectation: str = "recovery"
    distortion: str = "none"
    variant_structure: StructureSpec | None = None
    distortion_strength: float = 0.0
    model_error_class: str | None = None


def _mono_beam() -> BeamSpec:
    return BeamSpec(kind="monochromatic")


def _mixed_beam() -> BeamSpec:
    return BeamSpec(kind="mixed_kalpha")


def _mono_fit_for_mixed() -> BeamSpec:
    mixed = _mixed_beam()
    return BeamSpec(kind="monochromatic", wavelength_a=mixed.wavelength_1_a)


def _option_dict(case: SyntheticCase) -> dict[str, float]:
    return dict(case.generation_options)


def _theta_grid(kind: str) -> np.ndarray:
    if kind == "low_q":
        return np.linspace(0.03, 0.22, 300)
    if kind == "periodic":
        return np.linspace(0.04, 4.2, 650)
    return np.linspace(0.03, 3.0, 520)


def _noise_for_index(index: int) -> str:
    return NOISE_KINDS[index % len(NOISE_KINDS)]


def _instrument_options(
    *,
    angle_offset_deg: float = 0.0,
    scale: float = 0.9,
    background: float = 2e-8,
    relative_sigma: float = 0.002,
    footprint_spill_angle_deg: float = 0.0,
) -> tuple[tuple[str, float], ...]:
    return (
        ("angle_offset_deg", float(angle_offset_deg)),
        ("scale", float(scale)),
        ("background", float(background)),
        ("relative_sigma", float(relative_sigma)),
        ("footprint_spill_angle_deg", float(footprint_spill_angle_deg)),
    )


def _simple_metric(
    label: str,
    parameter_name: str,
    truth: float,
    family: str,
) -> RecoveryMetric:
    return RecoveryMetric(
        label=label,
        family=family,
        truth=float(truth),
        parameter_truths=((parameter_name, float(truth)),),
        value=lambda values, name=parameter_name: values[name],
    )


def _period_metric(
    label: str,
    first_name: str,
    second_name: str,
    first_truth: float,
    second_truth: float,
) -> RecoveryMetric:
    return RecoveryMetric(
        label=label,
        family="thickness_period",
        truth=float(first_truth + second_truth),
        parameter_truths=((first_name, float(first_truth)), (second_name, float(second_truth))),
        value=lambda values, first=first_name, second=second_name: values[first] + values[second],
    )


def _fraction_metric(
    label: str,
    numerator_name: str,
    denominator_name: str,
    numerator_truth: float,
    denominator_truth: float,
) -> RecoveryMetric:
    total_truth = float(numerator_truth + denominator_truth)
    return RecoveryMetric(
        label=label,
        family="fraction",
        truth=float(numerator_truth / total_truth),
        parameter_truths=(
            (numerator_name, float(numerator_truth)),
            (denominator_name, float(denominator_truth)),
        ),
        value=lambda values, first=numerator_name, second=denominator_name: values[first]
        / (values[first] + values[second]),
    )


def _single_layer_metrics(
    *,
    thickness: float,
    density: float,
    roughness: float,
    prefix: str = "component.0",
) -> tuple[RecoveryMetric, ...]:
    return (
        _simple_metric(f"{prefix}.thickness_a", f"{prefix}.thickness_a", thickness, "thickness_period"),
        _simple_metric(f"{prefix}.density_scale", f"{prefix}.density_scale", density, "density"),
        _simple_metric(f"{prefix}.roughness_a", f"{prefix}.roughness_a", roughness, "roughness"),
    )


def _instrument_metrics(
    *,
    angle_offset_deg: float,
    scale: float,
    background: float,
    relative_sigma: float,
) -> tuple[RecoveryMetric, ...]:
    return (
        _simple_metric(
            "instrument.angle_offset_deg",
            "instrument.angle_offset_deg",
            angle_offset_deg,
            "angle_offset",
        ),
        _simple_metric("instrument.scale", "instrument.scale", scale, "scale"),
        _simple_metric(
            "instrument.background",
            "instrument.background",
            background,
            "background",
        ),
        _simple_metric(
            "instrument.relative_sigma",
            "instrument.relative_sigma",
            relative_sigma,
            "resolution",
        ),
    )


def _geometry_instrument(spill_angle_deg: float) -> InstrumentSpec:
    sample_length_mm = 10.0
    beam_width_mm = sample_length_mm * np.sin(np.deg2rad(spill_angle_deg))
    return InstrumentSpec(
        footprint_mode="geometry",
        footprint_spill_angle_deg=float(spill_angle_deg),
        sample_length_mm=sample_length_mm,
        beam_width_mm=float(beam_width_mm),
    )
