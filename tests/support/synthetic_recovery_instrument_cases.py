from __future__ import annotations

import numpy as np

from tests.support.synthetic_recovery_model import (
    AIR,
    MOLYBDENUM,
    SILICON,
    SyntheticCase,
    _geometry_instrument,
    _instrument_metrics,
    _instrument_options,
    _mixed_beam,
    _mono_beam,
    _mono_fit_for_mixed,
    _noise_for_index,
    _simple_metric,
    _single_layer_metrics,
    _theta_grid,
)
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, StructureSpec


def _instrument_effect_cases() -> tuple[SyntheticCase, ...]:
    cases: list[SyntheticCase] = []
    for index in range(20):
        seed = 15000 + index
        rng = np.random.default_rng(seed)
        thickness = float(rng.uniform(125.0, 235.0))
        density = float(rng.uniform(0.86, 1.06))
        roughness = float(rng.uniform(2.0, 6.0))
        angle_offset = float(rng.uniform(-0.010, 0.010))
        scale = float(rng.uniform(0.70, 1.22))
        background = float(10 ** rng.uniform(-7.4, -5.9))
        relative_sigma = float(rng.uniform(0.002, 0.008))
        generating = StructureSpec(
            AIR,
            (LayerSpec("film", MOLYBDENUM, thickness, density_scale=density, roughness_a=roughness),),
            SILICON,
            backing_roughness_a=2.0,
        )
        fit = StructureSpec(
            AIR,
            (LayerSpec("film", MOLYBDENUM, thickness * 0.96, roughness_a=max(2.0, roughness * 1.05)),),
            SILICON,
            backing_roughness_a=2.0,
        )
        cases.append(
            SyntheticCase(
                case_id=f"instrument-{seed}",
                category="instrument_effects",
                seed=seed,
                fit_seed=25000 + index,
                theta_deg=_theta_grid("default"),
                generating_structure=generating,
                fit_structure=fit,
                generation_beam=_mono_beam(),
                fit_beam=_mono_beam(),
                generation_options=_instrument_options(
                    angle_offset_deg=angle_offset,
                    scale=scale,
                    background=background,
                    relative_sigma=relative_sigma,
                ),
                fit_instrument=InstrumentSpec(footprint_mode="none"),
                noise_kind=_noise_for_index(index),
                metrics=(
                    *_single_layer_metrics(thickness=thickness, density=density, roughness=roughness),
                    *_instrument_metrics(
                        angle_offset_deg=angle_offset,
                        scale=scale,
                        background=background,
                        relative_sigma=relative_sigma,
                    ),
                ),
            )
        )
    return tuple(cases)

def _ambiguous_cases() -> tuple[SyntheticCase, ...]:
    cases: list[SyntheticCase] = []
    for index in range(20):
        seed = 16000 + index
        rng = np.random.default_rng(seed)
        thickness = float(rng.uniform(120.0, 300.0))
        roughness = float(rng.uniform(2.0, 8.0))
        generating = StructureSpec(
            AIR,
            (LayerSpec("film", MOLYBDENUM, thickness, roughness_a=roughness),),
            SILICON,
            backing_roughness_a=2.0,
        )
        fit = StructureSpec(
            AIR,
            (LayerSpec("film", MOLYBDENUM, 180.0, roughness_a=4.0),),
            SILICON,
            backing_roughness_a=2.0,
        )
        cases.append(
            SyntheticCase(
                case_id=f"ambiguous-low-q-{seed}",
                category="ambiguous",
                seed=seed,
                fit_seed=26000 + index,
                theta_deg=_theta_grid("low_q"),
                generating_structure=generating,
                fit_structure=fit,
                generation_beam=_mono_beam(),
                fit_beam=_mono_beam(),
                generation_options=_instrument_options(
                    scale=float(rng.uniform(0.80, 1.05)),
                    background=float(10 ** rng.uniform(-7.8, -6.2)),
                    relative_sigma=float(rng.uniform(0.002, 0.006)),
                ),
                fit_instrument=InstrumentSpec(footprint_mode="none"),
                noise_kind="lognormal_1pct",
                metrics=(),
                expectation="ambiguous",
            )
        )
    return tuple(cases)


def _footprint_cases() -> tuple[SyntheticCase, ...]:
    cases: list[SyntheticCase] = []
    spill_angles = (0.08, 0.12, 0.18, 0.24, 0.30)
    for index in range(20):
        seed = 17000 + index
        rng = np.random.default_rng(seed)
        spill = float(spill_angles[index % len(spill_angles)])
        thickness = float(rng.uniform(130.0, 245.0))
        density = float(rng.uniform(0.84, 1.06))
        roughness = float(rng.uniform(2.0, 6.0))
        generating = StructureSpec(
            AIR,
            (
                LayerSpec(
                    "film",
                    MOLYBDENUM,
                    thickness,
                    density_scale=density,
                    roughness_a=roughness,
                ),
            ),
            SILICON,
            backing_roughness_a=2.0,
        )
        fit = StructureSpec(
            AIR,
            (
                LayerSpec(
                    "film",
                    MOLYBDENUM,
                    thickness * 1.05,
                    roughness_a=max(2.0, roughness),
                ),
            ),
            SILICON,
            backing_roughness_a=2.0,
        )
        generation_options = _instrument_options(
            scale=float(rng.uniform(0.80, 1.10)),
            background=float(10 ** rng.uniform(-8.0, -6.4)),
            relative_sigma=float(rng.uniform(0.002, 0.006)),
            footprint_spill_angle_deg=spill,
        )
        noise_kind = _noise_for_index(index)
        for mode in ("locked", "released"):
            metrics = list(_single_layer_metrics(thickness=thickness, density=density, roughness=roughness))
            if mode == "released":
                metrics.append(
                    _simple_metric(
                        "instrument.footprint_spill_angle_deg",
                        "instrument.footprint_spill_angle_deg",
                        spill,
                        "footprint_angle",
                    )
                )
            cases.append(
                SyntheticCase(
                    case_id=f"footprint-{seed}-{mode}-tfp{spill:.2f}",
                    category=f"footprint_{mode}",
                    seed=seed,
                    fit_seed=(27000 if mode == "locked" else 27100) + index,
                    theta_deg=_theta_grid("default"),
                    generating_structure=generating,
                    fit_structure=fit,
                    generation_beam=_mono_beam(),
                    fit_beam=_mono_beam(),
                    generation_options=generation_options,
                    fit_instrument=(
                        _geometry_instrument(spill)
                        if mode == "locked"
                        else InstrumentSpec(footprint_mode="fit")
                    ),
                    noise_kind=noise_kind,
                    metrics=tuple(metrics),
                    expectation=f"footprint_{mode}",
                )
            )
    return tuple(cases)
def _mixed_kalpha_cases() -> tuple[SyntheticCase, ...]:
    cases: list[SyntheticCase] = []
    for index in range(20):
        seed = 18000 + index
        rng = np.random.default_rng(seed)
        thickness = float(rng.uniform(120.0, 245.0))
        density = float(rng.uniform(0.86, 1.06))
        roughness = float(rng.uniform(2.0, 6.0))
        generating = StructureSpec(
            AIR,
            (
                LayerSpec(
                    "film",
                    MOLYBDENUM,
                    thickness,
                    density_scale=density,
                    roughness_a=roughness,
                ),
            ),
            SILICON,
            backing_roughness_a=2.0,
        )
        fit = StructureSpec(
            AIR,
            (
                LayerSpec(
                    "film",
                    MOLYBDENUM,
                    thickness * 0.96,
                    roughness_a=max(2.0, roughness * 1.05),
                ),
            ),
            SILICON,
            backing_roughness_a=2.0,
        )
        generation_options = _instrument_options(
            scale=float(rng.uniform(0.78, 1.08)),
            background=float(10 ** rng.uniform(-8.0, -6.3)),
            relative_sigma=float(rng.uniform(0.002, 0.006)),
        )
        noise_kind = _noise_for_index(index + 1)
        for mode in ("dual", "mono"):
            cases.append(
                SyntheticCase(
                    case_id=f"mixed-kalpha-{seed}-{mode}",
                    category=f"mixed_kalpha_{mode}",
                    seed=seed,
                    fit_seed=(28000 if mode == "dual" else 28100) + index,
                    theta_deg=_theta_grid("default"),
                    generating_structure=generating,
                    fit_structure=fit,
                    generation_beam=_mixed_beam(),
                    fit_beam=_mixed_beam() if mode == "dual" else _mono_fit_for_mixed(),
                    generation_options=generation_options,
                    fit_instrument=InstrumentSpec(footprint_mode="none"),
                    noise_kind=noise_kind,
                    metrics=_single_layer_metrics(
                        thickness=thickness,
                        density=density,
                        roughness=roughness,
                    ),
                    expectation=("mixed_dual" if mode == "dual" else "mixed_mono_mismatch"),
                )
            )
    return tuple(cases)
