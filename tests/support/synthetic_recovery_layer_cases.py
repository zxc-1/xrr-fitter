from __future__ import annotations

import numpy as np

from tests.support.synthetic_recovery_model import (
    AIR,
    MOLYBDENUM,
    SILICA,
    SILICON,
    SyntheticCase,
    _fraction_metric,
    _instrument_options,
    _mono_beam,
    _noise_for_index,
    _period_metric,
    _simple_metric,
    _single_layer_metrics,
    _theta_grid,
)
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import LayerSpec, PeriodicBlock, StructureSpec


def _single_layer_cases() -> tuple[SyntheticCase, ...]:
    cases: list[SyntheticCase] = []
    for index in range(20):
        seed = 11000 + index
        rng = np.random.default_rng(seed)
        thickness = float(rng.uniform(115.0, 255.0))
        density = float(rng.uniform(0.82, 1.06))
        roughness = float(rng.uniform(2.0, 7.0))
        scale = float(rng.uniform(0.78, 1.12))
        background = float(10 ** rng.uniform(-8.2, -6.4))
        relative_sigma = float(rng.uniform(0.002, 0.0055))
        generating = StructureSpec(
            AIR,
            (LayerSpec("film", MOLYBDENUM, thickness, density_scale=density, roughness_a=roughness),),
            SILICON,
            backing_roughness_a=2.0,
        )
        fit = StructureSpec(
            AIR,
            (LayerSpec("film", MOLYBDENUM, thickness * 1.04, roughness_a=max(2.0, roughness * 0.85)),),
            SILICON,
            backing_roughness_a=2.0,
        )
        cases.append(
            SyntheticCase(
                case_id=f"single-{seed}",
                category="single_layer",
                seed=seed,
                fit_seed=21000 + index,
                theta_deg=_theta_grid("default"),
                generating_structure=generating,
                fit_structure=fit,
                generation_beam=_mono_beam(),
                fit_beam=_mono_beam(),
                generation_options=_instrument_options(
                    scale=scale,
                    background=background,
                    relative_sigma=relative_sigma,
                ),
                fit_instrument=InstrumentSpec(footprint_mode="none"),
                noise_kind=_noise_for_index(index),
                metrics=_single_layer_metrics(
                    thickness=thickness,
                    density=density,
                    roughness=roughness,
                ),
            )
        )
    return tuple(cases)


def _double_layer_cases() -> tuple[SyntheticCase, ...]:
    cases: list[SyntheticCase] = []
    for index in range(20):
        seed = 12000 + index
        rng = np.random.default_rng(seed)
        top_t = float(rng.uniform(35.0, 95.0))
        bottom_t = float(rng.uniform(95.0, 230.0))
        top_density = float(rng.uniform(0.88, 1.06))
        bottom_density = float(rng.uniform(0.86, 1.06))
        top_rough = float(rng.uniform(2.0, 6.0))
        bottom_rough = float(rng.uniform(2.0, 7.5))
        generating = StructureSpec(
            AIR,
            (
                LayerSpec("cap", SILICA, top_t, density_scale=top_density, roughness_a=top_rough),
                LayerSpec("film", MOLYBDENUM, bottom_t, density_scale=bottom_density, roughness_a=bottom_rough),
            ),
            SILICON,
            backing_roughness_a=2.5,
        )
        fit = StructureSpec(
            AIR,
            (
                LayerSpec("cap", SILICA, top_t * 0.92, roughness_a=max(2.0, top_rough * 1.10)),
                LayerSpec("film", MOLYBDENUM, bottom_t * 1.06, roughness_a=max(2.0, bottom_rough * 0.9)),
            ),
            SILICON,
            backing_roughness_a=2.5,
        )
        metrics = (
            *_single_layer_metrics(thickness=top_t, density=top_density, roughness=top_rough, prefix="component.0"),
            *_single_layer_metrics(
                thickness=bottom_t, density=bottom_density, roughness=bottom_rough, prefix="component.1"
            ),
        )
        cases.append(
            SyntheticCase(
                case_id=f"double-{seed}",
                category="double_layer",
                seed=seed,
                fit_seed=22000 + index,
                theta_deg=_theta_grid("default"),
                generating_structure=generating,
                fit_structure=fit,
                generation_beam=_mono_beam(),
                fit_beam=_mono_beam(),
                generation_options=_instrument_options(
                    scale=float(rng.uniform(0.82, 1.05)),
                    background=float(10 ** rng.uniform(-8.0, -6.6)),
                    relative_sigma=float(rng.uniform(0.002, 0.005)),
                ),
                fit_instrument=InstrumentSpec(footprint_mode="none"),
                noise_kind=_noise_for_index(index + 1),
                metrics=metrics,
            )
        )
    return tuple(cases)


def _periodic_cases() -> tuple[SyntheticCase, ...]:
    cases: list[SyntheticCase] = []
    for index in range(20):
        seed = 13000 + index
        rng = np.random.default_rng(seed)
        repeats = 10 + int(round(index * 90 / 19))
        mo_t = float(rng.uniform(24.0, 34.0))
        si_t = float(rng.uniform(36.0, 50.0))
        mo_rough = float(rng.uniform(2.0, 5.5))
        si_rough = float(rng.uniform(2.0, 6.5))
        block = PeriodicBlock(
            "Mo/Si",
            (
                LayerSpec("Mo", MOLYBDENUM, mo_t, roughness_a=mo_rough),
                LayerSpec("Si", SILICON, si_t, roughness_a=si_rough),
            ),
            repeats=repeats,
            top_roughness_a=2.0,
        )
        fit_block = PeriodicBlock(
            "Mo/Si",
            (
                LayerSpec("Mo", MOLYBDENUM, mo_t * 1.03, roughness_a=max(2.0, mo_rough * 0.9)),
                LayerSpec("Si", SILICON, si_t * 0.97, roughness_a=max(2.0, si_rough * 1.05)),
            ),
            repeats=repeats,
            top_roughness_a=2.0,
        )
        metrics = (
            _period_metric(
                "component.0.period_a",
                "component.0.layer.0.thickness_a",
                "component.0.layer.1.thickness_a",
                mo_t,
                si_t,
            ),
            _fraction_metric(
                "component.0.layer.0.fraction",
                "component.0.layer.0.thickness_a",
                "component.0.layer.1.thickness_a",
                mo_t,
                si_t,
            ),
            _simple_metric(
                "component.0.layer.0.roughness_a",
                "component.0.layer.0.roughness_a",
                mo_rough,
                "roughness",
            ),
            _simple_metric(
                "component.0.layer.1.roughness_a",
                "component.0.layer.1.roughness_a",
                si_rough,
                "roughness",
            ),
        )
        cases.append(
            SyntheticCase(
                case_id=f"periodic-{seed}-n{repeats}",
                category="periodic_mosi",
                seed=seed,
                fit_seed=23000 + index,
                theta_deg=_theta_grid("periodic"),
                generating_structure=StructureSpec(AIR, (block,), SILICON, backing_roughness_a=3.0),
                fit_structure=StructureSpec(AIR, (fit_block,), SILICON, backing_roughness_a=3.0),
                generation_beam=_mono_beam(),
                fit_beam=_mono_beam(),
                generation_options=_instrument_options(
                    scale=float(rng.uniform(0.80, 1.05)),
                    background=float(10 ** rng.uniform(-8.3, -6.7)),
                    relative_sigma=float(rng.uniform(0.002, 0.005)),
                ),
                fit_instrument=InstrumentSpec(footprint_mode="none"),
                noise_kind=_noise_for_index(index + 2),
                metrics=metrics,
            )
        )
    return tuple(cases)


def _oxide_cap_cases() -> tuple[SyntheticCase, ...]:
    cases: list[SyntheticCase] = []
    for index in range(20):
        seed = 14000 + index
        rng = np.random.default_rng(seed)
        oxide_t = float(rng.uniform(8.0, 28.0))
        film_t = float(rng.uniform(120.0, 260.0))
        oxide_density = float(rng.uniform(0.90, 1.06))
        film_density = float(rng.uniform(0.84, 1.06))
        interface_roughness_limit = float(np.nextafter(0.49 * oxide_t, 0.0))
        oxide_rough = float(rng.uniform(2.0, min(4.5, interface_roughness_limit)))
        film_rough = float(rng.uniform(2.0, min(7.0, interface_roughness_limit)))
        generating = StructureSpec(
            AIR,
            (
                LayerSpec("native-oxide", SILICA, oxide_t, density_scale=oxide_density, roughness_a=oxide_rough),
                LayerSpec("film", MOLYBDENUM, film_t, density_scale=film_density, roughness_a=film_rough),
            ),
            SILICON,
            backing_roughness_a=2.0,
        )
        fit = StructureSpec(
            AIR,
            (
                LayerSpec("native-oxide", SILICA, oxide_t * 1.15, roughness_a=max(2.0, oxide_rough)),
                LayerSpec("film", MOLYBDENUM, film_t * 0.94, roughness_a=max(2.0, film_rough * 0.9)),
            ),
            SILICON,
            backing_roughness_a=2.0,
        )
        metrics = (
            *_single_layer_metrics(
                thickness=oxide_t, density=oxide_density, roughness=oxide_rough, prefix="component.0"
            ),
            *_single_layer_metrics(thickness=film_t, density=film_density, roughness=film_rough, prefix="component.1"),
        )
        cases.append(
            SyntheticCase(
                case_id=f"oxide-cap-{seed}",
                category="oxide_cap",
                seed=seed,
                fit_seed=24000 + index,
                theta_deg=_theta_grid("default"),
                generating_structure=generating,
                fit_structure=fit,
                generation_beam=_mono_beam(),
                fit_beam=_mono_beam(),
                generation_options=_instrument_options(
                    scale=float(rng.uniform(0.76, 1.10)),
                    background=float(10 ** rng.uniform(-8.2, -6.5)),
                    relative_sigma=float(rng.uniform(0.002, 0.006)),
                ),
                fit_instrument=InstrumentSpec(footprint_mode="none"),
                noise_kind=_noise_for_index(index + 3),
                metrics=metrics,
            )
        )
    return tuple(cases)
