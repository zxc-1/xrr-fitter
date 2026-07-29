from __future__ import annotations

import numpy as np

from tests.support.synthetic_recovery_model import (
    AIR,
    MOLYBDENUM,
    SILICA,
    SILICON,
    SyntheticCase,
    _instrument_options,
    _mono_beam,
    _single_layer_metrics,
    _theta_grid,
)
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.structure import GradientLayerSpec, LayerSpec, MaterialSpec, StructureSpec
from xrr_fitter.physics.stack import expand_structure


def _structure_sld(material: MaterialSpec, wavelength_a: float = 1.5406) -> complex:
    stack = expand_structure(
        StructureSpec(AIR, (LayerSpec("probe", material, 10.0),), SILICON),
        wavelength_a,
    )
    return complex(stack.sld_a2[1])
def _model_error_cases() -> tuple[SyntheticCase, ...]:
    cases: list[SyntheticCase] = []
    model_classes = (
        "interdiffusion_gradient",
        "non_gaussian_roughness",
        "hidden_oxide",
        "kalpha_satellite",
        "detector_nonlinearity",
    )
    mo_sld = _structure_sld(MOLYBDENUM)
    si_sld = _structure_sld(SILICON)
    for class_index, model_class in enumerate(model_classes):
        for offset in range(4):
            seed = 19000 + class_index * 10 + offset
            rng = np.random.default_rng(seed)
            thickness = float(rng.uniform(135.0, 235.0))
            density = float(rng.uniform(0.88, 1.06))
            roughness = float(rng.uniform(2.0, 5.0))
            nominal = StructureSpec(
                AIR,
                (LayerSpec("film", MOLYBDENUM, thickness, density_scale=density, roughness_a=roughness),),
                SILICON,
                backing_roughness_a=2.0,
            )
            fit = StructureSpec(
                AIR,
                (LayerSpec("film", MOLYBDENUM, thickness, roughness_a=roughness),),
                SILICON,
                backing_roughness_a=2.0,
            )
            generating = nominal
            variant = None
            distortion = "none"
            strength = 0.0
            if model_class == "interdiffusion_gradient":
                generating = StructureSpec(
                    AIR,
                    (
                        GradientLayerSpec(
                            "Mo-Si-gradient",
                            upper_sld_a2=mo_sld * density,
                            lower_sld_a2=0.88 * mo_sld * density + 0.12 * si_sld,
                            thickness_a=thickness,
                            roughness_a=roughness,
                            microslab_max_a=1.0,
                        ),
                    ),
                    SILICON,
                    backing_roughness_a=2.0,
                )
            elif model_class == "non_gaussian_roughness":
                variant = StructureSpec(
                    AIR,
                    (LayerSpec("film", MOLYBDENUM, thickness, density_scale=density, roughness_a=roughness + 7.0),),
                    SILICON,
                    backing_roughness_a=2.0,
                )
                distortion = "non_gaussian_roughness"
                strength = 0.35
            elif model_class == "hidden_oxide":
                oxide_t = float(rng.uniform(5.0, 13.0))
                generating = StructureSpec(
                    AIR,
                    (
                        LayerSpec("undisclosed-oxide", SILICA, oxide_t, roughness_a=2.0),
                        LayerSpec("film", MOLYBDENUM, thickness, density_scale=density, roughness_a=roughness),
                    ),
                    SILICON,
                    backing_roughness_a=2.0,
                )
            elif model_class == "kalpha_satellite":
                distortion = "kalpha_satellite"
                strength = 0.045
            elif model_class == "detector_nonlinearity":
                distortion = "detector_nonlinearity"
                strength = 0.04
            cases.append(
                SyntheticCase(
                    case_id=f"model-error-{model_class}-{seed}",
                    category="model_error",
                    seed=seed,
                    fit_seed=29000 + class_index * 10 + offset,
                    theta_deg=_theta_grid("default"),
                    generating_structure=generating,
                    fit_structure=fit,
                    generation_beam=_mono_beam(),
                    fit_beam=_mono_beam(),
                    generation_options=_instrument_options(
                        scale=float(rng.uniform(0.82, 1.08)),
                        background=float(10 ** rng.uniform(-8.0, -6.4)),
                        relative_sigma=float(rng.uniform(0.002, 0.006)),
                    ),
                    fit_instrument=InstrumentSpec(footprint_mode="none"),
                    noise_kind="lognormal_1pct",
                    metrics=_single_layer_metrics(
                        thickness=thickness,
                        density=density,
                        roughness=roughness,
                    ),
                    expectation="model_error",
                    distortion=distortion,
                    variant_structure=variant,
                    distortion_strength=strength,
                    model_error_class=model_class,
                )
            )
    return tuple(cases)
