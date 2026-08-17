"""NumPy scalar normalization at the full-project JSON boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.support.model_cases import dataset_project, project, simple_structure

from xrr_fitter.io.project_codec import project_from_bytes, project_to_bytes
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.fitting import ConfidenceThresholds, FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterPrior, ParameterSetting, PriorSpec
from xrr_fitter.model.structure import (
    DriftSpec,
    GradientLayerSpec,
    InterfaceTransition,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    TransitionBranch,
)


def _field_types(subject: object, fields: tuple[str, ...]) -> tuple[type, ...]:
    return tuple(type(getattr(subject, field)) for field in fields)


def _dataclass_field_types(subject: object) -> tuple[type, ...]:
    return _field_types(subject, tuple(subject.__dataclass_fields__))


def _value_types(values: tuple[object, ...]) -> tuple[type, ...]:
    return tuple(type(value) for value in values)


def _numpy_scalar_fit_config() -> FitConfig:
    budget = SearchBudget(4, 8, 200, 30, 8)
    for field, value in {
        "short_de_maxiter": np.int32(4),
        "full_de_maxiter": np.int64(8),
        "local_min_nfev": np.int32(200),
        "local_nfev_per_parameter": np.int64(30),
        "bootstrap_samples": np.int32(8),
    }.items():
        object.__setattr__(budget, field, value)
    confidence = ConfidenceThresholds(
        cluster_join_distance=np.float32(0.125),
        distinct_cluster_distance=np.float64(0.25),
        equivalent_cost_fraction=np.float32(0.03125),
        equivalent_cost_floor=np.float64(0.0009765625),
        boundary_fraction=np.float32(0.0078125),
        strong_correlation=np.float64(0.875),
        prior_conflict_sigmas=np.float32(4.0),
    )
    config = FitConfig(
        master_seed=1201,
        objective_name="robust_log_soft_l1",
        objective_version="1",
        c_decades=np.float32(0.125),
        final_seed_count=4,
        budget=budget,
        local_workers=2,
        scale_prior_enabled=True,
        scale_prior_tau_decades=np.float64(0.25),
        confidence=confidence,
    )
    object.__setattr__(config, "master_seed", np.int64(1201))
    object.__setattr__(config, "final_seed_count", np.int32(4))
    object.__setattr__(config, "local_workers", np.int64(2))
    return config


def _numpy_scalar_parameter_setting() -> ParameterSetting:
    return ParameterSetting(
        "component.0.thickness_a",
        initial=np.float32(24.0),
        lower=np.float64(2.0),
        upper=np.float32(48.0),
        locked=True,
    )


def _numpy_scalar_parameter_prior() -> ParameterPrior:
    return ParameterPrior(
        "component.0.thickness_a",
        PriorSpec("soft_range", (np.float32(12.0), np.float64(36.0), np.float32(4.0))),
    )


def _roundtrip_numpy_scalar_parameter_project() -> tuple[object, ParameterSetting, ParameterPrior]:
    setting = _numpy_scalar_parameter_setting()
    prior = _numpy_scalar_parameter_prior()
    dataset = replace(
        dataset_project("sample-1"),
        beam=BeamSpec(
            "mixed_kalpha",
            wavelength_a=np.float32(1.5),
            wavelength_1_a=np.float64(1.25),
            wavelength_2_a=np.float32(1.75),
            intensity_ratio_21=np.float64(0.5),
        ),
        instrument=InstrumentSpec(footprint_spill_angle_deg=np.float32(0.25)),
        parameter_settings=(setting,),
        parameter_priors=(prior,),
    )
    original = replace(project(dataset), fit_config=_numpy_scalar_fit_config())

    restored = project_from_bytes(project_to_bytes(original))
    return restored, setting, prior


def test_project_bytes_normalize_numpy_scalars_in_parameter_declarations() -> None:
    restored, setting, prior = _roundtrip_numpy_scalar_parameter_project()
    restored_dataset = restored.datasets[0]
    restored_setting = restored_dataset.parameter_settings[0]
    restored_prior = restored_dataset.parameter_priors[0].prior

    assert restored_setting == setting
    assert restored_setting.locked is True
    assert _field_types(restored_setting, ("initial", "lower", "upper")) == (float, float, float)
    assert restored_prior == prior.prior
    assert _value_types(restored_prior.parameters) == (float, float, float)


def test_project_bytes_normalize_numpy_scalars_in_dataset_declarations() -> None:
    restored, _, _ = _roundtrip_numpy_scalar_parameter_project()
    restored_dataset = restored.datasets[0]

    assert _field_types(
        restored_dataset.beam,
        ("wavelength_a", "wavelength_1_a", "wavelength_2_a", "intensity_ratio_21"),
    ) == (float, float, float, float)
    assert type(restored_dataset.instrument.footprint_spill_angle_deg) is float


def test_project_bytes_normalize_numpy_scalars_in_fit_declarations() -> None:
    restored, _, _ = _roundtrip_numpy_scalar_parameter_project()
    restored_budget = restored.fit_config.budget
    restored_confidence = restored.fit_config.confidence

    assert _field_types(restored.fit_config, ("master_seed", "final_seed_count", "local_workers")) == (int, int, int)
    assert _dataclass_field_types(restored_budget) == (int, int, int, int, int)
    assert _dataclass_field_types(restored_confidence) == (
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    )
    assert type(restored.fit_config.c_decades) is float
    assert restored.fit_config.scale_prior_enabled is True
    assert type(restored.fit_config.scale_prior_tau_decades) is float


def test_project_bytes_normalize_numpy_scalars_in_drifted_structure() -> None:
    material = MaterialSpec("SiO2", "SiO2", np.float32(2.2))
    transition = InterfaceTransition(
        (TransitionBranch("erf", np.float32(1.0), np.float32(2.0)),),
        microslab_max_a=np.float32(1.0),
    )
    cap = LayerSpec("cap", material, np.float32(10.0), transition=transition)
    layer = LayerSpec(
        "film",
        material,
        np.float32(20.0),
        density_scale=np.float32(1.0),
        roughness_a=np.float32(2.0),
    )
    block = PeriodicBlock(
        "cell",
        (layer,),
        repeats=3,
        top_roughness_a=np.float32(1.5),
        drift=DriftSpec("linear", "thickness", amount=np.float32(0.1)),
    )
    gradient = GradientLayerSpec(
        "gradient",
        np.complex64(2.0e-6 + 0.1e-6j),
        np.complex64(3.0e-6 + 0.2e-6j),
        np.float32(12.0),
        roughness_a=np.float32(0.5),
        microslab_max_a=np.float32(1.0),
    )
    structure = replace(
        simple_structure(),
        fronting=MaterialSpec("Air", None, None, np.complex64(0.0j)),
        components=(cap, block, gradient),
        backing_roughness_a=np.float32(3.0),
    )
    original = project(replace(dataset_project("sample-1"), structure=structure))

    restored = project_from_bytes(project_to_bytes(original))
    restored_structure = restored.datasets[0].structure
    restored_cap = restored_structure.components[0]
    restored_block = restored_structure.components[1]
    restored_layer = restored_block.layers[0]
    restored_transition = restored_cap.transition
    restored_gradient = restored_structure.components[2]
    scalar_types = {
        "fronting.sld_override_a2": type(restored_structure.fronting.sld_override_a2),
        "backing_roughness_a": type(restored_structure.backing_roughness_a),
        "cap.material.bulk_density_g_cm3": type(restored_cap.material.bulk_density_g_cm3),
        "cap.thickness_a": type(restored_cap.thickness_a),
        "transition.branch.weight": type(restored_transition.branches[0].weight),
        "transition.branch.thickness_a": type(restored_transition.branches[0].thickness_a),
        "transition.microslab_max_a": type(restored_transition.microslab_max_a),
        "block.repeats": type(restored_block.repeats),
        "block.top_roughness_a": type(restored_block.top_roughness_a),
        "block.drift.amount": type(restored_block.drift.amount),
        "layer.material.bulk_density_g_cm3": type(restored_layer.material.bulk_density_g_cm3),
        "layer.thickness_a": type(restored_layer.thickness_a),
        "layer.density_scale": type(restored_layer.density_scale),
        "layer.roughness_a": type(restored_layer.roughness_a),
        "gradient.upper_sld_a2": type(restored_gradient.upper_sld_a2),
        "gradient.lower_sld_a2": type(restored_gradient.lower_sld_a2),
        "gradient.thickness_a": type(restored_gradient.thickness_a),
        "gradient.roughness_a": type(restored_gradient.roughness_a),
        "gradient.microslab_max_a": type(restored_gradient.microslab_max_a),
    }

    assert restored == original
    assert restored_block == block
    assert scalar_types == {
        "fronting.sld_override_a2": complex,
        "backing_roughness_a": float,
        "cap.material.bulk_density_g_cm3": float,
        "cap.thickness_a": float,
        "transition.branch.weight": float,
        "transition.branch.thickness_a": float,
        "transition.microslab_max_a": float,
        "block.repeats": int,
        "block.top_roughness_a": float,
        "block.drift.amount": float,
        "layer.material.bulk_density_g_cm3": float,
        "layer.thickness_a": float,
        "layer.density_scale": float,
        "layer.roughness_a": float,
        "gradient.upper_sld_a2": complex,
        "gradient.lower_sld_a2": complex,
        "gradient.thickness_a": float,
        "gradient.roughness_a": float,
        "gradient.microslab_max_a": float,
    }
