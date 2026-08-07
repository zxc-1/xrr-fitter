from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.support.synthetic_recovery_model import AIR, SILICON, SyntheticCase, _option_dict
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.data import BeamSpec, DataColumnMapping, PreparedData
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.structure import StructureSpec
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure
from xrr_fitter.services.fitting import PreparedDatasetFit, fit_prepared_dataset


def _fit_config(case: SyntheticCase) -> FitConfig:
    return FitConfig.fast(master_seed=case.fit_seed)

def _render_structure_reflectivity(
    theta_deg: np.ndarray,
    structure: StructureSpec,
    beam: BeamSpec,
    options: dict[str, float],
) -> np.ndarray:
    theta_model = theta_deg + options.get("angle_offset_deg", 0.0)
    primary_wavelength = beam.wavelength_a if beam.kind == "monochromatic" else beam.wavelength_1_a
    secondary_stack = (
        expand_structure(structure, beam.wavelength_2_a)
        if beam.kind == "mixed_kalpha"
        else None
    )
    return instrument_reflectivity(
        theta_model,
        expand_structure(structure, primary_wavelength),
        beam,
        secondary_stack=secondary_stack,
        scale=options.get("scale", 1.0),
        background=options.get("background", 0.0),
        relative_sigma=options.get("relative_sigma", 0.0),
        footprint_spill_angle_deg=options.get("footprint_spill_angle_deg", 0.0),
    )


def _generate_case_intensity(case: SyntheticCase) -> np.ndarray:
    options = _option_dict(case)
    base = _render_structure_reflectivity(
        case.theta_deg,
        case.generating_structure,
        case.generation_beam,
        options,
    )
    if case.distortion == "non_gaussian_roughness":
        assert case.variant_structure is not None
        variant = _render_structure_reflectivity(
            case.theta_deg,
            case.variant_structure,
            case.generation_beam,
            options,
        )
        base = (1.0 - case.distortion_strength) * base + case.distortion_strength * variant
    elif case.distortion == "kalpha_satellite":
        satellite_beam = BeamSpec(kind="monochromatic", wavelength_a=1.53475)
        satellite = _render_structure_reflectivity(
            case.theta_deg,
            case.generating_structure,
            satellite_beam,
            options,
        )
        weight = case.distortion_strength
        base = (base + weight * satellite) / (1.0 + weight)
    elif case.distortion == "detector_nonlinearity":
        normalized = base / max(float(np.max(base)), 1e-30)
        base = base * (1.0 + case.distortion_strength * np.sqrt(np.clip(normalized, 0.0, None)))
    return _apply_noise(base, case.theta_deg, case.seed, case.noise_kind)


def _apply_noise(
    noiseless: np.ndarray,
    theta_deg: np.ndarray,
    seed: int,
    noise_kind: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if noise_kind == "none":
        observed = np.array(noiseless, dtype=float, copy=True)
    elif noise_kind == "lognormal_1pct":
        observed = noiseless * np.exp(rng.normal(0.0, 0.01, size=noiseless.size))
    elif noise_kind == "lognormal_5pct":
        observed = noiseless * np.exp(rng.normal(0.0, 0.05, size=noiseless.size))
    elif noise_kind == "high_angle_background":
        high_angle = (theta_deg / float(np.max(theta_deg))) ** 2
        jitter = 1.0 + rng.normal(0.0, 0.15, size=noiseless.size)
        additive = 4e-8 * high_angle * np.clip(jitter, 0.25, None)
        observed = noiseless * np.exp(rng.normal(0.0, 0.01, size=noiseless.size)) + additive
    else:
        raise AssertionError(f"unhandled noise kind: {noise_kind}")
    return np.maximum(observed, 1e-12)


def _prepared_case_data(case: SyntheticCase, observed: np.ndarray) -> PreparedData:
    theta = np.asarray(case.theta_deg, dtype=float)
    two_theta = 2.0 * theta
    qz = (
        4.0
        * np.pi
        * np.sin(np.deg2rad(theta))
        / case.fit_beam.effective_wavelength_a
    )
    size = theta.size
    fit_mask = np.ones(size, dtype=bool)
    fit_mask[[0, -1]] = False
    digest = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()
    return PreparedData(
        source_path=Path(f"{case.case_id}.xy"),
        source_sha256=digest,
        raw_rows=tuple(
            f"{angle:.17g} {value:.17g}"
            for angle, value in zip(two_theta, observed, strict=True)
        ),
        raw_parse_status=("data",) * size,
        source_row_groups=tuple((index,) for index in range(size)),
        beam=case.fit_beam,
        import_angle_offset_deg=0.0,
        two_theta_deg=two_theta,
        intensity_raw=observed,
        intensity_sigma_raw=None,
        resolution_raw=None,
        qz_a_inv=qz,
        intensity_normalized=observed,
        intensity_sigma_normalized=None,
        sigma_q_a_inv=None,
        validation_mask=np.ones(size, dtype=bool),
        fit_mask=fit_mask,
        normalization=1.0,
        r_floor=1e-10,
        fit_ready=True,
        warnings=(),
        column_mapping=DataColumnMapping(),
    )


def _fit_case(
    case: SyntheticCase,
    *,
    local_workers: int | None = None,
    profile_names: tuple[str, ...] | None = None,
):
    observed = _generate_case_intensity(case)
    data = _prepared_case_data(case, observed)
    problem = compile_fit_problem(
        data,
        case.fit_structure,
        case.fit_instrument,
        _fit_config(case),
    )
    prepared = PreparedDatasetFit(
        case.case_id,
        0,
        SimpleNamespace(checkpoint=None),
        problem,
    )
    result = fit_prepared_dataset(
        prepared,
        local_workers=local_workers,
        profile_names=profile_names,
    )
    return result, data
