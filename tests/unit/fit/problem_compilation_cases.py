"""Fit-problem compilation contracts.

Coverage keeps parameter layout, stage locks, geometry-dependent bounds, and
analytic evaluation tied to one immutable compiled snapshot.
The suite also proves that the shared context survives serialization without
exposing writable arrays or changing coordinate identity.
Automatic direct-SLD and staged-release cases additionally bind generated
parameter modes to the same physical-vector contract.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import replace
from importlib import import_module
from pathlib import PurePosixPath

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import (
    encode_physical_vector,
    least_squares_system,
    unit_to_physical,
    values_by_name,
)
from xrr_fitter.fit.candidates import bounded_perturbations
from xrr_fitter.fit.initialization import estimate_initial_candidates
from xrr_fitter.fit.objective import evaluate_jacobian, evaluate_vector
from xrr_fitter.fit.problem import (
    compile_fit_problem,
    compile_fixed_parameter_problem,
    compile_stage_problem,
)
from xrr_fitter.io.xy import read_xy_bytes
from xrr_fitter.model.data import BeamSpec, DataColumnMapping
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec, resolution_to_sigma_q
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.structure import (
    GradientLayerSpec,
    LayerSpec,
    MaterialSpec,
    PeriodicBlock,
    StructureSpec,
)


def _config(seed: int = 11) -> FitConfig:
    return replace(FitConfig.fast(master_seed=seed), scale_prior_enabled=False)


def _problem(
    *,
    data=None,
    structure=None,
    instrument=None,
    settings: tuple[ParameterSetting, ...] = (),
    seed: int = 11,
):
    return compile_fit_problem(
        data or prepared_data(size=72),
        structure or simple_structure(),
        instrument or InstrumentSpec(footprint_mode="fit"),
        _config(seed),
        settings,
    )


def _initial_values(problem) -> dict[str, float]:
    return {definition.name: definition.initial for definition in problem.parameter_definitions}


def _extreme_high_angle_tail_data():
    maximum = np.finfo(float).max
    rows = (f"{0.1 + 0.1 * index} {1.0 if index < 32 else maximum}" for index in range(40))
    return read_xy_bytes(
        ("\n".join(rows) + "\n").encode(),
        source_path=PurePosixPath("extreme-tail.xy"),
        beam=BeamSpec("monochromatic"),
        column_mapping=DataColumnMapping(),
    )


def _richardson(problem, unit: np.ndarray) -> np.ndarray:
    def residual(value: np.ndarray) -> np.ndarray:
        result = evaluate_vector(problem, value)
        assert result.valid
        return result.fit_log_residuals_decades

    output = np.empty((np.count_nonzero(problem.data.fit_mask), unit.size))
    step = 5e-5
    for index in range(unit.size):
        coarse_plus = unit.copy()
        coarse_minus = unit.copy()
        fine_plus = unit.copy()
        fine_minus = unit.copy()
        coarse_plus[index] += step
        coarse_minus[index] -= step
        fine_plus[index] += step / 2.0
        fine_minus[index] -= step / 2.0
        coarse = (residual(coarse_plus) - residual(coarse_minus)) / (2.0 * step)
        fine = (residual(fine_plus) - residual(fine_minus)) / step
        output[:, index] = (4.0 * fine - coarse) / 3.0
    return output


def _periodic_structure() -> StructureSpec:
    base = simple_structure()
    film = base.components[0]
    assert isinstance(film, LayerSpec)
    block = PeriodicBlock(
        "cell",
        (
            replace(film, name="a", thickness_a=22.0),
            replace(film, name="b", thickness_a=38.0),
        ),
        repeats=5,
        top_roughness_a=1.0,
    )
    return StructureSpec(base.fronting, (block,), base.backing, backing_roughness_a=2.0)


def _theta_resolution_problem():
    return _problem(
        instrument=InstrumentSpec(footprint_mode="none", resolution_domain="theta"),
        settings=(ParameterSetting("instrument.sigma_theta_deg", 0.01, 0.0, 0.05, locked=False),),
        seed=18,
    )


def _periodic_jacobian_problem():
    return _problem(
        structure=_periodic_structure(),
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),),
        seed=19,
    )


def _mixed_kalpha_problem():
    data = prepared_data(size=72, beam=BeamSpec(kind="mixed_kalpha"))
    return _problem(
        data=data,
        instrument=InstrumentSpec(footprint_mode="none"),
        seed=24,
    )


def _linear_background_jacobian_problem():
    return _problem(
        instrument=InstrumentSpec(footprint_mode="none", background_kind="linear"),
        settings=(
            ParameterSetting("instrument.linear_background_per_a_inv", 0.0, -0.01, 0.01),
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=20,
    )


def _powerlaw_background_jacobian_problem():
    return _problem(
        instrument=InstrumentSpec(footprint_mode="none", background_kind="powerlaw"),
        settings=(
            ParameterSetting("instrument.powerlaw_background_amplitude", 1e-7, 0.0, 1e-6),
            ParameterSetting("instrument.powerlaw_background_exponent", 2.5, 1.0, 4.0),
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=21,
    )


def _direct_sld_jacobian_problem():
    base = simple_structure()
    direct = MaterialSpec("direct film", None, None, 60e-6 + 2e-6j)
    structure = StructureSpec(
        base.fronting,
        (LayerSpec("direct film", direct, 140.0, roughness_a=3.0),),
        base.backing,
        backing_roughness_a=2.0,
    )
    return _problem(
        structure=structure,
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(
            ParameterSetting("component.0.sld_real_a2", 60e-6, 30e-6, 90e-6),
            ParameterSetting("component.0.sld_imag_a2", 2e-6, 0.5e-6, 4e-6),
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=22,
    )


def _direct_backing_jacobian_problem():
    base = simple_structure()
    direct_backing = MaterialSpec("direct backing", None, None, 30e-6 + 1e-6j)
    structure = StructureSpec(
        base.fronting,
        base.components,
        direct_backing,
        backing_roughness_a=2.0,
    )
    return _problem(
        structure=structure,
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(
            ParameterSetting("backing.sld_real_a2", 30e-6, 10e-6, 90e-6),
            ParameterSetting("backing.sld_imag_a2", 1e-6, 0.0, 4e-6),
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=28,
    )


def _gradient_jacobian_problem():
    base = simple_structure()
    structure = StructureSpec(
        base.fronting,
        (
            GradientLayerSpec(
                "gradient",
                upper_sld_a2=25e-6 + 0.5e-6j,
                lower_sld_a2=55e-6 + 2e-6j,
                thickness_a=100.0,
                roughness_a=2.0,
                microslab_max_a=20.0,
            ),
        ),
        base.backing,
        backing_roughness_a=2.0,
    )
    return _problem(
        structure=structure,
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(
            ParameterSetting("component.0.upper_sld_real_a2", 25e-6, 10e-6, 40e-6),
            ParameterSetting("component.0.upper_sld_imag_a2", 0.5e-6, 0.1e-6, 2e-6),
            ParameterSetting("component.0.lower_sld_real_a2", 55e-6, 40e-6, 70e-6),
            ParameterSetting("component.0.lower_sld_imag_a2", 2e-6, 0.5e-6, 4e-6),
            ParameterSetting("instrument.relative_sigma", 0.0, 0.0, 0.0, locked=True),
        ),
        seed=23,
    )


def _angular_point_resolution_jacobian_problem():
    data = prepared_data(size=72)
    raw = np.full(data.qz_a_inv.size, 0.01)
    sigma = resolution_to_sigma_q(
        data.two_theta_deg,
        raw,
        "sigma_two_theta_deg",
        data.beam.effective_wavelength_a,
        data.import_angle_offset_deg,
    )
    data = replace(
        data,
        resolution_raw=raw,
        sigma_q_a_inv=sigma,
        column_mapping=DataColumnMapping(
            two_theta=0,
            intensity=1,
            resolution=2,
            resolution_kind="sigma_two_theta_deg",
        ),
    )
    return _problem(
        data=data,
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(ParameterSetting("instrument.absolute_sigma_a_inv", 0.001, 0.0, 0.005),),
        seed=25,
    )


def _direct_q_point_resolution_jacobian_problem():
    data = prepared_data(size=72)
    sigma = np.linspace(1e-4, 2e-4, data.qz_a_inv.size)
    data = replace(
        data,
        resolution_raw=sigma,
        sigma_q_a_inv=sigma,
        column_mapping=DataColumnMapping(
            two_theta=0,
            intensity=1,
            resolution=2,
            resolution_kind="sigma_q_a_inv",
        ),
    )
    return _problem(
        data=data,
        instrument=InstrumentSpec(footprint_mode="none"),
        settings=(ParameterSetting("instrument.absolute_sigma_a_inv", 0.001, 0.0, 0.005),),
        seed=26,
    )


__all__ = [
    "BeamSpec",
    "DataColumnMapping",
    "FitConfig",
    "GradientLayerSpec",
    "InstrumentSpec",
    "LayerSpec",
    "MaterialSpec",
    "ParameterSetting",
    "PeriodicBlock",
    "PurePosixPath",
    "StructureSpec",
    "_angular_point_resolution_jacobian_problem",
    "_config",
    "_direct_backing_jacobian_problem",
    "_direct_q_point_resolution_jacobian_problem",
    "_direct_sld_jacobian_problem",
    "_extreme_high_angle_tail_data",
    "_gradient_jacobian_problem",
    "_initial_values",
    "_linear_background_jacobian_problem",
    "_mixed_kalpha_problem",
    "_periodic_jacobian_problem",
    "_periodic_structure",
    "_powerlaw_background_jacobian_problem",
    "_problem",
    "_richardson",
    "_theta_resolution_problem",
    "bounded_perturbations",
    "compile_fit_problem",
    "compile_fixed_parameter_problem",
    "compile_stage_problem",
    "encode_physical_vector",
    "estimate_initial_candidates",
    "evaluate_jacobian",
    "evaluate_vector",
    "import_module",
    "least_squares_system",
    "np",
    "pickle",
    "prepared_data",
    "pytest",
    "read_xy_bytes",
    "replace",
    "resolution_to_sigma_q",
    "simple_structure",
    "unit_to_physical",
    "values_by_name",
    "warnings",
]
