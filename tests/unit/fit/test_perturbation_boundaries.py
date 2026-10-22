"""Seeded local proposals must not collapse onto stationary box boundaries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.fit.local_search import solve_local
from xrr_fitter.fit.perturbations import bounded_perturbations
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterFreedom, ParameterSetting
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure


def test_boundary_proposals_retain_inward_distance_without_changing_interior_draws() -> None:
    center = np.array([0.0, 0.5, 1.0])
    displacement = np.random.default_rng(42).normal(0.0, 0.002, size=(7, 3))

    actual = np.stack(bounded_perturbations(center, 7, seed=42, sigma=0.002))

    np.testing.assert_allclose(actual[:, 0], np.abs(displacement[:, 0]), rtol=0.0, atol=2e-16)
    np.testing.assert_array_equal(actual[:, 1], center[1] + displacement[:, 1])
    np.testing.assert_allclose(actual[:, 2], 1.0 - np.abs(displacement[:, 2]), rtol=0.0, atol=2e-16)
    assert np.all((actual > 0.0) & (actual < 1.0))
    np.testing.assert_array_equal(center, [0.0, 0.5, 1.0])


@pytest.mark.parametrize("sigma", [0.3, 3.0])
def test_large_proposals_reflect_repeatedly_without_accumulating_at_bounds(sigma: float) -> None:
    center = np.array([0.0, 0.4, 1.0])
    proposals = center + np.random.default_rng(42).normal(0.0, sigma, size=(7, 3))
    phase = np.remainder(proposals, 2.0)
    expected = np.where(phase <= 1.0, phase, 2.0 - phase)

    actual = np.stack(bounded_perturbations(center, 7, seed=42, sigma=sigma))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-16)
    assert np.all((actual > 0.0) & (actual < 1.0))


def test_zero_sigma_preserves_exact_boundaries_and_readonly_replay() -> None:
    center = np.array([0.0, 0.5, 1.0])

    starts = bounded_perturbations(center, 3, seed=42, sigma=0.0)

    assert len(starts) == 3
    for start in starts:
        np.testing.assert_array_equal(start, center)
        assert not start.flags.writeable


def test_nonfinite_sampled_displacement_is_rejected() -> None:
    with pytest.raises(ValueError, match="perturbation.*finite"):
        with np.errstate(over="ignore"):
            bounded_perturbations(np.zeros(3), 7, seed=42, sigma=np.finfo(float).max)


def test_resolution_restart_leaves_zero_width_and_recovers_smearing() -> None:
    beam = BeamSpec(kind="monochromatic")
    structure = simple_structure()
    structure = replace(structure, components=(replace(structure.components[0], thickness_a=150.0),))
    theta = np.linspace(0.04, 3.0, 200)
    truth = 0.008
    intensity = instrument_reflectivity(
        theta, expand_structure(structure, beam.wavelength_a), beam, relative_sigma=truth
    )
    data = prepared_data(size=theta.size, two_theta_deg=2.0 * theta, intensity_raw=intensity, beam=beam)
    data = replace(data, intensity_normalized=intensity, normalization=1.0)
    instrument = InstrumentSpec(footprint_mode="none")
    config = replace(FitConfig.fast(4), scale_prior_enabled=False)
    base = compile_fit_problem(data, structure, instrument, config)
    target = "instrument.relative_sigma"
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            freedom=ParameterFreedom.from_locked(definition.name != target),
        )
        for definition in base.parameter_definitions
    )
    problem = compile_fit_problem(data, structure, instrument, config, settings)
    assert tuple(variable.name for variable in problem.variables) == (target,)
    start = bounded_perturbations(np.zeros(1), 1, seed=4, sigma=0.002)[0]

    result = solve_local(problem, start, max_nfev=200)

    recovered = next(parameter.value for parameter in result.evaluation.parameters if parameter.name == target)
    assert result.evaluation.valid
    assert recovered == pytest.approx(truth, rel=1e-4)
