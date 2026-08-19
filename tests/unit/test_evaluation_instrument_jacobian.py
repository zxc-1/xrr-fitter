from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import EvaluationConstraintError, encode_physical_vector
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec


def test_mixed_kalpha_jacobian_normalizes_extreme_intensity_ratio_without_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beam = BeamSpec(kind="mixed_kalpha", intensity_ratio_21=1e308)
    problem = SimpleNamespace(data=SimpleNamespace(beam=beam))
    traversals = iter(
        (
            (np.array([1.0]), np.array([[1.0]])),
            (np.array([2.0]), np.array([[2.0]])),
        )
    )
    monkeypatch.setattr(
        evaluation,
        "_single_wavelength_smeared_jacobian",
        lambda *_args, **_kwargs: next(traversals),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observed, jacobian = evaluation._smeared_beam_jacobian(
            problem,
            np.array([1.0]),
            np.zeros((1, 1)),
            {},
            {},
            object(),
            object(),
            None,
            None,
            None,
            None,
        )

    np.testing.assert_array_equal(observed, np.array([2.0]))
    np.testing.assert_array_equal(jacobian, np.array([[2.0]]))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_scaled_signal_jacobian_rejects_finite_product_overflow_without_warning() -> None:
    values = {
        "instrument.scale": 1e308,
        "instrument.footprint_spill_angle_deg": 0.0,
    }
    value_jacobians = {
        "instrument.scale": np.zeros(1),
        "instrument.footprint_spill_angle_deg": np.zeros(1),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="scaled signal"):
            evaluation._scaled_signal_jacobian(
                np.array([1.0]),
                np.zeros((1, 1)),
                np.array([2.0]),
                np.zeros((1, 1)),
                values,
                value_jacobians,
            )

    assert not any(item.category is RuntimeWarning for item in caught)


def test_footprint_jacobian_avoids_tiny_denominator_square_underflow() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        footprint, jacobian = evaluation._footprint_jacobian(
            np.array([1e-310]),
            np.array([[1e-300]]),
            1e-300,
            np.zeros(1),
        )

    assert footprint[0] == pytest.approx(1e-10, rel=1e-4)
    assert jacobian[0, 0] == pytest.approx(1.0, rel=1e-12)
    assert np.all(np.isfinite(jacobian))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_instrument_model_jacobian_rejects_finite_addition_overflow_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = (np.array([1e308]), np.array([[1e308]]))
    monkeypatch.setattr(evaluation, "_smeared_beam_jacobian", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(evaluation, "_scaled_signal_jacobian", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(evaluation, "_background_jacobian", lambda *_args, **_kwargs: block)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="instrument model Jacobian"):
            evaluation._instrument_model_jacobian(
                SimpleNamespace(),
                np.array([1.0]),
                np.zeros((1, 1)),
                {},
                {},
                object(),
                None,
                (None, None, None, None),
            )

    assert not any(item.category is RuntimeWarning for item in caught)


def test_powerlaw_background_jacobian_avoids_intermediate_power_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation,
        "_qz_and_jacobian",
        lambda *_args, **_kwargs: (np.array([1e-200]), np.zeros((1, 2))),
    )
    values = {
        "instrument.background": 0.0,
        "instrument.linear_background_per_a_inv": 0.0,
        "instrument.powerlaw_background_amplitude": 1e-200,
        "instrument.powerlaw_background_exponent": 2.0,
    }
    value_jacobians = {
        "instrument.background": np.zeros(2),
        "instrument.linear_background_per_a_inv": np.zeros(2),
        "instrument.powerlaw_background_amplitude": np.array([1e-200, 0.0]),
        "instrument.powerlaw_background_exponent": np.array([0.0, 1.0]),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampled, jacobian = evaluation._background_jacobian(
            SimpleNamespace(data=SimpleNamespace(beam=BeamSpec("monochromatic"))),
            np.array([1.0]),
            np.zeros((1, 2)),
            values,
            value_jacobians,
        )

    np.testing.assert_allclose(sampled, np.array([1e200]), rtol=5e-14)
    np.testing.assert_allclose(
        jacobian,
        np.array([[1e200, -np.log(1e-200) * 1e200]]),
        rtol=5e-14,
    )
    assert np.all(np.isfinite(jacobian))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_stack_diagnostic_numeric_failure_is_an_invalid_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=1805), scale_prior_enabled=False),
    )
    unit = encode_physical_vector(problem, {})

    def fail_before_instrument(*_args, **_kwargs):
        raise FloatingPointError("Nevot-Croce exponent is nonfinite")

    monkeypatch.setattr(evaluation, "parratt_reflectivity", fail_before_instrument)

    result = evaluate_vector(problem, unit)

    assert result.valid is False
    assert result.reason == "constraint_violation:FloatingPointError"


def test_background_jacobian_rejects_nonfinite_power_derivative(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = SimpleNamespace(data=SimpleNamespace(beam=SimpleNamespace(effective_wavelength_a=1.0)))
    values = {
        "instrument.background": 0.0,
        "instrument.linear_background_per_a_inv": 0.0,
        "instrument.powerlaw_background_amplitude": 1e-320,
        "instrument.powerlaw_background_exponent": 4.0,
    }
    zeros = np.zeros(1)
    value_jacobians = {
        "instrument.background": zeros,
        "instrument.linear_background_per_a_inv": zeros,
        "instrument.powerlaw_background_amplitude": zeros,
        "instrument.powerlaw_background_exponent": zeros,
    }
    monkeypatch.setattr(
        evaluation,
        "_qz_and_jacobian",
        lambda *_args: (np.array([1e-77]), np.array([[1e308]])),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(EvaluationConstraintError, match="FloatingPointError"):
            evaluation._background_jacobian(
                problem,
                np.array([1.0]),
                np.zeros((1, 1)),
                values,
                value_jacobians,
            )

    assert not any(item.category is RuntimeWarning for item in caught)


def test_background_jacobian_rejects_linear_derivative_overflow_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = SimpleNamespace(data=SimpleNamespace(beam=SimpleNamespace(effective_wavelength_a=1.0)))
    values = {
        "instrument.background": 0.0,
        "instrument.linear_background_per_a_inv": 0.0,
        "instrument.powerlaw_background_amplitude": 0.0,
        "instrument.powerlaw_background_exponent": 3.0,
    }
    zeros = np.zeros(1)
    value_jacobians = {
        "instrument.background": zeros,
        "instrument.linear_background_per_a_inv": np.array([1e308]),
        "instrument.powerlaw_background_amplitude": zeros,
        "instrument.powerlaw_background_exponent": zeros,
    }
    monkeypatch.setattr(
        evaluation,
        "_qz_and_jacobian",
        lambda *_args: (np.array([1e308]), np.zeros((1, 1))),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(EvaluationConstraintError, match="FloatingPointError"):
            evaluation._background_jacobian(
                problem,
                np.array([1.0]),
                np.zeros((1, 1)),
                values,
                value_jacobians,
            )

    assert not any(item.category is RuntimeWarning for item in caught)
