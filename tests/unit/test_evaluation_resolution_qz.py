from __future__ import annotations

import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data

import xrr_fitter.evaluation as evaluation
from xrr_fitter.evaluation import EvaluationConstraintError, encode_physical_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.slab_stack import SlabStack
from xrr_fitter.model.structure import LayerSpec, MaterialSpec, StructureSpec


def test_resolution_width_jacobian_rejects_overflow_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="finite"):
            evaluation._resolution_width_jacobian(
                np.array([1e308]),
                np.zeros((1, 1)),
                1e308,
                np.zeros(1),
                0.0,
                np.zeros(1),
                None,
                None,
            )

    assert not any(item.category is RuntimeWarning for item in caught)


def test_resolution_width_jacobian_avoids_intermediate_square_overflow() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        widths, jacobian = evaluation._resolution_width_jacobian(
            np.array([1.0]),
            np.zeros((1, 1)),
            1e200,
            np.ones(1),
            0.0,
            np.zeros(1),
            None,
            None,
        )

    np.testing.assert_array_equal(widths, np.array([1e200]))
    np.testing.assert_array_equal(jacobian, np.array([[1.0]]))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_point_resolution_jacobian_avoids_intermediate_inverse_wavelength_overflow() -> None:
    problem = SimpleNamespace(
        data=SimpleNamespace(
            resolution_raw=np.array([1e-10]),
            two_theta_deg=np.array([2.0]),
            column_mapping=SimpleNamespace(resolution_kind="sigma_two_theta_deg"),
        ),
        instrument=SimpleNamespace(resolution_domain="q"),
        variables=(object(),),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        point, jacobian = evaluation._point_resolution_with_jacobian(
            problem,
            0.0,
            np.ones(1),
            1e-308,
        )

    assert np.all(np.isfinite(point))
    assert np.all(np.isfinite(jacobian))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_qz_jacobian_avoids_intermediate_inverse_wavelength_overflow() -> None:
    theta = np.array([1e-300])
    theta_jacobian = np.array([[1e-10]])
    wavelength = 1e-308

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qz, jacobian = evaluation._qz_and_jacobian(
            theta,
            theta_jacobian,
            wavelength,
        )

    expected = 4.0 * np.pi * (np.pi / 180.0) * np.cos(np.deg2rad(theta))[:, None] * theta_jacobian / wavelength
    assert np.all(np.isfinite(qz))
    np.testing.assert_allclose(jacobian, expected, rtol=1e-15)
    assert not any(item.category is RuntimeWarning for item in caught)


def test_angle_layout_rejects_qz_overflow_without_warning() -> None:
    problem = SimpleNamespace(
        data=SimpleNamespace(
            two_theta_deg=np.array([2.0]),
            qz_a_inv=np.array([0.0]),
            beam=BeamSpec("monochromatic", wavelength_a=1e-310),
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="qz.*finite"):
            evaluation._angle_layout(problem, {"instrument.angle_offset_deg": 0.0})

    assert not any(item.category is RuntimeWarning for item in caught)


def test_model_evaluation_converts_finite_qz_overflow_to_constraint_error() -> None:
    air = MaterialSpec("air", None, None, 0.0j)
    film = MaterialSpec("film", None, None, 1e-5 + 1e-7j)
    backing = MaterialSpec("backing", None, None, 2e-5 + 2e-7j)
    structure = StructureSpec(
        air,
        (LayerSpec("film", film, 20.0),),
        backing,
    )
    data = replace(
        prepared_data(size=40),
        beam=BeamSpec("monochromatic", wavelength_a=1e-310),
        qz_a_inv=np.linspace(1e-4, 0.2, 40),
        fit_ready=True,
    )
    problem = compile_fit_problem(
        data,
        structure,
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(master_seed=187), scale_prior_enabled=False),
    )
    unit = encode_physical_vector(problem, {})

    with pytest.raises(EvaluationConstraintError, match="constraint_violation:ValueError"):
        evaluation.evaluate_model(problem, unit)


def test_qz_jacobian_rejects_derivative_overflow_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="qz Jacobian"):
            evaluation._qz_and_jacobian(
                np.array([1e-300]),
                np.ones((1, 1)),
                1e-310,
            )

    assert not any(item.category is RuntimeWarning for item in caught)


def test_primal_theta_reflectivity_rejects_qz_overflow_without_warning() -> None:
    stack = SlabStack(
        [0.0, 20.0, 0.0],
        [0.0j, 1e-5 + 1e-7j, 2e-5 + 2e-7j],
        [0.0, 0.0],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FloatingPointError, match="theta-domain qz"):
            evaluation._primal_theta_reflectivity(
                stack,
                1e-310,
                np.array([0.1, 0.2]),
            )

    assert not any(item.category is RuntimeWarning for item in caught)
