"""Uncertainty-report contracts needed by ORSO covariance export."""

from __future__ import annotations

import warnings
from dataclasses import replace
from importlib import import_module

import numpy as np
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import encode_physical_vector, evaluate_model
from xrr_fitter.fit.candidates import candidate_from_evaluation
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting


def _joint_report(*, fixed: bool = False):
    joint = import_module("xrr_fitter.analysis.joint")
    if fixed:
        return joint.analyze_joint_ensemble(
            variable_names=("shared", "fixed"),
            candidate_ids=("E-0", "E-1", "E-2"),
            unit_vectors=np.asarray(((0.2, 0.5), (0.5, 0.5), (0.8, 0.5))),
            physical_values=np.asarray(((1.0, 7.0), (2.0, 7.0), (3.0, 7.0))),
            objectives=(1.0, 1.1, 1.2),
            valid=(True, True, True),
            diagnostics=((), (), ()),
            thresholds=FitConfig.fast(1701).confidence,
        )[0]
    return joint.analyze_joint_ensemble(
        variable_names=("shared",),
        candidate_ids=("E-0",),
        unit_vectors=np.asarray(((0.5,),)),
        physical_values=np.asarray(((1.0,),)),
        objectives=(1.0,),
        valid=(True,),
        diagnostics=((),),
        thresholds=FitConfig.fast(1701).confidence,
    )[0]


def _local_report():
    initial = compile_fit_problem(
        prepared_data(size=48),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        replace(FitConfig.fast(937), scale_prior_enabled=False),
    )
    targets = {"component.0.thickness_a", "component.0.density_scale"}
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name not in targets,
        )
        for definition in initial.parameter_definitions
    )
    problem = compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        settings,
    )
    unit = encode_physical_vector(problem, {})
    candidate = candidate_from_evaluation(
        problem,
        unit,
        evaluate_model(problem, unit),
        candidate_id="E-0",
        seed_index=0,
        stop_reason="test candidate",
        nfev=1,
    )
    report = import_module("xrr_fitter.analysis.report")
    return report.build_uncertainty_report(problem, (candidate,), profile_names=())


def test_joint_ensemble_marks_bootstrap_as_not_performed() -> None:
    assert _joint_report().bootstrap_performed is False


def test_joint_report_leaves_prior_conflicts_empty() -> None:
    assert _joint_report().prior_conflicts == ()


def test_joint_singleton_does_not_claim_zero_parameter_uncertainty() -> None:
    report = _joint_report()

    assert report.parameter_sigma is None
    np.testing.assert_array_equal(report.correlation_matrix, np.eye(1))


def test_joint_ensemble_populates_parameter_sigma_from_physical_spread() -> None:
    report = _joint_report(fixed=True)

    np.testing.assert_allclose(report.parameter_sigma, (1.0, 0.0))
    np.testing.assert_array_equal(np.diag(report.correlation_matrix), (1.0, 1.0))


def test_joint_ensemble_handles_repeated_finite_extreme_physical_values() -> None:
    joint = import_module("xrr_fitter.analysis.joint")
    maximum = np.finfo(float).max

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report, _confidence, _evidence = joint.analyze_joint_ensemble(
            variable_names=("shared",),
            candidate_ids=("E-0", "E-1", "E-2"),
            unit_vectors=np.asarray(((0.5,), (0.5,), (0.5,))),
            physical_values=np.full((3, 1), maximum),
            objectives=(1.0, 1.1, 1.2),
            valid=(True, True, True),
            diagnostics=((), (), ()),
            thresholds=FitConfig.fast(1701).confidence,
        )

    np.testing.assert_array_equal(report.parameter_sigma, (0.0,))
    np.testing.assert_array_equal(report.correlation_matrix, np.eye(1))
    assert not any(item.category is RuntimeWarning for item in caught)


def test_local_report_parameter_sigma_matches_correlation_diagonal() -> None:
    report = _local_report()
    sigma = report.parameter_sigma
    diagonal = np.diag(report.correlation_matrix)

    assert sigma is not None
    assert sigma.shape == (len(report.correlation_names),)
    assert np.all(sigma[diagonal == 1.0] > 0.0)
    assert np.all(sigma[diagonal == 0.0] == 0.0)
