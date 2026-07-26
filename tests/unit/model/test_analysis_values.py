from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.support.model_cases import fit_result
from xrr_fitter.model.analysis import (
    BootstrapResult,
    ConfidenceClass,
    EnsembleSamples,
    FitResult,
    McmcConfig,
    McmcReport,
    ParameterProfile,
    StructureEvidence,
    UncertaintyReport,
)
from xrr_fitter.model.instrument import PhysicsDiagnostic


def test_analysis_arrays_are_copied_read_only_and_shape_checked() -> None:
    values = np.array([1.0, 2.0, 3.0])
    objectives = np.array([3.0, 2.0, 3.0])
    profile = ParameterProfile("thickness", values, objectives, True, False)
    matrix = np.eye(1)
    uncertainty = UncertaintyReport(
        correlation_names=("thickness",),
        correlation_matrix=matrix,
        profiles=(profile,),
        bootstrap_intervals=(("thickness", 1.0, 3.0),),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(PhysicsDiagnostic("residual", "pattern"),),
    )

    values[0] = 99.0
    matrix[0, 0] = 99.0

    assert profile.values[0] == 1.0
    assert uncertainty.correlation_matrix[0, 0] == 1.0
    assert profile.values.flags.writeable is False
    with pytest.raises(ValueError, match="same shape"):
        ParameterProfile("bad", np.ones(2), np.ones(3), True, True)


def test_uncertainty_report_rejects_empty_candidate_owner() -> None:
    report = UncertaintyReport(
        correlation_names=(),
        correlation_matrix=np.empty((0, 0)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
    )

    with pytest.raises(ValueError, match="candidate_id"):
        replace(report, candidate_id="")


def test_parameter_profile_preserves_nan_objective_evidence() -> None:
    profile = ParameterProfile(
        "thickness",
        np.array([1.0, 2.0]),
        np.array([np.nan, 1.0]),
        False,
        True,
    )

    assert np.isnan(profile.objectives[0])


def test_bootstrap_and_ensemble_results_copy_aligned_arrays() -> None:
    bootstrap_source = np.ones((2, 2))
    bootstrap = BootstrapResult(
        parameter_names=("a", "b"),
        samples=bootstrap_source,
        intervals=(("a", 0.5, 1.5), ("b", 0.5, 1.5)),
        failure_rate=0.0,
    )
    ensemble_source = np.ones((2, 4, 2))
    ensemble = EnsembleSamples(
        samples_unit=ensemble_source,
        log_probability=np.ones((2, 4)),
        acceptance_fraction=np.ones(4),
        split_rhat=np.ones(2),
        effective_sample_size=np.ones(2) * 8,
    )

    bootstrap_source[0, 0] = 99.0
    ensemble_source[0, 0, 0] = 99.0

    assert bootstrap.samples[0, 0] == 1.0
    assert ensemble.samples_unit[0, 0, 0] == 1.0
    assert bootstrap.samples.flags.writeable is False
    assert ensemble.samples_unit.flags.writeable is False


def test_mcmc_config_and_report_validate_sampling_geometry() -> None:
    config = McmcConfig.standard(3)
    report = McmcReport(
        config=config,
        child_seed=42,
        parameter_names=("a", "b"),
        samples_physical=np.ones((8, 2)),
        log_probability=np.ones(8),
        acceptance_fraction=np.ones(config.walkers),
        split_rhat=np.ones(2),
        effective_sample_size=np.ones(2) * 8,
        boundary_hits=(),
    )

    assert config.walkers >= 32 and config.walkers % 2 == 0
    assert report.label == "目标函数伪后验"
    assert report.samples_physical.flags.writeable is False
    with pytest.raises(ValueError, match="parameter"):
        McmcReport(
            config=config,
            child_seed=42,
            parameter_names=("a",),
            samples_physical=np.ones((8, 2)),
            log_probability=np.ones(8),
            acceptance_fraction=np.ones(config.walkers),
            split_rhat=np.ones(2),
            effective_sample_size=np.ones(2),
            boundary_hits=(),
        )
    with pytest.raises(ValueError, match="label"):
        replace(report, label="")
    with pytest.raises(ValueError, match="candidate_id"):
        replace(report, candidate_id="")
    invalid_samples = report.samples_physical.copy()
    invalid_samples[0, 0] = np.nan
    with pytest.raises(ValueError, match="values"):
        replace(report, samples_physical=invalid_samples)


def test_mcmc_config_accepts_zero_burn_in() -> None:
    config = McmcConfig(walkers=32, burn_in=0, production_steps=4)

    assert config.burn_in == 0


def test_final_fit_result_owns_confidence_without_back_edge_from_fitting() -> None:
    result = FitResult.from_search(
        fit_result(),
        confidence=ConfidenceClass.TRUSTED,
        uncertainty=None,
        classification_evidence=("single basin",),
    )
    evidence = StructureEvidence(1, 1, None, (20.0,))

    assert result.best_candidate is not None
    assert result.confidence is ConfidenceClass.TRUSTED
    assert evidence.peak_positions_a == (20.0,)


def test_structure_evidence_preserves_observation_order_and_validates_schema() -> None:
    evidence = StructureEvidence(2, 1, None, (20.0, 10.0))

    assert evidence.peak_positions_a == (20.0, 10.0)
    with pytest.raises(ValueError, match="m_data"):
        StructureEvidence(1, 1, None, (20.0, 10.0))
    with pytest.raises(TypeError, match="warning"):
        StructureEvidence(1, 1, 3, (20.0,))
