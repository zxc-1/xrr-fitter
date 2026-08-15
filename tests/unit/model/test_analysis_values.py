from __future__ import annotations

import pickle
from dataclasses import replace
from importlib import import_module

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


def _published_arrays(value: object):
    if isinstance(value, np.ndarray):
        yield value
    elif hasattr(value, "__dataclass_fields__"):
        for field in value.__dataclass_fields__:
            yield from _published_arrays(getattr(value, field))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _published_arrays(item)


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

    assert report.bootstrap_performed is True
    with pytest.raises(TypeError, match="bootstrap_performed"):
        replace(report, bootstrap_performed=1)


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


@pytest.mark.parametrize(
    ("names", "intervals"),
    [
        (("a", "a"), (("a", 0.5, 1.5), ("a", 0.5, 1.5))),
        (("a", "b"), (("b", 0.5, 1.5), ("a", 0.5, 1.5))),
        (("a", "b"), (("a", 1.5, 0.5), ("b", 0.5, 1.5))),
        (("a", "b"), (("a", 0.5, float("nan")), ("b", 0.5, 1.5))),
    ],
    ids=["duplicate-names", "interval-order", "reversed-bounds", "nonfinite-bound"],
)
def test_bootstrap_result_rejects_invalid_parameter_intervals(
    names: tuple[str, ...],
    intervals: tuple[tuple[str, float, float], ...],
) -> None:
    with pytest.raises(ValueError, match="parameter|interval"):
        BootstrapResult(names, np.ones((2, 2)), intervals, 0.0)


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
    with pytest.raises(ValueError, match="derived"):
        replace(
            report,
            derived_parameter_names=("derived",),
            derived_samples_physical=np.ones((7, 1)),
        )
    with pytest.raises(ValueError, match="derived"):
        replace(
            report,
            derived_parameter_names=("",),
            derived_samples_physical=np.ones((8, 1)),
        )
    fixed = replace(report, fixed_parameter_values=(("locked", 2.0),))
    assert fixed.fixed_parameter_values == (("locked", 2.0),)
    with pytest.raises(ValueError, match="fixed"):
        replace(report, fixed_parameter_values=(("a", 2.0),))
    with pytest.raises(ValueError, match="fixed"):
        replace(report, fixed_parameter_values=(("locked", float("nan")),))
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


def test_published_analysis_arrays_remain_read_only_after_pickle() -> None:
    config = McmcConfig(walkers=4, burn_in=0, production_steps=2)
    profile = ParameterProfile("a", np.array([0.0, 1.0]), np.array([1.0, 2.0]), True, False)
    bootstrap = BootstrapResult(("a",), np.ones((2, 1)), (("a", 0.5, 1.5),), 0.0)
    ensemble = EnsembleSamples(
        np.ones((2, 4, 1)),
        np.ones((2, 4)),
        np.ones(4),
        np.ones(1),
        np.ones(1),
    )
    mcmc = McmcReport(
        config,
        42,
        ("a",),
        np.ones((8, 1)),
        np.ones(8),
        np.ones(4),
        np.ones(1),
        np.ones(1),
        (),
    )
    uncertainty = UncertaintyReport(
        ("a",),
        np.eye(1),
        (profile,),
        bootstrap.intervals,
        bootstrap.failure_rate,
        (),
        (),
        False,
        (),
        mcmc=mcmc,
    )
    result = FitResult.from_search(
        fit_result(),
        confidence=ConfidenceClass.TRUSTED,
        uncertainty=uncertainty,
    )

    restored = pickle.loads(pickle.dumps((profile, bootstrap, ensemble, mcmc, uncertainty, result)))

    arrays = tuple(_published_arrays(restored))
    assert arrays
    assert all(not array.flags.writeable for array in arrays)


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


def test_profile_basin_decision_is_immutable_pickle_safe_evidence() -> None:
    analysis = import_module("xrr_fitter.model.analysis")
    source = np.array([0.25, 0.75])

    decision = analysis.ProfileBasinDecision(
        parameter_name="component.0.thickness_a",
        unit_vector=source,
        objective=0.125,
        evidence=("materially_better_profile_basin",),
    )
    source[0] = 0.5
    restored = pickle.loads(pickle.dumps(decision))

    np.testing.assert_array_equal(decision.unit_vector, np.array([0.25, 0.75]))
    np.testing.assert_array_equal(restored.unit_vector, decision.unit_vector)
    assert decision.unit_vector.flags.writeable is False
    assert restored.unit_vector.flags.writeable is False
    assert restored.evidence == ("materially_better_profile_basin",)

    with pytest.raises(ValueError, match="unit vector"):
        replace(decision, unit_vector=np.array([1.1, 0.5]))
    with pytest.raises(ValueError, match="objective"):
        replace(decision, objective=float("nan"))


def _bands(count: int = 4) -> object:
    from xrr_fitter.model.analysis import SldUncertaintyBands

    depth = np.linspace(-10.0, 50.0, count)
    levels = (0.025, 0.16, 0.5, 0.84, 0.975)
    real = np.tile(np.arange(len(levels), dtype=float)[:, None], (1, count))
    return SldUncertaintyBands(
        depth_a=depth,
        quantiles=levels,
        real=real,
        imaginary=real * 0.5,
        align_label="基底界面",
        sample_count=500,
        total_samples=2000,
        failure_rate=0.0,
    )


def test_sld_bands_expose_readonly_arrays_bound_to_the_quantile_axis() -> None:
    bands = _bands()

    assert bands.real.shape == (len(bands.quantiles), bands.depth_a.size)
    assert bands.imaginary.shape == bands.real.shape
    assert not bands.depth_a.flags.writeable
    assert not bands.real.flags.writeable
    assert not bands.imaginary.flags.writeable


def test_sld_bands_reject_a_quantile_axis_that_is_not_sorted_and_unique() -> None:
    from xrr_fitter.model.analysis import SldUncertaintyBands

    with pytest.raises(ValueError, match="quantiles"):
        SldUncertaintyBands(
            depth_a=np.linspace(0.0, 1.0, 3),
            quantiles=(0.5, 0.16),
            real=np.zeros((2, 3)),
            imaginary=np.zeros((2, 3)),
            align_label="基底界面",
            sample_count=10,
            total_samples=10,
            failure_rate=0.0,
        )


def test_sld_bands_reject_a_thinned_count_above_the_total() -> None:
    from xrr_fitter.model.analysis import SldUncertaintyBands

    with pytest.raises(ValueError, match="sample_count"):
        SldUncertaintyBands(
            depth_a=np.linspace(0.0, 1.0, 3),
            quantiles=(0.5,),
            real=np.zeros((1, 3)),
            imaginary=np.zeros((1, 3)),
            align_label="基底界面",
            sample_count=11,
            total_samples=10,
            failure_rate=0.0,
        )


def test_sld_bands_caption_names_quantiles_alignment_and_thinning() -> None:
    caption = _bands().caption()

    assert "16–84%" in caption
    assert "2.5–97.5%" in caption
    assert "基底界面" in caption
    assert "500/2000" in caption


def _uncertainty_report(**overrides: object) -> UncertaintyReport:
    defaults = dict(
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
    return UncertaintyReport(**{**defaults, **overrides})


def test_uncertainty_report_defaults_sld_bands_to_none() -> None:
    assert _uncertainty_report().sld_bands is None


def test_uncertainty_report_defaults_prior_conflicts_to_empty() -> None:
    assert _uncertainty_report().prior_conflicts == ()


def test_uncertainty_report_retains_a_supplied_sld_band() -> None:
    bands = _bands()

    assert _uncertainty_report(sld_bands=bands).sld_bands is bands


def test_uncertainty_report_rejects_a_wrongly_typed_sld_band() -> None:
    with pytest.raises(TypeError, match="sld_bands"):
        replace(_uncertainty_report(), sld_bands=object())


def test_uncertainty_report_defaults_parameter_sigma_to_none() -> None:
    assert _uncertainty_report().parameter_sigma is None


def test_uncertainty_report_retains_supplied_parameter_sigma() -> None:
    sigma = np.array([1.5])
    report = _uncertainty_report(
        correlation_names=("thickness",),
        correlation_matrix=np.eye(1),
        parameter_sigma=sigma,
    )

    sigma[0] = 99.0

    assert report.parameter_sigma[0] == 1.5
    assert report.parameter_sigma.flags.writeable is False


def test_uncertainty_report_rejects_parameter_sigma_length_mismatch() -> None:
    with pytest.raises(ValueError, match="parameter_sigma"):
        _uncertainty_report(
            correlation_names=("a", "b"),
            correlation_matrix=np.eye(2),
            parameter_sigma=np.array([1.0]),
        )


@pytest.mark.parametrize("sigma", (np.array([np.nan]), np.array([np.inf]), np.array([-0.1])))
def test_uncertainty_report_rejects_invalid_parameter_sigma(sigma: np.ndarray) -> None:
    with pytest.raises(ValueError, match="parameter_sigma"):
        _uncertainty_report(
            correlation_names=("a",),
            correlation_matrix=np.eye(1),
            parameter_sigma=sigma,
        )
