"""Immutable Poisson diagnostic evidence, independent from interval bootstrap.

The fixture is deliberately numerical rather than a physical model fit. These
tests exercise the value boundary, not the statistical adequacy of the max
calibration or the optimizer used to populate it. End-to-end calibration tests
must supply their own generated observations and replayable null statistics.

The default available fixture has B=999 and a family p-value of 8/1000. Its
background detector rejects at the fixed family alpha, while its ACF detector
does not. The same complete evidence is attached to several dataset members to
check that the family decision is never copied blindly to an innocent member.

Raw trend warnings remain advisory even after calibrated nonrejection. An
unavailable result has unknown effective flags, not false flags; the finite
raw statistic is still available to explain why calibration was requested.
Neither an unfinished Monte Carlo axis nor failed refits may publish p-values.

Invalid-value matrices cover integer/bool ambiguity, exact count accounting,
inclusive ties, fixed method versions, and independent bootstrap budgets.
Pickle tests intentionally corrupt frozen values before serialization so the
reconstruction path must reapply invariants instead of trusting slot state.

Full ModelEvaluation reporting axes can contain matched q/model NaN outside
the model mask. This is not a fitted-residual failure. The refit contract owns
read-only copies and rejects nonfinite fitted residuals without silently
discarding legitimate masked reporting rows.
"""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, fields, replace
from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest
from tests.support.diagnostic_work_cases import completed_work, failed_work, zero_work
from tests.support.model_cases import fit_candidate

from xrr_fitter.model.diagnostic_work import combine_refit_work
from xrr_fitter.model.evaluation import ModelEvaluation
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.inference import ResidualEvidence
from xrr_fitter.model.instrument import PhysicsDiagnostic


def _module():
    name = "xrr_fitter.model.diagnostic_calibration"
    assert find_spec(name) is not None, "the immutable diagnostic evidence model must exist"
    return import_module(name)


def _statistic(dataset_id="curve", kind="background", adjusted_p_value=None, **changes):
    values = dict(dataset_id=dataset_id, kind=kind, observed=2.0)
    if adjusted_p_value is not None:
        values.update(center=0.0, scale=1.0, adjusted_p_value=adjusted_p_value)
    return _module().DiagnosticStatistic(**(values | changes))


def _available(**changes):
    samples = changes.get("sample_count", 999)
    samples = samples if isinstance(samples, int) and samples > 0 else 999
    values = dict(
        status="available",
        sample_count=999,
        child_seed=7,
        owner_sha256="a" * 64,
        attempted_count=999,
        successful_count=999,
        statistics=(_statistic(adjusted_p_value=0.008), _statistic(kind="acf", adjusted_p_value=0.2)),
        tail_count=8,
        tie_count=1,
        observed_score=2.0,
        null_statistics_sha256="b" * 64,
        refit_nfev=12 * (samples + 1),
        observed_work=completed_work(),
        null_work=completed_work(4 * samples),
        refit_discrepancy=(1e-8, 1e-9, 1e-9),
    )
    return _module().DiagnosticCalibration(**(values | changes))


def _unavailable(**changes):
    values = dict(
        status="unavailable",
        sample_count=999,
        child_seed=7,
        owner_sha256="a" * 64,
        statistics=(_statistic(),),
        unavailable_reason="diagnostic_refitter_missing",
    )
    return _module().DiagnosticCalibration(**(values | changes))


def _evaluation(**changes):
    candidate = fit_candidate()
    values = dict(
        valid=True,
        reason="evaluated",
        parameters=candidate.parameters,
        qz_a_inv=candidate.qz_a_inv,
        model_normalized=candidate.model_normalized,
        fit_residuals=candidate.residuals,
        fit_weighted_residuals=candidate.weighted_residuals,
        objective=candidate.objective,
        expanded_stack=None,
        diagnostics=(),
        noise_model="poisson",
    )
    return ModelEvaluation(**(values | changes))


def _residual(calibration=None, **changes):
    values = dict(
        dataset_id="curve",
        executed=True,
        systematic=True,
        autocorrelation=False,
        point_count=100,
        raw_systematic=True,
        raw_autocorrelation=False,
        advisories=(PhysicsDiagnostic("suspected_diffuse_background", "raw tail screen"),),
        calibration=calibration,
        owner_sha256="a" * 64,
    )
    return ResidualEvidence(**(values | changes))


def test_diagnostic_budget_is_separate_from_interval_bootstrap() -> None:
    config = FitConfig.fast(7)

    assert config.budget.bootstrap_samples == 8
    assert getattr(config.budget, "diagnostic_samples", None) == 999
    assert getattr(config, "diagnostic_version", None) == "poisson-refit-null-v4"
    assert getattr(FitConfig.standard(7).budget, "diagnostic_samples", None) == 999


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_diagnostic_budget_rejects_nonpositive_or_noninteger_counts(value) -> None:
    assert "diagnostic_samples" in SearchBudget.__dataclass_fields__
    with pytest.raises(ValueError, match="diagnostic_samples"):
        replace(FitConfig.fast(7).budget, diagnostic_samples=value)


def test_diagnostic_budget_accepts_explicit_small_budget_for_unavailable_evidence() -> None:
    assert "diagnostic_samples" in SearchBudget.__dataclass_fields__
    assert replace(FitConfig.fast(7).budget, diagnostic_samples=8).diagnostic_samples == 8


@pytest.mark.parametrize("value", ["", "poisson-refit-null-v1", "poisson-refit-null-v2", None, 1])
def test_diagnostic_version_rejects_unknown_algorithm(value) -> None:
    assert "diagnostic_version" in FitConfig.__dataclass_fields__
    with pytest.raises(ValueError, match="diagnostic_version"):
        replace(FitConfig.fast(7), diagnostic_version=value)


def test_budget_and_config_revalidate_new_fields_after_pickle() -> None:
    config = FitConfig.fast(7)
    assert "diagnostic_version" in config.__dataclass_fields__
    assert pickle.loads(pickle.dumps(config)) == config
    corrupt_budget = replace(config.budget)
    object.__setattr__(corrupt_budget, "diagnostic_samples", 0)
    with pytest.raises(ValueError, match="diagnostic_samples"):
        pickle.loads(pickle.dumps(corrupt_budget))
    object.__setattr__(config, "diagnostic_version", "unknown")
    with pytest.raises(ValueError, match="diagnostic_version"):
        pickle.loads(pickle.dumps(config))


def test_raw_statistic_accepts_signed_observed_and_no_calibration() -> None:
    statistic = _statistic(dataset_id=None, observed=-2.0)

    assert statistic.observed == -2.0
    assert statistic.center is statistic.scale is statistic.adjusted_p_value is None
    assert pickle.loads(pickle.dumps(statistic)) == statistic
    with pytest.raises(FrozenInstanceError):
        statistic.observed = 0.0


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_id": " "},
        {"dataset_id": 7},
        {"kind": "unknown"},
        {"observed": float("nan")},
        {"observed": float("inf")},
        {"observed": True},
        {"center": 0.0},
        {"center": 0.0, "scale": 1.0},
        {"center": float("nan"), "scale": 1.0, "adjusted_p_value": 0.1},
        {"center": 0.0, "scale": -0.01, "adjusted_p_value": 0.1},
        {"center": 0.0, "scale": float("inf"), "adjusted_p_value": 0.1},
        {"center": 0.0, "scale": 1.0, "adjusted_p_value": -0.1},
        {"center": 0.0, "scale": 1.0, "adjusted_p_value": 1.1},
    ],
)
def test_statistic_rejects_invalid_or_partial_calibration(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        _statistic(**changes)


def test_available_calibration_derives_tail_probability_decision_and_resolution() -> None:
    calibration = _available()

    assert calibration.p_value == 0.008
    assert calibration.rejected is True
    assert calibration.resolution == 0.001
    assert calibration.method == "poisson_refit_null_rms_v4"
    assert calibration.refit_policy == "declared_sobol4_lbfgsb_trf_v3"
    assert pickle.loads(pickle.dumps(calibration)) == calibration
    with pytest.raises(FrozenInstanceError):
        calibration.tail_count = 999


def test_available_calibration_preserves_all_ties_and_boundary_alpha() -> None:
    tied = _available(
        statistics=(_statistic(adjusted_p_value=1.0),), tail_count=1000, tie_count=1000, observed_score=0.0
    )
    at_alpha = _available(
        sample_count=99,
        attempted_count=99,
        successful_count=99,
        statistics=(_statistic(adjusted_p_value=0.01),),
        tail_count=1,
    )

    assert tied.p_value == 1.0
    assert tied.rejected is False
    assert at_alpha.rejected is True
    assert at_alpha.resolution == 0.01


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "pending"},
        {"sample_count": True},
        {"sample_count": 98, "attempted_count": 98, "successful_count": 98},
        {"attempted_count": 998},
        {"successful_count": 998},
        {"child_seed": -1},
        {"child_seed": True},
        {"tail_count": 0},
        {"tail_count": 1001},
        {"tail_count": True},
        {"tail_count": 8.0},
        {"tie_count": 0},
        {"tie_count": 9},
        {"tie_count": True},
        {"observed_score": None},
        {"observed_score": -1.0},
        {"observed_score": float("inf")},
        {"owner_sha256": "A" * 64},
        {"null_statistics_sha256": None},
        {"null_statistics_sha256": "not a hash"},
        {"provenance_sha256": "bad"},
        {"method": "other"},
        {"method": "poisson_refit_null_v1"},
        {"refit_policy": "warm_start"},
        {"refit_policy": "declared_sobol4_v1"},
        {"alpha": 0.05},
        {"alpha": float("nan")},
        {"failure_reasons": ((1, "failed"),)},
        {"unavailable_reason": "failed"},
        {"refit_nfev": -1},
        {"refit_nfev": True},
        {"refit_discrepancy": (0.0, 0.0)},
        {"refit_discrepancy": (0.0, -1.0, 0.0)},
        {"refit_discrepancy": (0.0, float("nan"), 0.0)},
    ],
)
def test_available_calibration_rejects_invalid_complete_evidence(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        _available(**changes)


@pytest.mark.parametrize("statistics", ["empty", "raw", "duplicate", "below", "not-minimum", "wrong-type"])
def test_available_requires_unique_complete_statistics_and_matching_omnibus_p(statistics) -> None:
    values = {
        "empty": (),
        "raw": (_statistic(),),
        "duplicate": (_statistic(adjusted_p_value=0.008),) * 2,
        "below": (_statistic(adjusted_p_value=0.007),),
        "not-minimum": (_statistic(adjusted_p_value=0.009),),
        "wrong-type": (object(),),
    }
    with pytest.raises((TypeError, ValueError)):
        _available(statistics=values[statistics])


def test_unavailable_calibration_never_publishes_tail_or_member_probabilities() -> None:
    calibration = _unavailable(
        attempted_count=3,
        successful_count=2,
        failure_reasons=[[2, "diagnostic_refit_failed"]],
        unavailable_reason="diagnostic_refit_failed",
        refit_nfev=37,
        observed_work=completed_work(),
        null_work=combine_refit_work(completed_work(8), failed_work()),
    )

    assert calibration.p_value is calibration.rejected is None
    assert calibration.tail_count is calibration.tie_count is calibration.observed_score is None
    assert calibration.failure_reasons == ((2, "diagnostic_refit_failed"),)
    assert calibration.statistics[0].adjusted_p_value is None
    assert pickle.loads(pickle.dumps(calibration)) == calibration


def test_unavailable_accepts_observed_failure_and_budget_insufficiency_without_null_attempts() -> None:
    observed = _unavailable(failure_reasons=((-1, "diagnostic_refit_failed"),))
    budget = _unavailable(sample_count=8, unavailable_reason="diagnostic_budget_insufficient", statistics=())

    assert observed.attempted_count == observed.successful_count == 0
    assert budget.sample_count == 8
    assert budget.p_value is None


@pytest.mark.parametrize(
    "changes",
    [
        {"unavailable_reason": None},
        {"unavailable_reason": " "},
        {"tail_count": 1},
        {"tie_count": 1},
        {"observed_score": 0.0},
        {"attempted_count": 1},
        {"successful_count": 1},
        {"attempted_count": 1000, "successful_count": 1000},
        {"failure_reasons": ((0, "failed"),)},
        {"failure_reasons": ((-2, "failed"),)},
        {"failure_reasons": ((-1, ""),)},
        {"failure_reasons": ((-1, "failed"), (-1, "failed"))},
        {"failure_reasons": ((True, "failed"),)},
        {"failure_reasons": ((-1, "failed", "extra"),)},
        {"attempted_count": 1, "successful_count": 1, "failure_reasons": ((-1, "failed"),)},
        {"attempted_count": 1, "successful_count": 0, "failure_reasons": ((1, "failed"),)},
    ],
)
def test_unavailable_rejects_fabricated_tails_or_unaccounted_failures(changes) -> None:
    with pytest.raises((TypeError, ValueError)):
        _unavailable(**changes)


def test_unavailable_rejects_a_calibrated_statistic() -> None:
    with pytest.raises(ValueError):
        _unavailable(statistics=(_statistic(adjusted_p_value=0.008),))


def test_calibration_owns_sequence_inputs_and_revalidates_on_pickle() -> None:
    statistics = [_statistic(adjusted_p_value=0.008)]
    discrepancy = [0.0, 0.0, 0.0]
    calibration = _available(statistics=statistics, refit_discrepancy=discrepancy)
    statistics.clear()
    discrepancy[0] = 1.0

    assert len(calibration.statistics) == 1
    assert calibration.refit_discrepancy == (0.0, 0.0, 0.0)
    object.__setattr__(calibration, "tail_count", 999)
    with pytest.raises(ValueError):
        pickle.loads(pickle.dumps(calibration))


def test_statistic_revalidates_on_pickle() -> None:
    statistic = _statistic()
    object.__setattr__(statistic, "observed", float("nan"))
    with pytest.raises(ValueError):
        pickle.loads(pickle.dumps(statistic))


def test_diagnostic_refit_owns_unit_and_evaluation_arrays_after_pickle() -> None:
    unit = np.array([0.5])
    evaluations = [_evaluation()]
    refit = _module().DiagnosticRefit(unit, evaluations, nfev=14, attempted_paths=4, work=completed_work(nfev=14))
    unit[0] = 0.9
    evaluations.clear()
    restored = pickle.loads(pickle.dumps(refit))

    for value in (refit, restored):
        np.testing.assert_array_equal(value.unit_vector, [0.5])
        assert not value.unit_vector.flags.writeable
        assert len(value.evaluations) == 1
        assert not value.evaluations[0].fit_residuals.flags.writeable
        assert value.failure_reason is None
    with pytest.raises(FrozenInstanceError):
        refit.nfev = 1


def test_diagnostic_refit_zero_dimension_can_succeed_without_optimizer_evaluations() -> None:
    refit = _module().DiagnosticRefit(np.empty(0), (_evaluation(),), nfev=0, attempted_paths=1, work=zero_work())
    assert refit.unit_vector.shape == (0,)
    assert refit.nfev == 0


def test_diagnostic_refit_failure_has_no_partial_success_payload() -> None:
    refit = _module().DiagnosticRefit(
        None,
        (),
        nfev=14,
        attempted_paths=2,
        failure_reason="no_convergence",
        work=combine_refit_work(completed_work(1, nfev=7), failed_work(7)),
    )
    assert refit.unit_vector is None
    assert refit.evaluations == ()
    assert refit.nfev == 14
    assert refit.attempted_paths == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"unit_vector": None},
        {"unit_vector": np.array([np.nan])},
        {"unit_vector": np.array([-0.1])},
        {"unit_vector": np.array([1.1])},
        {"unit_vector": np.ones((1, 1))},
        {"evaluations": ()},
        {"evaluations": (object(),)},
        {"nfev": -1},
        {"nfev": True},
        {"attempted_paths": 0},
        {"attempted_paths": 5},
        {"attempted_paths": True},
        {"failure_reason": ""},
        {"failure_reason": "failed"},
    ],
)
def test_diagnostic_refit_rejects_invalid_success_or_partial_failure(changes) -> None:
    values = dict(
        unit_vector=np.array([0.5]), evaluations=(_evaluation(),), nfev=3, attempted_paths=1, work=completed_work(1)
    )
    with pytest.raises((TypeError, ValueError)):
        _module().DiagnosticRefit(**(values | changes))


@pytest.mark.parametrize("changes", [{"valid": False}, {"fit_residuals": np.full(4, np.nan)}])
def test_diagnostic_refit_rejects_invalid_or_nonfinite_fitted_evaluations(changes) -> None:
    evaluation = _evaluation(**changes)
    with pytest.raises(ValueError):
        _module().DiagnosticRefit(np.array([0.5]), (evaluation,), nfev=3, attempted_paths=1, work=completed_work(1))


def test_diagnostic_refit_retains_legal_unmodeled_axis_nan() -> None:
    evaluation = _evaluation(
        qz_a_inv=np.array([np.nan, 0.1, 0.2, 0.3]),
        model_normalized=np.array([np.nan, 0.1, 0.2, 0.3]),
        fit_residuals=np.zeros(3),
        fit_weighted_residuals=np.zeros(3),
    )
    refit = _module().DiagnosticRefit(np.array([0.5]), (evaluation,), nfev=3, attempted_paths=1, work=completed_work(1))

    assert np.isnan(refit.evaluations[0].model_normalized[0])
    assert np.all(np.isfinite(refit.evaluations[0].fit_residuals))


def test_diagnostic_refit_revalidates_on_pickle() -> None:
    refit = _module().DiagnosticRefit(
        np.array([0.5]), (_evaluation(),), nfev=3, attempted_paths=1, work=completed_work(1)
    )
    object.__setattr__(refit, "unit_vector", np.array([float("inf")]))
    with pytest.raises(ValueError):
        pickle.loads(pickle.dumps(refit))


def test_residual_fields_append_without_changing_standalone_unexecuted_evidence() -> None:
    assert [field.name for field in fields(ResidualEvidence)][7:] == [
        "raw_systematic",
        "raw_autocorrelation",
        "advisories",
        "calibration",
        "owner_sha256",
    ]
    residual = ResidualEvidence(None, False, None, None, 0, unavailable_reason="not_requested")
    assert residual.calibration is None
    assert residual.raw_systematic is residual.raw_autocorrelation is None
    assert residual.advisories == ()


def test_calibrated_nonrejection_preserves_raw_advisory_without_effective_alarm() -> None:
    calibration = _available(statistics=(_statistic(adjusted_p_value=0.2),), tail_count=200)
    residual = _residual(calibration, systematic=False)

    assert residual.raw_systematic is True
    assert residual.systematic is False
    assert residual.diagnostics == ()
    assert residual.advisories[0].code == "suspected_diffuse_background"
    assert pickle.loads(pickle.dumps(residual)) == residual


def test_residual_effective_flags_are_specific_to_the_member_not_omnibus() -> None:
    calibration = _available(
        statistics=(
            _statistic("first", adjusted_p_value=0.008),
            _statistic("first", "acf", adjusted_p_value=0.5),
            _statistic("second", adjusted_p_value=0.5),
            _statistic("second", "acf", adjusted_p_value=0.5),
        )
    )

    first = _residual(calibration, dataset_id="first")
    second = _residual(calibration, dataset_id="second", systematic=False)
    assert first.systematic is True
    assert second.systematic is False
    assert second.autocorrelation is False
    with pytest.raises(ValueError, match="calibrat|statistic|member"):
        _residual(calibration, dataset_id="unknown")


def test_calibrated_acf_rejection_also_contributes_to_systematic_flag() -> None:
    calibration = _available(statistics=(_statistic(kind="acf", adjusted_p_value=0.008),))
    residual = _residual(calibration, autocorrelation=True)
    assert residual.systematic is residual.autocorrelation is True
    with pytest.raises(ValueError):
        _residual(calibration, systematic=False, autocorrelation=True)


def test_unavailable_calibration_cannot_be_interpreted_as_false_diagnostic() -> None:
    calibration = _unavailable()
    residual = _residual(
        calibration,
        executed=False,
        systematic=None,
        autocorrelation=None,
        unavailable_reason=calibration.unavailable_reason,
    )
    assert not residual.executed
    assert residual.systematic is residual.autocorrelation is None
    assert residual.raw_systematic is True
    with pytest.raises(ValueError):
        _residual(calibration, systematic=False)
    with pytest.raises(ValueError):
        replace(residual, unavailable_reason="other")


@pytest.mark.parametrize(
    "changes",
    [
        {"raw_systematic": 1},
        {"raw_autocorrelation": "false"},
        {"advisories": ("message",)},
        {"calibration": "available"},
        {"owner_sha256": "bad"},
    ],
)
def test_residual_rejects_untyped_raw_evidence(changes) -> None:
    assert "raw_systematic" in ResidualEvidence.__dataclass_fields__
    with pytest.raises((TypeError, ValueError)):
        _residual(**changes)


def test_residual_owns_advisories_and_revalidates_calibrated_flags_after_pickle() -> None:
    source = [PhysicsDiagnostic("suspected_diffuse_background", "raw screen")]
    residual = _residual(_available(), advisories=source)
    source.clear()
    assert len(residual.advisories) == 1
    object.__setattr__(residual, "systematic", False)
    with pytest.raises(ValueError):
        pickle.loads(pickle.dumps(residual))


def test_derived_advisory_codes_are_shared_and_exclude_physical_failures() -> None:
    codes = _module().RESIDUAL_ADVISORY_CODES
    assert isinstance(codes, frozenset)
    assert codes == frozenset(
        {"suspected_unmodeled_footprint", "suspected_diffuse_background", "surface_thin_layer_residual"}
    )
    assert "nevot_croce_applicability_exceeded" not in codes
