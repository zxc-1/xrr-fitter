"""Current calibration seals and counts include the complete path-work evidence."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest
from tests.support.diagnostic_work_cases import completed_work, failed_work, zero_work
from tests.unit.model.test_diagnostic_calibration import _evaluation
from tests.unit.model.test_diagnostic_work import _work

from xrr_fitter.io import codec_diagnostic_calibration as codec
from xrr_fitter.model.diagnostic_calibration import DiagnosticCalibration, DiagnosticRefit, DiagnosticStatistic
from xrr_fitter.model.diagnostic_work import DiagnosticRefitWork, combine_refit_work
from xrr_fitter.model.provenance import diagnostic_calibration_sha256

EMPTY_REFIT_WORK = DiagnosticRefitWork()


def _calibration(**changes):
    assert {"observed_work", "null_work"} <= DiagnosticCalibration.__dataclass_fields__.keys()
    values = dict(
        status="available",
        sample_count=99,
        child_seed=5,
        owner_sha256="a" * 64,
        attempted_count=99,
        successful_count=99,
        statistics=(DiagnosticStatistic("curve", "acf", 0.5, 0.0, 1.0, 0.01),),
        tail_count=1,
        tie_count=1,
        observed_score=2.0,
        null_statistics_sha256="b" * 64,
        observed_work=_work(count=4),
        null_work=_work(count=396),
        refit_nfev=2000,
    )
    result = DiagnosticCalibration(**(values | changes))
    return replace(result, provenance_sha256=diagnostic_calibration_sha256(result))


def _failed_calibration(observed, null=EMPTY_REFIT_WORK, *, attempted=16, successful=0):
    indices = range(successful, attempted) if attempted else (-1,)
    result = DiagnosticCalibration(
        "unavailable",
        99,
        5,
        "a" * 64,
        attempted_count=attempted,
        successful_count=successful,
        unavailable_reason="diagnostic_null_refit_failed" if attempted else "observed_failure",
        failure_reasons=tuple((index, "fixture failure") for index in indices),
        observed_work=observed,
        null_work=null,
        refit_nfev=observed.nfev + null.nfev,
    )
    return replace(result, provenance_sha256=diagnostic_calibration_sha256(result))


def test_runtime_refit_cannot_claim_paths_without_their_actual_work():
    with pytest.raises(ValueError, match="work"):
        DiagnosticRefit(np.empty(0), (_evaluation(),), 0, 1)


@pytest.mark.parametrize("unit,work", [([0.5], zero_work()), ([], completed_work(1))])
def test_runtime_refit_rejects_a_unit_dimension_inconsistent_with_its_work(unit, work):
    with pytest.raises(ValueError, match="dimension"):
        DiagnosticRefit(np.array(unit), (_evaluation(),), work.nfev, work.path_count, work=work)


@pytest.mark.parametrize("zero", [True, False])
def test_runtime_refit_rechecks_unit_and_work_dimension_on_pickle(zero):
    work = zero_work() if zero else completed_work(1)
    unit = np.empty(0) if zero else np.array([0.5])
    refit = DiagnosticRefit(unit, (_evaluation(),), work.nfev, work.path_count, work=work)
    object.__setattr__(refit, "unit_vector", np.array([0.5]) if zero else np.empty(0))

    with pytest.raises(ValueError, match="dimension"):
        pickle.loads(pickle.dumps(refit))


@pytest.mark.parametrize("unit", [(), (0.5,), (0.2, 0.7)])
def test_runtime_refit_keeps_valid_zero_and_nonzero_successful_prefixes(unit):
    work = zero_work() if not unit else completed_work(1)
    refit = DiagnosticRefit(np.array(unit), (_evaluation(),), work.nfev, work.path_count, work=work)
    restored = pickle.loads(pickle.dumps(refit))

    np.testing.assert_array_equal(restored.unit_vector, unit)
    assert not restored.unit_vector.flags.writeable
    assert restored.work == work and restored.failure_reason is None


@pytest.mark.parametrize("work", [zero_work(), failed_work()])
def test_failed_refit_keeps_work_without_publishing_a_unit_dimension(work):
    refit = DiagnosticRefit(None, (), work.nfev, work.path_count, "publication_failure", work)
    restored = pickle.loads(pickle.dumps(refit))

    assert restored.unit_vector is None and restored.evaluations == ()
    assert restored.work == work and restored.failure_reason == "publication_failure"


@pytest.mark.parametrize("observed", [DiagnosticRefitWork(), failed_work(), completed_work(2)])
def test_even_generation_only_null_attempts_require_a_complete_observed_refit(observed):
    with pytest.raises(ValueError, match="observed"):
        _failed_calibration(observed)


@pytest.mark.parametrize("observed", [completed_work(), zero_work()])
def test_generation_only_failures_keep_observed_work_and_no_invented_null_work(observed):
    calibration = _failed_calibration(observed)
    assert calibration.refit_nfev == observed.nfev
    assert calibration.null_work == DiagnosticRefitWork()
    assert codec.calibration_from_dict(codec.calibration_to_dict(calibration)) == calibration


def test_runtime_refit_cannot_contain_work_after_its_first_failed_path():
    failed = failed_work()
    multiple = combine_refit_work(failed, failed)
    with pytest.raises(ValueError, match="failed|failure"):
        DiagnosticRefit(None, (), multiple.nfev, multiple.path_count, "failed", multiple)


def test_runtime_refit_cannot_attempt_more_than_its_declared_axis():
    work = completed_work(declared_paths=3)
    with pytest.raises(ValueError, match="declared"):
        DiagnosticRefit(np.array([0.5]), (_evaluation(),), work.nfev, work.path_count, work=work)


@pytest.mark.parametrize("owner", ["observed", "null"])
def test_single_refit_cannot_store_multiple_incomplete_paths(owner):
    failed = failed_work()
    multiple = combine_refit_work(failed, failed, failed, failed)
    with pytest.raises(ValueError, match="failed|failure"):
        if owner == "observed":
            _failed_calibration(multiple, attempted=0)
        else:
            _failed_calibration(completed_work(), multiple, attempted=1)


def test_null_incomplete_paths_cannot_exceed_the_failed_replicate_count():
    multiple = combine_refit_work(completed_work(60), failed_work(), failed_work())
    with pytest.raises(ValueError, match="failed|failure"):
        _failed_calibration(completed_work(), multiple, attempted=16, successful=15)


def test_post_solver_failure_can_retain_completed_paths_without_a_failed_native_exit():
    null = completed_work(64)
    calibration = _failed_calibration(completed_work(), null, attempted=16, successful=15)
    assert calibration.null_work.completed_paths == 64
    assert calibration.p_value is None
    assert codec.calibration_from_dict(codec.calibration_to_dict(calibration)) == calibration


def test_current_calibration_preserves_joint_work_histograms_in_its_strict_codec():
    calibration = _calibration()
    payload = codec.calibration_to_dict(calibration)
    assert payload["observed_work"]["declared_paths"] == 4
    assert payload["null_work"]["paths"][0]["count"] == 396
    assert codec.calibration_from_dict(payload) == calibration
    for field in ("observed_work", "null_work"):
        missing = dict(payload)
        del missing[field]
        with pytest.raises(ValueError):
            codec.calibration_from_dict(missing)


@pytest.mark.parametrize(
    "change",
    [
        {"refit_nfev": 1999},
        {"observed_count": 3},
        {"null_count": 395},
        {"null_count": 397},
    ],
)
def test_available_calibration_requires_exact_observed_null_and_nfev_accounting(change):
    values = dict(change)
    if "observed_count" in values:
        values["observed_work"] = _work(count=values.pop("observed_count"))
    if "null_count" in values:
        values["null_work"] = _work(count=values.pop("null_count"))
    with pytest.raises(ValueError):
        _calibration(**values)


def test_abnormal_localization_is_retained_without_invalidating_successful_refinement():
    work = _work()
    path = work.paths[0]
    abnormal = replace(path.stages[0], status=2, success=False, message="ABNORMAL")
    changed = replace(work, paths=(replace(path, stages=(abnormal, *path.stages[1:])),))
    calibration = _calibration(observed_work=changed)
    assert calibration.status == "available"
    assert not calibration.observed_work.paths[0].stages[0].success
    assert calibration.observed_work.paths[0].stages[-1].success


def test_stage_work_is_sealed_and_nested_fields_are_never_defaulted():
    calibration = _calibration()
    payload = codec.calibration_to_dict(calibration)
    payload["null_work"]["paths"][0]["stages"][0]["message"] = "changed exit"
    with pytest.raises(ValueError, match="seal|provenance"):
        codec.calibration_from_dict(payload)
    payload = codec.calibration_to_dict(calibration)
    del payload["null_work"]["paths"][0]["stages"][0]["kind"]
    with pytest.raises(ValueError):
        codec.calibration_from_dict(payload)


@pytest.mark.parametrize("corruption", ["native_status", "declared_axis"])
def test_even_a_recomputed_seal_cannot_make_impossible_solver_work_loadable(corruption):
    calibration = _calibration()
    payload = codec.calibration_to_dict(calibration)
    if corruption == "native_status":
        object.__setattr__(calibration.observed_work.paths[0].stages[-1], "status", 0)
        payload["observed_work"]["paths"][0]["stages"][-1]["status"] = 0
    else:
        object.__setattr__(calibration.observed_work, "declared_paths", 1)
        payload["observed_work"]["declared_paths"] = 1
    payload["provenance_sha256"] = diagnostic_calibration_sha256(calibration)
    with pytest.raises(ValueError, match="status|dimensional"):
        codec.calibration_from_dict(payload)


@pytest.mark.parametrize("work_field", ["observed_work", "null_work"])
def test_project_preflight_allows_only_the_nested_solver_status_to_be_null(work_field):
    from xrr_fitter.io.codec_common import _allows_null

    path = (
        "datasets",
        0,
        "last_valid_result",
        "uncertainty",
        "member_residuals",
        0,
        "calibration",
        work_field,
        "paths",
        0,
        "stages",
        1,
        "status",
    )
    assert _allows_null(path)
    assert not _allows_null(("calibration", "status"))
    assert not _allows_null((*path[:-1], "kind"))
    assert not _allows_null((*path[:-7], "calibration", "unrelated_work", *path[-5:]))
    assert not _allows_null((*path[:-2], "not_an_index", "status"))
