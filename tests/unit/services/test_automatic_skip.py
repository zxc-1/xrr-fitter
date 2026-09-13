"""Automatic retries must not undo explicit user stop decisions."""

from dataclasses import replace

import pytest
from tests.unit.services.test_automatic_joint import (
    FIT_GROUP_ID,
    _constant_joint_analysis,
    _fit_joint_group,
    _fit_result,
    _objective_outlier_group,
    _prefit,
    _prepared,
    _stub_joint_searches,
)

from xrr_fitter.model.analysis import ConfidenceClass
from xrr_fitter.services.fitting_phases.joint_selection import _accepted_material_values
from xrr_fitter.services.fitting_phases.sharing import automatic_sharing_rules


def _skipped_result(result, stage="E"):
    return replace(result, confidence=ConfidenceClass.UNTRUSTED, skipped_stages=(stage,))


@pytest.mark.parametrize("stage", ["B", "E"])
def test_automatic_joint_does_not_retry_or_accept_terminal_skip(stage) -> None:
    prepared = (_prepared("left", 0, released_imag=()), _prepared("right", 1, released_imag=()))
    prefits = tuple(_prefit(item, 1.0) for item in prepared)
    incomplete = tuple(_skipped_result(_fit_result(item, 2.0), stage) for item in prepared)
    attempts = []

    def run_joint(request, **kwargs):
        attempts.append(request)
        return _stub_joint_searches(request, **kwargs)

    results = _fit_joint_group(
        prepared, prefits, run_joint_fit=run_joint, analyze_joint_searches=_constant_joint_analysis(incomplete)
    )
    assert len(attempts) == 1
    assert all(not result.passed for result in results)
    assert all(result.fit_result.skipped_stages == (stage,) for result in results)
    assert all("skip" in result.reason for result in results)


def test_automatic_isolation_does_not_retry_a_skipped_prefit() -> None:
    prepared, prefits, joint_results = _objective_outlier_group()
    skipped = replace(prefits[2], fit_result=_skipped_result(prefits[2].fit_result), passed=False, reason="skip E")
    prefits = (*prefits[:2], skipped)
    results = _fit_joint_group(
        prepared,
        prefits,
        run_joint_fit=_stub_joint_searches,
        analyze_joint_searches=_constant_joint_analysis(joint_results),
    )
    assert results[2] is skipped


def test_skipped_joint_results_do_not_supply_material_values_to_other_datasets() -> None:
    prepared = (_prepared("left", 0), _prepared("right", 1))
    rules = automatic_sharing_rules(prepared, FIT_GROUP_ID, share_roughness=False)
    results = tuple(_skipped_result(_fit_result(item, 1.0)) for item in prepared)
    assert _accepted_material_values(prepared, results, rules) == {}
