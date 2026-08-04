from __future__ import annotations

from math import sqrt

import pytest
from tests.support.automatic_recovery import (
    parameter_value,
    run_direct_sld_recovery,
    run_isolated_outlier_recovery,
    run_roughness_release_recovery,
    run_shared_local_recovery,
)
from tests.support.synthetic_recovery import build_corpus

import xrr_fitter.api as api


def test_direct_sld_recovery_is_accurate_without_inventing_mass_density(
    tmp_path,
) -> None:
    run = run_direct_sld_recovery(tmp_path)
    dataset = run.project.datasets[0]
    layer = run.summary.datasets[0].layers[0]

    assert dataset.automation.status is api.AutomaticStatus.PASSED
    assert parameter_value(dataset, "component.0.thickness_a") == pytest.approx(
        130.0,
        rel=0.02,
    )
    assert parameter_value(dataset, "component.0.sld_real_a2") == pytest.approx(
        24e-6,
        rel=0.03,
    )
    assert layer.nominal_density_g_cm3 is None
    assert layer.fitted_density_g_cm3 is None
    assert layer.density_note == "配比未知，无法换算"
    assert run.bootstrap_count == 0
    assert run.profile_count <= 4


def test_shared_material_recovery_keeps_thickness_local_and_statistics_exact(
    tmp_path,
) -> None:
    run = run_shared_local_recovery(tmp_path)
    expected_thicknesses = (90.0, 100.0, 110.0, 120.0)
    recovered_thicknesses = tuple(
        parameter_value(dataset, "component.0.thickness_a")
        for dataset in run.project.datasets
    )
    recovered_densities = tuple(
        parameter_value(dataset, "component.0.density_scale")
        for dataset in run.project.datasets
    )

    assert all(
        dataset.automation.status is api.AutomaticStatus.PASSED
        and dataset.automation.role is api.AutomaticRole.JOINT
        and dataset.automation.statistics_member
        for dataset in run.project.datasets
    )
    assert recovered_thicknesses == pytest.approx(expected_thicknesses, rel=0.02)
    assert recovered_densities == pytest.approx((0.93,) * 4, rel=0.03)
    assert len(set(recovered_densities)) == 1

    uniformity = run.summary.uniformity[0]
    assert uniformity.count == 4
    assert uniformity.mean_thickness_a == pytest.approx(105.0, abs=0.05)
    assert uniformity.minimum_thickness_a == pytest.approx(90.0, abs=0.05)
    assert uniformity.maximum_thickness_a == pytest.approx(120.0, abs=0.05)
    assert uniformity.population_std_a == pytest.approx(sqrt(125.0), abs=0.05)


def test_model_error_outlier_is_isolated_and_excluded_from_statistics(
    tmp_path,
) -> None:
    run = run_isolated_outlier_recovery(tmp_path)
    members = run.project.datasets[:3]
    outlier = run.project.datasets[3]

    assert all(
        dataset.automation.status is api.AutomaticStatus.PASSED
        and dataset.automation.statistics_member
        for dataset in members
    )
    assert outlier.automation.role is api.AutomaticRole.ISOLATED_RETRY
    assert outlier.automation.status is api.AutomaticStatus.REVIEW
    assert outlier.automation.statistics_member is False
    assert "systematic residual" in outlier.automation.reason
    assert run.summary.uniformity[0].count == 3


def test_roughness_conflict_releases_sharing_and_recovers_local_values(
    tmp_path,
) -> None:
    run = run_roughness_release_recovery(tmp_path)
    expected = (2.0, 3.0, 8.0, 9.0)
    recovered = tuple(
        parameter_value(dataset, "component.0.roughness_a")
        for dataset in run.project.datasets
    )

    assert all(
        dataset.automation.status is api.AutomaticStatus.PASSED
        and dataset.automation.role is api.AutomaticRole.JOINT
        for dataset in run.project.datasets
    )
    assert recovered == pytest.approx(expected, abs=1.0)
    assert max(recovered) - min(recovered) > 5.0


def test_automatic_cases_do_not_change_the_existing_220_case_corpus() -> None:
    assert len(build_corpus()) == 220


def test_automatic_work_counts_are_deterministic_across_two_runs(tmp_path) -> None:
    first = run_direct_sld_recovery(tmp_path / "first")
    second = run_direct_sld_recovery(tmp_path / "second")

    assert first.work_signature == second.work_signature
    assert first.bootstrap_count == 0
    assert first.profile_count <= 4
