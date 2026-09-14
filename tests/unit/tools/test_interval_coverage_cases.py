"""Real physical inputs, local normalization, and public fitting declarations."""

from __future__ import annotations

from dataclasses import replace
from math import ceil
from types import SimpleNamespace

import numpy as np
import pytest

import xrr_fitter.api as api

TARGET = "component.0.thickness_a"


def test_case_builder_declares_fixed_nuisance_parameters_with_current_api(load_tool_module, tmp_path):
    module = load_tool_module("check_interval_coverage").cases
    fixture, failure = None, None
    try:
        fixture = module.build_case(tmp_path, "poisson_low", 0)
    except TypeError as error:
        failure = str(error)

    assert failure is None, f"Current-API case construction failed: {failure}"
    assert fixture is not None
    settings = fixture.project.datasets[0].parameter_settings
    assert {item.freedom for item in settings if item.name != TARGET} == {api.ParameterFreedom.FIXED}
    assert next(item for item in settings if item.name == TARGET).freedom is api.ParameterFreedom.FREE


@pytest.mark.parametrize("group", ("gaussian_regular", "poisson_low", "poisson_regular", "shared_gaussian"))
def test_real_fixture_uses_physics_truth_and_keeps_count_normalization_local(load_tool_module, tmp_path, group):
    module = load_tool_module("check_interval_coverage").cases
    fixture = module.build_case(tmp_path, group, 0)

    assert fixture.project.master_seed == fixture.project.fit_config.master_seed == 0
    assert fixture.truth == 100.0
    assert fixture.parameter_name == TARGET
    assert fixture.project.fit_config.scale_prior_enabled is False
    assert fixture.project.fit_config.local_workers == 1
    assert len(fixture.project.datasets) == (2 if group == "shared_gaussian" else 1)
    for dataset, metadata in zip(fixture.project.datasets, fixture.observations, strict=True):
        _assert_fixture_dataset(module, fixture, dataset, metadata)


def _assert_fixture_dataset(module, fixture, dataset, metadata):
    free = [item.name for item in dataset.parameter_settings if not item.locked]
    settings = {item.name: item for item in dataset.parameter_settings}
    data = api.import_data(
        dataset.source_path,
        dataset.beam,
        column_mapping=dataset.column_mapping,
        noise_model=fixture.project.fit_config.noise_model,
    )
    assert free == [TARGET]
    assert settings["instrument.scale"].initial * data.normalization == pytest.approx(metadata["raw_amplitude"])
    assert metadata["normalization"] == data.normalization
    _assert_imported_identity(data, metadata)
    assert api.preflight_fit(fixture.project).ready


def _assert_imported_identity(data, metadata):
    assert data.fit_mask.all()
    assert metadata["point_count"] >= 30
    assert metadata["source_sha256"] == data.source_sha256


def test_shared_fixture_only_shares_normalization_invariant_thickness(load_tool_module, tmp_path):
    module = load_tool_module("check_interval_coverage").cases
    fixture = module.build_case(tmp_path, "shared_gaussian", 0)
    rule = fixture.project.sharing_rules[0]

    assert fixture.project.batch_mode == "joint"
    assert tuple(member.parameter_name for member in rule.members) == (TARGET, TARGET)
    assert fixture.project.fit_config.budget.bootstrap_samples == 200
    assert fixture.observations[0]["normalization"] != fixture.observations[1]["normalization"]


def test_robust_fixture_is_fixed_correlated_noise_with_real_block_bootstrap_budget(load_tool_module, tmp_path):
    module = load_tool_module("check_interval_coverage").cases
    fixture = module.build_case(tmp_path, "robust_correlated", 0)

    assert fixture.project.fit_config.noise_model == "robust_log"
    assert fixture.project.fit_config.budget.bootstrap_samples == 200
    assert fixture.observations[0]["noise"]["kind"] == "stationary_ar1_log10"
    assert fixture.observations[0]["noise"]["rho"] == 0.75


def test_fixture_seed_independence_and_replay_are_deterministic(load_tool_module, tmp_path):
    module = load_tool_module("check_interval_coverage").cases
    first = module.build_case(tmp_path / "first", "poisson_low", 0)
    replay = module.build_case(tmp_path / "replay", "poisson_low", 0)
    changed = module.build_case(tmp_path / "changed", "poisson_low", 1)

    assert first.observations == replay.observations
    assert first.observations[0]["source_sha256"] != changed.observations[0]["source_sha256"]
    assert replace(first.project.fit_config, master_seed=1) == changed.project.fit_config


def test_missing_public_result_keeps_exact_fit_warning(load_tool_module, monkeypatch):
    module = load_tool_module("check_interval_coverage").cases
    warning = "ValueError: scored starts must be unique"

    def rejected(project):
        return SimpleNamespace(updated_project=project, warnings=(warning,), cancelled=False)

    monkeypatch.setattr(module.api, "fit_project", rejected)
    row = module.run_case("poisson_regular", 0)

    assert row["fit_available"] is False
    assert row.get("fit_warnings") == [warning]
    assert warning in row["failure_reason"]
    assert row["failure_stage"] == "fit"


def test_real_fit_estimate_is_preserved_if_evidence_persistence_fails(load_tool_module, monkeypatch):
    module = load_tool_module("check_interval_coverage").cases

    def cannot_save(*_args):
        raise ValueError("injected persistence failure")

    monkeypatch.setattr(module.api, "save_project", cannot_save)
    row = module.run_case("gaussian_regular", 0)

    assert row["fit_available"] is True
    assert isinstance(row["estimate"], float)
    assert row["interval"]["available"] is False
    assert row["failure_stage"] == "evidence_roundtrip"
    assert "injected persistence failure" in row["failure_reason"]


def test_shared_target_axis_uses_exact_member_identity_not_global_label(load_tool_module):
    module = load_tool_module("check_interval_coverage").cases
    report = SimpleNamespace(
        correlation_names=("local-looking.component.0.thickness_a", "arbitrary-global-name"),
        parameter_members=(
            (api.ParameterReference("other", TARGET),),
            (api.ParameterReference("member-0", TARGET), api.ParameterReference("member-1", TARGET)),
        ),
    )

    assert module._target_axis(report, "member-0") == "arbitrary-global-name"


@pytest.mark.parametrize("members", (None, ((api.ParameterReference("other", TARGET),),)))
def test_missing_shared_target_identity_is_explicit_not_label_fallback(load_tool_module, members):
    module = load_tool_module("check_interval_coverage").cases
    report = SimpleNamespace(correlation_names=(TARGET,), parameter_members=members)

    with pytest.raises(ValueError, match="joint_"):
        module._target_axis(report, "member-0")


def test_report_preserves_saved_joint_axis_identity_for_external_audit(load_tool_module):
    module = load_tool_module("check_interval_coverage").cases
    members = (api.ParameterReference("member-0", TARGET), api.ParameterReference("member-1", TARGET))
    uncertainty = api.UncertaintyReport(
        ("arbitrary-axis",), [[1.0]], (), (), 0.0, (), (), None, (), parameter_members=(members,)
    )

    report = module._diagnostics(uncertainty)

    assert report.get("correlation_names") == ["arbitrary-axis"]
    assert report["parameter_members"] == [
        [{"dataset_id": value.dataset_id, "parameter_name": TARGET} for value in members]
    ]


def test_normalization_recipe_matches_the_reader_positive_head_percentile(load_tool_module, tmp_path):
    module = load_tool_module("check_interval_coverage").cases
    fixture = module.build_case(tmp_path, "gaussian_regular", 0)
    values = np.asarray(fixture.observations[0]["raw_observations"])
    positive = values[values > 0.0]
    head_count = min(positive.size, max(20, ceil(0.10 * positive.size)))
    expected = float(np.percentile(positive[:head_count], 95))

    assert fixture.observations[0]["normalization"] == expected
    assert expected != float(np.max(values))
    assert "95th percentile" in module.experiment_definition()["normalization"]
    assert "observed maximum" not in module.experiment_definition()["normalization"]
