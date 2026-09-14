"""Non-holdout generation tests; these never run the product fitter."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import final_fit_result, fit_candidate
from tests.unit.tools.poisson_diagnostic_fixtures import tool as tool

from xrr_fitter.model.parameters import ParameterValue


def test_nonholdout_rng_and_exact_poisson_block_recipe(tool):
    mean = np.arange(1.0, 20.0)
    actual, recipe = tool.cases.draw_counts(mean, "acf", scenario_id=5, seed=37, member_id=0)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260912, 5, 37, 0])))
    expected = np.empty(mean.size, dtype=np.int64)
    for start in range(0, mean.size, 8):
        block = mean[start : start + 8]
        shared_mean = 0.9 * block.min()
        expected[start : start + 8] = rng.poisson(shared_mean) + rng.poisson(block - shared_mean)

    assert np.array_equal(actual, expected)
    assert recipe["kind"] == "shared_poisson_blocks"
    assert recipe["block_size"] == 8
    assert recipe["marginal_distribution"] == "Poisson(mu_i)"


def test_independent_nonholdout_counts_use_the_declared_member_rng(tool):
    mean = np.linspace(0.0, 4.0, 20)
    actual, _recipe = tool.cases.draw_counts(mean, "null_joint", scenario_id=1, seed=37, member_id=1)
    rng = np.random.default_rng(np.random.SeedSequence([20260912, 1, 37, 1]))

    assert np.array_equal(actual, rng.poisson(mean))


def test_surface_and_footprint_generation_use_existing_physics_only(tool):
    theta = np.linspace(0.1, 0.9, 17)
    base = tool.cases.mean_counts(theta, 40.0, "null_single")
    footprint = tool.cases.mean_counts(theta, 40.0, "footprint")
    from xrr_fitter.physics.footprint import footprint_factor

    assert footprint == pytest.approx(base * footprint_factor(theta, 0.30))
    structure = tool.cases.truth_structure("surface")
    assert [layer.thickness_a for layer in structure.components] == [30.0, 100.0]
    assert structure.components[0].material.sld_override_a2.real == 18.9e-6
    assert structure.components[0].roughness_a == 1.0


def test_background_injection_is_nonnegative_raw_mean_not_continuous_count_noise(tool):
    theta = np.linspace(0.1, 0.9, 17)
    base = tool.cases.mean_counts(theta, 40.0, "null_single")
    background = tool.cases.mean_counts(theta, 40.0, "background")
    from xrr_fitter.physics.reflectivity import qz_from_theta_deg

    q = qz_from_theta_deg(theta, 1.5406)
    assert background - base == pytest.approx(2000.0 * q / q.max())


def test_case_builder_rejects_undeclared_seed_before_creating_inputs(tool, tmp_path):
    target = tmp_path / "unused"
    with pytest.raises(ValueError, match="seed"):
        tool.cases.build_case(target, "null_single", 37)
    assert not target.exists()


def test_nonholdout_joint_builder_has_shared_thickness_and_two_local_free_scales(tool, tmp_path):
    fixture = tool.cases.build_observations(tmp_path, "null_joint", 37)
    project = fixture.project

    assert project.master_seed == 10000037
    assert project.batch_mode == "joint"
    assert project.fit_config.budget.diagnostic_samples == 999
    assert project.fit_config.budget.bootstrap_samples == 200
    assert project.fit_config.scale_prior_enabled is False
    assert len(project.sharing_rules) == 1
    assert {ref.parameter_name for ref in project.sharing_rules[0].members} == {tool.cases.TARGET}
    for dataset, observation in zip(project.datasets, fixture.observations, strict=True):
        _assert_local_scale(tool, dataset, observation)


def _assert_local_scale(tool, dataset, observation):
    settings = {value.name: value for value in dataset.parameter_settings}
    free = {value.name for value in settings.values() if not value.locked}
    assert free == {tool.cases.TARGET, "instrument.scale"}
    scale = settings["instrument.scale"]
    amplitude = observation["raw_amplitude"]
    normalization = observation["normalization"]
    assert scale.initial * normalization == pytest.approx(0.9 * amplitude)
    assert scale.lower * normalization == pytest.approx(0.5 * amplitude)
    assert scale.upper * normalization == pytest.approx(1.5 * amplitude)


def test_replay_builder_delegates_current_input_recipe_without_reading_old_projects(tool, monkeypatch, tmp_path):
    sentinel = object()
    calls = []

    def current_builder(root, group, seed):
        calls.append((root, group, seed))
        return sentinel

    monkeypatch.setattr(tool.cases.coverage, "build_case", current_builder)
    assert tool.cases.build_case(tmp_path, "replay_low", 8) is sentinel
    assert calls == [(tmp_path, "poisson_low", 8)]


def _toy_fit(tool, root):
    fixture = tool.cases.build_observations(root / "toy-source", "null_single", 37)
    candidate = replace(fit_candidate(), parameters=(ParameterValue(tool.cases.TARGET, 100.0, 75.0, 125.0),))
    dataset = replace(fixture.project.datasets[0], last_valid_result=final_fit_result(candidate))
    fitted = replace(fixture.project, datasets=(dataset,))
    return SimpleNamespace(updated_project=fitted, warnings=(), cancelled=False)


def test_case_result_is_taken_from_actual_current_api_save_load_not_live_object(tool, monkeypatch, tmp_path):
    fitted = _toy_fit(tool, tmp_path)
    monkeypatch.setattr(tool.cases.api, "fit_project", lambda _project: fitted)
    load = tool.cases.api.load_project
    readbacks = []

    def loaded(path):
        result = load(path)
        readbacks.append(result)
        return result

    monkeypatch.setattr(tool.cases.api, "load_project", loaded)
    fixture = SimpleNamespace(project=fitted.updated_project)
    result = tool.cases._fit_case(fixture, tmp_path, "null_single")

    assert result["fit_available"] is True
    assert result["estimate"] == 100.0
    assert len(readbacks) == 1
    assert readbacks[0] is not fitted.updated_project
    assert result["evidence_source"] == "api.fit_project -> api.save_project -> api.load_project"


def test_roundtrip_failure_keeps_fitted_estimate_and_records_diagnostic_u(tool, monkeypatch, tmp_path):
    fitted = _toy_fit(tool, tmp_path)
    monkeypatch.setattr(tool.cases.api, "fit_project", lambda _project: fitted)

    def broken(*_args):
        raise ValueError("injected persistence failure")

    monkeypatch.setattr(tool.cases.api, "load_project", broken)
    result = tool.cases._fit_case(SimpleNamespace(project=fitted.updated_project), tmp_path, "null_single")

    assert result["fit_available"] is True
    assert result["estimate"] == 100.0
    assert result["failure_stage"] == "evidence_roundtrip"
    assert result["diagnostic"]["status"] == "unavailable"
    assert (tmp_path / "fitted.xrrproj.json").is_file()


def test_nonholdout_input_and_generation_metadata_survive_product_failure(tool, monkeypatch, tmp_path):
    fixture = tool.cases.build_observations(tmp_path / "raw", "null_single", 37)
    monkeypatch.setattr(tool.cases, "build_case", lambda *_args: fixture)
    warning = "optimizer_nonconvergence: injected failure"
    failed = SimpleNamespace(updated_project=fixture.project, warnings=(warning,), cancelled=False)
    monkeypatch.setattr(tool.cases.api, "fit_project", lambda _project: failed)
    result = tool.cases.run_case(tmp_path, "null_single", 37)

    assert result["fit_available"] is False
    assert result["fit_warnings"] == [warning]
    assert warning in result["failure_reason"]
    assert (tmp_path / "observations.json").is_file()
    assert (tmp_path / "input.xrrproj.json").is_file()
    assert (tmp_path / "raw" / "member-0.xy").is_file()


def test_product_exception_preserves_actual_fit_time_in_failed_seed(tool, monkeypatch, tmp_path):
    def broken(_project):
        raise ValueError("numeric product exception")

    times = iter((10.0, 13.0))
    monkeypatch.setattr(tool.cases, "monotonic", lambda: next(times))
    monkeypatch.setattr(tool.cases.api, "fit_project", broken)
    row = tool.cases._fit_case(SimpleNamespace(project=object()), tmp_path, "null_single")

    assert row["fit_available"] is False
    assert row["fit_seconds"] == 3.0
    assert row["diagnostic"]["status"] == "unavailable"
