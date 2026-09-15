"""Public configuration changes invalidate evidence, not user declarations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, final_fit_result, project, simple_structure

import xrr_fitter.api as api
from xrr_fitter.model.fitting import FitCheckpoint
from xrr_fitter.services import fitting, projects


def _completed_project(batch_mode: str = "independent") -> api.XrrProject:
    result = final_fit_result()
    if batch_mode == "joint":
        result = replace(
            result, candidates=tuple(replace(item, ranking_objective=item.objective) for item in result.candidates)
        )
    checkpoint = FitCheckpoint("a" * 64, "b" * 64, "c" * 64, "E", result.candidates, (101,))
    state = api.DatasetAutomation(
        import_batch_id="batch-1",
        fit_group_id="group-1",
        role=api.AutomaticRole.JOINT,
        status=api.AutomaticStatus.PASSED,
        statistics_member=True,
    )
    datasets = tuple(
        replace(
            dataset_project(dataset_id, result=result),
            fit_mask=tuple(index != 5 for index in range(32)),
            checkpoint=checkpoint,
            structure_evidence=api.StructureEvidence(1, 1, None, (20.0,)),
            scale_prior=api.ScalePriorState(enabled=True, s_hat=1.0, tau_s_decades=0.1),
            parameter_settings=(api.ParameterSetting("component.0.thickness_a", 20.0, 10.0, 30.0),),
            parameter_priors=(api.ParameterPrior("component.0.thickness_a", api.PriorSpec("normal", (20.0, 2.0))),),
            automation=state,
        )
        for dataset_id in ("first", "second")
    )
    value = project(*datasets)
    return replace(
        value,
        batch_mode=batch_mode,
        sharing_rules=(
            api.SharingRule(
                "thickness",
                tuple(api.ParameterReference(item.dataset_id, "component.0.thickness_a") for item in datasets),
            ),
        ),
        ui_state=replace(
            value.ui_state,
            active_dataset_id="first",
            selected_candidate_ids=tuple((item.dataset_id, result.candidates[0].candidate_id) for item in datasets),
            expert_mode=True,
            dock_state="saved-layout",
        ),
    )


def test_set_fit_config_is_one_direct_public_service() -> None:
    assert api.set_fit_config is projects.set_fit_config


@pytest.mark.parametrize("equivalent_copy", (False, True))
def test_equal_config_preserves_the_project_and_all_completed_evidence(equivalent_copy: bool) -> None:
    value = _completed_project()
    config = replace(value.fit_config) if equivalent_copy else value.fit_config

    assert api.set_fit_config(value, config) is value


def _assert_dataset_invalidation(before, after, *, mode_changed: bool) -> None:
    assert (after.last_valid_result, after.checkpoint, after.scale_prior) == (None, None, api.ScalePriorState(False))
    expected_automation = replace(
        before.automation, status=api.AutomaticStatus.PENDING, statistics_member=False, reason=None
    )
    assert after.automation == expected_automation
    assert after == replace(
        before,
        last_valid_result=None,
        checkpoint=None,
        scale_prior=api.ScalePriorState(False),
        automation=expected_automation,
        structure_evidence=None if mode_changed else before.structure_evidence,
    )
    assert (
        before.last_valid_result is not None,
        before.checkpoint is not None,
        before.automation.status,
    ) == (True, True, api.AutomaticStatus.PASSED)


@pytest.mark.parametrize("batch_mode", ("independent", "joint"))
@pytest.mark.parametrize(
    "change",
    (
        {"noise_model": "gaussian"},
        {"noise_model": "poisson"},
        {"master_seed": 7},
        {"profile_steps": 17},
        {"scale_prior_enabled": False},
        {"budget": api.FitConfig.fast(1).budget},
    ),
)
def test_config_changes_clear_all_evidence_and_preserve_declarations(batch_mode: str, change: dict) -> None:
    value = _completed_project(batch_mode)
    config = replace(value.fit_config, **change)

    updated = api.set_fit_config(value, config)

    assert updated.fit_config is config
    assert updated.batch_mode == value.batch_mode
    assert updated.sharing_rules == value.sharing_rules
    assert updated.constraint_rules == value.constraint_rules
    assert updated.ui_state == replace(value.ui_state, selected_candidate_ids=())
    for before, after in zip(value.datasets, updated.datasets, strict=True):
        _assert_dataset_invalidation(before, after, mode_changed="noise_model" in change)


@pytest.mark.parametrize("config", (None, {}, False, "gaussian"))
def test_set_fit_config_rejects_non_config_values(config) -> None:
    with pytest.raises(TypeError, match="config must be.*FitConfig"):
        api.set_fit_config(project(), config)


def test_empty_project_accepts_config_before_any_data_are_imported() -> None:
    value = api.new_project()
    config = replace(value.fit_config, noise_model="poisson")

    updated = api.set_fit_config(value, config)

    assert updated.datasets == ()
    assert updated.fit_config is config


def _source_project(tmp_path: Path, *, sigma: bool = True, bad: str | None = None) -> api.XrrProject:
    angles = np.linspace(0.1, 3.2, 48)
    counts = np.arange(100, 52, -1, dtype=float)
    errors = np.ones(48)
    if bad == "fractional":
        counts[4] += 0.5
    elif bad == "merged":
        angles[4] = angles[3]
        counts[4] = counts[3]
    elif bad == "sigma":
        errors[4] = 0.0
    columns = (angles, counts, errors) if sigma else (angles, counts)
    source = tmp_path / "curve.xy"
    np.savetxt(source, np.column_stack(columns), fmt="%.17g")
    value = api.add_dataset(
        api.new_project(),
        source,
        api.InstrumentSpec(instrument_id="mode-preflight", footprint_mode="none"),
        column_mapping=api.DataColumnMapping(intensity_sigma=2 if sigma and bad != "sigma" else None),
    )
    value = api.set_structure(value, "curve", simple_structure())
    mask = np.array(value.datasets[0].fit_mask)
    mask[11] = False
    value = api.set_fit_mask(value, "curve", mask)
    if bad == "sigma":
        value = replace(
            value,
            datasets=(replace(value.datasets[0], column_mapping=api.DataColumnMapping(intensity_sigma=2)),),
        )
    return replace(value, fit_config=replace(value.fit_config, scale_prior_enabled=False))


@pytest.mark.parametrize(
    ("mode", "sigma", "bad", "message"),
    (
        ("gaussian", False, None, "requires known intensity sigma"),
        ("gaussian", True, "sigma", "invalid intensity uncertainty"),
        ("poisson", False, "fractional", "integer raw counts"),
        ("poisson", False, "merged", "merged duplicate rows"),
    ),
)
def test_changed_mode_rejects_invalid_data_before_search(
    tmp_path: Path, monkeypatch, mode: str, sigma: bool, bad: str | None, message: str
) -> None:
    value = _source_project(tmp_path, sigma=sigma, bad=bad)
    updated = api.set_fit_config(value, replace(value.fit_config, noise_model=mode))
    calls = []

    def unexpected_search(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("optimizer must not run for invalid input")

    monkeypatch.setattr(fitting, "run_fit_search", unexpected_search)

    readiness = api.preflight_fit(updated)

    assert not readiness.ready
    assert message in readiness.message
    with pytest.raises(ValueError, match=message):
        api.fit_project(updated)
    assert calls == []
    assert updated.datasets[0].fit_mask == value.datasets[0].fit_mask


@pytest.mark.parametrize("mode", ("robust_log", "gaussian", "poisson"))
def test_mode_selection_roundtrips_and_reprepares_only_the_persisted_mask(tmp_path: Path, mode: str) -> None:
    value = _source_project(tmp_path)
    updated = api.set_fit_config(value, replace(value.fit_config, noise_model=mode))
    target = tmp_path / "project.xrrproj.json"

    api.save_project(updated, target)
    loaded = api.load_project(target)
    prepared = fitting.prepare_dataset_fit(loaded, "curve", loaded.master_seed)

    assert loaded == updated
    assert loaded.schema_version == 5
    assert loaded.algorithm_version == "xrr-fit-v2-poisson-5"
    assert loaded.fit_config.noise_model == mode
    assert prepared.problem.config.noise_model == mode
    assert loaded.datasets[0].fit_mask == value.datasets[0].fit_mask
    np.testing.assert_array_equal(prepared.problem.data.fit_mask, value.datasets[0].fit_mask)
    assert api.preflight_fit(loaded).ready
