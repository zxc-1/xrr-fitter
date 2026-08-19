from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import xrr_fitter.api as api

ROOT = Path(__file__).resolve().parents[2]


def _resumable_project() -> api.XrrProject:
    value = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    budget = replace(
        value.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=5,
        local_nfev_per_parameter=1,
        bootstrap_samples=1,
    )
    value = replace(
        value,
        fit_config=replace(
            api.FitConfig.fast(value.master_seed),
            budget=budget,
            local_workers=1,
            scale_prior_enabled=False,
        ),
    )
    dataset_id = value.datasets[0].dataset_id
    free_name = "component.0.thickness_a"
    settings = tuple(
        api.ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name == free_name else definition.initial,
            definition.upper if definition.name == free_name else definition.initial,
            locked=definition.name != free_name,
        )
        for definition in api.describe_parameters(value, dataset_id)
    )
    return api.set_parameter_settings(value, dataset_id, settings)


def test_independent_batch_resume_reuses_the_persisted_stage_e_graph() -> None:
    first = api.fit_project(_resumable_project())
    dataset = first.updated_project.datasets[0]
    assert dataset.checkpoint is not None and dataset.checkpoint.stage == "E"
    resumable = replace(
        first.updated_project,
        datasets=(replace(dataset, last_valid_result=None),),
    )

    resumed = api.fit_project(resumable)

    fresh_result = first.datasets[0].fit_result
    resumed_result = resumed.datasets[0].fit_result
    assert fresh_result.child_seeds == resumed_result.child_seeds
    assert fresh_result.stage_summaries == resumed_result.stage_summaries
    assert tuple(item.candidate_id for item in fresh_result.candidates) == tuple(
        item.candidate_id for item in resumed_result.candidates
    )
    assert tuple(item.objective for item in fresh_result.candidates) == tuple(
        item.objective for item in resumed_result.candidates
    )
