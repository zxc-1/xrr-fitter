from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import xrr_fitter.api as api

ROOT = Path(__file__).resolve().parents[2]


def _fast_single_project() -> api.XrrProject:
    value = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    budget = replace(
        value.fit_config.budget,
        short_de_maxiter=0,
        full_de_maxiter=0,
        local_min_nfev=5,
        local_nfev_per_parameter=1,
        bootstrap_samples=1,
    )
    config = replace(
        api.FitConfig.fast(value.master_seed),
        budget=budget,
        local_workers=1,
        scale_prior_enabled=False,
    )
    value = replace(value, fit_config=config)
    dataset_id = value.datasets[0].dataset_id
    definitions = api.describe_parameters(value, dataset_id)
    free_name = "component.0.thickness_a"
    settings = tuple(
        api.ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower if definition.name == free_name else definition.initial,
            definition.upper if definition.name == free_name else definition.initial,
            locked=definition.name != free_name,
        )
        for definition in definitions
    )
    return api.set_parameter_settings(value, dataset_id, settings)


def _single_fit_snapshot(readiness, result, progress, checkpoints) -> dict[str, object]:
    fit_result = result.datasets[0].fit_result
    final_dataset = result.updated_project.datasets[0]
    final_checkpoint = final_dataset.checkpoint
    best = fit_result.best_candidate
    checkpoint_best = next(
        candidate for candidate in final_checkpoint.candidates if candidate.candidate_id == best.candidate_id
    )
    return {
        "readiness": readiness,
        "mode": result.mode,
        "cancelled": result.cancelled,
        "dataset_count": len(result.datasets),
        "published_result": final_dataset.last_valid_result is fit_result,
        "has_best_candidate": best is not None,
        "has_uncertainty": fit_result.uncertainty is not None,
        "has_required_progress": {item.stage for item in progress}
        >= {
            "A",
            "B",
            "C",
            "D",
            "E",
            "basin-recovery",
            "bootstrap",
            "profile",
            "finalizing",
        },
        "has_checkpoints": bool(checkpoints),
        "checkpoints_populated": all(item.datasets[0].checkpoint is not None for item in checkpoints),
        "final_checkpoint_stage": final_checkpoint.stage,
        "best_has_diagnostics": bool(best.diagnostics),
        "checkpoint_diagnostics_match": checkpoint_best.diagnostics == best.diagnostics,
    }


def test_single_fit_workflow_publishes_progress_checkpoints_and_analysis() -> None:
    value = _fast_single_project()
    progress: list[api.FitProgress] = []
    checkpoints: list[api.XrrProject] = []

    readiness = api.preflight_fit(value)
    result = api.fit_project(value, progress.append, checkpoints.append)

    snapshot = _single_fit_snapshot(readiness, result, progress, checkpoints)
    assert snapshot == {
        "readiness": api.FitReadiness(True, "ready"),
        "mode": "independent",
        "cancelled": False,
        "dataset_count": 1,
        "published_result": True,
        "has_best_candidate": True,
        "has_uncertainty": True,
        "has_required_progress": True,
        "has_checkpoints": True,
        "checkpoints_populated": True,
        "final_checkpoint_stage": "E",
        "best_has_diagnostics": True,
        "checkpoint_diagnostics_match": True,
    }
