from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import xrr_fitter.api as api


ROOT = Path(__file__).resolve().parents[2]


def _joint_project() -> api.XrrProject:
    value = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    original = value.datasets[0]
    left = replace(original, dataset_id="left", display_name="left")
    right = replace(original, dataset_id="right", display_name="right")
    value = replace(
        value,
        datasets=(left, right),
        ui_state=api.ProjectUiState(active_dataset_id="left"),
    )
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
    free_name = "component.0.thickness_a"
    for dataset_id in ("left", "right"):
        definitions = api.describe_parameters(value, dataset_id)
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
        value = api.set_parameter_settings(value, dataset_id, settings)
    value = api.set_sharing_rules(
        value,
        (
            api.SharingRule(
                "shared-thickness",
                (
                    api.ParameterReference("left", free_name),
                    api.ParameterReference("right", free_name),
                ),
            ),
        ),
    )
    return api.set_batch_mode(value, "joint")


def _aligned_fit_snapshot(left, right) -> dict[str, object]:
    uncertainty = left.uncertainty
    return {
        "best_index_aligned": left.best_index == right.best_index,
        "child_seeds_aligned": left.child_seeds == right.child_seeds,
        "confidence_aligned": left.confidence == right.confidence,
        "evidence_aligned": (
            left.classification_evidence == right.classification_evidence
        ),
        "shared_uncertainty": uncertainty is not None and uncertainty is right.uncertainty,
        "correlation_names": getattr(uncertainty, "correlation_names", None),
        "stages": tuple(summary.stage for summary in left.stage_summaries),
        "candidate_ids_aligned": (
            tuple(item.candidate_id for item in left.candidates)
            == tuple(item.candidate_id for item in right.candidates)
        ),
    }


def _joint_transaction_snapshot(result, progress, checkpoints) -> dict[str, object]:
    left, right = (item.fit_result for item in result.datasets)
    return {
        "mode": result.mode,
        "cancelled": result.cancelled,
        "dataset_order": tuple(item.dataset_id for item in result.datasets),
        "aligned_fit": _aligned_fit_snapshot(left, right),
        "published_results": tuple(
            dataset.last_valid_result is not None
            for dataset in result.updated_project.datasets
        ),
        "has_checkpoints": bool(checkpoints),
        "checkpoint_stage_widths": {
            len({dataset.checkpoint.stage for dataset in checkpoint.datasets})
            for checkpoint in checkpoints
        },
        "progress_stages": tuple(item.stage for item in progress),
        "progress_dataset_ids": {item.dataset_id for item in progress},
    }


def test_joint_fit_workflow_publishes_one_aligned_result_transaction() -> None:
    value = _joint_project()
    checkpoints: list[api.XrrProject] = []
    progress: list[api.FitProgress] = []

    ready = api.preflight_fit(value).ready
    result = api.fit_project(value, progress.append, checkpoints.append)

    assert (ready, _joint_transaction_snapshot(result, progress, checkpoints)) == (
        True,
        {
            "mode": "joint",
            "cancelled": False,
            "dataset_order": ("left", "right"),
            "aligned_fit": {
                "best_index_aligned": True,
                "child_seeds_aligned": True,
                "confidence_aligned": True,
                "evidence_aligned": True,
                "shared_uncertainty": True,
                "correlation_names": ("shared-thickness",),
                "stages": ("A", "B", "C", "D", "E"),
                "candidate_ids_aligned": True,
            },
            "published_results": (True, True),
            "has_checkpoints": True,
            "checkpoint_stage_widths": {1},
            "progress_stages": (
                "A",
                "B",
                "C",
                "D",
                "E",
                "E",
                "E",
                "E",
                "finalizing",
                "finalizing",
            ),
            "progress_dataset_ids": {None},
        },
    )
