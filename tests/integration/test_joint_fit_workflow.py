from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

import xrr_fitter.api as api
from xrr_fitter.io.xy import xy_bytes
from xrr_fitter.model.automation import (
    AutomaticRole,
    AutomaticStatus,
    DatasetAutomation,
    MeasurementPreset,
)
from xrr_fitter.physics.reflectivity import instrument_reflectivity
from xrr_fitter.physics.stack import expand_structure
from xrr_fitter.services.fitting import fit_automatically

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


def _automatic_joint_project(tmp_path: Path) -> api.XrrProject:
    value = api.load_project(ROOT / "examples/single-layer.xrrproj.json")
    original = value.datasets[0]
    instrument = api.InstrumentSpec(
        instrument_id="automatic-integration",
        footprint_mode="none",
    )
    two_theta_deg = np.linspace(0.08, 6.0, 320)
    intensity = instrument_reflectivity(
        two_theta_deg / 2.0,
        expand_structure(
            original.structure,
            original.beam.effective_wavelength_a,
        ),
        original.beam,
    )
    intensity *= np.exp(np.random.default_rng(817).normal(0.0, 0.01, len(two_theta_deg)))
    source = tmp_path / "automatic-joint.xy"
    content = xy_bytes(two_theta_deg, intensity)
    source.write_bytes(content)
    automation = DatasetAutomation(
        import_batch_id="automatic-integration",
        role=AutomaticRole.UNROUTED,
        status=AutomaticStatus.PENDING,
    )
    left = replace(
        original,
        dataset_id="left",
        display_name="left",
        source_path=str(source),
        source_sha256=sha256(content).hexdigest(),
        fit_mask=(True,) * len(two_theta_deg),
        fit_range_two_theta_deg=(
            float(two_theta_deg[0]),
            float(two_theta_deg[-1]),
        ),
        instrument=instrument,
        automation=automation,
    )
    right = replace(
        original,
        dataset_id="right",
        display_name="right",
        source_path=str(source),
        source_sha256=sha256(content).hexdigest(),
        fit_mask=(True,) * len(two_theta_deg),
        fit_range_two_theta_deg=(
            float(two_theta_deg[0]),
            float(two_theta_deg[-1]),
        ),
        instrument=instrument,
        automation=automation,
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
        datasets=(left, right),
        fit_config=replace(
            api.FitConfig.fast(value.master_seed),
            budget=budget,
            local_workers=1,
            scale_prior_enabled=False,
        ),
        measurement_preset=MeasurementPreset(
            "automatic-integration",
            original.beam,
            instrument,
        ),
        ui_state=api.ProjectUiState(active_dataset_id="left"),
    )
    free_name = "component.0.density_scale"
    for dataset_id in ("left", "right"):
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
        value = api.set_parameter_settings(value, dataset_id, settings)
    return value


def _candidate_parameter(result, name: str) -> float:
    candidate = result.best_candidate
    assert candidate is not None
    return next(parameter.value for parameter in candidate.parameters if parameter.name == name)


def _dataset_ids(items) -> tuple[str, ...]:
    return tuple(item.dataset_id for item in items)


def _published_candidates(items) -> tuple[bool, ...]:
    return tuple(item.fit_result.best_candidate is not None for item in items)


def _automation_roles(items) -> set[AutomaticRole]:
    return {item.automation.role for item in items}


def _published_results(items) -> tuple[bool, ...]:
    return tuple(item.last_valid_result is not None for item in items)


def _joint_checkpoint_widths(checkpoints) -> set[int]:
    widths = set()
    for project in checkpoints:
        values = tuple(dataset.checkpoint for dataset in project.datasets)
        if not values or any(value is None for value in values):
            continue
        fingerprints = {value.joint_layout_fingerprint for value in values}
        if len(fingerprints) == 1 and next(iter(fingerprints)):
            widths.add(len(values))
    return widths


def _aligned_fit_snapshot(left, right) -> dict[str, object]:
    uncertainty = left.uncertainty
    return {
        "best_index_aligned": left.best_index == right.best_index,
        "child_seeds_aligned": left.child_seeds == right.child_seeds,
        "confidence_aligned": left.confidence == right.confidence,
        "evidence_aligned": (left.classification_evidence == right.classification_evidence),
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
        "dataset_order": _dataset_ids(result.datasets),
        "aligned_fit": _aligned_fit_snapshot(left, right),
        "published_results": _published_results(result.updated_project.datasets),
        "has_checkpoints": bool(checkpoints),
        "checkpoint_stage_widths": {
            len({dataset.checkpoint.stage for dataset in checkpoint.datasets}) for checkpoint in checkpoints
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


def test_automatic_joint_workflow_refines_matching_points_with_shared_material(
    tmp_path: Path,
) -> None:
    value = _automatic_joint_project(tmp_path)
    checkpoints: list[api.XrrProject] = []

    result = fit_automatically(value, checkpoint_callback=checkpoints.append)

    left, right = result.datasets
    final_datasets = result.updated_project.datasets
    density_name = "component.0.density_scale"
    assert _dataset_ids(result.datasets) == ("left", "right")
    assert _published_candidates(result.datasets) == (True, True)
    assert _candidate_parameter(left.fit_result, density_name) == _candidate_parameter(
        right.fit_result,
        density_name,
    )
    assert _automation_roles(final_datasets) == {AutomaticRole.JOINT}
    assert _published_results(final_datasets) == (True, True)
    assert checkpoints
    assert _joint_checkpoint_widths(checkpoints) == {2}
