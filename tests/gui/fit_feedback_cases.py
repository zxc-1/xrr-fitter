"""Typed GUI publication fixtures; not optimizer or checkpoint-resume benchmarks."""

from dataclasses import replace
from pathlib import Path

from tests.gui.test_export_dialog import _fitted_project
from tests.support.model_cases import fit_candidate

import xrr_fitter.api as api
from xrr_fitter.model.fitting import FitCheckpoint
from xrr_fitter.model.operations import DatasetFitResult


def with_checkpoint(project: api.XrrProject) -> api.XrrProject:
    checkpoint = FitCheckpoint(
        data_sha256="a" * 64,
        structure_fingerprint="b" * 64,
        config_fingerprint="c" * 64,
        stage="B",
        candidates=(fit_candidate("retained"),),
        child_seeds=(17,),
    )
    return replace(project, datasets=tuple(replace(row, checkpoint=checkpoint) for row in project.datasets))


def _unselected_project(tmp_path: Path, count: int) -> api.XrrProject:
    fitted = _fitted_project(tmp_path, datasets=count)
    # The fixture searches scale only. Persist the other layer values as fixed
    # declarations, so the real automatic summary can resolve every parameter.
    fixed = tuple(
        api.ParameterSetting(name, value, lower, upper, api.ParameterFreedom.FIXED)
        for name, value, lower, upper in (
            ("component.0.thickness_a", 20.0, 1.0, 100.0),
            ("component.0.roughness_a", 2.0, 0.0, 10.0),
            ("component.0.density_scale", 1.0, 0.5, 1.5),
        )
    )
    rows = tuple(replace(row, parameter_settings=(*row.parameter_settings, *fixed)) for row in fitted.datasets)
    return replace(fitted, datasets=rows, ui_state=replace(fitted.ui_state, selected_candidate_ids=()))


def terminal_case(
    tmp_path: Path, states: tuple[api.AutomaticStatus, ...]
) -> tuple[api.XrrProject, api.ProjectFitResult]:
    project = _unselected_project(tmp_path, len(states))
    rows = []
    outcomes = []
    for dataset, state in zip(project.datasets, states, strict=True):
        result = dataset.last_valid_result
        reason = None
        if state is api.AutomaticStatus.FAILED:
            reason = f"{dataset.dataset_id}: joint member qualification rejected"
            result = api.FitResult.failed(reason, result.best_candidate.qz_a_inv.size)
        elif state is api.AutomaticStatus.REVIEW:
            reason = f"{dataset.dataset_id}: profile interval open"
        automation = api.DatasetAutomation(
            import_batch_id="feedback-batch",
            fit_group_id=f"single-{dataset.dataset_id}",
            role=api.AutomaticRole.SINGLE,
            status=state,
            statistics_member=state is api.AutomaticStatus.PASSED,
            reason=reason,
        )
        rows.append(
            replace(
                dataset,
                automation=automation,
                last_valid_result=None if state is api.AutomaticStatus.FAILED else result,
            )
        )
        outcomes.append(DatasetFitResult(dataset.dataset_id, result))
    updated = replace(project, datasets=tuple(rows))
    initial = replace(
        updated,
        datasets=tuple(
            replace(
                row,
                last_valid_result=None,
                automation=replace(
                    row.automation,
                    status=api.AutomaticStatus.PENDING,
                    statistics_member=False,
                    reason=None,
                ),
            )
            for row in rows
        ),
    )
    return initial, api.ProjectFitResult("automatic", tuple(outcomes), (), updated)


def manual_case(
    tmp_path: Path, confidence: api.ConfidenceClass, **evidence
) -> tuple[api.XrrProject, api.ProjectFitResult]:
    project = _unselected_project(tmp_path, 1)
    dataset = project.datasets[0]
    fit_result = replace(dataset.last_valid_result, confidence=confidence, **evidence)
    updated = replace(project, datasets=(replace(dataset, last_valid_result=fit_result),))
    initial = replace(project, datasets=(replace(dataset, last_valid_result=None),))
    return initial, api.ProjectFitResult(
        "independent",
        (DatasetFitResult(dataset.dataset_id, fit_result),),
        (),
        updated,
    )
