from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

from xrr_fitter.model.analysis import ConfidenceClass, McmcConfig, McmcReport, UncertaintyReport
from xrr_fitter.model.parameters import ParameterReference, SharingRule
from xrr_fitter.model.project import (
    DatasetProject,
    DatasetSourceValidation,
    ProjectUiState,
    ProjectValidation,
    ScalePriorState,
    SourceStatus,
    XrrProject,
    validate_project,
    with_active_dataset,
    with_batch_mode,
    with_dataset_fit_mask,
    with_workspace_state,
)


def _uncertainty(candidate_id: str, *, mcmc_candidate_id: str | None = None) -> UncertaintyReport:
    mcmc = None
    if mcmc_candidate_id is not None:
        config = McmcConfig.standard(1)
        mcmc = McmcReport(
            config=config,
            child_seed=1,
            parameter_names=("scale",),
            samples_physical=[[1.0]],
            log_probability=[0.0],
            acceptance_fraction=[0.5] * config.walkers,
            split_rhat=[1.0],
            effective_sample_size=[1.0],
            boundary_hits=(),
            candidate_id=mcmc_candidate_id,
        )
    return UncertaintyReport(
        correlation_names=(),
        correlation_matrix=np.empty((0, 0)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        mcmc=mcmc,
        candidate_id=candidate_id,
    )


def test_project_and_dataset_serialization_field_order_is_stable() -> None:
    assert [field.name for field in fields(XrrProject)] == [
        "schema_version",
        "algorithm_version",
        "fit_config",
        "input_angle_kind",
        "batch_mode",
        "datasets",
        "sharing_rules",
        "ui_state",
        "measurement_preset",
        "base_directory",
    ]
    assert [field.name for field in fields(DatasetProject)] == [
        "dataset_id",
        "source_path",
        "source_sha256",
        "beam",
        "import_angle_offset_deg",
        "column_mapping",
        "fit_mask",
        "fit_range_two_theta_deg",
        "structure",
        "instrument",
        "structure_evidence",
        "scale_prior",
        "oxide_decisions",
        "parameter_settings",
        "last_valid_result",
        "checkpoint",
        "display_name",
        "automation",
        "parameter_priors",
    ]


def test_dataset_defaults_parameter_priors_to_empty() -> None:
    assert dataset_project().parameter_priors == ()


@pytest.mark.parametrize("dataset_ids", [("",), ("duplicate", "duplicate")])
def test_project_rejects_empty_or_duplicate_dataset_ids(dataset_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="dataset_id"):
        datasets = tuple(dataset_project(value) for value in dataset_ids)
        project(*datasets)


@pytest.mark.parametrize(
    ("field", "value"),
    (("algorithm_version", "future"), ("input_angle_kind", "theta_deg"), ("batch_mode", "parallel")),
)
def test_project_rejects_unsupported_header_values(field: str, value: object) -> None:
    current = project()
    with pytest.raises(ValueError, match=field):
        replace(current, **{field: value})


@pytest.mark.parametrize("schema", [1, 1.0, True])
def test_project_rejects_unsupported_or_noninteger_schema_version(schema: object) -> None:
    current = project()
    with pytest.raises(ValueError, match="schema_version"):
        replace(current, schema_version=schema)


def test_project_rejects_duplicate_or_missing_sharing_references() -> None:
    datasets = (dataset_project("first"), dataset_project("second"))
    rule = SharingRule(
        "shared",
        (ParameterReference("first", "scale"), ParameterReference("second", "scale")),
    )
    current = replace(project(*datasets), sharing_rules=(rule,))

    validate_project(current)
    with pytest.raises(ValueError, match="sharing_key"):
        replace(current, sharing_rules=(rule, rule))
    missing = SharingRule(
        "missing",
        (ParameterReference("first", "scale"), ParameterReference("third", "scale")),
    )
    with pytest.raises(ValueError, match="missing dataset"):
        replace(project(*datasets), sharing_rules=(missing,))


@pytest.mark.parametrize(
    "state",
    [
        ScalePriorState(True),
        ScalePriorState(False, 1.0, 0.1, None),
        ScalePriorState(True, 1.0, 0.1, "disabled"),
        ScalePriorState(False, None, None, "disabled"),
        ScalePriorState(1, 1.0, 0.1, None),
    ],
)
def test_project_rejects_partial_or_stale_scale_prior_states(state: ScalePriorState) -> None:
    with pytest.raises(ValueError, match="scale prior"):
        replace(dataset_project(), scale_prior=state)


def test_project_accepts_compiled_inactive_scale_prior() -> None:
    state = ScalePriorState(False, None, 0.1, "insufficient low-q support")

    assert replace(dataset_project(), scale_prior=state).scale_prior == state


def test_project_state_transitions_return_new_valid_values() -> None:
    first = dataset_project("first")
    second = dataset_project("second")
    current = project(first, second)
    active = with_active_dataset(current, "second")
    joint = with_batch_mode(active, "joint")
    mask = (False,) + first.fit_mask[1:]
    masked = with_dataset_fit_mask(joint, "first", mask)
    workspace = ProjectUiState(
        active_dataset_id="second",
        expert_mode=True,
        workspace_splitter_sizes=(300, 700, 400),
        left_splitter_sizes=(260, 500),
        plot_tab_index=1,
    )
    updated = with_workspace_state(masked, workspace)

    assert current.ui_state.active_dataset_id is None
    assert updated.batch_mode == "joint"
    assert updated.datasets[0].fit_mask == mask
    assert updated.ui_state == workspace


def test_project_rejects_ui_candidate_reference_without_persisted_result() -> None:
    state = ProjectUiState(selected_candidate_ids=(("curve", "candidate-0"),))
    with pytest.raises(ValueError, match="persisted result"):
        replace(project(), ui_state=state)


def test_project_accepts_candidate_reference_from_final_fit_result() -> None:
    dataset = dataset_project(result=final_fit_result())
    state = ProjectUiState(selected_candidate_ids=(("curve", "candidate-0"),))

    value = replace(project(dataset), ui_state=state)

    assert value.ui_state.selected_candidate_ids == (("curve", "candidate-0"),)


def test_project_validation_copies_record_sequence() -> None:
    records = [DatasetSourceValidation("curve", SourceStatus.OK, "a" * 64, "a" * 64, "ok")]
    validation = ProjectValidation(records)

    records.clear()

    assert len(validation.datasets) == 1
    assert validation.valid


def test_oxide_decision_rejects_unknown_location() -> None:
    from xrr_fitter.model.project import OxideDecision

    with pytest.raises(ValueError, match="location"):
        OxideDecision("Si", "SiO2", "middle", True, "oxide-v1")


def test_source_validation_preserves_public_compatibility_properties() -> None:
    record = DatasetSourceValidation(
        "curve",
        SourceStatus.HASH_MISMATCH,
        "a" * 64,
        "b" * 64,
        "source changed",
    )

    assert record.expected_hash == record.expected_sha256
    assert record.actual_hash == record.actual_sha256
    assert record.user_message == record.message


@pytest.mark.parametrize("owner", ["uncertainty", "mcmc"])
def test_project_rejects_missing_analysis_evidence_owner(owner: str) -> None:
    uncertainty = (
        _uncertainty("missing") if owner == "uncertainty" else _uncertainty("candidate-0", mcmc_candidate_id="missing")
    )
    result = replace(final_fit_result(), uncertainty=uncertainty)

    with pytest.raises(ValueError, match=f"{owner} candidate_id"):
        dataset_project(result=result)


def test_joint_project_requires_results_for_every_dataset() -> None:
    first = dataset_project("first", result=final_fit_result())
    second = dataset_project("second")

    with pytest.raises(ValueError, match="every dataset"):
        replace(project(first, second), batch_mode="joint")


def test_joint_project_rejects_candidate_identity_drift() -> None:
    first = dataset_project("first", result=final_fit_result(fit_candidate("shared", 1.0)))
    second = dataset_project("second", result=final_fit_result(fit_candidate("drifted", 2.0)))

    with pytest.raises(ValueError, match="coherent across datasets"):
        replace(project(first, second), batch_mode="joint")


def test_joint_project_validates_global_rank_against_local_mean() -> None:
    first_candidate = replace(fit_candidate("shared", 1.0), ranking_objective=1.5)
    second_candidate = replace(fit_candidate("shared", 2.0), ranking_objective=1.5)
    first = dataset_project("first", result=final_fit_result(first_candidate))
    second = dataset_project("second", result=final_fit_result(second_candidate))

    joint = replace(project(first, second), batch_mode="joint")

    assert joint.batch_mode == "joint"
    bad_first_candidate = replace(first_candidate, ranking_objective=1.6)
    bad_second_candidate = replace(second_candidate, ranking_objective=1.6)
    bad_first = dataset_project("first", result=final_fit_result(bad_first_candidate))
    bad_second = dataset_project("second", result=final_fit_result(bad_second_candidate))
    with pytest.raises(ValueError, match="global ranking"):
        replace(project(bad_first, bad_second), batch_mode="joint")


def test_joint_project_allows_dataset_local_classification_and_warnings() -> None:
    first_candidate = replace(fit_candidate("shared", 1.0), ranking_objective=1.5)
    second_candidate = replace(fit_candidate("shared", 2.0), ranking_objective=1.5)
    first_result = replace(
        final_fit_result(first_candidate),
        confidence=ConfidenceClass.CORRELATED,
        warnings=("shared fit warning", "left diagnostic"),
    )
    second_result = replace(
        final_fit_result(second_candidate),
        confidence=ConfidenceClass.MULTIPLE,
        warnings=("shared fit warning", "right diagnostic"),
    )

    joint = replace(
        project(
            dataset_project("first", result=first_result),
            dataset_project("second", result=second_result),
        ),
        batch_mode="joint",
    )

    validate_project(joint)
