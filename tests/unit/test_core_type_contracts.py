from __future__ import annotations

from collections.abc import Callable
from typing import get_type_hints

import numpy as np
from numpy.typing import ArrayLike

from xrr_fitter import evaluation_parameters
from xrr_fitter.analysis import mcmc, profile_selection, profile_tasks
from xrr_fitter.analysis.bootstrap_samples import TaskRunner
from xrr_fitter.model.analysis import McmcConfig, UncertaintyReport
from xrr_fitter.model.data import BeamSpec
from xrr_fitter.model.fitting import FitCandidate, FitCheckpoint, FitEvaluationContext, ModelEvaluation
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ParameterCoordinate,
    ParameterDefinition,
    ParameterPrior,
    ParameterSetting,
    SharingRule,
)
from xrr_fitter.model.project import DatasetProject
from xrr_fitter.model.structure import MaterialSpec, StructureComponent
from xrr_fitter.services import batch, batch_routing, fitting
from xrr_fitter.services.fitting_phases.common import PreparedDatasetFit


def test_evaluation_parameter_helpers_use_fit_context_contracts() -> None:
    validated_hints = get_type_hints(evaluation_parameters._validated_unit)
    declared_hints = get_type_hints(evaluation_parameters._declared_values)
    decode_hints = get_type_hints(evaluation_parameters._decode_nonrough_values)
    readonly_hints = get_type_hints(evaluation_parameters._readonly_vector)

    assert validated_hints["problem"] is FitEvaluationContext
    assert declared_hints["problem"] is FitEvaluationContext
    assert decode_hints["problem"] is FitEvaluationContext
    assert readonly_hints["value"] == ArrayLike


def test_fitting_sharing_problem_view_uses_domain_types() -> None:
    view_hints = get_type_hints(fitting._sharing_problem_view)
    local_view_hints = get_type_hints(fitting._SharingProblemView)

    assert view_hints["project"] is fitting.XrrProject
    assert view_hints["dataset"] is DatasetProject
    assert view_hints["return"] is fitting._SharingProblemView
    assert local_view_hints["parameter_definitions"] == tuple[ParameterDefinition, ...]
    assert local_view_hints["variables"] == tuple[ParameterCoordinate, ...]
    assert local_view_hints["instrument"] is InstrumentSpec


def test_fitting_parameter_sidecar_helpers_use_existing_domain_types() -> None:
    prior_validation_hints = get_type_hints(fitting._validate_parameter_priors)
    settings_hints = get_type_hints(fitting._reconciled_settings)
    priors_hints = get_type_hints(fitting._reconciled_priors)
    sharing_hints = get_type_hints(fitting.reconciled_sharing_rules)

    assert prior_validation_hints["definitions"] == tuple[ParameterDefinition, ...]
    assert prior_validation_hints["priors"] == tuple[ParameterPrior, ...]
    assert settings_hints["settings"] == tuple[ParameterSetting, ...]
    assert settings_hints["return"] == tuple[ParameterSetting, ...]
    assert priors_hints["definitions"] == tuple[ParameterDefinition, ...]
    assert priors_hints["priors"] == tuple[ParameterPrior, ...]
    assert priors_hints["return"] == tuple[ParameterPrior, ...]
    assert sharing_hints["return"] == tuple[SharingRule, ...]


def test_mcmc_helpers_use_concrete_domain_types() -> None:
    update_hints = get_type_hints(mcmc._update_group)
    initial_hints = get_type_hints(mcmc._validated_initial_state)
    candidate_hints = get_type_hints(mcmc._validated_candidate)

    assert update_hints["config"] is McmcConfig
    assert initial_hints["config"] is McmcConfig
    assert candidate_hints["problem"] is FitEvaluationContext
    assert candidate_hints["candidate"] is FitCandidate


def test_independent_preparation_uses_concrete_domain_types() -> None:
    preparation_hints = get_type_hints(batch._IndependentPreparation)

    assert preparation_hints["original"] is DatasetProject
    assert preparation_hints["prepared"] == PreparedDatasetFit | None


def test_buffered_fit_uses_concrete_domain_types() -> None:
    parameters = getattr(batch._BufferedFit, "__type_params__", ())
    assert len(parameters) == 1, "independent and automatic results need distinct buffered types"
    buffered_hints = get_type_hints(batch._BufferedFit)

    assert buffered_hints["result"] == parameters[0] | None
    assert buffered_hints["checkpoints"] == tuple[FitCheckpoint | None, ...]


def test_automatic_preparation_uses_concrete_domain_types() -> None:
    automatic_hints = get_type_hints(batch._AutomaticPreparation)

    assert automatic_hints["original"] is DatasetProject
    assert automatic_hints["prepared"] == PreparedDatasetFit | None


def test_automatic_routing_uses_concrete_domain_inputs() -> None:
    material_hints = get_type_hints(batch_routing._material_signature)
    component_hints = get_type_hints(batch_routing._component_signature)
    dataset_hints = get_type_hints(batch_routing.automatic_physical_signature)

    assert material_hints.get("material") is MaterialSpec
    assert material_hints["return"] == tuple[str, str | None, bool]
    assert component_hints.get("component") == StructureComponent
    assert dataset_hints.get("dataset") is DatasetProject


def test_automatic_routing_field_values_accept_only_acquisition_models() -> None:
    hints = get_type_hints(batch_routing._dataclass_values)

    assert hints.get("value") == BeamSpec | InstrumentSpec


def test_profile_selection_uses_uncertainty_report_contract() -> None:
    selected_hints = get_type_hints(profile_selection.select_profile_names)
    reported_hints = get_type_hints(profile_selection._reported_profile_names)

    assert selected_hints["preliminary_report"] == UncertaintyReport | None
    assert reported_hints["preliminary_report"] == UncertaintyReport | None


def test_profile_task_graph_retains_model_evaluation_and_runner_types() -> None:
    plan_hints = get_type_hints(profile_tasks._problem_profile_plan)
    graph_hints = get_type_hints(profile_tasks.build_problem_profiles)

    assert plan_hints["evaluate"] == Callable[[FitEvaluationContext, np.ndarray], ModelEvaluation]
    assert graph_hints["evaluate"] == plan_hints["evaluate"]
    assert graph_hints.get("task_runner") == TaskRunner | None
    assert plan_hints["return"] is not object
