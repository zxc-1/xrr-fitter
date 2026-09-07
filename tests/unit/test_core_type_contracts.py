from __future__ import annotations

from typing import get_type_hints

from numpy.typing import ArrayLike

from xrr_fitter import evaluation_parameters
from xrr_fitter.analysis import mcmc
from xrr_fitter.model.analysis import FitResult, McmcConfig
from xrr_fitter.model.fitting import FitCandidate, FitCheckpoint, FitEvaluationContext
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import (
    ParameterCoordinate,
    ParameterDefinition,
    ParameterPrior,
    ParameterSetting,
    SharingRule,
)
from xrr_fitter.model.project import DatasetProject
from xrr_fitter.services import batch, fitting
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
    buffered_hints = get_type_hints(batch._BufferedFit)

    assert buffered_hints["result"] == FitResult | None
    assert buffered_hints["checkpoints"] == tuple[FitCheckpoint, ...]


def test_automatic_preparation_uses_concrete_domain_types() -> None:
    automatic_hints = get_type_hints(batch._AutomaticPreparation)

    assert automatic_hints["original"] is DatasetProject
    assert automatic_hints["prepared"] == PreparedDatasetFit | None
