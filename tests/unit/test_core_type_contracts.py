from __future__ import annotations

from typing import get_type_hints

from xrr_fitter.analysis import mcmc
from xrr_fitter.model.analysis import FitResult, McmcConfig
from xrr_fitter.model.fitting import FitCandidate, FitCheckpoint, FitEvaluationContext
from xrr_fitter.model.project import DatasetProject
from xrr_fitter.services import batch
from xrr_fitter.services.fitting_phases.common import PreparedDatasetFit


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
