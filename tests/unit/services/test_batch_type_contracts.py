from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import get_type_hints

from tests.support.model_cases import final_fit_result

from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.fitting import FitCheckpoint
from xrr_fitter.model.project import DatasetSourceValidation, ProjectValidation
from xrr_fitter.services import batch, datasets
from xrr_fitter.services.fitting_phases.common import AutomaticPreparedResult, PreparedDatasetFit


def test_batch_source_records_keep_validation_types():
    source = get_type_hints(batch._source_records)
    assert source.get("validation") is ProjectValidation
    assert source["return"] == dict[str, DatasetSourceValidation]
    assert get_type_hints(batch._source_error)["records"] == source["return"]


def test_batch_seed_callback_uses_the_seed_owner_contract():
    branch_type = getattr(datasets, "ServiceSeedBranches", None)
    assert branch_type is not None, "missing seed owner return contract"
    assert get_type_hints(datasets.service_seed_branches)["return"] == branch_type
    assert batch._SeedBranches == Callable[[batch.XrrProject], branch_type]


def test_independent_rows_preserve_the_buffered_result_type():
    hints = get_type_hints(batch._run_independent_rows)
    assert getattr(batch._BufferedFit, "__type_params__", ()), "missing generic result buffer"
    assert hints["return"] == dict[int, batch._BufferedFit[FitResult]]


def test_automatic_publication_uses_prepared_service_results():
    hints = get_type_hints(batch._automatic_fit_parts)
    assert hints["return"] == tuple[PreparedDatasetFit, FitResult, bool, str | None]
    assert get_type_hints(batch._automatic_joint_groups)["prefit_results"] == dict[int, AutomaticPreparedResult]


def test_dataset_fit_callback_keeps_keyword_and_checkpoint_contracts():
    callback = getattr(batch, "_DatasetFit", None)
    assert callback is not None, "missing typed dataset fit callback"
    parameters = {parameter.__name__: parameter for parameter in callback.__type_params__}
    hints = get_type_hints(callback.__call__, localns=parameters)
    assert hints["prepared"] is PreparedDatasetFit
    assert hints["checkpoint"] == Callable[[FitCheckpoint | None], None] | None


def test_buffered_checkpoint_clear_remains_publishable():
    def fit(prepared, *, checkpoint, **kwargs):
        checkpoint(None)
        return final_fit_result()

    outcome = batch._dataset_fit(SimpleNamespace(), 1, fit, None, None)
    assert outcome.checkpoints == (None,)
    assert outcome.error is None
