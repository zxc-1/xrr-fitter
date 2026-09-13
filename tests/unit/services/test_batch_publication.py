from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support.model_cases import dataset_project, final_fit_result


def _publication():
    name = "xrr_fitter.services.batch_publication"
    assert find_spec(name) is not None, "dataset publication still belongs to batch execution"
    return import_module(name)


def test_structural_automatic_result_guard_keeps_existing_acceptance():
    publication = _publication()
    prepared = SimpleNamespace(updated_dataset=dataset_project("sample"))
    result = SimpleNamespace(prepared=prepared, fit_result=final_fit_result(), passed=True, reason=None)
    assert publication._automatic_fit_parts(result) == (prepared, result.fit_result, True, None)


@pytest.mark.parametrize("passed", [None, 0, 1, "yes"])
def test_structural_automatic_result_guard_rejects_non_boolean_quality(passed):
    publication = _publication()
    result = SimpleNamespace(prepared=object(), fit_result=final_fit_result(), passed=passed)
    with pytest.raises(TypeError, match="passed flag"):
        publication._automatic_fit_parts(result)


def test_failure_arrays_keep_source_length_and_readonly_values():
    publication = _publication()
    dataset = dataset_project("sample")
    result = publication._failure_result(dataset, ValueError("source failed"))
    np.testing.assert_array_equal(result.region_labels, [-1] * len(dataset.fit_mask))
    np.testing.assert_array_equal(result.region_weights, [0.0] * len(dataset.fit_mask))
    assert not result.region_labels.flags.writeable
    assert not result.region_weights.flags.writeable


def test_publication_has_one_owner_used_by_batch():
    from xrr_fitter.services import batch

    publication = _publication()
    assert batch._commit_automatic_result is publication._commit_automatic_result
    assert batch._commit_success is publication._commit_success
    assert batch._replay_checkpoints is publication._replay_checkpoints
