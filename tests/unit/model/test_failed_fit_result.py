from __future__ import annotations

import numpy as np
import pytest

from xrr_fitter.model.analysis import ConfidenceClass, FitResult


def _failed(count):
    assert hasattr(FitResult, "failed"), "failure result arrays must be constructed by the model owner"
    return FitResult.failed("ValueError: source failed", count)


def test_failed_result_has_no_publishable_candidate():
    result = _failed(4)
    assert result.confidence is ConfidenceClass.UNTRUSTED
    assert result.best_index is None
    assert result.candidates == ()
    assert result.uncertainty is None


def test_failed_result_preserves_diagnostic_and_empty_search_evidence():
    result = _failed(4)
    assert result.warnings == ("ValueError: source failed",)
    assert result.parameter_definitions == ()
    assert result.child_seeds == ()
    assert result.stage_summaries == ()


@pytest.mark.parametrize("count", [0, 1, 4])
def test_failed_result_owns_readonly_source_aligned_region_arrays(count):
    result = _failed(count)
    np.testing.assert_array_equal(result.region_labels, [-1] * count)
    np.testing.assert_array_equal(result.region_weights, [0.0] * count)
    assert not result.region_labels.flags.writeable
    assert not result.region_weights.flags.writeable
