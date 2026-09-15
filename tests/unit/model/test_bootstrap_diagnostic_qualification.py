"""A sufficient sample axis is not a passed Poisson model diagnostic."""

from dataclasses import replace

import numpy as np
import pytest

from xrr_fitter.analysis.bootstrap import bootstrap_local
from xrr_fitter.model.analysis import BootstrapResult


@pytest.mark.parametrize("reason", ["residual_diagnostics_failed", "diagnostic_budget_insufficient"])
def test_diagnostic_gate_preserves_quantiles_but_withholds_formal_confidence(reason):
    assert "diagnostic_unavailable_reason" in BootstrapResult.__dataclass_fields__, (
        "bootstrap needs diagnostic eligibility"
    )
    original = bootstrap_local(lambda rng, _index: rng.normal(size=1), ("scale",), sample_count=200, child_seed=21)
    qualified = replace(original, diagnostic_unavailable_reason=reason, unavailable_reason=reason)
    np.testing.assert_array_equal(qualified.samples, original.samples)
    assert qualified.intervals == original.intervals
    assert qualified.successful_samples == original.successful_samples
    assert qualified.interval_kind == "exploratory_bootstrap"
    assert qualified.confidence_level is None
    assert qualified.unavailable_reason == reason


def test_sample_sufficiency_remains_an_independent_gate():
    assert "diagnostic_unavailable_reason" in BootstrapResult.__dataclass_fields__, (
        "bootstrap needs diagnostic eligibility"
    )
    original = bootstrap_local(lambda rng, _index: rng.normal(size=1), ("scale",), sample_count=8, child_seed=21)
    qualified = replace(original, diagnostic_unavailable_reason="residual_diagnostics_failed")
    assert qualified.intervals == ()
    assert qualified.unavailable_reason == "insufficient_successful_samples"
    assert qualified.confidence_level is None


def test_diagnostic_reason_and_display_qualification_must_agree():
    assert "diagnostic_unavailable_reason" in BootstrapResult.__dataclass_fields__, (
        "bootstrap needs diagnostic eligibility"
    )
    original = bootstrap_local(lambda rng, _index: rng.normal(size=1), ("scale",), sample_count=200, child_seed=21)
    with pytest.raises(ValueError, match="calibration gates"):
        replace(original, diagnostic_unavailable_reason="residual_diagnostics_failed")
