"""Explicitly calibrated sampling evidence for presentation-only fixtures."""

from __future__ import annotations

import numpy as np

from xrr_fitter.model.bootstrap import BootstrapResult, bootstrap_calibration_reason


def _fixture_samples(intervals, count):
    bounds = np.array([[lower, upper] for _name, lower, upper in intervals])
    span = bounds[:, 1] - bounds[:, 0]
    return np.linspace(bounds[:, 0] - span * (0.025 / 0.95), bounds[:, 1] + span * (0.025 / 0.95), count)


def bootstrap_evidence(intervals, *, failure_rate=0.0, attempted_count=400):
    count = attempted_count
    failures = int(count * failure_rate)
    assert failures / count == failure_rate
    names = tuple(name for name, _lower, _upper in intervals)
    samples = _fixture_samples(intervals, count - failures)
    reason = bootstrap_calibration_reason(len(samples), failure_rate)
    return BootstrapResult(
        names,
        samples,
        tuple(intervals) if reason is None else (),
        failures / count,
        count,
        tuple((index, "fixture_fit_failed") for index in range(failures)),
        method="custom_resampling",
        unavailable_reason=reason,
    )
