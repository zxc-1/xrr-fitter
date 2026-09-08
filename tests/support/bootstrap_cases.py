"""Explicitly calibrated sampling evidence for presentation-only fixtures."""

from __future__ import annotations

import numpy as np

from xrr_fitter.model.analysis import BootstrapResult


def bootstrap_evidence(intervals, *, failure_rate=0.0):
    count = 400
    failures = int(count * failure_rate)
    names = tuple(name for name, _lower, _upper in intervals)
    bounds = np.array([[lower, upper] for _name, lower, upper in intervals])
    span = bounds[:, 1] - bounds[:, 0]
    samples = np.linspace(bounds[:, 0] - span * (0.025 / 0.95), bounds[:, 1] + span * (0.025 / 0.95), count - failures)
    return BootstrapResult(
        names,
        samples,
        tuple(intervals),
        failures / count,
        count,
        tuple((index, "fixture_fit_failed") for index in range(failures)),
        method="custom_resampling",
    )
