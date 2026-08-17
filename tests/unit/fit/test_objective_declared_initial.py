from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

import xrr_fitter.evaluation as evaluation_module
from xrr_fitter.fit.objective import evaluate_declared_initial
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import PhysicalValueError
from xrr_fitter.model.structure import ExpandedSlabLimitError


def _problem():
    config = replace(FitConfig.fast(master_seed=7), scale_prior_enabled=False)
    return compile_fit_problem(
        prepared_data(size=64),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (
            PhysicalValueError("roughness outside candidate geometry"),
            "constraint_violation:PhysicalValueError",
        ),
        (
            ExpandedSlabLimitError("expanded slab count exceeds the limit"),
            "constraint_violation:ExpandedSlabLimitError",
        ),
    ],
    ids=("physical-value", "expanded-slab-limit"),
)
def test_declared_initial_marks_known_physical_encoding_failures_invalid(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    reason: str,
) -> None:
    problem = _problem()

    def fail_encoding(_problem, _physical_values):
        raise error

    monkeypatch.setattr(evaluation_module, "encode_physical_vector", fail_encoding)

    result = evaluate_declared_initial(problem)

    assert result.valid is False
    assert result.reason == reason
    assert result.objective == float("inf")
    assert result.fit_log_residuals_decades.shape == (np.count_nonzero(problem.data.fit_mask),)


def test_declared_initial_propagates_unexpected_encoding_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    sentinel = RuntimeError("unexpected initial encoding failure")

    def fail_encoding(_problem, _physical_values):
        raise sentinel

    monkeypatch.setattr(evaluation_module, "encode_physical_vector", fail_encoding)

    with pytest.raises(RuntimeError, match="unexpected initial encoding failure") as excinfo:
        evaluate_declared_initial(problem)

    assert excinfo.value is sentinel
