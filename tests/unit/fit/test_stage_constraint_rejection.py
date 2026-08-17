from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.evaluation import EvaluationConstraintError
from xrr_fitter.fit.candidates import CandidateStart
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec


def _stages_api():
    return import_module("xrr_fitter.fit.stages")


def _problem(seed: int):
    config = replace(
        FitConfig.fast(seed),
        final_seed_count=4,
        budget=SearchBudget(0, 0, 5, 1, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    return compile_fit_problem(
        prepared_data(size=40),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )


def test_stage_a_rejects_constraint_encoding_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _stages_api()

    def fail(*_args, **_kwargs):
        raise EvaluationConstraintError("constraint_out_of_bounds:repeat")

    monkeypatch.setattr(api, "encode_physical_vector", fail)

    assert api._stage_a_candidate(_problem(756), CandidateStart((), "declared"), 0) is None
