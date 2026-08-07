from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import prepared_data, simple_structure

from xrr_fitter.fit.automatic import refit_from_physical_values
from xrr_fitter.fit.local_search import LocalSearchResult
from xrr_fitter.fit.objective import evaluate_vector
from xrr_fitter.fit.problem import compile_fit_problem
from xrr_fitter.model.fitting import FitConfig, SearchBudget
from xrr_fitter.model.instrument import InstrumentSpec
from xrr_fitter.model.parameters import ParameterSetting
from xrr_fitter.model.provenance import fit_search_provenance_sha256


def _problem():
    config = replace(
        FitConfig.fast(1201),
        budget=SearchBudget(0, 0, 8, 2, 1),
        local_workers=1,
        scale_prior_enabled=False,
    )
    initial = compile_fit_problem(
        prepared_data(size=40),
        simple_structure(),
        InstrumentSpec(footprint_mode="none"),
        config,
    )
    settings = tuple(
        ParameterSetting(
            definition.name,
            definition.initial,
            definition.lower,
            definition.upper,
            locked=definition.name != "component.0.thickness_a",
        )
        for definition in initial.parameter_definitions
    )
    return compile_fit_problem(
        initial.data,
        initial.structure,
        initial.instrument,
        initial.config,
        settings,
    )


def test_automatic_refit_rejects_empty_starts() -> None:
    with pytest.raises(ValueError, match="start"):
        refit_from_physical_values(_problem(), (), max_nfev=4)


def test_automatic_refit_publishes_owned_candidates_and_provenance() -> None:
    problem = _problem()
    starts = (
        {"component.0.thickness_a": 18.0},
        {"component.0.thickness_a": 35.0},
    )

    result = refit_from_physical_values(problem, starts, max_nfev=4)

    assert tuple(candidate.candidate_id for candidate in result.candidates) == (
        "automatic-refit-0",
        "automatic-refit-1",
    )
    assert result.stage_summaries[-1].stage == "automatic-refit"
    assert result.stage_summaries[-1].candidate_ids == (
        "automatic-refit-0",
        "automatic-refit-1",
    )
    assert result.provenance_sha256 == fit_search_provenance_sha256(problem, result)
    assert result.best_candidate is not None
    assert result.best_candidate.valid


def test_automatic_refit_caps_solver_work_and_preserves_start_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__("xrr_fitter.fit.automatic", fromlist=["refit_from_physical_values"])
    problem = _problem()
    observed: list[tuple[np.ndarray, int]] = []

    def fake_solve(_problem, start, *, max_nfev, cancelled=None):
        del cancelled
        observed.append((np.asarray(start), max_nfev))
        return LocalSearchResult(
            start,
            evaluate_vector(_problem, start),
            "synthetic",
            1,
        )

    monkeypatch.setattr(module, "solve_local", fake_solve)

    refit_from_physical_values(
        problem,
        ({"component.0.thickness_a": 18.0},),
        max_nfev=10_000,
    )

    assert len(observed) == 1
    assert observed[0][1] == problem.config.budget.local_min_nfev
