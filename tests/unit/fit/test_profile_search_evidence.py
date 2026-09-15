"""Profile-basin continuation appends its real work without erasing the search."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.unit.fit.test_stage_search import _candidate, _problem

from xrr_fitter.fit import profile_rescue
from xrr_fitter.model.fitting import SearchAllocation, SearchEvidence


def _original_candidates(problem, seeds):
    poor = _candidate(problem, "source", np.full(len(problem.variables), 0.8))
    return tuple(
        replace(
            poor,
            candidate_id=f"E-{index}",
            seed_index=index,
            search_evidence=(
                SearchEvidence(
                    f"E-{index}",
                    seed,
                    (),
                    (SearchAllocation(f"E-{index}", f"E-{index}-local", 1, 1, 1),),
                    "previous_local",
                ),
            ),
        )
        for index, seed in enumerate(seeds)
    )


def _assert_recorded_work(before, after, work):
    assert len(after.search_evidence) == len(before.search_evidence) + 1
    assert after.search_evidence[:-1] == before.search_evidence
    added = after.search_evidence[-1]
    (allocation,) = added.budget_allocations
    assert (allocation.max_nfev, allocation.nfev, added.stop_reason) == work
    assert after.nfev == before.nfev + work[1]
    return added, allocation


def test_profile_rescue_appends_each_real_optimizer_allocation_and_preserves_prior_work(monkeypatch) -> None:
    problem = _problem(seed=775)
    center = np.full(len(problem.variables), 0.4)
    seeds = (101, 202, 303, 404)
    originals = _original_candidates(problem, seeds)
    original_solve = profile_rescue.solve_local
    observed = []

    def solve(context, start, *, max_nfev, **kwargs):
        result = original_solve(context, start, max_nfev=max_nfev, **kwargs)
        observed.append((max_nfev, result.nfev, result.stop_reason))
        return result

    monkeypatch.setattr(profile_rescue, "solve_local", solve)
    rebuilt = profile_rescue.reconverge_profile_basin(
        problem, originals, center, seeds, parameter_name="component.0.thickness_a"
    )

    assert rebuilt is not None
    assert len(observed) == 4
    for index, (before, after, work) in enumerate(zip(originals, rebuilt, observed, strict=True)):
        added, allocation = _assert_recorded_work(before, after, work)
        assert allocation.lineage_id == before.candidate_id
        assert allocation.round_index == 2
        assert added.seed == int(
            np.random.SeedSequence([seeds[index], ord("P"), index]).generate_state(1, dtype=np.uint64)[0]
        )
