from __future__ import annotations

from collections import Counter

import pytest
from tests.support import synthetic_recovery
from tests.support.synthetic_recovery_runs import _case_work_estimate, _CaseOutcome


def test_shards_cover_all_canonical_cases_once_in_original_order(load_tool_module) -> None:
    module = load_tool_module("statistical_partition")
    cases = synthetic_recovery.build_corpus()

    shards = tuple(module.shard_cases(cases, index) for index in range(8))
    repeated = tuple(module.shard_cases(cases, index) for index in range(8))

    assert module.SHARD_COUNT == 8
    assert tuple(tuple(case.case_id for case in shard) for shard in shards) == tuple(
        tuple(case.case_id for case in shard) for shard in repeated
    )
    assert Counter(case.case_id for shard in shards for case in shard) == Counter(case.case_id for case in cases)
    assert all(27 <= len(shard) <= 28 for shard in shards)
    positions = {case.case_id: index for index, case in enumerate(cases)}
    for shard in shards:
        assert [positions[case.case_id] for case in shard] == sorted(positions[case.case_id] for case in shard)


def test_shards_spread_expensive_optical_work_instead_of_contiguous_categories(load_tool_module) -> None:
    module = load_tool_module("statistical_partition")
    cases = synthetic_recovery.build_corpus()
    shards = tuple(module.shard_cases(cases, index) for index in range(8))

    work = tuple(sum(_case_work_estimate(case) for case in shard) for shard in shards)

    assert max(work) - min(work) <= max(_case_work_estimate(case) for case in cases)
    assert all(any(case.category == "periodic_mosi" for case in shard) for shard in shards)
    assert all(any(case.category == "model_error" for case in shard) for shard in shards)


@pytest.mark.parametrize("index", [-1, 8, True, 1.0, "0"])
def test_shard_index_must_be_an_exact_bounded_integer(index, load_tool_module) -> None:
    module = load_tool_module("statistical_partition")
    with pytest.raises(ValueError, match="shard index"):
        module.shard_cases(synthetic_recovery.build_corpus(), index)


def test_sharding_rejects_an_incomplete_corpus(load_tool_module) -> None:
    module = load_tool_module("statistical_partition")
    with pytest.raises(ValueError, match="220"):
        module.shard_cases(synthetic_recovery.build_corpus()[:-1], 0)


def _aggregation(monkeypatch):
    function = getattr(synthetic_recovery, "validate_corpus_outcomes", None)
    assert callable(function), "missing independent full-corpus aggregation"
    calls = []

    def record(name):
        def run(cases, outcomes):
            calls.append((name, tuple(case.case_id for case in cases), tuple(value.case_id for value in outcomes)))

        return run

    for name in ("statistical_recovery", "ambiguous", "model_error"):
        monkeypatch.setattr(synthetic_recovery, f"_run_slow_{name}_corpus", record(name))
    return function, calls


def test_aggregation_reuses_all_three_original_global_assertions(monkeypatch) -> None:
    function, calls = _aggregation(monkeypatch)
    cases = synthetic_recovery.build_corpus()
    outcomes = tuple(_CaseOutcome(case.case_id) for case in cases)

    report = function(cases, outcomes)

    assert (report.schema, report.status, report.case_count, report.fit_count, report.failed_case_ids) == (
        "xrr-r23-synthetic-recovery-v1",
        "PASS",
        220,
        220,
        (),
    )
    assert tuple((name, len(ids)) for name, ids, _outcomes in calls) == (
        ("statistical_recovery", 180),
        ("ambiguous", 20),
        ("model_error", 20),
    )
    assert all(ids == outcome_ids for _name, ids, outcome_ids in calls)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_aggregation_rejects_misaligned_outcomes_before_thresholds(monkeypatch, mutation) -> None:
    function, calls = _aggregation(monkeypatch)
    cases = synthetic_recovery.build_corpus()
    outcomes = [_CaseOutcome(case.case_id) for case in cases]
    if mutation == "missing":
        outcomes.pop()
    elif mutation == "duplicate":
        outcomes[-1] = outcomes[0]
    else:
        outcomes[0], outcomes[1] = outcomes[1], outcomes[0]

    with pytest.raises(ValueError, match="align"):
        function(cases, tuple(outcomes))
    assert calls == []


def test_direct_execution_still_fits_every_case_before_aggregation(monkeypatch) -> None:
    function, calls = _aggregation(monkeypatch)
    cases = synthetic_recovery.build_corpus()
    outcomes = tuple(_CaseOutcome(case.case_id) for case in cases)
    dispatched = []

    def fit(requested):
        dispatched.append(tuple(case.case_id for case in requested))
        return outcomes

    monkeypatch.setattr(synthetic_recovery, "_parallel_case_outcomes", fit)
    report = synthetic_recovery.run_corpus(cases)

    assert dispatched == [tuple(case.case_id for case in cases)]
    assert report == function(cases, outcomes)
    assert len(calls) == 6
