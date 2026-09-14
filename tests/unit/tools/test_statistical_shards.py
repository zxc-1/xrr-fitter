from __future__ import annotations

import copy
import json

import pytest
from tests.support.statistical_fixtures import BUDGET, IDENTITY
from tests.support.synthetic_recovery import build_corpus
from tests.support.synthetic_recovery_runs import _CaseOutcome, _initialize_worker_cases
from tools.statistical_partition import shard_cases


@pytest.fixture
def producer(tmp_path, monkeypatch, load_tool_module):
    def prepare():
        module = load_tool_module("statistical_shards")
        monkeypatch.setattr(module, "capture_identity", lambda _root: copy.deepcopy(IDENTITY))
        monkeypatch.setattr(module.os, "cpu_count", lambda: 3)
        root = tmp_path / "repo"
        root.mkdir()
        return module, root, tmp_path / "report"

    return prepare


def _complete(cases, budget, publish) -> None:
    assert budget == BUDGET
    for case in reversed(cases):
        publish(case, _CaseOutcome(case.case_id), 1.25)


def test_producer_emits_only_a_complete_shard_and_all_case_byte_hashes(producer, monkeypatch) -> None:
    module, root, report = producer()
    monkeypatch.setattr(module, "_compute_cases", _complete)

    value = module.run_shard(root, report, 0, allow_compute=True)

    assert value == json.loads((report / "result.json").read_bytes())
    assert value["state"] == "SHARD_COMPLETE"
    assert value["identity"] == IDENTITY
    assert value["worker_budget"] == BUDGET
    expected = shard_cases(build_corpus(), 0)
    assert [item["case_id"] for item in value["cases"]] == [case.case_id for case in expected]
    assert len(tuple(report.glob("case-*.json"))) == len(expected)


def test_producer_preserves_completed_case_evidence_but_never_success_after_failure(producer, monkeypatch) -> None:
    module, root, report = producer()

    def fail(cases, budget, publish):
        publish(cases[0], _CaseOutcome(cases[0].case_id), 1.0)
        raise RuntimeError("fit failed")

    monkeypatch.setattr(module, "_compute_cases", fail)

    with pytest.raises(RuntimeError, match="fit failed"):
        module.run_shard(root, report, 0, allow_compute=True)
    assert not (report / "result.json").exists()
    assert len(tuple(report.glob("case-*.json"))) == 1
    assert json.loads((report / "failure.json").read_bytes())["state"] == "FAIL"


def test_producer_refuses_source_changes_during_computation(producer, monkeypatch) -> None:
    module, root, report = producer()

    def changed(cases, budget, publish):
        _complete(cases, budget, publish)
        monkeypatch.setattr(module, "capture_identity", lambda _root: {**IDENTITY, "source_commit": "f" * 40})

    monkeypatch.setattr(module, "_compute_cases", changed)
    with pytest.raises(ValueError, match="source"):
        module.run_shard(root, report, 0, allow_compute=True)
    assert not (report / "result.json").exists()


@pytest.mark.parametrize("kind", ["existing", "internal", "symlink"])
def test_producer_requires_a_new_regular_external_output(producer, monkeypatch, kind, tmp_path) -> None:
    module, root, report = producer()
    if kind == "existing":
        report.mkdir()
    elif kind == "internal":
        report = root / "report"
    else:
        target = tmp_path / "target"
        target.mkdir()
        report.symlink_to(target, target_is_directory=True)
    called = []
    monkeypatch.setattr(module, "_compute_cases", lambda *_args: called.append(True))

    with pytest.raises(ValueError):
        module.run_shard(root, report, 0, allow_compute=True)
    assert called == []


@pytest.mark.parametrize(
    "cpu,expected",
    [
        (None, {"cpu_count": 1, "case_workers": 1, "local_workers": 1}),
        (3, BUDGET),
        (10, {"cpu_count": 10, "case_workers": 5, "local_workers": 2}),
    ],
)
def test_shard_worker_budget_matches_existing_full_corpus(cpu, expected, producer, monkeypatch) -> None:
    module, _root, _report = producer()
    monkeypatch.setattr(module.os, "cpu_count", lambda: cpu)
    assert module.worker_budget() == expected


def test_compute_spawns_bounded_original_fits_and_publishes_in_completion_order(producer, monkeypatch) -> None:
    module, _root, _report = producer()
    submitted = []
    observed = {}

    class Future:
        def __init__(self, case_id):
            self.case_id = case_id

        def result(self):
            return _CaseOutcome(self.case_id), 1.0

    class Executor:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def submit(self, function, case_id):
            assert function is module._timed_case
            submitted.append(case_id)
            return Future(case_id)

    monkeypatch.setattr(module, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(module, "as_completed", lambda futures: reversed(tuple(futures)))
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda name: f"{name}-context")
    cases = shard_cases(build_corpus(), 0)
    published = []

    module._compute_cases(
        cases, BUDGET, lambda case, outcome, _elapsed: published.append((case.case_id, outcome.case_id))
    )

    assert observed == {
        "max_workers": 3,
        "mp_context": "spawn-context",
        "initializer": _initialize_worker_cases,
        "initargs": (1,),
    }
    assert published == [(case_id, case_id) for case_id in reversed(submitted)]
    assert set(submitted) == {case.case_id for case in cases}
