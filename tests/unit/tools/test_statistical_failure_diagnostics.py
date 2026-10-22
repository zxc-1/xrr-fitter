"""Failure diagnostics from controlled results; these tests perform no fits."""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tests.support import synthetic_recovery_runs as runs
from tests.support.statistical_fixtures import BUDGET, IDENTITY
from tests.support.synthetic_recovery import build_corpus

from xrr_fitter.model.analysis import ConfidenceClass, FitResult, UncertaintyReport
from xrr_fitter.model.bootstrap import BootstrapResult
from xrr_fitter.model.fitting import FitCandidate, FitStageSummary
from xrr_fitter.model.parameters import ParameterDefinition, ParameterValue


def _candidate() -> FitCandidate:
    return FitCandidate(
        candidate_id="E-0",
        seed_index=0,
        unit_vector=np.array([0.4]),
        parameters=(ParameterValue("thickness", 40.0, 1.0, 100.0),),
        objective=1.5,
        valid=True,
        stop_reason="converged",
        nfev=12,
        qz_a_inv=np.array([0.1, 0.2, 0.3]),
        model_normalized=np.array([1.0, 0.1, 0.01]),
        log_residuals_decades=np.array([0.01, -0.01, np.nan]),
        residuals=np.array([0.01, -0.01, np.nan]),
        weighted_residuals=np.array([0.01, -0.01, np.nan]),
        expanded_stack=None,
        sld_depth_a=np.array([]),
        sld_profile_a2=np.array([], dtype=complex),
        diagnostics=(),
        ranking_objective=1.25,
    )


def _result() -> FitResult:
    bootstrap = BootstrapResult(
        ("thickness",),
        np.full((6, 1), 40.0),
        (),
        0.25,
        8,
        ((1, "invalid_candidate"), (6, "optimizer_limit")),
        unavailable_reason="excessive_fit_failures",
    )
    uncertainty = UncertaintyReport(
        correlation_names=("thickness",),
        correlation_matrix=np.ones((1, 1)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.25,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        bootstrap_performed=True,
        bootstrap_evidence=bootstrap,
    )
    candidate = _candidate()
    rejected = replace(
        candidate,
        candidate_id="E-1",
        seed_index=1,
        valid=False,
        objective=float("inf"),
        ranking_objective=None,
        unit_vector=np.array([0.8]),
        stop_reason="invalid_evaluation",
    )
    return FitResult(
        parameter_definitions=(
            ParameterDefinition("thickness", "Thickness", "A", "layer", 40.0, 1.0, 100.0, "linear", False),
        ),
        candidates=(candidate, rejected),
        best_index=0,
        confidence=ConfidenceClass.UNTRUSTED,
        warnings=("retained warning",),
        child_seeds=(101, 102),
        stage_summaries=(
            FitStageSummary("A", ("E-0",), 1.5, 2, ("reviewed",)),
            FitStageSummary("E", ("E-0", "E-1"), 1.25, 24, ("converged", "invalid_evaluation")),
        ),
        region_labels=np.array([0, 0, -1]),
        region_weights=np.array([0.5, 0.5, 0.0]),
        uncertainty=uncertainty,
        classification_evidence=("bootstrap_failure_rate",),
    )


@pytest.fixture
def controlled_failure(monkeypatch):
    case = next(case for case in build_corpus() if case.case_id == "oxide-cap-14012")
    result = _result()
    data = SimpleNamespace(fit_mask=np.array([True, True, False]))
    calls = []

    def stored_result(selected, **kwargs):
        calls.append((selected.case_id, kwargs))
        return result, data

    monkeypatch.setattr(runs, "WORKER_CASES", {case.case_id: case})
    monkeypatch.setattr(runs, "_fit_case", stored_result)
    with pytest.raises(AssertionError) as caught:
        runs._fit_worker_case(case.case_id)
    assert len(calls) == 1
    assert caught.value.args[0]["case_id"] == case.case_id
    return case, result, caught.value


def test_worker_keeps_existing_classification_and_case_on_assertion(controlled_failure) -> None:
    case, result, error = controlled_failure
    diagnostic = getattr(error, "fit_diagnostics", None)
    assert diagnostic is not None, "post-fit failure lost already-computed classification evidence"
    assert diagnostic["case"] == {
        "case_id": case.case_id,
        "category": case.category,
        "seed": case.seed,
        "fit_seed": case.fit_seed,
        "expectation": case.expectation,
    }
    assert diagnostic["classification_evidence"] == ["bootstrap_failure_rate"]
    assert diagnostic["confidence"] == result.confidence.value
    assert diagnostic["best_index"] == 0 and diagnostic["skipped_stages"] == []
    assert diagnostic["child_seeds"] == [101, 102]


def test_worker_keeps_candidate_and_stage_classification_inputs(controlled_failure) -> None:
    diagnostic = controlled_failure[2].fit_diagnostics
    assert diagnostic["candidates"][0]["selection_objective"] == 1.25
    assert diagnostic["candidates"][1]["valid"] is False
    assert diagnostic["candidates"][1]["objective"] == "+inf"
    assert diagnostic["candidates"][1]["unit_vector"] == [0.8]
    assert diagnostic["stage_summaries"][-1]["candidate_ids"] == ["E-0", "E-1"]


def test_worker_keeps_actual_bootstrap_evidence_and_serializable_exception(controlled_failure) -> None:
    error = controlled_failure[2]
    diagnostic = error.fit_diagnostics
    bootstrap = diagnostic["uncertainty"]["bootstrap"]
    assert (bootstrap["attempted_count"], bootstrap["successful_samples"], bootstrap["failure_rate"]) == (8, 6, 0.25)
    assert bootstrap["failure_reasons"] == [[1, "invalid_candidate"], [6, "optimizer_limit"]]
    assert json.loads(json.dumps(diagnostic, allow_nan=False)) == diagnostic
    restored = pickle.loads(pickle.dumps(error))
    assert restored.args == error.args and restored.fit_diagnostics == diagnostic


def _rethrow(error):
    raise error


@pytest.mark.spawn
def test_original_assertion_diagnostics_survive_real_spawn_transport(controlled_failure) -> None:
    _case, _result_value, error = controlled_failure
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        future = pool.submit(_rethrow, error)
        with pytest.raises(AssertionError) as caught:
            future.result(timeout=30)
    diagnostic = getattr(caught.value, "fit_diagnostics", None)
    assert diagnostic is not None, "spawn transport lost failure diagnostics"
    assert diagnostic == error.fit_diagnostics and caught.value.args == error.args


def test_shard_binds_failure_sidecar_without_publishing_a_success(
    controlled_failure,
    monkeypatch,
    tmp_path,
    load_tool_module,
) -> None:
    case, _result_value, error = controlled_failure
    module = load_tool_module("statistical_shards")
    monkeypatch.setattr(module, "capture_identity", lambda _root: copy.deepcopy(IDENTITY))
    monkeypatch.setattr(module.os, "cpu_count", lambda: 3)
    root, report = tmp_path / "repo", tmp_path / "report"
    root.mkdir()

    def fail(cases, budget, publish):
        assert budget == BUDGET
        assert case.case_id in {item.case_id for item in cases}
        publish(cases[0], runs._CaseOutcome(cases[0].case_id), 1.0)
        raise pickle.loads(pickle.dumps(error))

    monkeypatch.setattr(module, "_compute_cases", fail)
    with pytest.raises(AssertionError):
        module.run_shard(root, report, 4, allow_compute=True)
    assert not (report / "result.json").exists()
    assert len(tuple(report.glob("case-*.json"))) == 1
    failure = json.loads((report / "failure.json").read_bytes())
    assert failure["state"] == "FAIL" and failure["error_type"] == "AssertionError"
    path = report / "failure-diagnostics.json"
    assert path.exists(), "failure publication lost already-computed diagnostics"
    raw = path.read_bytes()
    sidecar = json.loads(raw)
    assert failure["diagnostics_sha256"] == hashlib.sha256(raw).hexdigest()
    assert sidecar["schema"] == "xrr-r23-statistical-failure-diagnostics-v1"
    assert sidecar["state"] == "FAIL" and sidecar["shard_index"] == 4
    assert sidecar["identity"] == IDENTITY and sidecar["worker_budget"] == BUDGET
    assert sidecar["diagnostics"] == error.fit_diagnostics


def test_output_contract_failure_also_retains_available_result(monkeypatch) -> None:
    case = build_corpus()[0]
    result = FitResult.failed("no valid candidate", 3)
    monkeypatch.setattr(runs, "WORKER_CASES", {case.case_id: case})
    monkeypatch.setattr(runs, "_fit_case", lambda *_args, **_kwargs: (result, None))
    with pytest.raises(AssertionError) as caught:
        runs._fit_worker_case(case.case_id)
    diagnostic = getattr(caught.value, "fit_diagnostics", None)
    assert diagnostic is not None, "output-contract failure discarded its FitResult"
    assert diagnostic["best_index"] is None and diagnostic["candidates"] == []
    assert diagnostic["uncertainty"] is None


def test_diagnostic_serialization_error_keeps_original_worker_assertion(controlled_failure, monkeypatch) -> None:
    case, _result_value, original = controlled_failure

    def broken_serializer(*_args):
        raise ValueError("diagnostic serialization failed")

    monkeypatch.setattr(runs, "failure_diagnostics", broken_serializer)
    with pytest.raises(AssertionError) as caught:
        runs._fit_worker_case(case.case_id)
    assert caught.value.args == original.args
    assert caught.value.fit_diagnostics_error == {
        "error_type": "ValueError",
        "error": "diagnostic serialization failed",
    }


@pytest.mark.parametrize("failure_kind", ["write", "serialize", "worker"])
def test_diagnostic_failure_does_not_replace_original_shard_failure(
    controlled_failure,
    monkeypatch,
    tmp_path,
    load_tool_module,
    failure_kind,
) -> None:
    _case, _result_value, error = controlled_failure
    module = load_tool_module("statistical_shards")
    monkeypatch.setattr(module, "capture_identity", lambda _root: copy.deepcopy(IDENTITY))
    root, report = tmp_path / "repo", tmp_path / "report"
    root.mkdir()
    original_write = module._write

    def write(path, identity, name, content):
        if name == "failure-diagnostics.json" and failure_kind == "write":
            raise OSError("sidecar storage failed")
        original_write(path, identity, name, content)

    if failure_kind == "serialize":
        error.fit_diagnostics = {"unencodable": object()}
    elif failure_kind == "worker":
        del error.fit_diagnostics
        error.fit_diagnostics_error = {"error_type": "ValueError", "error": "worker diagnostic failed"}

    def fail(*_args):
        raise error

    monkeypatch.setattr(module, "_write", write)
    monkeypatch.setattr(module, "_compute_cases", fail)
    with pytest.raises(AssertionError) as caught:
        module.run_shard(root, report, 4, allow_compute=True)
    assert caught.value is error
    failure = json.loads((report / "failure.json").read_bytes())
    assert failure["state"] == "FAIL" and failure["error_type"] == "AssertionError"
    assert (
        failure["diagnostics_error"]["error_type"]
        == {
            "write": "OSError",
            "serialize": "TypeError",
            "worker": "ValueError",
        }[failure_kind]
    )
    assert "diagnostics_sha256" not in failure
    assert not (report / "result.json").exists()
