from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _plugin():
    path = ROOT / "tests/statistical_gate.py"
    assert path.is_file(), "missing statistical pytest input plugin"
    spec = importlib.util.spec_from_file_location("r23_statistical_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _request(inputs=None, report=None):
    options = {"statistical_results": inputs, "statistical_report": report}
    return SimpleNamespace(config=SimpleNamespace(getoption=lambda name, **_kwargs: options[name]))


def test_default_fixture_ignores_environment_shortcuts(monkeypatch, tmp_path) -> None:
    module = _plugin()
    monkeypatch.setenv("XRR_STATISTICAL_RESULTS", str(tmp_path))
    calls = []
    monkeypatch.setattr(module, "load_results", lambda *_args: calls.append(True))

    assert module.statistical_evidence.__wrapped__(_request()) is None
    assert calls == []


@pytest.mark.parametrize("inputs,report", [("inputs", None), (None, "summary.json")])
def test_statistical_fixture_requires_explicit_paired_input_and_output(inputs, report) -> None:
    module = _plugin()
    with pytest.raises(ValueError, match="together"):
        module.statistical_evidence.__wrapped__(_request(inputs, report))


def test_statistical_fixture_loads_and_evaluates_only_the_explicit_input(monkeypatch, tmp_path) -> None:
    module = _plugin()
    calls = []
    report = object()
    loaded = SimpleNamespace(
        evaluate=lambda: report,
        publish=lambda value, path: calls.append((value, path)),
    )
    input_path = tmp_path / "shards"
    output_path = tmp_path / "summary.json"

    def load(root, path):
        assert root == ROOT
        assert path == input_path
        return loaded

    monkeypatch.setattr(module, "load_results", load)
    execution = module.statistical_evidence.__wrapped__(_request(str(input_path), str(output_path)))

    assert execution.evaluate() is report
    execution.publish(report)
    assert calls == [(report, output_path)]


def test_acceptance_test_reuses_full_expected_counts_without_refitting(monkeypatch) -> None:
    from collections import Counter

    from tests.acceptance import test_synthetic_recovery_corpus as acceptance
    from tests.support.synthetic_recovery import CorpusReport, build_corpus

    report = CorpusReport(
        "xrr-r23-synthetic-recovery-v1",
        "PASS",
        220,
        220,
        tuple(sorted(Counter(case.category for case in build_corpus()).items())),
        (),
    )
    published = []
    execution = SimpleNamespace(evaluate=lambda: report, publish=published.append)
    monkeypatch.setattr(acceptance, "run_corpus", lambda *_args: pytest.fail("must not refit supplied shards"))

    acceptance.test_synthetic_recovery_corpus_meets_approved_thresholds(statistical_evidence=execution)

    assert published == [report]
