"""Explicit byte-bound inputs for the unchanged full-corpus pytest assertion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from tools.statistical_results import StatisticalResults, load_results

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser) -> None:
    group = parser.getgroup("statistical evidence")
    group.addoption("--statistical-results", default=None)
    group.addoption("--statistical-report", default=None)


@dataclass(frozen=True)
class StatisticalExecution:
    results: StatisticalResults
    report_path: Path

    def evaluate(self):
        return self.results.evaluate()

    def publish(self, report) -> None:
        self.results.publish(report, self.report_path)


@pytest.fixture
def statistical_evidence(request):
    inputs = request.config.getoption("statistical_results", default=None)
    report = request.config.getoption("statistical_report", default=None)
    if (inputs is None) != (report is None):
        raise ValueError("statistical input and report must be provided together")
    if inputs is None:
        return None
    return StatisticalExecution(load_results(ROOT, Path(inputs)), Path(report))
