from __future__ import annotations

import sys

import pytest


FORBIDDEN_OUTCOMES = ("skipped", "xfailed", "xpassed", "deselected")
RECORDER_PLUGIN = "xrr-outcome-recorder"


class OutcomeRecorder:
    def __init__(self) -> None:
        self.stats = {name: 0 for name in FORBIDDEN_OUTCOMES}

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.stats["deselected"] += len(items)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        was_xfail = hasattr(report, "wasxfail")
        if report.skipped:
            outcome = "xfailed" if was_xfail else "skipped"
            self.stats[outcome] += 1
        elif report.passed and was_xfail:
            self.stats["xpassed"] += 1

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            self.stats["skipped"] += 1


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(OutcomeRecorder(), RECORDER_PLUGIN)


def _rejected_outcomes(session: pytest.Session, stats: dict[str, object]) -> list[str]:
    rejected = [name for name in FORBIDDEN_OUTCOMES if stats.get(name)]
    if session.testscollected == 0:
        rejected.append("empty collection")
    return rejected


def _report_failure(reporter: object, rejected: list[str]) -> None:
    detail = ", ".join(rejected)
    if reporter is not None:
        reporter.write_line(f"XRR outcome gate rejected: {detail}", red=True)
    else:
        sys.stderr.write(f"XRR outcome gate rejected: {detail}\n")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    recorder = session.config.pluginmanager.get_plugin(RECORDER_PLUGIN)
    stats = getattr(recorder, "stats", {})
    rejected = _rejected_outcomes(session, stats)
    if not rejected:
        return
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
    _report_failure(reporter, rejected)
