"""The root conftest's progress log names the running test and, when it stalls, where it is stuck.

The coverage audit captures pytest's output in memory and loses all of it when the job hits its
time limit, so this file is the only record of which test was running at that point.
"""

from __future__ import annotations

import importlib.util
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


def _root_conftest():
    spec = importlib.util.spec_from_file_location("xrr_root_conftest_for_progress", ROOT / "tests" / "conftest.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stall_in_the_test_body(seconds: float) -> None:
    time.sleep(seconds)


def test_progress_log_names_each_test_and_dumps_the_stack_of_a_stalled_one(tmp_path: Path) -> None:
    progress = _root_conftest().ProgressLog(stall_seconds=0.2, slow_seconds=0.5)
    # The first test starts before the session fixture knows where basetemp is.
    progress.start("tests/probe.py::test_first")
    progress.finish()
    progress.open(tmp_path / "progress.log")
    progress.start("tests/probe.py::test_stuck")
    _stall_in_the_test_body(0.7)
    progress.finish()
    progress.close()
    progress.write("after close")

    text = (tmp_path / "progress.log").read_text(encoding="utf-8")
    first = text.index("[start] tests/probe.py::test_first")
    stuck = text.index("[start] tests/probe.py::test_stuck")
    stalled = text.index("[stalled", stuck)
    assert first < stuck < stalled
    # Wall clock against process CPU tells a test that computes from one that waits or is throttled.
    timing = r"\d+s, cpu \d+s, \d+ threads"
    assert re.match(rf"\[stalled {timing}\] tests/probe.py::test_stuck$", text[stalled:].splitlines()[0])
    assert "_stall_in_the_test_body" in text[stalled:]
    slow = [line for line in text.splitlines() if "[slow" in line]
    assert len(slow) == 1 and re.search(rf"\[slow {timing}\] tests/probe.py::test_stuck$", slow[0])
    assert "after close" not in text


def test_a_stalled_test_also_reaches_the_job_log(tmp_path: Path) -> None:
    # pr-verify streams pytest's stderr into the job log and uploads no basetemp, so the stack of a
    # stuck GUI test has to be written there too, not only into progress.log.
    echo = tmp_path / "stderr.txt"
    descriptor = os.open(echo, os.O_WRONLY | os.O_CREAT, 0o644)
    progress = _root_conftest().ProgressLog(stall_seconds=0.2, slow_seconds=5.0, echo=descriptor)
    progress.open(tmp_path / "progress.log")
    progress.start("tests/probe.py::test_stuck")
    _stall_in_the_test_body(0.7)
    progress.finish()
    progress.close()
    os.close(descriptor)

    text = echo.read_text(encoding="utf-8")
    assert re.search(r"\[stalled \d+s, cpu \d+s, \d+ threads\] tests/probe.py::test_stuck", text)
    assert "_stall_in_the_test_body" in text


class _FaulthandlerCalls:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def dump_traceback_later(self, *_args, **_kwargs) -> None:
        self.calls.append("dump_traceback_later")

    def cancel_dump_traceback_later(self) -> None:
        self.calls.append("cancel_dump_traceback_later")


def test_running_a_test_arms_no_c_level_stall_timer(monkeypatch) -> None:
    # faulthandler's timer thread dumps the main thread's frames without the GIL. Under coverage a
    # 300 s test fired it mid-computation; the dump faulted, the thread kept its lock, and
    # cancel_dump_traceback_later then waited on that lock holding the GIL: the coverage audit froze
    # until its 60-minute limit (PR #31, four runs). The progress log's watcher dumps under the GIL.
    conftest = _root_conftest()
    recorder = _FaulthandlerCalls()
    monkeypatch.setattr(conftest, "faulthandler", recorder)
    progress = conftest.ProgressLog(stall_seconds=60, slow_seconds=60)
    stash = {conftest.PROGRESS_LOG: progress, conftest.STALL_STREAM: None}
    item = SimpleNamespace(nodeid="tests/probe.py::test_quick", config=SimpleNamespace(stash=stash))

    protocol = conftest.pytest_runtest_protocol(item, None)
    next(protocol)
    protocol.close()

    assert recorder.calls == []
