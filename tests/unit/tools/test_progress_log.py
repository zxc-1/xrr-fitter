"""The root conftest's progress log names the running test and, when it stalls, where it is stuck.

The coverage audit captures pytest's output in memory and loses all of it when the job hits its
time limit, so this file is the only record of which test was running at that point.
"""

from __future__ import annotations

import importlib.util
import re
import time
from pathlib import Path

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
