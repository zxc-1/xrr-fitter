"""Atomic evidence persistence, fixed-shard continuation, and completeness summaries."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.tools.poisson_diagnostic_fixtures import _row, _test_registration
from tests.unit.tools.poisson_diagnostic_fixtures import tool as tool
from tests.unit.tools.poisson_writer_fixtures import writer_lifecycle as writer_lifecycle


@pytest.mark.parametrize("value", [b"immutable evidence", {"stage": "sealed"}])
def test_write_new_syncs_and_closes_real_stream_before_link_or_unlink(tool, writer_lifecycle, tmp_path, value):
    target = tmp_path / "result.json"

    tool._write_new(target, value)

    expected = value if isinstance(value, bytes) else tool.protocol.canonical(value)
    assert target.read_bytes() == expected
    assert writer_lifecycle.events == ["write", "flush", "fsync", "link", "unlink"]
    assert writer_lifecycle.stream.closed
    assert not list(tmp_path.glob(".pending-*"))


@pytest.mark.parametrize("stage", ["write", "flush", "fsync", "link"])
def test_write_new_closes_and_cleans_temporary_on_io_failure(tool, writer_lifecycle, tmp_path, stage):
    target = tmp_path / "result.json"
    writer_lifecycle.fail_at = stage

    with pytest.raises(OSError) as caught:
        tool._write_new(target, b"must not be published")

    assert caught.value is writer_lifecycle.error
    assert writer_lifecycle.stream.closed
    assert writer_lifecycle.events[-1] == "unlink"
    assert not target.exists()
    assert not list(tmp_path.glob(".pending-*"))


def test_write_new_never_overwrites_existing_target_and_cleans_closed_temporary(tool, writer_lifecycle, tmp_path):
    target = tmp_path / "result.json"
    target.write_bytes(b"original sealed evidence")
    before = target.stat()

    with pytest.raises(FileExistsError):
        tool._write_new(target, b"replacement")

    assert target.read_bytes() == b"original sealed evidence"
    assert target.stat().st_ino == before.st_ino
    assert writer_lifecycle.events == ["write", "flush", "fsync", "link", "unlink"]
    assert not list(tmp_path.glob(".pending-*"))


def test_write_new_propagates_cleanup_error_without_removing_published_evidence(tool, writer_lifecycle, tmp_path):
    target = tmp_path / "result.json"
    writer_lifecycle.fail_at = "unlink"

    with pytest.raises(OSError) as caught:
        tool._write_new(target, b"published evidence")

    assert caught.value is writer_lifecycle.error
    assert target.read_bytes() == b"published evidence"
    assert writer_lifecycle.stream.closed
    temporary = Path(writer_lifecycle.stream.name)
    assert temporary.exists()
    writer_lifecycle.fail_at = None
    temporary.unlink()
    assert not list(tmp_path.glob(".pending-*"))


def test_seed_records_survive_resume_and_failed_seeds_are_never_retried(tool, monkeypatch, tmp_path):
    output, _manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)
    calls = []

    def runner(root, group, seed):
        calls.append((group, seed))
        (root / "raw-test-input.txt").write_text(str(seed))
        if len(calls) == 1:
            raise ValueError("declared numerical failure")
        return _row()

    tool.run_shard(output, 0, max_cases=2, case_runner=runner)
    with pytest.raises(FileExistsError):
        tool.run_shard(output, 0, max_cases=2, case_runner=runner)
    tool.run_shard(output, 0, resume=True, max_cases=1, case_runner=runner)
    report = tool.summarize_experiment(output)

    assert len(calls) == 3
    assert report["scenarios"]["null_single"]["unavailable"] == 1
    assert report["scenarios"]["null_single"]["attempted"] == 3
    assert report["acceptance_status"] == "UNVERIFIED"
    assert report["protocol_complete"] is False
    assert len(list((output / "cases").rglob("result.json"))) == 3


def test_changed_completed_record_or_input_never_becomes_a_cache_hit(tool, monkeypatch, tmp_path):
    output, _manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)

    def runner(root, *_args):
        (root / "input.xy").write_text("0.3 5\n")
        return _row()

    tool.run_shard(output, 0, max_cases=1, case_runner=runner)
    raw = next((output / "cases").rglob("input.xy"))
    raw.write_text("0.3 6\n")

    with pytest.raises(ValueError, match="artifact"):
        tool.run_shard(output, 0, resume=True, max_cases=1, case_runner=runner)


def test_interrupt_leaves_incomplete_evidence_and_requires_explicit_failure_sealing(tool, monkeypatch, tmp_path):
    output, _manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)

    def interrupt(*_args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        tool.run_shard(output, 0, max_cases=1, case_runner=interrupt)
    assert tool.summarize_experiment(output)["incomplete_attempts"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        tool.run_shard(output, 0, resume=True, max_cases=1, case_runner=interrupt)
    tool.run_shard(output, 0, resume=True, seal_interrupted=True, max_cases=1, case_runner=interrupt)
    report = tool.summarize_experiment(output)["scenarios"]["null_single"]
    assert report["unavailable"] == 1
    assert report["attempted"] == 1


def test_summary_report_is_never_overwritten(tool, monkeypatch, tmp_path):
    output, _manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)
    report = tmp_path / "summary.json"

    assert tool.main(["summarize", "--experiment", str(output), "--output", str(report)]) == 2
    before = report.read_bytes()
    with pytest.raises(FileExistsError):
        tool.main(["summarize", "--experiment", str(output), "--output", str(report)])
    assert report.read_bytes() == before


def test_completed_record_payload_tampering_is_rejected(tool, monkeypatch, tmp_path):
    output, _manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)
    tool.run_shard(output, 0, max_cases=1, case_runner=lambda *_args: _row())
    path = next((output / "cases").rglob("result.json"))
    record = json.loads(path.read_bytes())
    record["payload"]["diagnostic"]["status"] = "rejected"
    path.write_bytes(tool.protocol.canonical(record))

    with pytest.raises(ValueError, match="record hash"):
        tool.summarize_experiment(output)


def test_code_changed_during_seed_is_retained_as_incomplete_not_success(tool, monkeypatch, tmp_path):
    output, _manifest, identity = _test_registration(tool, monkeypatch, tmp_path)

    def changed(root, *_args):
        (root / "raw.txt").write_text("preserved")
        identity["sha256"] = "b" * 64
        return _row()

    with pytest.raises(ValueError, match="source"):
        tool.run_shard(output, 0, max_cases=1, case_runner=changed)
    assert not list((output / "cases").rglob("result.json"))
    assert next((output / "cases").rglob("raw.txt")).read_text() == "preserved"


def test_simultaneous_same_shard_cannot_reserve_or_overwrite_a_seed(tool, monkeypatch, tmp_path):
    output, _manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)

    with tool._shard_lock(output, 0):
        with pytest.raises(OSError):
            tool.run_shard(output, 0, max_cases=1, case_runner=lambda *_args: pytest.fail("runner started"))
    assert not (output / "cases").exists()


def test_out_of_protocol_seed_directories_are_not_ignored_in_summary(tool, monkeypatch, tmp_path):
    output, _manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)
    extra = output / "cases" / "null_single" / "123"
    extra.mkdir(parents=True)
    (extra / "result.json").write_text("{}")

    with pytest.raises(ValueError, match="undeclared"):
        tool.summarize_experiment(output)


def test_complete_population_can_fail_and_passing_holdout_can_have_incomplete_replay(tool, monkeypatch, tmp_path):
    output, _manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)

    def summaries(_root, manifest, group):
        kind = manifest["protocol"]["scenarios"][group]["kind"]
        complete = group != "replay_low"
        gates = [] if kind == "replay" else [{"passed": True}]
        return {"protocol_complete": complete, "endpoints": gates}, 0

    monkeypatch.setattr(tool.module, "_scenario_summary", summaries)
    report = tool.summarize_experiment(output)
    assert report["holdout_complete"] is True
    assert report["protocol_complete"] is False
    assert report["acceptance_status"] == "PASS"


def test_summary_refuses_to_present_old_source_evidence_as_current_acceptance(tool, monkeypatch, tmp_path):
    output, _manifest, identity = _test_registration(tool, monkeypatch, tmp_path)
    identity["sha256"] = "b" * 64

    with pytest.raises(ValueError, match="source"):
        tool.summarize_experiment(output)


def test_cli_import_does_not_require_a_posix_only_module_on_windows(tool):
    source = Path(tool.__file__).read_text()
    names = {alias.name for node in ast.parse(source).body if isinstance(node, ast.Import) for alias in node.names}

    assert "fcntl" not in names


def test_windows_shard_lock_uses_nonblocking_native_byte_lock(tool, monkeypatch, tmp_path):
    calls = []
    native = SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=lambda fd, mode, count: calls.append((fd, mode, count)))
    monkeypatch.setitem(tool.sys.modules, "msvcrt", native)
    with (tmp_path / "lock").open("w+b") as stream:
        stream.write(b"\0\0")
        monkeypatch.setattr(tool.sys, "platform", "win32")
        tool._platform_lock(stream, acquire=True)
        tool._platform_lock(stream, acquire=False)
        assert calls == [(stream.fileno(), 2, 1), (stream.fileno(), 0, 1)]
        assert stream.tell() == 0
