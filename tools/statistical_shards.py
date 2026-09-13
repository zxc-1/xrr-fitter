#!/usr/bin/env python3
"""Compute one complete, provenance-bound partition of the approved corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __name__ == "__main__":
    sys.dont_write_bytecode = True

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
ROOT = Path(__file__).resolve().parents[1]
for directory in (TOOL_DIRECTORY, str(ROOT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from tests.support.synthetic_recovery import build_corpus  # noqa: E402
from tests.support.synthetic_recovery_runs import (  # noqa: E402
    _case_work_estimate,
    _fit_worker_case,
    _initialize_worker_cases,
)

from installed_inputs import json_object  # noqa: E402
from statistical_partition import SHARD_COUNT, shard_cases  # noqa: E402
from statistical_provenance import capture_identity  # noqa: E402
from statistical_records import decode_record, encode_record  # noqa: E402
from verify_publish import _write_new_file_in_anchored_directory  # noqa: E402
from verify_report import (  # noqa: E402
    _directory_identity,
    _prepare_report_directory,
    _resolve_output_path,
)


def worker_budget() -> dict[str, int]:
    cpus = max(1, os.cpu_count() or 1)
    workers = min(5, cpus)
    return {"cpu_count": cpus, "case_workers": workers, "local_workers": min(4, max(1, cpus // workers))}


def _timed_case(case_id: str):
    started = time.perf_counter()
    outcome = _fit_worker_case(case_id)
    return outcome, time.perf_counter() - started


def _compute_cases(cases, budget: dict[str, int], publish) -> None:
    scheduled = sorted(cases, key=lambda case: -_case_work_estimate(case))
    with ProcessPoolExecutor(
        max_workers=budget["case_workers"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker_cases,
        initargs=(budget["local_workers"],),
    ) as executor:
        futures = {executor.submit(_timed_case, case.case_id): case for case in scheduled}
        try:
            for future in as_completed(futures):
                outcome, elapsed = future.result()
                publish(futures[future], outcome, elapsed)
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write(report: Path, identity: tuple[int, int], name: str, content: bytes) -> None:
    _write_new_file_in_anchored_directory(
        report,
        identity,
        name,
        content,
        directory_label="statistical report",
        file_label="statistical evidence",
    )


def _output_directory(root: Path, report: Path) -> tuple[Path, tuple[int, int]]:
    output = _resolve_output_path(report, "statistical report")
    if output.is_relative_to(root.resolve()) or os.path.lexists(output):
        raise ValueError("statistical report must be a new external directory")
    return output, _prepare_report_directory(output)


def _case_publisher(report: Path, directory_identity: tuple[int, int], hashes: dict):
    def publish(case, outcome, elapsed) -> None:
        content = _json_bytes(encode_record(outcome, elapsed))
        decode_record(case, json_object(content))
        _write(report, directory_identity, f"case-{case.case_id}.json", content)
        hashes[case.case_id] = hashlib.sha256(content).hexdigest()
        print(json.dumps({"event": "case_complete", "case_id": case.case_id, "elapsed_seconds": elapsed}), flush=True)

    return publish


def _manifest(index: int, identity: dict, budget: dict, cases, hashes: dict) -> dict:
    if set(hashes) != {case.case_id for case in cases}:
        raise ValueError("statistical shard has missing or unexpected case outcomes")
    return {
        "schema": "xrr-r23-statistical-shard-v1",
        "state": "SHARD_COMPLETE",
        "shard_index": index,
        "shard_count": SHARD_COUNT,
        "identity": identity,
        "worker_budget": budget,
        "cases": [{"case_id": case.case_id, "sha256": hashes[case.case_id]} for case in cases],
    }


def run_shard(root: Path, report: Path, index: int, *, allow_compute: bool = False) -> dict:
    if allow_compute is not True:
        raise ValueError("statistical shard fitting requires explicit --compute permission")
    cases = shard_cases(build_corpus(), index)
    identity = capture_identity(root)
    budget = worker_budget()
    report, directory_identity = _output_directory(root, report)
    hashes: dict[str, str] = {}
    print(
        json.dumps({"event": "shard_start", "index": index, "identity": identity, "worker_budget": budget}), flush=True
    )
    try:
        _compute_cases(cases, budget, _case_publisher(report, directory_identity, hashes))
        if capture_identity(root) != identity:
            raise ValueError("statistical source or runtime identity changed during computation")
        result = _manifest(index, identity, budget, cases, hashes)
        _write(report, directory_identity, "result.json", _json_bytes(result))
    except BaseException as error:
        failure = {
            "state": "FAIL",
            "shard_index": index,
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
        }
        _write(report, directory_identity, "failure.json", _json_bytes(failure))
        raise
    _directory_identity(report, "statistical report")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, choices=range(SHARD_COUNT), required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--compute", action="store_true", help="explicitly allow new shard fits")
    args = parser.parse_args(argv)
    if not args.compute:
        parser.error("statistical shard fitting requires explicit --compute permission")
    run_shard(ROOT, args.report_dir, args.shard_index, allow_compute=args.compute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
