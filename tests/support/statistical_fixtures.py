"""Synthetic receipt bytes for unit tests, never scientific acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from tools.statistical_partition import shard_cases

from tests.support.synthetic_recovery import build_corpus
from tests.support.synthetic_recovery_runs import _CaseOutcome

IDENTITY = {
    "source_commit": "a" * 40,
    "source_tree": "b" * 40,
    "input_sha256": {"lock": "c" * 64},
    "runtime": {"python": "3.12.13"},
    "workflow": {"provider": "github-actions", "repository": "owner/repository", "run_id": "1", "run_attempt": "1"},
}
BUDGET = {"cpu_count": 3, "case_workers": 3, "local_workers": 1}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_fixture_shards(directory: Path) -> tuple[Path, ...]:
    cases = build_corpus()
    directories = []
    for index in range(8):
        shard = directory / f"artifact-{index}"
        shard.mkdir(parents=True)
        records = []
        for case in shard_cases(cases, index):
            content = json_bytes({"outcome": asdict(_CaseOutcome(case.case_id)), "elapsed_seconds": 1.0})
            (shard / f"case-{case.case_id}.json").write_bytes(content)
            records.append({"case_id": case.case_id, "sha256": hashlib.sha256(content).hexdigest()})
        manifest = {
            "schema": "xrr-r23-statistical-shard-v1",
            "state": "SHARD_COMPLETE",
            "shard_index": index,
            "shard_count": 8,
            "identity": IDENTITY,
            "worker_budget": BUDGET,
            "cases": records,
        }
        (shard / "result.json").write_bytes(json_bytes(manifest))
        directories.append(shard)
    return tuple(directories)
