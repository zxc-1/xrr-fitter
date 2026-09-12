"""Validate the exact complete set of current-source statistical shard bytes."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from tests.support.synthetic_recovery import build_corpus, validate_corpus_outcomes  # noqa: E402

from installed_files import InstalledSnapshot  # noqa: E402
from installed_inputs import InputBindings, json_object  # noqa: E402
from statistical_handoff import StatisticalHandoff, load_handoff  # noqa: E402
from statistical_partition import SHARD_COUNT, shard_cases  # noqa: E402
from statistical_provenance import capture_identity  # noqa: E402
from statistical_records import decode_record, require_schema  # noqa: E402
from verify_publish import _write_new_file_in_anchored_directory  # noqa: E402
from verify_report import _directory_identity, _resolve_output_path  # noqa: E402

SCHEMA_KEYS = {"schema", "state", "shard_index", "shard_count", "identity", "worker_budget", "cases"}


def _worker_budget(value: object) -> None:
    budget = require_schema(value, {"cpu_count", "case_workers", "local_workers"}, "worker budget")
    if any(type(number) is not int or number < 1 for number in budget.values()):
        raise ValueError("statistical worker budget must contain positive integers")
    workers = min(5, budget["cpu_count"])
    local = min(4, max(1, budget["cpu_count"] // workers))
    if (budget["case_workers"], budget["local_workers"]) != (workers, local):
        raise ValueError("statistical worker budget differs from the full-corpus scheduler")


def _shard_manifest(content: bytes, identity: dict) -> tuple[dict, int]:
    value = require_schema(json_object(content), SCHEMA_KEYS, "statistical shard")
    if (value["schema"], value["state"]) != ("xrr-r23-statistical-shard-v1", "SHARD_COMPLETE"):
        raise ValueError("statistical shard is not complete")
    if type(value["shard_count"]) is not int or value["shard_count"] != SHARD_COUNT:
        raise ValueError("statistical shard count differs")
    index = value["shard_index"]
    if type(index) is not int or not 0 <= index < SHARD_COUNT:
        raise ValueError("statistical shard index differs")
    if value["identity"] != identity:
        raise ValueError("statistical shard source, runtime or workflow identity differs")
    _worker_budget(value["worker_budget"])
    return value, index


def _case_bindings(value: object, cases) -> tuple[dict, ...]:
    if not isinstance(value, list):
        raise ValueError("statistical case bindings must be a list")
    records = tuple(require_schema(item, {"case_id", "sha256"}, "case binding") for item in value)
    if tuple(item["case_id"] for item in records) != tuple(case.case_id for case in cases):
        raise ValueError("statistical shard cases do not match the canonical partition")
    return records


def _directory_members(directory: Path, names: set[str]) -> None:
    if {entry.name for entry in directory.iterdir()} != names:
        raise ValueError("statistical evidence directory contains missing or extra files")


@dataclass
class StatisticalResults:
    root: Path
    directory: Path
    identity: dict
    inputs: InputBindings
    memberships: dict[Path, set[str]]
    outcomes: tuple
    elapsed_seconds: dict[str, float]
    handoff: StatisticalHandoff | None = None

    @property
    def input_hashes(self) -> dict[str, str]:
        return self.inputs.hashes()

    def guard(self) -> None:
        self.inputs.guard()
        if self.handoff is not None:
            self.handoff.guard()
            if self.input_hashes != self.handoff.hashes:
                raise ValueError("statistical producer artifact bytes differ from the authenticated inputs")
        for directory, names in self.memberships.items():
            _directory_members(directory, names)
        if capture_identity(self.root) != self.identity:
            raise ValueError("statistical source or runtime identity changed during evaluation")

    def evaluate(self):
        self.guard()
        report = validate_corpus_outcomes(build_corpus(), self.outcomes)
        self.guard()
        return report

    def publish(self, report, output: Path) -> None:
        if report != self.evaluate():
            raise ValueError("statistical report differs from verified full-corpus evidence")
        path = _resolve_output_path(output, "statistical summary")
        if path.is_relative_to(self.root.resolve()):
            raise ValueError("statistical summary must be external")
        value = {
            **self._attribution(),
            "state": "PASS",
            "corpus": asdict(report),
            "input_sha256": self.input_hashes,
            "case_elapsed_seconds": self.elapsed_seconds,
        }
        content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        _write_new_file_in_anchored_directory(
            path.parent,
            _directory_identity(path.parent, "statistical summary"),
            path.name,
            content,
            directory_label="statistical summary",
            file_label="statistical summary",
        )
        self.guard()

    def _attribution(self) -> dict:
        if self.handoff is None:
            return {"schema": "xrr-r23-statistical-evidence-v1", "identity": self.identity}
        return {
            "schema": "xrr-r23-statistical-evidence-v2",
            "execution": "revalidated-existing-outcomes",
            "new_fit_count": 0,
            "producer_identity": self.handoff.producer,
            "consumer_identity": self.identity,
            "compatibility": self.handoff.proof,
            "producer_provenance": self.handoff.provenance,
        }


def _read_case(shard: Path, case, binding: dict, inputs: InputBindings):
    path = shard / f"case-{case.case_id}.json"
    name = f"{shard.name}/{path.name}"
    inputs.add({name: path})
    content = inputs.contents[name]
    if hashlib.sha256(content).hexdigest() != binding["sha256"]:
        raise ValueError("statistical case byte hash differs")
    return decode_record(case, json_object(content))


def _load_shard(shard: Path, cases, identity: dict, inputs: InputBindings):
    name = f"{shard.name}/result.json"
    inputs.add({name: shard / "result.json"})
    manifest, index = _shard_manifest(inputs.contents[name], identity)
    selected = shard_cases(cases, index)
    bindings = _case_bindings(manifest["cases"], selected)
    names = {"result.json", *(f"case-{case.case_id}.json" for case in selected)}
    _directory_members(shard, names)
    records = tuple(_read_case(shard, case, binding, inputs) for case, binding in zip(selected, bindings, strict=True))
    return index, names, records


def _input_shards(root: Path, directory: Path) -> tuple[Path, ...]:
    if directory.absolute().is_relative_to(root.resolve()):
        raise ValueError("statistical result inputs must be external")
    snapshot = InstalledSnapshot(directory)
    shards = tuple(sorted(directory.iterdir()))
    if len(shards) != SHARD_COUNT:
        raise ValueError("statistical results require exactly eight shard directories")
    for shard in shards:
        snapshot.anchor(shard / "result.json")
    snapshot.finish()
    return shards


def load_results(root: Path, directory: Path, *, producer_path: Path | None = None) -> StatisticalResults:
    directory = directory.absolute()
    shards = _input_shards(root, directory)
    identity = capture_identity(root)
    handoff = load_handoff(root, producer_path, identity) if producer_path is not None else None
    record_identity = handoff.producer if handoff is not None else identity
    cases = build_corpus()
    inputs = InputBindings({}, limit=2 * 1024**2)
    memberships = {directory: {shard.name for shard in shards}}
    seen: set[int] = set()
    outcomes = {}
    durations = {}
    for shard in shards:
        index, names, records = _load_shard(shard, cases, record_identity, inputs)
        if index in seen:
            raise ValueError("duplicate statistical shard index")
        seen.add(index)
        memberships[shard] = names
        for outcome, elapsed in records:
            outcomes[outcome.case_id] = outcome
            durations[outcome.case_id] = elapsed
    ordered = tuple(outcomes[case.case_id] for case in cases)
    evidence = StatisticalResults(root, directory, identity, inputs, memberships, ordered, durations, handoff)
    evidence.guard()
    return evidence
