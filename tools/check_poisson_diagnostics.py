#!/usr/bin/env python3
"""Create-once preregistration, fixed shards, and auditable Poisson acceptance.

Register only after the integrated source tree is final. Separate processes may
run distinct registered shard numbers; each fit keeps local_workers=1. Completed
seed records, including failures, are immutable. Resume verifies every cached
artifact and never refits a completed seed. An interrupted attempt is retained;
only explicit --seal-interrupted turns it into U, without a replacement draw.

No pilot outcome, partial scenario, conditional coverage increase, or bootstrap
availability is a PASS for the eight predeclared diagnostic assertions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

import poisson_diagnostic_cases as cases  # noqa: E402
import poisson_diagnostic_evidence as evidence  # noqa: E402
import poisson_diagnostic_protocol as protocol  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "xrr-poisson-diagnostic-preregistration-v1"
RECORD_SCHEMA = "xrr-poisson-diagnostic-case-v1"


def _external(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        raise ValueError("experiment evidence must be outside the repository")
    return resolved


def _write_new(path: Path, value: object) -> None:
    """Atomic publication without replace/overwrite; remove only our temp file."""
    payload = value if isinstance(value, bytes) else protocol.canonical(value)
    stream = tempfile.NamedTemporaryFile(prefix=".pending-", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _approved_review(path: Path) -> bytes:
    review = path.read_bytes()
    if hashlib.sha256(review).hexdigest() != protocol.REVIEW_SHA256:
        raise ValueError("statistical review hash does not match the fixed approved recipe")
    return review


def _check_versions() -> dict:
    versions = protocol.version_identity()
    if versions != protocol.EXPECTED_VERSIONS:
        raise ValueError(f"integrated version required before registration/execution: {versions}")
    return versions


def register(output: Path, review_path: Path, *, shard_count: int = 1) -> dict:
    root = _external(output)
    if root.exists():
        raise FileExistsError(root)
    if not isinstance(shard_count, int) or isinstance(shard_count, bool) or not 1 <= shard_count <= 4600:
        raise ValueError("shard_count must be an integer in 1..4600")
    review = _approved_review(review_path)
    versions = _check_versions()
    identity = protocol.source_identity(ROOT)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "registered_utc": datetime.now(UTC).isoformat(),
        "review_sha256": protocol.REVIEW_SHA256,
        "versions": versions,
        "source_identity": identity,
        "runtime_identity": protocol.runtime_identity(),
        "protocol": protocol.definition(),
        "shard_count": shard_count,
    }
    if identity != protocol.source_identity(ROOT):
        raise ValueError("source changed during preregistration")
    root.mkdir(parents=True)
    _write_new(root / "statistical-design-review.md", review)
    _write_new(root / "preregistration.json", manifest)
    _write_new(root / "preregistration.sha256", (protocol.digest(manifest) + "\n").encode())
    return manifest


def load_manifest(root: Path) -> dict:
    encoded = (root / "preregistration.json").read_bytes()
    expected_hash = (root / "preregistration.sha256").read_text().strip()
    if hashlib.sha256(encoded).hexdigest() != expected_hash:
        raise ValueError("preregistration hash mismatch")
    manifest = json.loads(encoded)
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["protocol"] != protocol.definition():
        raise ValueError("preregistration protocol differs from the fixed approved recipe")
    if manifest["review_sha256"] != protocol.REVIEW_SHA256:
        raise ValueError("preregistration review hash mismatch")
    _approved_review(root / "statistical-design-review.md")
    return manifest


def _check_environment(manifest: dict) -> None:
    if _check_versions() != manifest["versions"]:
        raise ValueError("registered version identity changed")
    if protocol.runtime_identity() != manifest["runtime_identity"]:
        raise ValueError("registered runtime identity changed")
    if protocol.source_identity(ROOT) != manifest["source_identity"]:
        raise ValueError("registered source identity changed")


def assigned_cases(manifest: dict, shard_index: int) -> list[tuple[str, int]]:
    count = manifest["shard_count"]
    if not isinstance(shard_index, int) or isinstance(shard_index, bool) or not 0 <= shard_index < count:
        raise ValueError("shard index is outside its preregistered range")
    groups = manifest["protocol"]["scenarios"]
    tasks = [(group, seed) for group in protocol.GROUPS for seed in groups[group]["seeds"]]
    return tasks[shard_index::count]


def _identity(manifest: dict, group: str, seed: int) -> dict:
    return {
        "schema": RECORD_SCHEMA,
        "manifest_sha256": protocol.digest(manifest),
        "source_sha256": manifest["source_identity"]["sha256"],
        "scenario": group,
        "seed": seed,
    }


def _artifacts(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact symlinks are not accepted: {path}")
        if path.is_file() and path != root / "result.json":
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return records


def _seal_record(root: Path, manifest: dict, group: str, seed: int, payload: dict) -> None:
    record = _identity(manifest, group, seed) | {"payload": payload, "artifacts": _artifacts(root)}
    _write_new(root / "result.json", record | {"record_sha256": protocol.digest(record)})


def _read_record(root: Path, manifest: dict, group: str, seed: int) -> dict:
    record = json.loads((root / "result.json").read_bytes())
    seal = record.pop("record_sha256")
    if protocol.digest(record) != seal:
        raise ValueError(f"case record hash mismatch: {root}")
    if any(record[key] != value for key, value in _identity(manifest, group, seed).items()):
        raise ValueError(f"case record identity mismatch: {root}")
    if record["artifacts"] != _artifacts(root):
        raise ValueError(f"case artifact hash mismatch: {root}")
    return record["payload"]


@contextmanager
def _shard_lock(root: Path, index: int):
    directory = root / "locks"
    directory.mkdir(exist_ok=True)
    with (directory / f"shard-{index}.lock").open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"\0")
            stream.flush()
        _platform_lock(stream, acquire=True)
        try:
            yield
        finally:
            _platform_lock(stream, acquire=False)


def _platform_lock(stream, *, acquire: bool) -> None:
    if sys.platform == "win32":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)


def _resume_case(root: Path, manifest: dict, group: str, seed: int, seal_interrupted: bool) -> bool:
    if (root / "result.json").exists():
        _read_record(root, manifest, group, seed)
        return False
    if not seal_interrupted:
        raise ValueError(f"incomplete attempt: {root}; audit it, then explicitly use --seal-interrupted (U, no refit)")
    started = json.loads((root / "started.json").read_bytes())
    if any(started[key] != value for key, value in _identity(manifest, group, seed).items()):
        raise ValueError("incomplete attempt identity mismatch")
    failure = evidence.failed_outcome(
        InterruptedError("interrupted_attempt_explicitly_sealed_without_refit"), stage="interrupted"
    )
    _seal_record(root, manifest, group, seed, failure)
    return True


def _attempt(
    root: Path, manifest: dict, group: str, seed: int, runner, *, resume: bool, seal_interrupted: bool
) -> bool:
    if root.exists():
        if not resume:
            raise FileExistsError(root)
        return _resume_case(root, manifest, group, seed, seal_interrupted)
    _check_environment(manifest)
    root.mkdir(parents=True)
    _write_new(
        root / "started.json",
        _identity(manifest, group, seed)
        | {
            "started_utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
        },
    )
    try:
        payload = runner(root, group, seed)
    except Exception as error:
        payload = evidence.failed_outcome(error) | {"traceback": traceback.format_exc()}
    _check_environment(manifest)
    _seal_record(root, manifest, group, seed, payload)
    return True


def _selected_cases(manifest: dict, shard_index: int, scenarios) -> list[tuple[str, int]]:
    selected = protocol.GROUPS if scenarios is None else tuple(scenarios)
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(protocol.GROUPS):
        raise ValueError("selected scenarios must be unique preregistered names")
    return [task for task in assigned_cases(manifest, shard_index) if task[0] in selected]


def _validate_run_options(max_cases: int | None, seal_interrupted: bool, resume: bool) -> None:
    if max_cases is not None and (not isinstance(max_cases, int) or isinstance(max_cases, bool) or max_cases < 1):
        raise ValueError("max_cases must be a positive integer")
    if seal_interrupted and not resume:
        raise ValueError("seal-interrupted requires explicit resume")


def run_shard(
    output: Path,
    shard_index: int,
    *,
    resume: bool = False,
    seal_interrupted: bool = False,
    max_cases: int | None = None,
    scenarios=None,
    case_runner=None,
) -> int:
    """Run a bounded prefix of one fixed shard; the bound never changes N."""
    root = _external(output)
    manifest = load_manifest(root)
    _check_environment(manifest)
    _validate_run_options(max_cases, seal_interrupted, resume)
    _validate_case_tree(root, manifest)
    tasks = _selected_cases(manifest, shard_index, scenarios)
    completed = 0
    runner = cases.run_case if case_runner is None else case_runner
    with _shard_lock(root, shard_index):
        for group, seed in tasks:
            changed = _attempt(
                root / "cases" / group / str(seed),
                manifest,
                group,
                seed,
                runner,
                resume=resume,
                seal_interrupted=seal_interrupted,
            )
            completed += int(changed)
            if max_cases is not None and completed >= max_cases:
                break
    return completed


def _scenario_summary(root: Path, manifest: dict, group: str) -> tuple[dict, int]:
    declaration = manifest["protocol"]["scenarios"][group]
    rows, missing = [], []
    incomplete = 0
    for seed in declaration["seeds"]:
        folder = root / "cases" / group / str(seed)
        if (folder / "result.json").exists():
            rows.append(_read_record(folder, manifest, group, seed))
        else:
            missing.append(seed)
            incomplete += int(folder.exists())
    summary = protocol.summarize(rows, required_count=len(declaration["seeds"]))
    summary.update(missing_seeds=missing, endpoints=protocol.acceptance(summary, declaration["kind"]))
    if declaration["kind"] == "null":
        summary["rejection_interval"] = protocol.rejection_interval(summary)
    return summary, incomplete


def _acceptance_status(reports: dict, complete: bool) -> str:
    if not complete:
        return "UNVERIFIED"
    endpoints = [endpoint for name in protocol.GROUPS[:-1] for endpoint in reports[name]["endpoints"]]
    return "PASS" if all(endpoint["passed"] for endpoint in endpoints) else "FAIL"


def _validate_case_tree(root: Path, manifest: dict) -> None:
    case_root = root / "cases"
    if case_root.is_symlink():
        raise ValueError("case artifact directory is a symlink")
    declarations = manifest["protocol"]["scenarios"]
    for group in case_root.glob("*"):
        if group.is_symlink() or not group.is_dir() or group.name not in declarations:
            raise ValueError(f"undeclared case group: {group}")
        _validate_seed_directories(group, declarations[group.name]["seeds"])


def _validate_seed_directories(group: Path, declared_seeds: list[int]) -> None:
    seeds = {str(seed) for seed in declared_seeds}
    for folder in group.iterdir():
        if folder.is_symlink() or not folder.is_dir() or folder.name not in seeds:
            raise ValueError(f"undeclared case seed: {folder}")


def summarize_experiment(output: Path) -> dict:
    root = _external(output)
    manifest = load_manifest(root)
    _check_environment(manifest)
    _validate_case_tree(root, manifest)
    reports, incomplete = {}, 0
    for group in protocol.GROUPS:
        reports[group], pending = _scenario_summary(root, manifest, group)
        incomplete += pending
    holdout_complete = all(reports[name]["protocol_complete"] for name in protocol.GROUPS[:-1])
    replay_complete = reports["replay_low"]["protocol_complete"]
    return {
        "schema": "xrr-poisson-diagnostic-summary-v1",
        "manifest_sha256": protocol.digest(manifest),
        "source_identity": manifest["source_identity"],
        "versions": manifest["versions"],
        "scenarios": reports,
        "incomplete_attempts": incomplete,
        "holdout_complete": holdout_complete,
        "replay_complete": replay_complete,
        "protocol_complete": holdout_complete and replay_complete,
        "acceptance_status": _acceptance_status(reports, holdout_complete),
        "interpretation": {
            "denominator": "all fixed seeds; partial rates are descriptive lower yields, not inferential endpoints",
            "conditional_coverage": "covered / available formal intervals; data-dependent selection remains",
            "all_seed_coverage": (
                "covered / all fixed seeds; unavailable and unattempted intervals are not silently deleted"
            ),
            "scope": (
                "unknown-parameter plug-in diagnostic calibration, not finite-sample exactness or general 95% coverage"
            ),
            "joint": (
                "profile and actual saved bootstrap are separate; no fabricated joint profile or bootstrap coverage claim"
            ),
            "MC": "0.001 is p-value resolution, not an independent-binomial error bar for the symmetric MC matrix",
        },
    }


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register_command = commands.add_parser(
        "register", help="freeze the final integrated code and complete fixed recipe"
    )
    register_command.add_argument("--output", type=Path, required=True)
    register_command.add_argument("--review", type=Path, required=True)
    register_command.add_argument("--shards", type=_positive_integer, default=1)
    run = commands.add_parser("run", help="run or resume one fixed shard; no ad hoc seed selection")
    run.add_argument("--experiment", type=Path, required=True)
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--scenario", choices=protocol.GROUPS, action="append")
    run.add_argument("--max-cases", type=_positive_integer)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--seal-interrupted", action="store_true")
    summary = commands.add_parser("summarize", help="write a new immutable summary; incomplete is not PASS")
    summary.add_argument("--experiment", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "register":
        manifest = register(args.output, args.review, shard_count=args.shards)
        print(f"registered {args.output}: {protocol.digest(manifest)}", file=sys.stderr)
        return 0
    if args.command == "run":
        completed = run_shard(
            args.experiment,
            args.shard,
            resume=args.resume,
            seal_interrupted=args.seal_interrupted,
            max_cases=args.max_cases,
            scenarios=args.scenario,
        )
        print(f"shard {args.shard}: {completed} new immutable records (not an acceptance result)", file=sys.stderr)
        return 0
    report = summarize_experiment(args.experiment)
    _write_new(_external(args.output), report)
    if not report["protocol_complete"]:
        return 2
    return 0 if report["acceptance_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
