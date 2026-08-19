#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as platform_module
import re
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "xrr-test-manifest-v1"
SUITES = ("tests", "tests_r21")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
PRUNED_PARTS = {".pytest_cache", "__pycache__"}


def canonical_json_bytes(value: object) -> bytes:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _test_tree(repo_root: Path, suite: str) -> list[dict[str, object]]:
    suite_root = repo_root / suite
    if suite_root.is_symlink() or not suite_root.is_dir():
        raise ValueError(f"suite directory is missing: {suite}")
    records: list[dict[str, object]] = []
    for path in suite_root.rglob("*"):
        relative = path.relative_to(repo_root)
        if any(part in PRUNED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"test tree contains a symlink: {relative.as_posix()}")
        if not path.is_file():
            continue
        content = path.read_bytes()
        records.append(
            {
                "path": relative.as_posix(),
                "size": len(content),
                "sha256": _sha256(content),
            }
        )
    records.sort(key=lambda item: str(item["path"]))
    return records


def _normalize_records(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    observed: set[str] = set()
    for record in records:
        nodeid = record.get("nodeid")
        markers = record.get("markers")
        if not isinstance(nodeid, str) or not nodeid or nodeid in observed:
            raise ValueError("duplicate or invalid test nodeid")
        if not isinstance(markers, (list, tuple)) or not all(isinstance(marker, str) and marker for marker in markers):
            raise ValueError(f"invalid marker metadata for {nodeid}")
        marker_list = sorted(set(markers))
        if len(marker_list) != len(markers):
            raise ValueError(f"duplicate marker metadata for {nodeid}")
        observed.add(nodeid)
        normalized.append({"nodeid": nodeid, "markers": marker_list})
    normalized.sort(key=lambda item: str(item["nodeid"]))
    return normalized


def build_manifest(
    *,
    repo_root: str | Path,
    source_commit: str,
    suite: str,
    lock_file: str | Path,
    records: Iterable[dict[str, object]],
    python_version: str,
    platform: str,
) -> dict[str, object]:
    root = Path(repo_root).resolve()
    if suite not in SUITES:
        raise ValueError(f"unknown suite: {suite}")
    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a lowercase 40-character commit")
    lock = _regular_file(Path(lock_file), "lock file")
    nodes = _normalize_records(records)
    base: dict[str, object] = {
        "schema": SCHEMA,
        "source_commit": source_commit,
        "suite": suite,
        "test_tree": _test_tree(root, suite),
        "node_count": len(nodes),
        "nodes": nodes,
        "python_version": python_version,
        "platform": platform,
        "lock_sha256": _sha256(lock.read_bytes()),
    }
    return {**base, "collection_sha256": _sha256(canonical_json_bytes(base))}


def _atomic_write(path: Path, content: bytes) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent must be an existing regular directory")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("output must be a regular file path")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(path: str | Path, manifest: dict[str, object]) -> None:
    _atomic_write(Path(path), canonical_json_bytes(manifest))


class CollectionRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def pytest_collection_finish(self, session: object) -> None:
        items = session.items
        self.records = [
            {
                "nodeid": str(item.nodeid),
                "markers": sorted({marker.name for marker in item.iter_markers()}),
            }
            for item in items
        ]


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "git command failed"
        raise ValueError(detail)
    return result.stdout.strip()


def _assert_r22_source(root: Path, expected_tag: str) -> str:
    head = _git(root, "rev-parse", "HEAD")
    tagged = _git(root, "rev-parse", f"{expected_tag}^{{commit}}")
    if head != tagged:
        raise ValueError("R22 HEAD does not match expected tag")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("R22 source worktree is not clean")
    return head


def _assert_r23_source(root: Path, source_commit: str, suite: str) -> str:
    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ValueError("source commit must be a lowercase 40-character commit")
    _git(root, "cat-file", "-e", f"{source_commit}^{{commit}}")
    head = _git(root, "rev-parse", "HEAD")
    try:
        _git(root, "merge-base", "--is-ancestor", source_commit, head)
    except ValueError as error:
        raise ValueError("source commit must be an ancestor of HEAD") from error
    changed = _git(root, "diff", "--name-only", source_commit, head, "--", suite)
    if changed:
        raise ValueError("test tree differs from source commit")
    return source_commit


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _is_interpreter_path(path: Path, roots: tuple[Path, ...]) -> bool:
    if any(path == root or root in path.parents for root in roots):
        return True
    return path.suffix == ".zip" and any(path.parent == root.parent for root in roots)


def _trusted_sys_path_entries(entries: Iterable[str]) -> list[str]:
    roots = tuple(
        Path(value).resolve()
        for key in ("stdlib", "platstdlib", "purelib", "platlib")
        if (value := sysconfig.get_path(key))
    )
    trusted: list[str] = []
    for entry in entries:
        if not entry:
            continue
        path = Path(entry).resolve()
        if _is_interpreter_path(path, roots) and str(path) not in trusted:
            trusted.append(str(path))
    return trusted


@contextmanager
def _isolated_import_state(python_path: Path, repo_root: Path) -> Iterator[None]:
    import_root = str(python_path.resolve())
    repository = str(repo_root.resolve())
    previous_pythonpath = os.environ.get("PYTHONPATH")
    previous_pytest_environment = {name: value for name, value in os.environ.items() if name.startswith("PYTEST_")}
    previous_sys_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    for name in previous_pytest_environment:
        os.environ.pop(name)
    os.environ["PYTHONPATH"] = import_root
    sys.path[:] = [import_root, repository, *_trusted_sys_path_entries(previous_sys_path)]
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        for name in tuple(os.environ):
            if name.startswith("PYTEST_"):
                os.environ.pop(name)
        os.environ.update(previous_pytest_environment)
        sys.path[:] = previous_sys_path
        sys.dont_write_bytecode = previous_bytecode
        if previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous_pythonpath


def _platform_identity() -> str:
    system = "macOS" if sys.platform == "darwin" else platform_module.system()
    return f"{system}-{platform_module.machine()}"


def collect_records(repo_root: Path, suite: str, python_path: Path) -> tuple[dict[str, object], ...]:
    recorder = CollectionRecorder()
    arguments = [
        "-o",
        "addopts=",
        "--import-mode=importlib",
        "--strict-config",
        "--strict-markers",
        "-p",
        "no:cacheprovider",
        "--collect-only",
        suite,
        "-q",
    ]
    with _isolated_import_state(python_path, repo_root), _working_directory(repo_root):
        import pytest

        exit_code = pytest.main(arguments, plugins=[recorder])
    if exit_code != pytest.ExitCode.OK:
        raise ValueError(f"pytest collection failed with exit code {int(exit_code)}")
    return tuple(recorder.records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--expected-tag")
    source.add_argument("--source-commit")
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output.absolute()
    if args.expected_tag:
        commit = _assert_r22_source(root, args.expected_tag)
        python_path = root
    else:
        commit = _assert_r23_source(root, args.source_commit, args.suite)
        python_path = root / "src"
    records = collect_records(root, args.suite, python_path)
    if args.expected_tag:
        _assert_r22_source(root, args.expected_tag)
    manifest = build_manifest(
        repo_root=root,
        source_commit=commit,
        suite=args.suite,
        lock_file=args.lock_file,
        records=records,
        python_version=platform_module.python_version(),
        platform=_platform_identity(),
    )
    write_manifest(output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
