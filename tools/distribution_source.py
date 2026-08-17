"""Clean-HEAD source selection and reproducible staging builds."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from distribution_archive import canonicalize_sdist
from distribution_manifest import git_oid, select_artifacts

BUILD_VERSIONS = {"setuptools": "75.8.2", "wheel": "0.45.1"}


@dataclass(frozen=True, slots=True)
class GitIdentity:
    head_commit: str
    head_tree: str
    source_date_epoch: int


@dataclass(frozen=True, slots=True)
class _GitTreeEntry:
    mode: str
    kind: str
    oid: str
    path: PurePosixPath


@dataclass(frozen=True, slots=True)
class _GitBlob:
    mode: str
    oid: str


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_bytes(repository: Path, *args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=repository,
        input=stdin,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _captured_commit(repository: Path, commit: str) -> str:
    value = git_oid(commit, "captured commit")
    try:
        _git_bytes(repository, "cat-file", "-e", f"{value}^{{commit}}")
        kind = _git(repository, "cat-file", "-t", value)
    except subprocess.CalledProcessError as error:
        raise ValueError("captured commit must exist as a Git commit object") from error
    if kind != "commit":
        raise ValueError("captured commit must be a Git commit object")
    return value


def clean_head_identity(repo_root: str | Path) -> GitIdentity:
    repository = Path(repo_root).resolve()
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("distribution verification requires a clean Git HEAD")
    commit = _captured_commit(repository, _git(repository, "rev-parse", "HEAD"))
    tree = git_oid(_git(repository, "rev-parse", f"{commit}^{{tree}}"), "head tree")
    timestamp = _git(repository, "show", "-s", "--format=%ct", commit)
    if not timestamp.isdecimal() or int(timestamp) <= 0:
        raise ValueError("HEAD commit timestamp must be a positive integer")
    return GitIdentity(commit, tree, int(timestamp))


def validate_bundle_paths(
    repo_root: str | Path,
    report_dir: str | Path,
    artifact_dir: str | Path,
) -> tuple[Path, Path]:
    repository = Path(repo_root).resolve()
    report_input = Path(report_dir).absolute()
    artifact_input = Path(artifact_dir).absolute()
    if report_input.is_symlink() or artifact_input.is_symlink():
        raise ValueError("distribution output paths must not be symlinks")
    report = report_input.resolve()
    artifacts = artifact_input.resolve()
    if artifacts != report / "artifacts":
        raise ValueError("artifact directory must equal report-dir/artifacts")
    if report == repository or report.is_relative_to(repository):
        raise ValueError("distribution report must be outside the repository")
    return report, artifacts


def _git_path(path: PurePosixPath) -> PurePosixPath:
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"distribution input path must be a relative Git path: {path}")
    return path


def _parse_tree_entry(raw: bytes) -> _GitTreeEntry:
    metadata, separator, path = raw.partition(b"\t")
    if not separator:
        raise ValueError("Git tree entry is missing a path")
    fields = metadata.decode("ascii").split(" ")
    if len(fields) != 3:
        raise ValueError("Git tree entry metadata drift")
    try:
        relative = PurePosixPath(path.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("tracked path is not UTF-8") from error
    mode, kind, oid = fields
    return _GitTreeEntry(mode, kind, git_oid(oid, "Git object"), _git_path(relative))


def _tree_entries(
    repository: Path,
    commit: str,
    paths: tuple[PurePosixPath, ...] = (),
) -> tuple[_GitTreeEntry, ...]:
    commit = _captured_commit(repository, commit)
    args = ["ls-tree", "-r", "-z", "--full-tree", commit]
    if paths:
        args.extend(("--", *(f":(literal){_git_path(path).as_posix()}" for path in paths)))
    return tuple(_parse_tree_entry(raw) for raw in _git_bytes(repository, *args).split(b"\0") if raw)


def _regular_blob(entry: _GitTreeEntry) -> _GitBlob:
    if entry.kind != "blob" or entry.mode not in {"100644", "100755"}:
        raise ValueError(f"distribution input must be a regular Git blob: {entry.path}")
    return _GitBlob(entry.mode, entry.oid)


def _regular_blob_oid(entry: _GitTreeEntry) -> str:
    return _regular_blob(entry).oid


def _regular_blobs(
    repository: Path,
    commit: str,
    paths: tuple[PurePosixPath, ...],
) -> dict[PurePosixPath, _GitBlob]:
    requested = {_git_path(path) for path in paths}
    if len(requested) != len(paths):
        raise ValueError("duplicate distribution input")
    blobs: dict[PurePosixPath, _GitBlob] = {}
    for entry in _tree_entries(repository, commit, paths):
        if entry.path not in requested:
            raise ValueError(f"distribution input must be an exact Git file: {entry.path}")
        if entry.path in blobs:
            raise ValueError(f"duplicate distribution input: {entry.path}")
        blobs[entry.path] = _regular_blob(entry)
    if set(blobs) != requested:
        missing = sorted(requested - set(blobs), key=PurePosixPath.as_posix)
        raise ValueError(f"distribution input missing from captured commit: {missing}")
    return blobs


def _regular_blob_oids(
    repository: Path,
    commit: str,
    paths: tuple[PurePosixPath, ...],
) -> dict[PurePosixPath, str]:
    return {path: blob.oid for path, blob in _regular_blobs(repository, commit, paths).items()}


def _batch_header(
    output: bytes,
    offset: int,
    path: PurePosixPath,
    expected_oid: str,
) -> tuple[int, int]:
    line_end = output.find(b"\n", offset)
    if line_end < 0:
        raise ValueError("Git blob batch header is missing")
    header = output[offset:line_end].decode("ascii").split(" ")
    if len(header) != 3:
        raise ValueError("Git blob batch header drift")
    observed_oid, kind, size_text = header
    if observed_oid != expected_oid or kind != "blob" or not size_text.isdecimal():
        raise ValueError(f"Git blob batch object drift: {path}")
    return line_end + 1, int(size_text)


def _batch_content(
    output: bytes,
    offset: int,
    size: int,
    path: PurePosixPath,
) -> tuple[int, bytes]:
    content = output[offset : offset + size]
    if len(content) != size:
        raise ValueError(f"Git blob batch content truncated: {path}")
    offset += size
    if output[offset : offset + 1] != b"\n":
        raise ValueError(f"Git blob batch delimiter drift: {path}")
    return offset + 1, content


def _cat_file_blobs(repository: Path, blob_oids: dict[PurePosixPath, str]) -> dict[PurePosixPath, bytes]:
    payload = "".join(f"{oid}\n" for oid in blob_oids.values()).encode("ascii")
    output = _git_bytes(repository, "cat-file", "--batch", stdin=payload)
    result: dict[PurePosixPath, bytes] = {}
    offset = 0
    for path, expected_oid in blob_oids.items():
        offset, size = _batch_header(output, offset, path, expected_oid)
        offset, content = _batch_content(output, offset, size, path)
        result[path] = content
    if offset != len(output):
        raise ValueError("Git blob batch output had trailing data")
    return result


def _committed_blob(repository: Path, commit: str, path: PurePosixPath) -> bytes:
    return _cat_file_blobs(
        repository,
        _regular_blob_oids(repository, commit, (_git_path(path),)),
    )[path]


def committed_blob(
    repository: Path,
    commit: str,
    path: str | PurePosixPath,
) -> bytes:
    return _committed_blob(repository, commit, _git_path(PurePosixPath(path)))


def committed_tree_files(
    repository: Path,
    commit: str,
    root: str | PurePosixPath,
) -> dict[PurePosixPath, bytes]:
    prefix = _git_path(PurePosixPath(root))
    entries = _tree_entries(repository, commit, (prefix,))
    blob_oids = {entry.path: _regular_blob_oid(entry) for entry in entries}
    return _cat_file_blobs(repository, blob_oids)


def release_spec(repository: Path, commit: str) -> dict[str, object]:
    content = _committed_blob(
        repository,
        _captured_commit(repository, commit),
        PurePosixPath("verification/release-spec.json"),
    )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("release spec JSON must be UTF-8") from error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid release spec JSON") from error
    if not isinstance(value, dict):
        raise ValueError("release spec must be a JSON object")
    return value


def _tracked_paths(repository: Path, commit: str) -> tuple[PurePosixPath, ...]:
    return tuple(entry.path for entry in _tree_entries(repository, commit))


def _input_policy(policy_value: object) -> tuple[set[object], set[object]]:
    if not isinstance(policy_value, dict):
        raise ValueError("sdist content policy must be a mapping")
    return (
        set(policy_value.get("input_directories", ())),
        set(policy_value.get("input_files", ())),
    )


def _validate_input_coverage(
    selected: tuple[PurePosixPath, ...],
    directories: set[object],
    files: set[object],
) -> None:
    root_files = {path.as_posix() for path in selected if len(path.parts) == 1}
    selected_directories = {path.parts[0] for path in selected if len(path.parts) > 1}
    if root_files != files:
        raise ValueError("sdist root input allowlist drift")
    if selected_directories != directories:
        raise ValueError("sdist directory input allowlist drift")


def distribution_inputs(
    repository: Path,
    policy_value: object,
    commit: str,
) -> tuple[PurePosixPath, ...]:
    commit = _captured_commit(repository, commit)
    directories, files = _input_policy(policy_value)
    selected_entries = tuple(
        entry
        for entry in _tree_entries(repository, commit)
        if entry.path.as_posix() in files or entry.path.parts[0] in directories
    )
    for entry in selected_entries:
        _regular_blob_oid(entry)
    selected = tuple(entry.path for entry in selected_entries)
    _validate_input_coverage(selected, directories, files)
    return selected


def _copy_inputs(
    repository: Path,
    staging: Path,
    paths: tuple[PurePosixPath, ...],
    epoch: int,
    commit: str,
) -> None:
    """Copy immutable Git blobs rather than reading the mutable worktree.

    The clean-HEAD check and the build are separate operations. Reading from
    ``repository / relative`` after the check therefore permits a concurrent
    worktree edit to be packaged under the old identity. Raw blob reads from
    the captured commit keep staged bytes bound to that immutable identity
    without archive attribute filters such as ``export-subst``.
    """
    commit = _captured_commit(repository, commit)
    if not paths:
        raise ValueError("distribution inputs must not be empty")
    ordered_paths = tuple(_git_path(path) for path in paths)
    blobs = _regular_blobs(repository, commit, ordered_paths)
    payloads = _cat_file_blobs(
        repository,
        {path: blob.oid for path, blob in blobs.items()},
    )
    for relative in ordered_paths:
        target = staging.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[relative])
        target.chmod(0o755 if blobs[relative].mode == "100755" else 0o644)
        os.utime(target, (epoch, epoch))


def build_environment(root: Path, epoch: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SOURCE_DATE_EPOCH"] = str(epoch)
    environment["HOME"] = str(root / "home")
    environment["XDG_CACHE_HOME"] = str(root / "xdg-cache")
    environment["MPLCONFIGDIR"] = str(root / "mpl-cache")
    for name in ("HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        Path(environment[name]).mkdir(parents=True, exist_ok=True)
    return environment


def check_build_versions() -> None:
    observed = {name: importlib.metadata.version(name) for name in BUILD_VERSIONS}
    if observed != BUILD_VERSIONS:
        raise ValueError(f"pinned build tool drift: {observed!r}")


def build_once(
    repository: Path,
    destination: Path,
    inputs: tuple[PurePosixPath, ...],
    epoch: int,
    commit: str,
) -> Path:
    commit = _captured_commit(repository, commit)
    staging = destination / "staging"
    artifacts = destination / "artifacts"
    staging.mkdir(parents=True)
    artifacts.mkdir()
    _copy_inputs(repository, staging, inputs, epoch, commit)
    source = staging.resolve()
    output = artifacts.resolve()
    subprocess.run(
        (
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
            str(source),
        ),
        cwd=source,
        env=build_environment(destination, epoch),
        check=True,
        capture_output=True,
        text=True,
    )
    selected = select_artifacts(artifacts)
    canonicalize_sdist(selected["sdist"], epoch)
    return artifacts
