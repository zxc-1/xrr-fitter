"""Clean-HEAD source selection and reproducible staging builds."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

from distribution_archive import canonicalize_sdist
from distribution_manifest import git_oid, select_artifacts


BUILD_VERSIONS = {"setuptools": "75.8.2", "wheel": "0.45.1"}


@dataclass(frozen=True, slots=True)
class GitIdentity:
    head_commit: str
    head_tree: str
    source_date_epoch: int


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def clean_head_identity(repo_root: str | Path) -> GitIdentity:
    repository = Path(repo_root).resolve()
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("distribution verification requires a clean Git HEAD")
    commit = git_oid(_git(repository, "rev-parse", "HEAD"), "head commit")
    tree = git_oid(_git(repository, "rev-parse", "HEAD^{tree}"), "head tree")
    timestamp = _git(repository, "show", "-s", "--format=%ct", "HEAD")
    if not timestamp.isdecimal() or int(timestamp) <= 0:
        raise ValueError("HEAD commit timestamp must be a positive integer")
    return GitIdentity(commit, tree, int(timestamp))


def validate_bundle_paths(
    repo_root: str | Path,
    report_dir: str | Path,
    artifact_dir: str | Path,
) -> tuple[Path, Path]:
    repository = Path(repo_root).resolve()
    report = Path(report_dir).resolve()
    artifacts = Path(artifact_dir).resolve()
    if artifacts != report / "artifacts":
        raise ValueError("artifact directory must equal report-dir/artifacts")
    if report == repository or report.is_relative_to(repository):
        raise ValueError("distribution report must be outside the repository")
    return report, artifacts


def release_spec(repository: Path) -> dict[str, object]:
    path = repository / "verification" / "release-spec.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("release spec must be a JSON object")
    return value


def _tracked_paths(repository: Path) -> tuple[PurePosixPath, ...]:
    result = subprocess.run(
        ("git", "ls-tree", "-r", "-z", "--name-only", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    try:
        return tuple(
            PurePosixPath(item.decode("utf-8"))
            for item in result.stdout.split(b"\0")
            if item
        )
    except UnicodeDecodeError as error:
        raise ValueError("tracked path is not UTF-8") from error


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
    selected_directories = {
        path.parts[0] for path in selected if len(path.parts) > 1
    }
    if root_files != files:
        raise ValueError("sdist root input allowlist drift")
    if selected_directories != directories:
        raise ValueError("sdist directory input allowlist drift")


def distribution_inputs(
    repository: Path,
    policy_value: object,
) -> tuple[PurePosixPath, ...]:
    directories, files = _input_policy(policy_value)
    selected = tuple(
        path
        for path in _tracked_paths(repository)
        if path.as_posix() in files or path.parts[0] in directories
    )
    _validate_input_coverage(selected, directories, files)
    return selected


def _copy_inputs(
    repository: Path,
    staging: Path,
    paths: tuple[PurePosixPath, ...],
    epoch: int,
) -> None:
    for relative in paths:
        source = repository.joinpath(*relative.parts)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"distribution input must be a regular file: {relative}")
        target = staging.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
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
    observed = {
        name: importlib.metadata.version(name) for name in BUILD_VERSIONS
    }
    if observed != BUILD_VERSIONS:
        raise ValueError(f"pinned build tool drift: {observed!r}")


def build_once(
    repository: Path,
    destination: Path,
    inputs: tuple[PurePosixPath, ...],
    epoch: int,
) -> Path:
    staging = destination / "staging"
    artifacts = destination / "artifacts"
    staging.mkdir(parents=True)
    artifacts.mkdir()
    _copy_inputs(repository, staging, inputs, epoch)
    subprocess.run(
        (
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(artifacts),
            str(staging),
        ),
        cwd=repository,
        env=build_environment(destination, epoch),
        check=True,
        capture_output=True,
        text=True,
    )
    selected = select_artifacts(artifacts)
    canonicalize_sdist(selected["sdist"], epoch)
    return artifacts
