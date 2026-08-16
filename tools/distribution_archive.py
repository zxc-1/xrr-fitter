"""Deterministic archive normalization and wheel/sdist content policy."""

from __future__ import annotations

import gzip
import io
import os
import stat
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.markers import Marker
from packaging.requirements import Requirement

from distribution_manifest import ARTIFACT_KINDS, select_artifacts
from version_source import declared_project_version as _declared_project_version


def _declared_requirements(project: Mapping[str, object]) -> tuple[Requirement, ...]:
    expected = [Requirement(value) for value in project.get("dependencies", ())]
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ValueError("project optional-dependencies must be a mapping")
    for extra, values in optional.items():
        extra_marker = Marker(f'extra == "{extra}"')
        for value in values:
            requirement = Requirement(value)
            requirement.marker = (
                extra_marker if requirement.marker is None else Marker(f"({requirement.marker}) and ({extra_marker})")
            )
            expected.append(requirement)
    return tuple(expected)


def verify_wheel_dependencies(wheel: str | Path, pyproject: str | Path) -> None:
    project = tomllib.loads(Path(pyproject).read_text(encoding="utf-8"))["project"]
    expected = _declared_requirements(project)
    with zipfile.ZipFile(wheel) as archive:
        names = tuple(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        if len(names) != 1:
            raise ValueError("wheel must contain exactly one METADATA file")
        message = BytesParser(policy=policy.default).parsebytes(archive.read(names[0]))
    observed = tuple(Requirement(value) for value in message.get_all("Requires-Dist", ()))
    if observed != expected:
        raise ValueError("wheel Requires-Dist does not exactly match pyproject.toml")


def verify_reproducible_artifacts(first: str | Path, second: str | Path) -> None:
    first_selected = select_artifacts(Path(first))
    second_selected = select_artifacts(Path(second))
    for kind in ARTIFACT_KINDS:
        left = first_selected[kind]
        right = second_selected[kind]
        if left.name != right.name or left.read_bytes() != right.read_bytes():
            raise ValueError(f"{kind} build is not byte-for-byte reproducible")


def _canonical_sdist_entries(
    path: Path,
    epoch: int,
) -> tuple[tuple[tarfile.TarInfo, bytes | None], ...]:
    entries = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not (member.isfile() or member.isdir()):
                raise ValueError("sdist contains a non-file member")
            payload = archive.extractfile(member).read() if member.isfile() else None
            canonical = member.replace(
                mtime=epoch,
                uid=0,
                gid=0,
                uname="",
                gname="",
            )
            canonical.pax_headers = {}
            entries.append((canonical, payload))
    return tuple(sorted(entries, key=lambda item: item[0].name))


def _canonical_sdist_bytes(path: Path, epoch: int) -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for member, payload in _canonical_sdist_entries(path, epoch):
            archive.addfile(member, io.BytesIO(payload) if payload is not None else None)
    gzip_buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=gzip_buffer,
        mtime=epoch,
    ) as archive:
        archive.write(tar_buffer.getvalue())
    return gzip_buffer.getvalue()


def canonicalize_sdist(path: str | Path, epoch: int) -> None:
    target = Path(path)
    content = _canonical_sdist_bytes(target, epoch)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _project_identity(repository: Path) -> tuple[str, str]:
    payload = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    normalized = project["name"].replace("-", "_").replace(".", "_")
    version = _declared_project_version(repository, payload)
    return normalized, version


def _wheel_metadata(repository: Path) -> set[str]:
    name, version = _project_identity(repository)
    root = f"{name}-{version}.dist-info"
    result = {f"{root}/{item}" for item in ("METADATA", "RECORD", "WHEEL", "top_level.txt")}
    project = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project.get("gui-scripts") or project.get("scripts"):
        result.add(f"{root}/entry_points.txt")
    if project.get("license"):
        result.add(f"{root}/LICENSE")
    return result


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        members = tuple(archive.infolist())
    if any(member.is_dir() for member in members):
        raise ValueError("wheel contains a directory member")
    if any(stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK for member in members):
        raise ValueError("wheel contains a symlink")
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise ValueError("wheel contains duplicate members")
    return set(names)


def _verify_wheel_members(
    repository: Path,
    wheel: Path,
    inputs: tuple[PurePosixPath, ...],
    spec: dict[str, object],
) -> None:
    expected_package = {
        path.as_posix().removeprefix("src/") for path in inputs if path.parts[:2] == ("src", "xrr_fitter")
    }
    observed = _wheel_members(wheel)
    if observed != expected_package | _wheel_metadata(repository):
        raise ValueError("wheel member allowlist drift")
    policy_value = spec.get("wheel_content_policy")
    if not isinstance(policy_value, dict):
        raise ValueError("wheel content policy must be a mapping")
    forbidden = set(policy_value.get("forbidden_roots", ()))
    if {PurePosixPath(path).parts[0] for path in observed} & forbidden:
        raise ValueError("wheel contains a forbidden root")


def _sdist_members(path: Path) -> tuple[str, set[str]]:
    with tarfile.open(path, "r:gz") as archive:
        members = tuple(archive.getmembers())
    if any(not (member.isfile() or member.isdir()) for member in members):
        raise ValueError("sdist contains a non-file member")
    files = tuple(PurePosixPath(member.name) for member in members if member.isfile())
    if len(files) != len(set(files)):
        raise ValueError("sdist contains duplicate members")
    roots = {path.parts[0] for path in files}
    if len(roots) != 1:
        raise ValueError("sdist must have one canonical root")
    root = roots.pop()
    return root, {PurePosixPath(*path.parts[1:]).as_posix() for path in files}


def _verify_sdist_members(
    repository: Path,
    sdist: Path,
    inputs: tuple[PurePosixPath, ...],
    spec: dict[str, object],
) -> None:
    root, observed = _sdist_members(sdist)
    name, version = _project_identity(repository)
    if root != f"{name}-{version}":
        raise ValueError("sdist root drift")
    policy_value = spec.get("sdist_content_policy")
    if not isinstance(policy_value, dict):
        raise ValueError("sdist content policy must be a mapping")
    generated = set(policy_value.get("generated_metadata", ()))
    expected = {path.as_posix() for path in inputs} | generated
    if observed != expected:
        raise ValueError("sdist member allowlist drift")


def verify_archives(
    repository: Path,
    artifact_dir: Path,
    inputs: tuple[PurePosixPath, ...],
    spec: dict[str, object],
) -> None:
    selected = select_artifacts(artifact_dir)
    _verify_wheel_members(repository, selected["wheel"], inputs, spec)
    _verify_sdist_members(repository, selected["sdist"], inputs, spec)
    verify_wheel_dependencies(selected["wheel"], repository / "pyproject.toml")
