#!/usr/bin/env python3
"""Build the canonical release policy from audited package and lock inputs.

The output records stable policy rather than a snapshot of the evolving source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from typing import Iterable, Mapping, Sequence

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


SCHEMA = "xrr-r23-release-spec-v1"
EXPECTED_BUILD = ("setuptools==75.8.2", "wheel==0.45.1")
INPUT_DIRECTORIES = ("docs", "examples", "src", "tests", "tools", "verification")
INPUT_FILES = (
    "MANIFEST.in",
    "README.md",
    "pyproject.toml",
    "requirements-macos-arm64-py312.lock",
)
GENERATED_METADATA = (
    "PKG-INFO",
    "setup.cfg",
    "src/xrr_fitter.egg-info/PKG-INFO",
    "src/xrr_fitter.egg-info/SOURCES.txt",
    "src/xrr_fitter.egg-info/dependency_links.txt",
    "src/xrr_fitter.egg-info/requires.txt",
    "src/xrr_fitter.egg-info/top_level.txt",
)
ENTRY_POINT_METADATA = "src/xrr_fitter.egg-info/entry_points.txt"
VCS_REVISION = re.compile(r"[0-9a-f]{40}")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve()


def _pyproject(path: Path) -> dict[str, object]:
    source = _regular(path, "pyproject")
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("unable to read pyproject as UTF-8") from error
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError("invalid pyproject TOML") from error
    try:
        build = payload["build-system"]
        project = payload["project"]
        tests = project["optional-dependencies"]["test"]
        runtime = project["dependencies"]
    except (KeyError, TypeError) as error:
        raise ValueError("pyproject is missing dependency metadata") from error
    if not isinstance(build, dict) or tuple(build.get("requires", ())) != EXPECTED_BUILD:
        raise ValueError("build-system requirements differ from the pinned policy")
    if build.get("build-backend") != "setuptools.build_meta":
        raise ValueError("unexpected build backend")
    if not isinstance(runtime, list) or not isinstance(tests, list):
        raise ValueError("dependency groups must be arrays")
    _parse_requirements((*runtime, *tests))
    return payload


def _parse_requirements(values: Iterable[object]) -> tuple[Requirement, ...]:
    parsed: list[Requirement] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("dependency values must be strings")
        try:
            parsed.append(Requirement(value))
        except InvalidRequirement as error:
            raise ValueError(f"invalid dependency requirement: {value}") from error
    return tuple(parsed)


def _lock_bytes(path: Path) -> bytes:
    source = _regular(path, "dependency lock")
    try:
        return source.read_bytes()
    except OSError as error:
        raise ValueError("unable to read dependency lock") from error


def _lock_text(content: bytes) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("dependency lock is not UTF-8") from error
    if not text or "\r" in text or not text.endswith("\n"):
        raise ValueError("dependency lock is not canonical text")
    return text


def _lock_requirement(line: str) -> Requirement:
    try:
        requirement = Requirement(line)
    except InvalidRequirement as error:
        raise ValueError(f"invalid dependency lock line: {line}") from error
    if str(requirement) != line:
        raise ValueError("dependency lock is not canonical text")
    return requirement


def _validate_pin(requirement: Requirement) -> None:
    if requirement.extras or requirement.marker is not None:
        raise ValueError("dependency lock requirements cannot contain extras or markers")
    if requirement.url is not None:
        if not requirement.url.startswith("git+https://"):
            raise ValueError("dependency lock contains a non-portable direct URL")
        revision = requirement.url.rsplit("@", 1)[-1]
        if VCS_REVISION.fullmatch(revision) is None:
            raise ValueError("dependency lock VCS URL requires a full lowercase commit")
        return
    specs = list(requirement.specifier)
    if len(specs) != 1 or specs[0].operator != "==" or "*" in specs[0].version:
        raise ValueError("dependency lock contains a non-exact requirement")


def _lock(path: Path, declared: Sequence[Requirement]) -> tuple[bytes, tuple[str, ...]]:
    content = _lock_bytes(path)
    text = _lock_text(content)
    lines = tuple(text.splitlines())
    if not lines or lines != tuple(sorted(lines, key=str.casefold)):
        raise ValueError("dependency lock is not sorted")
    pins: dict[str, Requirement] = {}
    for line in lines:
        requirement = _lock_requirement(line)
        _validate_pin(requirement)
        name = canonicalize_name(requirement.name)
        if name in pins:
            raise ValueError("dependency lock contains duplicate requirements")
        pins[name] = requirement
    _validate_direct_closure(pins, declared)
    _bind_declared(pins, declared)
    return content, lines


def _validate_direct_closure(
    pins: Mapping[str, Requirement],
    declared: Sequence[Requirement],
) -> None:
    observed = {name for name, requirement in pins.items() if requirement.url is not None}
    expected = {
        canonicalize_name(requirement.name)
        for requirement in declared
        if requirement.url is not None
    }
    if observed != expected:
        raise ValueError("dependency lock contains an undeclared direct dependency")


def _bind_declared(pins: dict[str, Requirement], declared: Sequence[Requirement]) -> None:
    for requirement in declared:
        name = canonicalize_name(requirement.name)
        if name not in pins:
            raise ValueError(f"dependency lock is missing {requirement.name}")
        pin = pins[name]
        if requirement.url is not None:
            if pin.url != requirement.url:
                raise ValueError(f"direct dependency drift: {requirement.name}")
            continue
        version = next(iter(pin.specifier)).version
        if not requirement.specifier.contains(version, prereleases=True):
            raise ValueError(f"locked version violates pyproject: {requirement.name}")


def _toml_array(values: Sequence[object]) -> str:
    if not all(isinstance(value, str) for value in values):
        raise ValueError("fixture metadata arrays must contain strings")
    return json.dumps(values, ensure_ascii=True)


def _fixture_toml(payload: dict[str, object]) -> str:
    project = payload["project"]
    assert isinstance(project, dict)
    tests = project["optional-dependencies"]["test"]
    lines = [
        "[build-system]",
        f"requires = {_toml_array(EXPECTED_BUILD)}",
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f"name = {json.dumps(project['name'])}",
        f"version = {json.dumps(project['version'])}",
        f"requires-python = {json.dumps(project['requires-python'])}",
        f"dependencies = {_toml_array(project['dependencies'])}",
        "",
        "[project.optional-dependencies]",
        f"test = {_toml_array(tests)}",
    ]
    scripts = project.get("gui-scripts")
    if scripts is not None:
        if not isinstance(scripts, dict) or not scripts:
            raise ValueError("invalid gui-scripts metadata")
        lines.extend(("", "[project.gui-scripts]"))
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in sorted(scripts.items()))
    lines.extend(
        (
            "",
            "[tool.setuptools.packages.find]",
            'where = ["src"]',
            'include = ["xrr_fitter*"]',
            "namespaces = false",
            "",
        )
    )
    return "\n".join(lines)


def _build_fixture(root: Path, payload: dict[str, object]) -> Path:
    source = root / "src" / "xrr_fitter"
    source.mkdir(parents=True)
    (source / "__init__.py").write_bytes(b"")
    (root / "pyproject.toml").write_text(_fixture_toml(payload), encoding="utf-8")
    output = root / "dist"
    output.mkdir()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(output),
            str(root),
        ),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    archives = tuple(output.glob("*.tar.gz"))
    if len(archives) != 1:
        raise ValueError("fixture build did not create exactly one sdist")
    return archives[0]


def _archive_files(archive: Path) -> tuple[str, ...]:
    with tarfile.open(archive, "r:gz") as handle:
        names = [PurePosixPath(member.name) for member in handle.getmembers() if member.isfile()]
    roots = {path.parts[0] for path in names if path.parts}
    if len(roots) != 1:
        raise ValueError("fixture sdist has an invalid root")
    return tuple(PurePosixPath(*path.parts[1:]).as_posix() for path in names)


def _is_generated_metadata(path: str) -> bool:
    return path in {"PKG-INFO", "setup.cfg"} or path.startswith(
        "src/xrr_fitter.egg-info/"
    )


def _generated_metadata(archive: Path) -> tuple[str, ...]:
    selected = tuple(sorted(filter(_is_generated_metadata, _archive_files(archive))))
    if not selected:
        raise ValueError("fixture sdist generated no metadata")
    allowed = {*GENERATED_METADATA, ENTRY_POINT_METADATA}
    if not set(selected) <= allowed:
        raise ValueError("fixture sdist generated unexpected metadata")
    return selected


def _expected_generated_metadata(payload: dict[str, object]) -> tuple[str, ...]:
    project = payload["project"]
    assert isinstance(project, dict)
    if project.get("gui-scripts") is None:
        return GENERATED_METADATA
    return tuple(sorted((*GENERATED_METADATA, ENTRY_POINT_METADATA)))


def _assert_build_environment() -> None:
    expected = {"setuptools": "75.8.2", "wheel": "0.45.1"}
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected:
        raise ValueError(f"pinned build environment required: {observed}")


def build_generated_metadata(payload: dict[str, object]) -> tuple[str, ...]:
    _assert_build_environment()
    with tempfile.TemporaryDirectory(prefix="xrr-r23-sdist-fixture-") as directory:
        root = Path(directory)
        first = _generated_metadata(_build_fixture(root / "first", payload))
        second = _generated_metadata(_build_fixture(root / "second", payload))
    if first != second:
        raise ValueError("pinned sdist metadata is not deterministic")
    if first != _expected_generated_metadata(payload):
        raise ValueError("pinned sdist metadata differs from the audited policy")
    return first


def calculate_release_spec(
    pyproject: str | Path,
    lock_file: str | Path,
) -> dict[str, object]:
    payload = _pyproject(Path(pyproject))
    project = payload["project"]
    build = payload["build-system"]
    assert isinstance(project, dict) and isinstance(build, dict)
    runtime = tuple(project["dependencies"])
    tests = tuple(project["optional-dependencies"]["test"])
    declared = _parse_requirements((*build["requires"], *runtime, *tests))
    lock_content, _ = _lock(Path(lock_file), declared)
    metadata = build_generated_metadata(payload)
    return {
        "schema": SCHEMA,
        "build_system": {
            "requires": list(build["requires"]),
            "build_backend": build["build-backend"],
        },
        "requires_python": project["requires-python"],
        "runtime_dependencies": list(runtime),
        "test_dependencies": list(tests),
        "lock_sha256": hashlib.sha256(lock_content).hexdigest(),
        "wheel_content_policy": {
            "package_root": "xrr_fitter",
            "include_distribution_metadata": True,
            "forbidden_roots": [
                "docs",
                "examples",
                "tests",
                "tools",
                "verification",
            ],
        },
        "sdist_content_policy": {
            "input_directories": list(INPUT_DIRECTORIES),
            "input_files": list(INPUT_FILES),
            "generated_metadata": list(metadata),
        },
    }


def _atomic_write(path: Path, content: bytes) -> None:
    parent = path.parent.resolve()
    if path.parent.is_symlink() or not parent.is_dir():
        raise ValueError("release spec output parent must exist")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("release spec output must be a regular file path")
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


def write_release_spec(
    pyproject: str | Path,
    lock_file: str | Path,
    output: str | Path,
) -> dict[str, object]:
    spec = calculate_release_spec(pyproject, lock_file)
    _atomic_write(Path(output), _canonical(spec))
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_release_spec(args.pyproject, args.lock_file, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
