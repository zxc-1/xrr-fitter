#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

PIP_VERSION = "26.1.2"
VCS_PATTERN = re.compile(r"git\+(https://[^@]+)@([0-9a-f]{40})")
PYTHON_ENVIRONMENT_KEYS = {
    "PYTHONHOME",
    "PYTHONOPTIMIZE",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "__PYVENV_LAUNCHER__",
}


def _requirements(values: Iterable[str]) -> tuple[Requirement, ...]:
    parsed: list[Requirement] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("dependency entries must be strings")
        try:
            parsed.append(Requirement(value))
        except InvalidRequirement as error:
            raise ValueError(f"invalid dependency requirement: {value}") from error
    return tuple(parsed)


def _direct_requirement(requirement: Requirement) -> tuple[str, str, str]:
    if requirement.url is None:
        raise ValueError("direct reference URL is missing")
    match = VCS_PATTERN.fullmatch(requirement.url)
    if match is None:
        raise ValueError("VCS references require an HTTPS Git URL and full commit")
    url, commit = match.groups()
    return canonicalize_name(requirement.name), url, commit


def _declared_direct(dependencies: Sequence[Requirement]) -> tuple[str, str, str] | None:
    direct = [_direct_requirement(item) for item in dependencies if item.url is not None]
    if len(direct) > 1:
        raise ValueError("only one direct VCS dependency is supported")
    return direct[0] if direct else None


def _validate_pin(requirement: Requirement) -> None:
    if requirement.extras or requirement.marker is not None:
        raise ValueError("lock requirements cannot contain extras or markers")
    if requirement.url is not None:
        return
    specs = list(requirement.specifier)
    if len(specs) != 1 or specs[0].operator != "==" or "*" in specs[0].version:
        raise ValueError("ordinary lock requirements must use one exact == pin")


def _check_declared_versions(pins: dict[str, Requirement], dependencies: Sequence[Requirement]) -> None:
    for declared in dependencies:
        name = canonicalize_name(declared.name)
        if name not in pins:
            raise ValueError(f"lock is missing declared dependency: {declared.name}")
        locked = pins[name]
        if declared.url is not None:
            if locked.url is None or _direct_requirement(locked) != _direct_requirement(declared):
                raise ValueError(f"direct dependency drift: {declared.name}")
            continue
        version = next(iter(locked.specifier)).version
        if not declared.specifier.contains(version, prereleases=True):
            raise ValueError(f"locked version violates declaration: {declared.name}")


def _canonical_lock_lines(text: str) -> tuple[str, ...]:
    if not text or "\r" in text or not text.endswith("\n"):
        raise ValueError("lock must use canonical LF text with a final newline")
    lines = tuple(text.splitlines())
    if not lines or any(not line or line != line.strip() for line in lines):
        raise ValueError("lock contains blank or non-canonical lines")
    if list(lines) != sorted(lines, key=str.casefold):
        raise ValueError("lock requirements are not canonically sorted")
    return lines


def _parse_lock_requirement(line: str) -> Requirement:
    if line.startswith(("-", "/", ".")) or "file:" in line:
        raise ValueError("lock contains a local, editable, or option requirement")
    try:
        requirement = Requirement(line)
    except InvalidRequirement as error:
        raise ValueError(f"invalid lock requirement: {line}") from error
    if str(requirement) != line:
        raise ValueError(f"non-canonical lock requirement: {line}")
    _validate_pin(requirement)
    return requirement


def _parse_lock(
    lines: Sequence[str],
) -> tuple[dict[str, Requirement], tuple[str, str, str] | None]:
    pins: dict[str, Requirement] = {}
    observed_direct: tuple[str, str, str] | None = None
    for line in lines:
        requirement = _parse_lock_requirement(line)
        name = canonicalize_name(requirement.name)
        if name in pins:
            raise ValueError(f"duplicate lock requirement: {requirement.name}")
        pins[name] = requirement
        if requirement.url is None:
            continue
        if observed_direct is not None:
            raise ValueError("lock contains more than one VCS reference")
        observed_direct = _direct_requirement(requirement)
    return pins, observed_direct


def validate_lock_text(
    text: str,
    *,
    runtime_dependencies: Sequence[str] = (),
    test_dependencies: Sequence[str] = (),
    build_dependencies: Sequence[str] = (),
) -> dict[str, object]:
    lines = _canonical_lock_lines(text)
    declared = _requirements((*build_dependencies, *runtime_dependencies, *test_dependencies))
    expected_direct = _declared_direct(_requirements(test_dependencies))
    pins, observed_direct = _parse_lock(lines)
    if observed_direct != expected_direct:
        raise ValueError("lock direct VCS dependency does not match pyproject")
    _check_declared_versions(pins, declared)
    return {
        "requirement_count": len(lines),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "refnx_commit": observed_direct[2] if observed_direct else None,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    parent = path.parent.resolve()
    if path.parent.is_symlink() or not parent.is_dir():
        raise ValueError("lock output parent must exist and not be a symlink")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("lock output must be a regular file path")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_lock(
    output: str | Path,
    pins: Iterable[str],
    *,
    runtime_dependencies: Sequence[str] = (),
    test_dependencies: Sequence[str] = (),
    build_dependencies: Sequence[str] = (),
) -> dict[str, object]:
    lines = sorted(tuple(pins), key=str.casefold)
    text = "\n".join(lines) + "\n"
    report = validate_lock_text(
        text,
        runtime_dependencies=runtime_dependencies,
        test_dependencies=test_dependencies,
        build_dependencies=build_dependencies,
    )
    _atomic_write(Path(output), text.encode("utf-8"))
    return report


def read_project_dependencies(path: str | Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    project_path = Path(path)
    if project_path.is_symlink() or not project_path.is_file():
        raise ValueError("pyproject must be a regular file")
    try:
        payload = tomllib.loads(project_path.read_text(encoding="utf-8"))
        build = tuple(payload["build-system"]["requires"])
        runtime = tuple(payload["project"]["dependencies"])
        tests = tuple(payload["project"]["optional-dependencies"]["test"])
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ValueError("invalid dependency metadata in pyproject") from error
    _requirements((*build, *runtime, *tests))
    return build, runtime, tests


def _run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def _resolver_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name in PYTHON_ENVIRONMENT_KEYS or name.startswith("PIP_"):
            environment.pop(name)
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def resolve_lock(pyproject: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    build, runtime, tests = read_project_dependencies(pyproject)
    resolver_environment = _resolver_environment()
    with tempfile.TemporaryDirectory(prefix="xrr-r23-lock-") as directory:
        environment = Path(directory) / "venv"
        if environment.resolve().is_relative_to(pyproject.resolve().parent):
            raise ValueError("resolver environment must be outside the repository")
        _run((sys.executable, "-m", "venv", str(environment)), env=resolver_environment)
        python = environment / "bin" / "python"
        _run(
            (str(python), "-m", "pip", "install", f"pip=={PIP_VERSION}"),
            env=resolver_environment,
        )
        _run(
            (str(python), "-m", "pip", "install", *build, *runtime, *tests),
            env=resolver_environment,
        )
        result = _run(
            (str(python), "-m", "pip", "freeze", "--exclude-editable"),
            capture_output=True,
            env=resolver_environment,
        )
    pins = tuple(line for line in result.stdout.splitlines() if line)
    validate_lock_text(
        "\n".join(sorted(pins, key=str.casefold)) + "\n",
        build_dependencies=build,
        runtime_dependencies=runtime,
        test_dependencies=tests,
    )
    return pins, build, runtime, tests


def _read_lock(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("lock path must be a regular file")
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("lock must be readable UTF-8") from error


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--check", type=Path)
    parser.add_argument("--pyproject", type=Path, default=root / "pyproject.toml")
    args = parser.parse_args(argv)
    build, runtime, tests = read_project_dependencies(args.pyproject)
    if args.check is not None:
        report = validate_lock_text(
            _read_lock(args.check),
            build_dependencies=build,
            runtime_dependencies=runtime,
            test_dependencies=tests,
        )
    else:
        pins, build, runtime, tests = resolve_lock(args.pyproject)
        report = write_lock(
            args.output,
            pins,
            build_dependencies=build,
            runtime_dependencies=runtime,
            test_dependencies=tests,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
