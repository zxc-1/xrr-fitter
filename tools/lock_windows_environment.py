#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402

from lock_environment import (  # noqa: E402
    PIP_VERSION,
    _read_lock,
    _requirements,
    _resolver_environment,
    validate_lock_text,
    write_lock,
)

# pip's --platform switch only changes wheel tag selection; environment markers are
# still evaluated against the running host. Resolving the Windows closure from any
# other platform therefore requires evaluating markers ourselves against this table.
WINDOWS_ENVIRONMENT = {
    "implementation_name": "cpython",
    "implementation_version": "3.12.0",
    "os_name": "nt",
    "platform_machine": "AMD64",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Windows",
    "platform_version": "",
    "python_full_version": "3.12.0",
    "python_version": "3.12",
    "sys_platform": "win32",
}

TARGET_ARGUMENTS = (
    "--only-binary=:all:",
    "--platform",
    "win_amd64",
    "--python-version",
    "3.12",
    "--implementation",
    "cp",
    "--abi",
    "cp312",
)


def _regular_file(path: str | Path) -> Path:
    project_path = Path(path)
    if project_path.is_symlink() or not project_path.is_file():
        raise ValueError("pyproject must be a regular file")
    return project_path


def _packaging_tables(project_path: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    try:
        payload = tomllib.loads(project_path.read_text(encoding="utf-8"))
        build = tuple(payload["build-system"]["requires"])
        runtime = tuple(payload["project"]["dependencies"])
        packaging = tuple(payload["tool"]["xrr"]["windows-packaging"]["requires"])
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ValueError("invalid Windows packaging metadata in pyproject") from error
    return build, runtime, packaging


def read_windows_dependencies(path: str | Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    build, runtime, packaging = _packaging_tables(_regular_file(path))
    declared = _requirements((*build, *runtime, *packaging))
    if any(item.url is not None for item in declared):
        raise ValueError("Windows packaging closure cannot contain direct references")
    return build, runtime, packaging


def _run_resolver(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True, text=True, capture_output=True, env=_resolver_environment())


def _installation_report(report: Path) -> tuple[dict[str, object], ...]:
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        entries = tuple(item["metadata"] for item in payload["install"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("pip did not produce a usable installation report") from error
    return entries


def _fetch_metadata(python: Path, specifiers: Sequence[str], workdir: Path) -> tuple[dict[str, object], ...]:
    requirement_file = workdir / "requirements.txt"
    requirement_file.write_text("\n".join(specifiers) + "\n", encoding="utf-8")
    report = workdir / "report.json"
    _run_resolver(
        (
            str(python),
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-deps",
            "--ignore-installed",
            *TARGET_ARGUMENTS,
            "--report",
            str(report),
            "--requirement",
            str(requirement_file),
        )
    )
    return _installation_report(report)


def _applies_to_windows(requirement: Requirement) -> bool:
    if requirement.extras:
        return False
    marker = requirement.marker
    return marker is None or marker.evaluate(WINDOWS_ENVIRONMENT)


def _dependencies_for_windows(metadata: dict[str, object]) -> tuple[str, ...]:
    required: list[str] = []
    for raw in metadata.get("requires_dist") or ():
        requirement = Requirement(raw)
        if _applies_to_windows(requirement):
            # The marker must be dropped before handing the requirement back to pip:
            # pip re-evaluates it against the host and silently omits win32-only
            # packages while still exiting zero.
            required.append(f"{requirement.name}{requirement.specifier}")
    return tuple(required)


def _lock_text(pins: Iterable[str]) -> str:
    return "\n".join(sorted(pins, key=str.casefold)) + "\n"


def _resolver_python(root: Path) -> Path:
    environment = root / "venv"
    _run_resolver((sys.executable, "-m", "venv", str(environment)))
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    _run_resolver((str(python), "-m", "pip", "install", f"pip=={PIP_VERSION}"))
    return python


def _absorb(metadata: dict[str, object], resolved: dict[str, str]) -> tuple[str, ...]:
    # Key on the canonical name but pin under the distribution's own spelling,
    # matching what `pip freeze` writes into the macOS lock.
    declared_name = str(metadata["name"])
    name = canonicalize_name(declared_name)
    if name in resolved:
        return ()
    resolved[name] = f"{declared_name}=={metadata['version']}"
    return _dependencies_for_windows(metadata)


def _resolve_closure(python: Path, root: Path, seeds: Iterable[str]) -> tuple[str, ...]:
    resolved: dict[str, str] = {}
    pending = sorted(set(seeds))
    while pending:
        workdir = root / f"round-{len(resolved)}"
        workdir.mkdir()
        discovered: list[str] = []
        for metadata in _fetch_metadata(python, pending, workdir):
            discovered.extend(_absorb(metadata, resolved))
        pending = sorted({item for item in discovered if canonicalize_name(Requirement(item).name) not in resolved})
    return tuple(resolved.values())


def resolve_windows_lock(pyproject: Path) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    build, runtime, packaging = read_windows_dependencies(pyproject)
    with tempfile.TemporaryDirectory(prefix="xrr-r23-windows-lock-") as directory:
        root = Path(directory)
        if root.resolve().is_relative_to(pyproject.resolve().parent):
            raise ValueError("resolver environment must be outside the repository")
        pins = _resolve_closure(_resolver_python(root), root, (*build, *runtime, *packaging))
    validate_lock_text(
        _lock_text(pins),
        build_dependencies=(*build, *packaging),
        runtime_dependencies=runtime,
    )
    return pins, build, runtime, packaging


def _check(lock: Path, pyproject: Path) -> dict[str, object]:
    build, runtime, packaging = read_windows_dependencies(pyproject)
    return validate_lock_text(
        _read_lock(lock),
        build_dependencies=(*build, *packaging),
        runtime_dependencies=runtime,
    )


def _report_drift(committed: str, expected: str) -> None:
    observed = dict(line.partition("==")[::2] for line in committed.splitlines())
    resolved = dict(line.partition("==")[::2] for line in expected.splitlines())
    for name in sorted(set(observed) | set(resolved), key=str.casefold):
        if observed.get(name) != resolved.get(name):
            print(f"drift {name}: locked={observed.get(name)} resolved={resolved.get(name)}", file=sys.stderr)


def _verify(lock: Path, pyproject: Path) -> dict[str, object]:
    # --check only proves the committed pins satisfy the declared ranges. Drift
    # against the current upstream index (a newer patch release inside the same
    # range) is invisible to it, so re-resolve and compare byte for byte.
    committed = _read_lock(lock)
    pins, build, runtime, packaging = resolve_windows_lock(pyproject)
    expected = _lock_text(pins)
    if committed != expected:
        _report_drift(committed, expected)
        raise SystemExit("Windows lock does not match a fresh resolution of pyproject")
    return validate_lock_text(
        committed,
        build_dependencies=(*build, *packaging),
        runtime_dependencies=runtime,
    )


def _write(output: Path, pyproject: Path) -> dict[str, object]:
    pins, build, runtime, packaging = resolve_windows_lock(pyproject)
    return write_lock(
        output,
        pins,
        build_dependencies=(*build, *packaging),
        runtime_dependencies=runtime,
    )


def _parser(default_pyproject: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--check", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--pyproject", type=Path, default=default_pyproject)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    args = _parser(root / "pyproject.toml").parse_args(argv)
    if args.check is not None:
        report = _check(args.check, args.pyproject)
    elif args.verify is not None:
        report = _verify(args.verify, args.pyproject)
    else:
        report = _write(args.output, args.pyproject)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
