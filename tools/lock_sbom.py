#!/usr/bin/env python3
"""Emit an offline CycloneDX inventory of one validated Python lock file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402

from lock_environment import read_project_dependencies, validate_lock_text  # noqa: E402
from lock_windows_environment import read_windows_dependencies  # noqa: E402
from version_source import declared_project_version  # noqa: E402

TARGETS = ("macos-arm64-py312", "windows-x64-py312")


def _read_input(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"SBOM input must be a regular file: {path.name}")
    return path.read_bytes()


def _validate_lock(root: Path, target: str, text: str) -> None:
    project = root / "pyproject.toml"
    if target == "macos-arm64-py312":
        build, runtime, tests = read_project_dependencies(project)
    else:
        build, runtime, packaging = read_windows_dependencies(project)
        build = (*build, *packaging)
        tests = ()
    validate_lock_text(text, build_dependencies=build, runtime_dependencies=runtime, test_dependencies=tests)


def _versioned_component(name: str, version: str, kind: str) -> dict[str, object]:
    purl = f"pkg:pypi/{name}@{quote(version, safe='')}"
    return {"type": kind, "bom-ref": purl, "name": name, "version": version, "purl": purl}


def _locked_component(line: str) -> dict[str, object]:
    requirement = Requirement(line)
    name = canonicalize_name(requirement.name)
    if requirement.url is None:
        return _versioned_component(name, next(iter(requirement.specifier)).version, "library")
    # validate_lock_text has already required an HTTPS Git URL and full commit.
    url, _separator, commit = requirement.url.removeprefix("git+").rpartition("@")
    return {
        "type": "library",
        "bom-ref": f"vcs:{name}:{commit}",
        "name": name,
        "externalReferences": [{"type": "vcs", "url": url}],
        "properties": [{"name": "xrr:vcs:commit", "value": commit}],
    }


def _metadata(
    root: Path, target: str, lock_name: str, lock_content: bytes, project_content: bytes
) -> dict[str, object]:
    payload = tomllib.loads(project_content.decode("utf-8"))
    name = canonicalize_name(payload["project"]["name"], validate=True)
    version = declared_project_version(root, payload)
    properties = {
        "xrr:inventory:scope": "locked-python-environment",
        "xrr:inventory:target": target,
        "xrr:lock:path": lock_name,
        "xrr:lock:sha256": hashlib.sha256(lock_content).hexdigest(),
        "xrr:pyproject:sha256": hashlib.sha256(project_content).hexdigest(),
    }
    return {
        "component": _versioned_component(name, version, "application"),
        "properties": [{"name": name, "value": value} for name, value in sorted(properties.items())],
    }


def build_lock_sbom(repo_root: str | Path, *, target: str) -> dict[str, object]:
    """Inventory pins, not installed wheels, licenses, or bundled native code.

    A flat lock cannot prove dependency edges or artifact hashes. Leave those
    fields absent and explicitly mark the composition incomplete.
    """
    if target not in TARGETS:
        raise ValueError(f"unsupported lock SBOM target: {target}")
    root = Path(repo_root).resolve()
    lock = root / f"requirements-{target}.lock"
    project = root / "pyproject.toml"
    lock_content = _read_input(lock)
    project_content = _read_input(project)
    text = lock_content.decode("utf-8")
    _validate_lock(root, target, text)
    metadata = _metadata(root, target, lock.name, lock_content, project_content)
    components = sorted((_locked_component(line) for line in text.splitlines()), key=lambda item: item["name"])
    if _read_input(project) != project_content or _read_input(lock) != lock_content:
        raise ValueError("SBOM inputs changed during inventory generation")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": metadata,
        "components": components,
        "compositions": [{"aggregate": "incomplete"}],
    }


def canonical_sbom_bytes(report: dict[str, object]) -> bytes:
    return (
        json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target", choices=TARGETS, required=True)
    args = parser.parse_args(argv)
    try:
        content = canonical_sbom_bytes(build_lock_sbom(args.repo_root, target=args.target))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    sys.stdout.buffer.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
