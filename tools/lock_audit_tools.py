#!/usr/bin/env python3
"""Generate the macOS ARM64/Python 3.12 audit-tool wheel hash lock."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.requirements import Requirement  # noqa: E402
from packaging.tags import sys_tags  # noqa: E402
from packaging.utils import canonicalize_name, parse_wheel_filename  # noqa: E402
from packaging.version import Version  # noqa: E402

from lock_environment import PIP_VERSION, _atomic_write  # noqa: E402


def _wheel_filename(value: str) -> str:
    url = urlsplit(value)
    if url.scheme != "https" or url.netloc != "files.pythonhosted.org" or url.query or url.fragment:
        raise ValueError("audit wheel must come from the HTTPS PyPI file host")
    return Path(unquote(url.path)).name


def _wheel_digest(download: dict) -> str:
    digest = download.get("archive_info", {}).get("hashes", {}).get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("audit wheel is missing a canonical SHA-256 digest")
    return digest


def _wheel_entry(item: dict) -> tuple[str, str, str]:
    if item.get("is_direct") is not False or item.get("is_yanked") is not False:
        raise ValueError("audit lock requires non-yanked index wheels")
    metadata = item["metadata"]
    name = canonicalize_name(metadata["name"], validate=True)
    version = Version(metadata["version"])
    download = item["download_info"]
    wheel_name, wheel_version, _build, tags = parse_wheel_filename(_wheel_filename(download["url"]))
    if wheel_name != name or wheel_version != version or not tags.intersection(sys_tags()):
        raise ValueError("audit wheel identity or target tags do not match resolution")
    return name, str(version), _wheel_digest(download)


def _require_version(requirement: Requirement, entries: dict[str, tuple[str, str]]) -> str:
    name = canonicalize_name(requirement.name)
    if name not in entries:
        raise ValueError(f"audit resolution is missing dependency: {name}")
    if requirement.url or not requirement.specifier.contains(entries[name][0], prereleases=True):
        raise ValueError(f"audit resolution violates dependency constraint: {requirement}")
    return name


def _active_dependencies(metadata: dict, extras: set[str], environment: dict[str, str]) -> list[Requirement]:
    dependencies = []
    for value in metadata.get("requires_dist", ()):
        dependency = Requirement(value)
        if dependency.marker is None or any(
            dependency.marker.evaluate({**environment, "extra": extra}) for extra in ("", *extras)
        ):
            dependencies.append(dependency)
    return dependencies


def _validate_closure(
    entries: dict[str, tuple[str, str]],
    metadata: dict[str, dict],
    requirements: Sequence[str],
    environment: dict[str, str],
) -> None:
    pending = [Requirement(value) for value in requirements]
    visited: set[tuple[str, tuple[str, ...]]] = set()
    while pending:
        requirement = pending.pop()
        name = _require_version(requirement, entries)
        key = name, tuple(sorted(requirement.extras))
        if key in visited:
            continue
        visited.add(key)
        pending.extend(_active_dependencies(metadata[name], requirement.extras, environment))
    if {name for name, _extras in visited} != set(entries):
        raise ValueError("audit resolution includes packages outside the declared closure")


def _report_environment(report: dict) -> dict[str, str]:
    if report.get("version") != "1" or report.get("pip_version") != PIP_VERSION:
        raise ValueError("audit resolution requires the pinned pip report schema")
    environment = report["environment"]
    expected = {"python_version": "3.12", "sys_platform": "darwin", "platform_machine": "arm64"}
    if any(environment.get(name) != value for name, value in expected.items()):
        raise ValueError("audit tool lock requires macOS ARM64/Python 3.12 resolution")
    return environment


def _resolved_packages(report: dict) -> tuple[dict[str, tuple[str, str]], dict[str, dict]]:
    entries: dict[str, tuple[str, str]] = {}
    metadata = {}
    for item in report["install"]:
        name, version, digest = _wheel_entry(item)
        if name in entries:
            raise ValueError(f"duplicate audit resolution package: {name}")
        entries[name] = version, digest
        metadata[name] = item["metadata"]
    return entries, metadata


def _validate_constraints(entries: dict[str, tuple[str, str]], constraints: Sequence[str]) -> None:
    for value in constraints:
        requirement = Requirement(value)
        if canonicalize_name(requirement.name) in entries:
            _require_version(requirement, entries)


def render_lock(report: dict, requirements: Sequence[str], constraints: Sequence[str]) -> bytes:
    environment = _report_environment(report)
    entries, metadata = _resolved_packages(report)
    _validate_constraints(entries, constraints)
    _validate_closure(entries, metadata, requirements, environment)
    return "".join(
        f"{name}=={version} --hash=sha256:{digest}\n" for name, (version, digest) in sorted(entries.items())
    ).encode("ascii")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        project = tomllib.loads((args.repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        requirements = project["tool"]["xrr"]["audit"]["requires"]
        constraints = (args.repo_root / "requirements-macos-arm64-py312.lock").read_text(encoding="utf-8").splitlines()
        content = render_lock(json.loads(args.report.read_text(encoding="utf-8")), requirements, constraints)
        if args.check:
            if args.output.is_symlink() or args.output.read_bytes() != content:
                raise ValueError("audit tool lock differs from the supplied resolution")
        else:
            _atomic_write(args.output, content)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
