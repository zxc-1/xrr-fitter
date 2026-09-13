#!/usr/bin/env python3
"""Reproducible non-GUI coverage, incremental types and advisory evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402

from lock_sbom import TARGETS, build_lock_sbom  # noqa: E402
from verify import _materialize  # noqa: E402
from verify_publish import _write_new_file_in_anchored_directory  # noqa: E402
from verify_registry import MODE_REGISTRY, PYTEST_PREFIX  # noqa: E402
from verify_report import (  # noqa: E402
    _directory_identity,
    _make_report_guard,
    _prepare_regular_report,
    _resolve_output_path,
    build_environment,
)

AUDITS = ("coverage", "typing", "advisories")
COVERAGE_MODES = ("unit", "integration", "spawn", "regression")


def _write(report: Path, name: str, content: str, identity: tuple[int, int] | None = None) -> None:
    _write_new_file_in_anchored_directory(
        report,
        identity or _directory_identity(report, "audit report"),
        name,
        content.encode("utf-8"),
        directory_label="audit report",
        file_label="audit evidence",
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"


def check_tool_versions(root: Path) -> dict[str, str]:
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    versions = {}
    for value in payload["tool"]["xrr"]["audit"]["requires"]:
        requirement = Requirement(value)
        expected = tuple(requirement.specifier)
        if len(expected) != 1 or expected[0].operator != "==":
            raise ValueError("audit tool version must have one exact pin")
        actual = metadata.version(requirement.name)
        if actual != expected[0].version:
            raise ValueError(f"audit tool version mismatch: {requirement.name}: {actual} != {expected[0].version}")
        versions[canonicalize_name(requirement.name)] = actual
    return versions


def run_logged(
    command: Sequence[str],
    *,
    root: Path,
    report: Path,
    label: str,
    environment: Mapping[str, str],
) -> int:
    identity = _directory_identity(report, "audit report")
    result = subprocess.run(
        tuple(command), cwd=root, env=dict(environment), capture_output=True, text=True, check=False
    )
    _write(report, f"{label}.stdout", result.stdout, identity)
    _write(report, f"{label}.stderr", result.stderr, identity)
    _write(report, f"{label}.command.json", _json({"argv": list(command), "exit_code": result.returncode}), identity)
    return result.returncode


def coverage_command(root: Path, report: Path) -> tuple[str, ...]:
    paths = tuple(
        token
        for mode in COVERAGE_MODES
        for command in MODE_REGISTRY[mode].commands
        for token in command
        if token.startswith("tests/")
    )
    if len(paths) != len(set(paths)):
        raise ValueError("coverage verifier selections overlap")
    prefix = _materialize(PYTEST_PREFIX, report, root, None, None)
    return (
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--rcfile",
        str(root / "pyproject.toml"),
        *prefix[1:],
        *paths,
        "-q",
    )


def report_coverage(root: Path, report: Path, environment: Mapping[str, str]) -> dict[str, int]:
    environment = {
        **environment,
        "COVERAGE_FILE": str(report / ".coverage"),
        "COVERAGE_RCFILE": str(root / "pyproject.toml"),
    }
    codes = {
        "tests": run_logged(
            coverage_command(root, report), root=root, report=report, label="tests", environment=environment
        )
    }
    commands = {
        "combine": ("combine", "--keep", str(report)),
        "json": ("json", "--keep-combined", "-o", str(report / "coverage.json")),
        "xml": ("xml", "--keep-combined", "-o", str(report / "coverage.xml")),
        "html": ("html", "--keep-combined", "-d", str(report / "html")),
    }
    for name, arguments in commands.items():
        codes[name] = run_logged(
            (sys.executable, "-m", "coverage", *arguments),
            root=root,
            report=report,
            label=name,
            environment=environment,
        )
    return codes


def report_types(root: Path, report: Path, environment: Mapping[str, str]) -> dict[str, int]:
    code = run_logged(
        (
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(root / "pyproject.toml"),
            "--no-incremental",
            "--cache-dir",
            str(report / "mypy-cache"),
            "--output=json",
        ),
        root=root,
        report=report,
        label="mypy",
        environment=environment,
    )
    return {"mypy": code}


def advisory_input(root: Path, target: str) -> tuple[tuple[str, ...], dict[str, object]]:
    inventory = build_lock_sbom(root, target=target)
    components = inventory["components"]
    pins = tuple(f"{item['name']}=={item['version']}" for item in components if "version" in item)
    properties = {item["name"]: item["value"] for item in inventory["metadata"]["properties"]}
    return pins, {
        "target": target,
        "pin_count": len(pins),
        "lock_sha256": properties["xrr:lock:sha256"],
        "excluded_vcs_components": [item for item in components if "version" not in item],
        "native_libraries_scanned": False,
        "service": "pypi",
    }


def advisory_findings(path: Path, pins: tuple[str, ...]) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("advisory report is missing dependencies")
    observed = []
    findings = 0
    for item in dependencies:
        if not isinstance(item, dict) or "skip_reason" in item or not isinstance(item.get("vulns"), list):
            raise ValueError("advisory report contains an incomplete dependency")
        name, version = item.get("name"), item.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError("advisory dependency identity is missing")
        observed.append(f"{canonicalize_name(name)}=={version}")
        findings += len(item["vulns"])
    if sorted(observed) != sorted(pins):
        raise ValueError("advisory report does not cover the exact requested pins")
    return findings


def report_advisories(root: Path, report: Path, environment: Mapping[str, str]) -> dict[str, int]:
    codes = {}
    for target in TARGETS:
        pins, scope = advisory_input(root, target)
        _write(report, f"{target}.requirements", "\n".join(pins) + "\n")
        _write(report, f"{target}.scope.json", _json(scope))
        output = report / f"{target}.advisories.json"
        codes[target] = run_logged(
            (
                sys.executable,
                "-m",
                "pip_audit",
                "--disable-pip",
                "--no-deps",
                "--strict",
                "--cache-dir",
                str(report / "advisory-cache" / target),
                "--vulnerability-service",
                "pypi",
                "--progress-spinner",
                "off",
                "--format",
                "json",
                "--output",
                str(output),
                "--requirement",
                str(report / f"{target}.requirements"),
            ),
            root=root,
            report=report,
            label=target,
            environment=environment,
        )
        try:
            findings = advisory_findings(output, pins)
        except (OSError, ValueError) as error:
            codes[target] = codes[target] or 2
            _write(report, f"{target}.validation-error.txt", f"{error}\n")
        else:
            codes[target] = codes[target] or int(findings > 0)
    return codes


def _input_identity(root: Path) -> dict[str, str]:
    paths = [
        root / name
        for name in ("pyproject.toml", "requirements-macos-arm64-py312.lock", "requirements-windows-x64-py312.lock")
    ]
    for directory in ("src", "tools", "tests", "examples", ".github"):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if {"gui", "__pycache__"}.isdisjoint(path.parts) and path.is_file()
        )
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def run_audit(kind: str, *, root: Path, report: Path) -> int:
    if kind not in AUDITS:
        raise ValueError(f"unknown audit: {kind}")
    root = root.resolve()
    report = _resolve_output_path(report, "audit report")
    if report.is_relative_to(root):
        raise ValueError("audit report must be external")
    identity = _prepare_regular_report(report)
    guard = _make_report_guard(report, identity)
    environment = build_environment(root, report, expected_identity=identity)
    inputs = _input_identity(root)
    _write(report, "inputs.json", _json(inputs), identity)
    versions = check_tool_versions(root)
    environment["MYPYPATH"] = str(root / "src")
    dispatch = {"coverage": report_coverage, "typing": report_types, "advisories": report_advisories}
    codes = dispatch[kind](root, report, environment)
    guard()
    if _input_identity(root) != inputs:
        raise ValueError("audit inputs changed during execution")
    passed = all(code == 0 for code in codes.values())
    _write(
        report,
        "summary.json",
        _json(
            {
                "schema": "xrr-audit-report-v1",
                "kind": kind,
                "state": "PASS" if passed else "FAIL",
                "completed_at": datetime.now(UTC).isoformat(),
                "exit_codes": codes,
                "tool_versions": versions,
                "inputs_sha256": hashlib.sha256(_json(inputs).encode("utf-8")).hexdigest(),
            }
        ),
        identity,
    )
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=AUDITS)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run_audit(args.kind, root=args.repo_root, report=args.report_dir)
    except (OSError, ValueError, KeyError, metadata.PackageNotFoundError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
