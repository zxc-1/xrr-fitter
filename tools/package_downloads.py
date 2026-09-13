#!/usr/bin/env python3
"""Resolve, download and verify target wheel bytes without changing lock versions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from audit_reports import _write, run_logged  # noqa: E402
from lock_environment import PIP_VERSION, _resolver_environment  # noqa: E402
from lock_sbom import TARGETS  # noqa: E402
from package_manifest import (  # noqa: E402
    build_manifest,
    download_requirements,
    locked_inputs,
    manifest_bytes,
    target_platforms,
    validate_manifest,
    verify_wheels,
)
from verify_report import (  # noqa: E402
    _directory_identity,
    _make_report_guard,
    _prepare_regular_report,
    _require_same_directory,
    _resolve_output_path,
    build_environment,
)


def _target_arguments(target: str) -> tuple[str, ...]:
    platforms = tuple(token for platform in target_platforms(target) for token in ("--platform", platform))
    return ("--only-binary=:all:", "--python-version", "3.12", "--implementation", "cp", "--abi", "cp312", *platforms)


def resolution_command(target: str, report: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-cache-dir",
        "--dry-run",
        "--no-deps",
        "--ignore-installed",
        "--index-url",
        "https://pypi.org/simple",
        *_target_arguments(target),
        "--report",
        str(report / "resolution.json"),
        "--requirement",
        str(report / "locked.requirements"),
    )


def download_command(target: str, report: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pip",
        "--isolated",
        "download",
        "--no-cache-dir",
        "--require-hashes",
        "--no-deps",
        "--no-index",
        *_target_arguments(target),
        "--dest",
        str(report / "wheels"),
        "--requirement",
        str(report / "download.requirements"),
    )


def prepare_report(root: Path, report: Path) -> tuple[Path, object, dict[str, str]]:
    report = _resolve_output_path(report, "package report")
    if report.is_relative_to(root.resolve()):
        raise ValueError("package report must be external")
    identity = _prepare_regular_report(report)
    environment = {
        **_resolver_environment(),
        **{
            name: value
            for name, value in build_environment(root, report, expected_identity=identity).items()
            if name in {"XDG_CACHE_HOME", "MPLCONFIGDIR"}
        },
    }
    return report, _make_report_guard(report, identity), environment


def _require_pip() -> None:
    if metadata.version("pip") != PIP_VERSION:
        raise ValueError(f"package resolution requires pip {PIP_VERSION}")


def _summary(report: Path, mode: str, code: int, **fields: object) -> None:
    content = manifest_bytes({"state": "PASS" if code == 0 else "FAIL", "mode": mode, "exit_code": code, **fields})
    _write(report, "summary.json", content.decode("ascii"))


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("package manifest/report input must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("package manifest/report must be a JSON object")
    return payload


def read_manifest(root: Path, path: Path) -> dict:
    return validate_manifest(root, _read_json(path))


def resolve_packages(root: Path, target: str, report: Path) -> int:
    pins, inputs = locked_inputs(root, target)
    report, guard, environment = prepare_report(root, report)
    _require_pip()
    _write(report, "locked.requirements", "".join(f"{name}=={version}\n" for name, version in sorted(pins.items())))
    _write(report, "inputs.json", manifest_bytes(inputs).decode("ascii"))
    code = run_logged(
        resolution_command(target, report), root=root, report=report, label="resolution", environment=environment
    )
    guard()
    if locked_inputs(root, target) != (pins, inputs):
        raise ValueError("package lock inputs changed during resolution")
    if code == 0:
        manifest = build_manifest(root, target, _read_json(report / "resolution.json"))
        _write(report, "manifest.json", manifest_bytes(manifest).decode("ascii"))
    _summary(report, "resolve", code, target=target)
    return code


def _download_stage(root: Path, manifest: dict, stage: Path, environment: dict[str, str]) -> int:
    _write(stage, "download.requirements", download_requirements(manifest).decode("ascii"))
    code = run_logged(
        download_command(manifest["target"], stage), root=root, report=stage, label="download", environment=environment
    )
    if code == 0:
        verify_wheels(stage / "wheels", manifest["wheels"])
    return code


def _publish_wheels(stage: Path, report: Path, manifest: dict) -> None:
    destination = report / "wheels"
    if os.path.lexists(destination):
        raise ValueError("package wheel destination already exists")
    identity = _directory_identity(stage / "wheels", "package directory")
    os.rename(stage / "wheels", destination)
    try:
        verify_wheels(destination, manifest["wheels"])
    except BaseException:
        try:
            _require_same_directory(destination, identity, "package directory")
        except (OSError, ValueError):
            pass
        else:
            shutil.rmtree(destination)
        raise


def download_packages(root: Path, manifest_path: Path, report: Path) -> int:
    manifest = read_manifest(root, manifest_path)
    report, guard, environment = prepare_report(root, report)
    _require_pip()
    _write(report, "manifest.json", manifest_bytes(manifest).decode("ascii"))
    with tempfile.TemporaryDirectory(prefix="download-stage-", dir=report) as temporary:
        stage = Path(temporary)
        try:
            code = _download_stage(root, manifest, stage, environment)
        finally:
            for path in stage.glob("download.*"):
                _write(report, path.name, path.read_text(encoding="utf-8"))
        guard()
        if read_manifest(root, manifest_path) != manifest:
            raise ValueError("package manifest changed during download")
        if code == 0:
            _publish_wheels(stage, report, manifest)
    _summary(report, "download", code, target=manifest["target"], vcs_verified=False)
    return code


def require_install_target(target: str) -> None:
    if sys.prefix == sys.base_prefix:
        raise ValueError("package installation requires an external virtual environment")
    expected = {"macos-arm64-py312": ("darwin", "arm64"), "windows-x64-py312": ("win32", "amd64")}
    if (sys.platform, platform.machine().lower()) != expected.get(target) or sys.version_info[:2] != (3, 12):
        raise ValueError("package install target does not match the running Python platform")
    _require_pip()


def install_command(report: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-cache-dir",
        "--no-index",
        "--no-deps",
        "--require-hashes",
        "--only-binary=:all:",
        "--force-reinstall",
        "--requirement",
        str(report / "install.requirements"),
    )


def _installed_versions(manifest: dict) -> None:
    for item in manifest["wheels"]:
        if metadata.version(item["name"]) != item["version"]:
            raise ValueError(f"installed package version differs from the manifest: {item['name']}")


def install_packages(root: Path, manifest_path: Path, wheel_directory: Path, report: Path) -> int:
    manifest = read_manifest(root, manifest_path)
    require_install_target(manifest["target"])
    if Path(sys.prefix).resolve().is_relative_to(root.resolve()):
        raise ValueError("package installation requires an external virtual environment")
    before = verify_wheels(wheel_directory, manifest["wheels"])
    report, guard, environment = prepare_report(root, report)
    requirements = "".join(
        f"{item['name']} @ {(wheel_directory / item['filename']).resolve().as_uri()} --hash=sha256:{item['sha256']}\n"
        for item in manifest["wheels"]
    )
    _write(report, "install.requirements", requirements)
    _write(report, "manifest.json", manifest_bytes(manifest).decode("ascii"))
    code = run_logged(install_command(report), root=root, report=report, label="install", environment=environment)
    guard()
    if read_manifest(root, manifest_path) != manifest or verify_wheels(wheel_directory, manifest["wheels"]) != before:
        raise ValueError("package inputs changed during installation")
    if code == 0:
        _installed_versions(manifest)
        code = run_logged(
            (sys.executable, "-m", "pip", "check"), root=root, report=report, label="pip-check", environment=environment
        )
    guard()
    _summary(report, "install", code, target=manifest["target"], vcs_verified=False)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("resolve", "download", "verify", "install"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--wheel-dir", type=Path)
    args = parser.parse_args(argv)
    required = {
        "resolve": ("target", "report_dir"),
        "download": ("manifest", "report_dir"),
        "verify": ("manifest", "wheel_dir"),
        "install": ("manifest", "wheel_dir", "report_dir"),
    }
    if any(getattr(args, name) is None for name in required[args.mode]):
        parser.error(f"{args.mode} requires {required[args.mode]}")
    root = args.repo_root.resolve()
    try:
        if args.mode == "resolve":
            return resolve_packages(root, args.target, args.report_dir)
        if args.mode == "download":
            return download_packages(root, args.manifest, args.report_dir)
        if args.mode == "install":
            return install_packages(root, args.manifest, args.wheel_dir, args.report_dir)
        manifest = read_manifest(root, args.manifest)
        records = verify_wheels(args.wheel_dir, manifest["wheels"])
        print(json.dumps({"state": "PASS", "target": manifest["target"], "wheels": records, "vcs_verified": False}))
    except (OSError, ValueError, KeyError, TypeError, metadata.PackageNotFoundError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
