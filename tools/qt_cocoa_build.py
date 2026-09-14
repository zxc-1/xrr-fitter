#!/usr/bin/env python3
"""Build and test the pinned Qt Cocoa plugin outside the application environment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from audit_reports import _write  # noqa: E402
from package_cache import copy_verified  # noqa: E402
from package_downloads import _summary, prepare_report, require_install_target  # noqa: E402
from package_manifest import manifest_bytes  # noqa: E402
from qt_cocoa_evidence import CASES, build_receipt, code_bindings, source_additions  # noqa: E402
from qt_cocoa_inputs import input_records, read_inputs, verify_inputs  # noqa: E402
from qt_cocoa_wheel import derive_wheel, verify_derivation  # noqa: E402
from refnx_build_runtime import build_environment, run_checked, toolchain_evidence  # noqa: E402


def compile_commands(sdk: Path, stage: Path) -> list[tuple[str, tuple[str, ...], Path]]:
    qmake = str(sdk / "bin/qmake")
    build = stage / "build"
    regression = stage / "regression"
    return [
        (
            "qmake",
            (
                qmake,
                str(stage / "source/cocoa.pro"),
                "CONFIG+=release",
                "CONFIG+=no_qt_rpath no_default_rpath",
                "QMAKE_RPATHDIR=@loader_path/../../lib",
            ),
            build,
        ),
        ("make", ("/usr/bin/make", "-j4"), build),
        (
            "signature",
            ("/usr/bin/codesign", "--verify", "--strict", "--verbose=2", str(build / "platforms/libqcocoa.dylib")),
            build,
        ),
        (
            "regression-qmake",
            (qmake, str(stage / "regression-source/ownership_regression.pro"), "CONFIG+=release"),
            regression,
        ),
        ("regression-make", ("/usr/bin/make", "-j4"), regression),
    ]


def sdk_member_names(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        name = line.removesuffix("/")
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
            raise ValueError("Qt SDK member is not a canonical relative path")
        names.append(name)
    if not names or len(names) != len(set(names)):
        raise ValueError("Qt SDK members are empty or duplicated")
    return names


def run_regressions(root: Path, report: Path, stage: Path, environment: dict) -> dict[str, int]:
    plugin = (stage / "build/platforms/libqcocoa.dylib").resolve()
    environment = {**environment, "QT_PLUGIN_PATH": str(stage / "build"), "QT_QPA_PLATFORM": "cocoa"}
    result = {}
    for case in CASES:
        command = (str(stage / "regression/native_ownership_regression"), case)
        output = run_checked(command, root=root, report=report, label=f"native-{case}", environment=environment)
        lines = output.splitlines()
        if [line for line in lines if line.startswith("PLUGIN:")] != [f"PLUGIN: {plugin}"]:
            raise ValueError("native Qt regression loaded a different plugin")
        if [line for line in lines if line.startswith("PASS:")] != [f"PASS: {case}"]:
            raise ValueError("native Qt regression did not confirm its requested case")
        result[case] = 0
    return result


def extract_sdk(root: Path, archive: Path, sdk: Path, report: Path, environment: dict) -> None:
    output = run_checked(
        ("/usr/bin/tar", "-tf", str(archive)), root=root, report=report, label="sdk-members", environment=environment
    )
    sdk_member_names(output)
    sdk.mkdir()
    run_checked(
        ("/usr/bin/tar", "-xf", str(archive), "-C", str(sdk), "--no-same-owner"),
        root=root,
        report=report,
        label="sdk-extract",
        environment=environment,
    )
    for path in sdk.rglob("*"):
        if path.is_symlink() and not path.resolve(strict=True).is_relative_to(sdk.resolve()):
            raise ValueError("Qt SDK link escapes the extracted input")
    version = run_checked(
        (str(sdk / "bin/qmake"), "-query", "QT_VERSION"),
        root=root,
        report=report,
        label="qt-version",
        environment=environment,
    )
    if version != "6.11.2":
        raise ValueError("Qt SDK version differs from the pinned build")


def _prepare_source(root: Path, manifest: dict, inputs: Path, stage: Path, report: Path, environment: dict) -> dict:
    additions = source_additions(root, manifest, lambda record: (inputs / record["filename"]).read_bytes())
    source = stage / "source"
    regression = stage / "regression-source"
    source.mkdir()
    regression.mkdir()
    for record in manifest["sources"]:
        path = source / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / record["filename"], path)
    for record in manifest["recipes"]:
        path = root / record["path"]
        destination = regression if path.name.startswith("ownership_regression.") else source
        shutil.copyfile(path, destination / path.name)
    command = ("/usr/bin/patch", "-p5", "-F0", "-t", "-N", "-i", str(source / "ownership.patch"))
    run_checked(command, root=source, report=report, label="source-patch", environment=environment)
    return additions


def _build_stage(
    root: Path, inputs: Path, wheels: Path, report: Path, stage: Path, manifest: dict, environment: dict
) -> dict:
    copy_verified(inputs, stage / "inputs", input_records(manifest))
    toolchain = toolchain_evidence(root, report, environment)
    environment = build_environment(environment, stage, toolchain)
    for name in ("home", "tmp", "cache", "build", "regression"):
        (stage / name).mkdir()
    extract_sdk(root, stage / "inputs" / manifest["sdk"]["filename"], stage / "sdk", report, environment)
    additions = _prepare_source(root, manifest, stage / "inputs", stage, report, environment)
    for label, command, directory in compile_commands(stage / "sdk", stage):
        run_checked(command, root=directory, report=report, label=label, environment=environment)
    regressions = run_regressions(root, report, stage, environment)
    verify_inputs(root, manifest, stage / "inputs")
    (report / "wheels").mkdir()
    upstream = wheels / manifest["upstream_wheel"]["filename"]
    plugin = stage / "build/platforms/libqcocoa.dylib"
    record = derive_wheel(upstream, manifest["upstream_wheel"], plugin, report / "wheels", additions)
    derivation = verify_derivation(
        upstream, manifest["upstream_wheel"], report / "wheels" / record["filename"], record, additions
    )
    return build_receipt(root, manifest, record, derivation, toolchain, regressions)


def build_cocoa(root: Path, inputs: Path, wheels: Path, report: Path) -> int:
    require_install_target("macos-arm64-py312")
    if Path(sys.prefix).resolve().is_relative_to(root.resolve()):
        raise ValueError("Qt build requires an external virtual environment")
    manifest = read_inputs(root)
    verify_inputs(root, manifest, inputs)
    builder = code_bindings(root)
    report, guard, environment = prepare_report(root, report)
    try:
        with tempfile.TemporaryDirectory(prefix="qt-cocoa-build-stage-", dir=report) as temporary:
            result = _build_stage(root, inputs, wheels, report, Path(temporary), manifest, environment)
        guard()
        verify_inputs(root, manifest, inputs)
        if code_bindings(root) != builder:
            raise ValueError("Qt builder code changed during the build")
        _write(report, "build.json", manifest_bytes(result).decode("ascii"))
    except subprocess.CalledProcessError as error:
        _summary(report, "qt-cocoa-build", error.returncode)
        return error.returncode
    _summary(report, "qt-cocoa-build", 0, offline=True, derived_wheel_reused=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return build_cocoa(args.repo_root, args.input_dir, args.wheel_dir, args.report_dir)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
