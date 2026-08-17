#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

if __name__ == "__main__":
    sys.dont_write_bytecode = True


TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from verify_registry import (  # noqa: E402
    ARTIFACT,
    ARTIFACT_MANIFEST,
    MODE_REGISTRY,
    PYTEST_PREFIX,  # noqa: F401 - re-exported for registry contract tests.
    PYTHON,
    RELEASE_ORDER,
    REPORT,
    ROOT,
    Mode,
)
from verify_report import (  # noqa: E402
    DirectoryIdentity,
    ReportAnchor,
    ReportGuard,
    _directory_identity,
    _make_report_anchor,
    _make_report_guard,
    _prepare_regular_report,
    _prepare_report_directory,
    _require_same_directory,
    _resolve_output_path,
    build_environment,
)

Runner = Callable[..., object]


class MissingApprovedEvidence(RuntimeError):
    """The post-delivery owner sign-off has not been frozen yet."""


def _materialize(
    command: Sequence[str],
    report_dir: Path,
    repo_root: Path,
    artifact_dir: Path | None,
    artifact_manifest: Path | None,
) -> tuple[str, ...]:
    _require_placeholder(command, ARTIFACT, artifact_dir, "artifact directory")
    _require_placeholder(command, ARTIFACT_MANIFEST, artifact_manifest, "artifact manifest")
    replacements = {
        PYTHON: sys.executable,
        REPORT: str(report_dir),
        ROOT: str(repo_root),
    }
    if artifact_dir is not None:
        replacements[ARTIFACT] = str(artifact_dir)
    if artifact_manifest is not None:
        replacements[ARTIFACT_MANIFEST] = str(artifact_manifest)
    return tuple(_replace_token(token, replacements) for token in command)


def _require_placeholder(
    command: Sequence[str],
    placeholder: str,
    value: object | None,
    label: str,
) -> None:
    if value is None and any(placeholder in token for token in command):
        raise ValueError(f"{label} is required for this mode")


def _replace_token(token: str, replacements: Mapping[str, str]) -> str:
    value = replacements.get(token, token)
    for placeholder, replacement in replacements.items():
        value = value.replace(placeholder, replacement)
    return value


def _invoke(
    runner: Runner,
    command: Sequence[str],
    *,
    root: Path,
    environment: Mapping[str, str],
    report_guard: ReportGuard | None = None,
) -> None:
    if report_guard is not None:
        report_guard()
    runner(tuple(command), cwd=root, env=dict(environment), check=True)


def _execute_mode(
    mode: Mode,
    *,
    root: Path,
    report: Path,
    artifact: Path | None,
    artifact_manifest: Path | None,
    environment: Mapping[str, str],
    runner: Runner,
    require_git_clean: bool = False,
    report_guard: ReportGuard | None = None,
) -> None:
    hygiene = (sys.executable, "tools/check_hygiene.py")
    if require_git_clean:
        hygiene = (*hygiene, "--require-git-clean")
    _invoke(runner, hygiene, root=root, environment=environment, report_guard=report_guard)
    failure: BaseException | None = None
    try:
        for command in mode.commands:
            args = _materialize(command, report, root, artifact, artifact_manifest)
            _invoke(runner, args, root=root, environment=environment, report_guard=report_guard)
    except BaseException as error:
        failure = error
    try:
        _invoke(runner, hygiene, root=root, environment=environment, report_guard=report_guard)
    except BaseException as error:
        if failure is None:
            failure = error
    if failure is not None:
        raise failure


def run_mode(
    name: str,
    mode: Mode,
    *,
    repo_root: str | Path,
    report_dir: str | Path,
    artifact_dir: str | Path | None = None,
    artifact_manifest: str | Path | None = None,
    approved_data_root: str | Path | None = None,
    capture_candidate: bool = False,
    expected_report_identity: DirectoryIdentity | None = None,
    runner: Runner = subprocess.run,
) -> None:
    root = Path(repo_root).resolve()
    report = _resolve_output_path(report_dir, "report directory")
    artifact = _resolve_output_path(artifact_dir, "artifact directory") if artifact_dir is not None else None
    manifest = _resolve_output_path(artifact_manifest, "artifact manifest") if artifact_manifest is not None else None
    approved = Path(approved_data_root).resolve() if approved_data_root is not None else None
    if expected_report_identity is not None:
        _require_same_directory(report, expected_report_identity, "report directory")
    report_anchor = _make_report_anchor(report)
    _validate_mode_inputs(name, root, report, artifact, manifest, approved, capture_candidate)
    if _run_special_mode(
        name,
        mode,
        root=root,
        report=report,
        artifact=artifact,
        manifest=manifest,
        approved=approved,
        capture_candidate=capture_candidate,
        report_anchor=report_anchor,
        expected_report_identity=expected_report_identity,
        runner=runner,
    ):
        return
    _run_regular_mode(
        name,
        mode,
        root=root,
        report=report,
        artifact=artifact,
        manifest=manifest,
        report_anchor=report_anchor,
        runner=runner,
    )


def _validate_mode_inputs(
    name: str,
    root: Path,
    report: Path,
    artifact: Path | None,
    manifest: Path | None,
    approved: Path | None,
    capture_candidate: bool,
) -> None:
    if report.is_relative_to(root):
        raise ValueError("report directory must be external")
    _require_option_scope(
        artifact is not None,
        name,
        {"distribution", "identity", "release"},
        "artifact directory is only valid for distribution, identity, or release",
    )
    if name == "distribution" and artifact != report / "artifacts":
        raise ValueError("distribution artifact directory must equal report-dir/artifacts")
    if name == "identity":
        _validate_identity_paths(root, report, artifact, manifest)
    _require_option_scope(
        capture_candidate,
        name,
        {"approved-data"},
        "candidate capture is only valid for approved-data mode",
    )
    _require_option_scope(
        approved is not None,
        name,
        {"approved-data"},
        "owner-data input is only valid for approved-data mode",
    )
    _require_option_scope(
        manifest is not None,
        name,
        {"identity"},
        "artifact manifest is only valid for identity mode",
    )


def _require_option_scope(
    active: bool,
    name: str,
    allowed_modes: set[str],
    message: str,
) -> None:
    if active and name not in allowed_modes:
        raise ValueError(message)


def _run_special_mode(
    name: str,
    mode: Mode,
    *,
    root: Path,
    report: Path,
    artifact: Path | None,
    manifest: Path | None,
    approved: Path | None,
    capture_candidate: bool,
    report_anchor: ReportAnchor,
    expected_report_identity: DirectoryIdentity | None,
    runner: Runner,
) -> bool:
    if name == "approved-data":
        _run_approved_data(
            mode,
            root=root,
            report=report,
            approved=approved,
            capture_candidate=capture_candidate,
            report_anchor=report_anchor,
            runner=runner,
        )
        return True
    if name == "release":
        if artifact is None:
            raise ValueError("release requires an artifact directory")
        run_release(root, report, artifact, runner=runner)
        return True
    if name == "identity":
        _run_isolated(
            mode,
            root=root,
            report=report,
            artifact=artifact,
            artifact_manifest=manifest,
            report_anchor=report_anchor,
            expected_report_identity=expected_report_identity,
            runner=runner,
            prefix="xrr-r23-identity-runtime-",
        )
        return True
    return False


def _run_regular_mode(
    name: str,
    mode: Mode,
    *,
    root: Path,
    report: Path,
    artifact: Path | None,
    manifest: Path | None,
    report_anchor: ReportAnchor,
    runner: Runner,
) -> None:
    report_identity = _prepare_regular_report(report, report_anchor)
    if name != "distribution":
        _require_same_directory(report, report_identity, "report directory")
        environment = build_environment(root, report, expected_identity=report_identity)
        _require_same_directory(report, report_identity, "report directory")
        report_guard = _make_report_guard(
            report,
            report_identity,
            watched=(
                (report / "mpl-cache", "matplotlib cache directory"),
                (report / "xdg-cache", "XDG cache directory"),
            ),
        )
        _execute_mode(
            mode,
            root=root,
            report=report,
            artifact=artifact,
            artifact_manifest=manifest,
            environment=environment,
            runner=runner,
            require_git_clean=name == "distribution",
            report_guard=report_guard,
        )
        return
    with tempfile.TemporaryDirectory(prefix="xrr-r23-distribution-runtime-") as directory:
        environment = build_environment(root, Path(directory))
        _require_same_directory(report, report_identity, "report directory")
        report_guard = _make_report_guard(report, report_identity)
        _execute_mode(
            mode,
            root=root,
            report=report,
            artifact=artifact,
            artifact_manifest=manifest,
            environment=environment,
            runner=runner,
            require_git_clean=True,
            report_guard=report_guard,
        )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_identity_paths(
    root: Path,
    report: Path,
    artifact: Path | None,
    manifest: Path | None,
) -> None:
    if artifact is None or manifest is None:
        raise ValueError("identity requires artifact-dir and artifact-manifest")
    if artifact.name != "artifacts" or manifest.name != "artifact-manifest.json":
        raise ValueError("identity artifact bundle names drift")
    if artifact.parent != manifest.parent:
        raise ValueError("identity artifact bundle must share one parent")
    if report.is_relative_to(root):
        raise ValueError("identity report directory must be external")


def _approved_binding(root: Path, approved: Path):
    evidence = root / "verification/approved-data"
    if not os.path.lexists(evidence / "manifest.json"):
        raise MissingApprovedEvidence("owner-approved manifest is not frozen")
    from freeze_approved_data import validate_approved_data

    return validate_approved_data(
        evidence,
        approved,
    )


def _run_approved_data(
    mode: Mode,
    *,
    root: Path,
    report: Path,
    approved: Path | None,
    capture_candidate: bool,
    report_anchor: ReportAnchor,
    runner: Runner,
) -> None:
    if approved is None:
        raise ValueError("approved-data mode requires an explicit owner-data root")
    if report.is_relative_to(root):
        raise ValueError("approved-data report directory must be external")
    if os.path.lexists(report):
        raise ValueError("approved-data report directory must not already exist")
    binding = None if capture_candidate else _approved_binding(root, approved)
    report_identity = _prepare_report_directory(report, report_anchor)
    report_guard = _make_report_guard(report, report_identity)
    with tempfile.TemporaryDirectory(prefix="xrr-r23-approved-runtime-") as directory:
        environment = build_environment(root, Path(directory))
        environment["XRR_APPROVED_DATA_ROOT"] = str(approved)
        environment["XRR_APPROVED_REPORT_DIR"] = str(report)
        _execute_mode(
            mode,
            root=root,
            report=report,
            artifact=None,
            artifact_manifest=None,
            environment=environment,
            runner=runner,
            report_guard=report_guard,
        )
    report_guard()
    if binding is not None:
        candidate = report / "approved-data-candidate.json"
        if not candidate.is_file():
            raise ValueError("approved-data workflow did not publish a candidate")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != binding.candidate_report_sha256:
            raise ValueError("approved-data workflow candidate drift")


def _run_isolated(
    mode: Mode,
    *,
    root: Path,
    report: Path,
    artifact: Path | None,
    artifact_manifest: Path | None,
    report_anchor: ReportAnchor,
    expected_report_identity: DirectoryIdentity | None,
    runner: Runner,
    prefix: str,
) -> None:
    report_identity = _prepare_report_directory(report, report_anchor)
    if expected_report_identity is not None and report_identity != expected_report_identity:
        raise ValueError("report directory changed during validation")
    report_guard = _make_report_guard(report, report_identity)
    with tempfile.TemporaryDirectory(prefix=prefix) as directory:
        environment = build_environment(root, Path(directory))
        _execute_mode(
            mode,
            root=root,
            report=report,
            artifact=artifact,
            artifact_manifest=artifact_manifest,
            environment=environment,
            runner=runner,
            report_guard=report_guard,
        )


def _release_mode_kwargs(
    name: str,
    root: Path,
    report: Path,
    artifact: Path,
    scratch: Path,
    report_identity: DirectoryIdentity | None,
    runner: Runner,
) -> dict[str, object]:
    subreport = report if name in {"distribution", "identity"} else scratch / name
    kwargs: dict[str, object] = {"repo_root": root, "report_dir": subreport, "runner": runner}
    if name in {"distribution", "identity"}:
        kwargs["artifact_dir"] = artifact
    if name == "identity":
        kwargs["artifact_manifest"] = report / "artifact-manifest.json"
        kwargs["expected_report_identity"] = report_identity
    return kwargs


def _advance_release_report_identity(
    name: str,
    report: Path,
    identity: DirectoryIdentity | None,
) -> DirectoryIdentity | None:
    if name == "distribution":
        return _directory_identity(report, "release report directory")
    if name == "identity" and identity is not None:
        _require_same_directory(report, identity, "release report directory")
    return identity


def run_release(
    repo_root: str | Path,
    report_dir: str | Path,
    artifact_dir: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> None:
    root = Path(repo_root).resolve()
    report = _resolve_output_path(report_dir, "release report directory")
    artifact = _resolve_output_path(artifact_dir, "release artifact directory")
    if artifact != report / "artifacts":
        raise ValueError("release artifact directory must equal report-dir/artifacts")
    if report.is_relative_to(root) or os.path.lexists(report):
        raise ValueError("release report directory must be a new external path")
    with tempfile.TemporaryDirectory(prefix="xrr-r23-release-gates-") as directory:
        scratch = Path(directory)
        report_identity: DirectoryIdentity | None = None
        for name in RELEASE_ORDER:
            kwargs = _release_mode_kwargs(name, root, report, artifact, scratch, report_identity, runner)
            run_mode(name, MODE_REGISTRY[name], **kwargs)
            report_identity = _advance_release_report_identity(name, report, report_identity)


def _run_with_report(
    name: str,
    report_dir: Path,
    artifact_dir: Path | None = None,
    artifact_manifest: Path | None = None,
    approved_data_root: Path | None = None,
    capture_candidate: bool = False,
) -> None:
    root = _repository_root()
    run_mode(
        name,
        MODE_REGISTRY[name],
        repo_root=root,
        report_dir=report_dir,
        artifact_dir=artifact_dir,
        artifact_manifest=artifact_manifest,
        approved_data_root=approved_data_root,
        capture_candidate=capture_candidate,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=tuple(MODE_REGISTRY))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--approved-data-root", type=Path)
    parser.add_argument("--capture-candidate", action="store_true")
    return parser


REQUIRED_ARGUMENT_ERRORS = {
    "distribution": (("report_dir", "artifact_dir"), "distribution requires --report-dir and --artifact-dir"),
    "approved-data": (
        ("report_dir", "approved_data_root"),
        "approved-data requires --report-dir and --approved-data-root",
    ),
    "identity": (
        ("report_dir", "artifact_dir", "artifact_manifest"),
        "identity requires --report-dir, --artifact-dir, and --artifact-manifest",
    ),
    "release": (("report_dir", "artifact_dir"), "release requires --report-dir and --artifact-dir"),
}

OPTION_SCOPE_ERRORS = (
    ("capture_candidate", {"approved-data"}, "--capture-candidate is only valid with approved-data"),
    ("approved_data_root", {"approved-data"}, "--approved-data-root is only valid with approved-data"),
    (
        "artifact_dir",
        {"distribution", "identity", "release"},
        "--artifact-dir is only valid with distribution, identity, or release",
    ),
    ("artifact_manifest", {"identity"}, "--artifact-manifest is only valid with identity"),
)


def _require_mode_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    required, message = REQUIRED_ARGUMENT_ERRORS.get(args.mode, ((), ""))
    if any(getattr(args, name) is None for name in required):
        parser.error(message)


def _validate_argument_scopes(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for attribute, modes, message in OPTION_SCOPE_ERRORS:
        value = getattr(args, attribute)
        if value is not None and value is not False and args.mode not in modes:
            parser.error(message)


def _run_explicit(args: argparse.Namespace) -> int:
    try:
        _run_with_report(
            args.mode,
            args.report_dir,
            args.artifact_dir,
            args.artifact_manifest,
            args.approved_data_root,
            args.capture_candidate,
        )
    except MissingApprovedEvidence as error:
        print(str(error), file=sys.stderr)
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _require_mode_arguments(args, parser)
    _validate_argument_scopes(args, parser)
    if args.report_dir is not None:
        return _run_explicit(args)
    with tempfile.TemporaryDirectory(prefix=f"xrr-r23-{args.mode}-") as directory:
        _run_with_report(args.mode, Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
