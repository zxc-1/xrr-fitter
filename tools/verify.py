#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence


if __name__ == "__main__":
    sys.dont_write_bytecode = True


TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from verify_registry import (  # noqa: E402
    ARTIFACT,
    ARTIFACT_MANIFEST,
    MODE_REGISTRY,
    PYTEST_PREFIX,
    PYTHON,
    RELEASE_ORDER,
    REPORT,
    ROOT,
    Mode,
)


Runner = Callable[..., object]


class MissingApprovedEvidence(RuntimeError):
    """The post-delivery owner sign-off has not been frozen yet."""


def build_environment(repo_root: str | Path, report_dir: str | Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    report = Path(report_dir).resolve()
    mpl = report / "mpl-cache"
    xdg = report / "xdg-cache"
    mpl.mkdir(parents=True, exist_ok=True)
    xdg.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PYTEST_") or name == "PYTHONOPTIMIZE":
            environment.pop(name)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(mpl)
    environment["XDG_CACHE_HOME"] = str(xdg)
    return environment


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
) -> None:
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
) -> None:
    hygiene = (sys.executable, "tools/check_hygiene.py")
    if require_git_clean:
        hygiene = (*hygiene, "--require-git-clean")
    _invoke(runner, hygiene, root=root, environment=environment)
    failure: BaseException | None = None
    try:
        for command in mode.commands:
            args = _materialize(command, report, root, artifact, artifact_manifest)
            _invoke(runner, args, root=root, environment=environment)
    except BaseException as error:
        failure = error
    try:
        _invoke(runner, hygiene, root=root, environment=environment)
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
    runner: Runner = subprocess.run,
) -> None:
    root = Path(repo_root).resolve()
    report = Path(report_dir).resolve()
    artifact = Path(artifact_dir).resolve() if artifact_dir is not None else None
    manifest = Path(artifact_manifest).resolve() if artifact_manifest is not None else None
    approved = Path(approved_data_root).resolve() if approved_data_root is not None else None
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
    runner: Runner,
) -> bool:
    if name == "approved-data":
        _run_approved_data(
            mode,
            root=root,
            report=report,
            approved=approved,
            capture_candidate=capture_candidate,
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
    runner: Runner,
) -> None:
    report.mkdir(parents=True, exist_ok=True)
    if name != "distribution":
        environment = build_environment(root, report)
        _execute_mode(
            mode,
            root=root,
            report=report,
            artifact=artifact,
            artifact_manifest=manifest,
            environment=environment,
            runner=runner,
            require_git_clean=name == "distribution",
        )
        return
    with tempfile.TemporaryDirectory(prefix="xrr-r23-distribution-runtime-") as directory:
        environment = build_environment(root, Path(directory))
        _execute_mode(
            mode,
            root=root,
            report=report,
            artifact=artifact,
            artifact_manifest=manifest,
            environment=environment,
            runner=runner,
            require_git_clean=True,
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
        root / "verification/r22/reference/manifest.json",
    )


def _run_approved_data(
    mode: Mode,
    *,
    root: Path,
    report: Path,
    approved: Path | None,
    capture_candidate: bool,
    runner: Runner,
) -> None:
    if approved is None:
        raise ValueError("approved-data mode requires an explicit owner-data root")
    if report.is_relative_to(root):
        raise ValueError("approved-data report directory must be external")
    if os.path.lexists(report):
        raise ValueError("approved-data report directory must not already exist")
    binding = None if capture_candidate else _approved_binding(root, approved)
    report.mkdir(parents=True)
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
        )
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
    runner: Runner,
    prefix: str,
) -> None:
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
        )


def run_release(
    repo_root: str | Path,
    report_dir: str | Path,
    artifact_dir: str | Path,
    *,
    runner: Runner = subprocess.run,
) -> None:
    root = Path(repo_root).resolve()
    report = Path(report_dir).resolve()
    artifact = Path(artifact_dir).resolve()
    if artifact != report / "artifacts":
        raise ValueError("release artifact directory must equal report-dir/artifacts")
    if report.is_relative_to(root) or os.path.lexists(report):
        raise ValueError("release report directory must be a new external path")
    with tempfile.TemporaryDirectory(prefix="xrr-r23-release-gates-") as directory:
        scratch = Path(directory)
        for name in RELEASE_ORDER:
            subreport = report if name in {"distribution", "identity"} else scratch / name
            kwargs: dict[str, object] = {
                "repo_root": root,
                "report_dir": subreport,
                "runner": runner,
            }
            if name in {"distribution", "identity"}:
                kwargs["artifact_dir"] = artifact
            if name == "identity":
                kwargs["artifact_manifest"] = report / "artifact-manifest.json"
            run_mode(name, MODE_REGISTRY[name], **kwargs)


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


def _require_mode_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _require_distribution_arguments(args, parser)
    _require_approved_arguments(args, parser)
    _require_identity_arguments(args, parser)
    _require_release_arguments(args, parser)


def _require_distribution_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.mode == "distribution" and None in (args.report_dir, args.artifact_dir):
        parser.error("distribution requires --report-dir and --artifact-dir")


def _require_approved_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.mode == "approved-data" and None in (args.report_dir, args.approved_data_root):
        parser.error("approved-data requires --report-dir and --approved-data-root")


def _require_identity_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    required = (args.report_dir, args.artifact_dir, args.artifact_manifest)
    if args.mode == "identity" and None in required:
        parser.error("identity requires --report-dir, --artifact-dir, and --artifact-manifest")


def _require_release_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.mode == "release" and None in (args.report_dir, args.artifact_dir):
        parser.error("release requires --report-dir and --artifact-dir")


def _validate_argument_scopes(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.capture_candidate and args.mode != "approved-data":
        parser.error("--capture-candidate is only valid with approved-data")
    if args.approved_data_root is not None and args.mode != "approved-data":
        parser.error("--approved-data-root is only valid with approved-data")
    if args.artifact_dir is not None and args.mode not in {"distribution", "identity", "release"}:
        parser.error("--artifact-dir is only valid with distribution, identity, or release")
    if args.artifact_manifest is not None and args.mode != "identity":
        parser.error("--artifact-manifest is only valid with identity")


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
