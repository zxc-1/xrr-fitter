from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_distribution_mode_materializes_exact_external_artifact_directory(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    report = tmp_path / "bundle"
    artifacts = report / "artifacts"
    (root / "src").mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    module.run_mode(
        "distribution",
        module.MODE_REGISTRY["distribution"],
        repo_root=root,
        report_dir=report,
        artifact_dir=artifacts,
        runner=lambda args, **_kwargs: calls.append(tuple(args)),
    )

    distribution = next(command for command in calls if "tools/verify_distribution.py" in command)
    assert distribution[-4:] == (
        "--report-dir",
        str(report.resolve()),
        "--artifact-dir",
        str(artifacts.resolve()),
    )

    with pytest.raises(ValueError, match="artifact"):
        module.run_mode(
            "distribution",
            module.MODE_REGISTRY["distribution"],
            repo_root=root,
            report_dir=report,
            runner=lambda *_args, **_kwargs: None,
        )


def test_approved_capture_passes_only_explicit_owner_paths_to_both_workflows(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    owner = tmp_path / "owner-data"
    report = tmp_path / "approved-report"
    (root / "src").mkdir(parents=True)
    owner.mkdir()
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    module.run_mode(
        "approved-data",
        module.MODE_REGISTRY["approved-data"],
        repo_root=root,
        report_dir=report,
        approved_data_root=owner,
        capture_candidate=True,
        runner=lambda args, **kwargs: calls.append((tuple(args), kwargs)),
    )

    pytest_calls = [(args, kwargs) for args, kwargs in calls if "pytest" in args]
    assert len(pytest_calls) == 2
    assert "tests/acceptance/test_real_data_workflows.py" in pytest_calls[0][0]
    assert "tests/acceptance/test_gui_real_data_workflows.py" in pytest_calls[1][0]
    assert all(kwargs["env"]["XRR_APPROVED_DATA_ROOT"] == str(owner.resolve()) for _args, kwargs in pytest_calls)
    assert all(kwargs["env"]["XRR_APPROVED_REPORT_DIR"] == str(report.resolve()) for _args, kwargs in pytest_calls)


def test_missing_signed_approved_evidence_returns_three_without_output(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify")
    owner = tmp_path / "owner-data"
    report = tmp_path / "report"
    owner.mkdir()
    monkeypatch.setattr(module, "_repository_root", lambda: tmp_path / "repo")

    status = module.main(
        (
            "approved-data",
            "--approved-data-root",
            str(owner),
            "--report-dir",
            str(report),
        )
    )

    assert status == 3
    assert not report.exists()


def test_malformed_approved_evidence_does_not_return_missing_status(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    owner = tmp_path / "owner-data"
    report = tmp_path / "report"
    manifest = root / "verification/approved-data/manifest.json"
    owner.mkdir()
    manifest.mkdir(parents=True)
    monkeypatch.setattr(module, "_repository_root", lambda: root)

    with pytest.raises(ValueError):
        module.main(
            (
                "approved-data",
                "--approved-data-root",
                str(owner),
                "--report-dir",
                str(report),
            )
        )

    assert not report.exists()


def test_identity_materializes_one_bound_artifact_bundle(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    artifacts = bundle / "artifacts"
    manifest = bundle / "artifact-manifest.json"
    report = tmp_path / "identity"
    (root / "src").mkdir(parents=True)
    artifacts.mkdir(parents=True)
    manifest.write_bytes(b"manifest")
    calls: list[tuple[str, ...]] = []

    module.run_mode(
        "identity",
        module.MODE_REGISTRY["identity"],
        repo_root=root,
        report_dir=report,
        artifact_dir=artifacts,
        artifact_manifest=manifest,
        runner=lambda args, **_kwargs: calls.append(tuple(args)),
    )

    identity_calls = [args for args in calls if "tools/release_identity.py" in args]
    assert len(identity_calls) == 2
    assert identity_calls[0][2] == "build"
    assert identity_calls[1][2] == "validate"
    assert all(str(artifacts.resolve()) in args for args in identity_calls)
    assert all(str(manifest.resolve()) in args for args in identity_calls)


def test_release_identity_build_accepts_a_precreated_empty_report(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("release_identity")
    report = tmp_path / "identity-report"
    report.mkdir()
    artifacts = tmp_path / "bundle/artifacts"
    manifest = tmp_path / "bundle/artifact-manifest.json"
    monkeypatch.setattr(module, "calculate_release_identity", lambda *_args: object())
    monkeypatch.setattr(module, "canonical_identity_bytes", lambda _identity: b"identity\n")

    target = module.build_release_identity(tmp_path / "repo", report, artifacts, manifest)

    assert target == report / "release-identity.json"
    assert target.read_bytes() == b"identity\n"


def test_identity_rejects_a_report_bundle_inside_the_repository(
    tmp_path: Path,
    load_tool_module,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    report = root / "verification/identity-bundle"
    artifacts = report / "artifacts"
    manifest = report / "artifact-manifest.json"
    (root / "src").mkdir(parents=True)

    with pytest.raises(ValueError, match="external"):
        module.run_mode(
            "identity",
            module.MODE_REGISTRY["identity"],
            repo_root=root,
            report_dir=report,
            artifact_dir=artifacts,
            artifact_manifest=manifest,
            runner=lambda *_args, **_kwargs: None,
        )


def test_release_runs_every_software_gate_in_order_without_approved_data(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify")
    root = tmp_path / "repo"
    report = tmp_path / "release"
    artifacts = report / "artifacts"
    root.mkdir()
    calls: list[tuple[str, Path, Path | None, Path | None]] = []

    def run(name, _mode, **kwargs):
        calls.append(
            (
                name,
                Path(kwargs["report_dir"]),
                None if kwargs.get("artifact_dir") is None else Path(kwargs["artifact_dir"]),
                None if kwargs.get("artifact_manifest") is None else Path(kwargs["artifact_manifest"]),
            )
        )
        if name == "distribution":
            report.mkdir()

    monkeypatch.setattr(module, "run_mode", run)
    module.run_release(root, report, artifacts)

    assert tuple(item[0] for item in calls) == module.RELEASE_ORDER
    assert "approved-data" not in module.RELEASE_ORDER
    assert calls[-2][1:] == (report, artifacts, None)
    assert calls[-1][1:] == (
        report,
        artifacts,
        report / "artifact-manifest.json",
    )


def test_release_stops_at_first_nonzero_submode(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify")
    calls: list[str] = []

    def run(name, _mode, **_kwargs):
        calls.append(name)
        if name == "statistical":
            raise subprocess.CalledProcessError(7, (name,))

    monkeypatch.setattr(module, "run_mode", run)
    with pytest.raises(subprocess.CalledProcessError):
        module.run_release(
            tmp_path / "repo",
            tmp_path / "release",
            tmp_path / "release/artifacts",
        )
    assert tuple(calls) == module.RELEASE_ORDER[:8]


@pytest.mark.parametrize("mode", ("identity", "release"))
def test_release_modes_reject_approved_data_arguments(
    tmp_path: Path,
    load_tool_module,
    mode: str,
) -> None:
    module = load_tool_module("verify")
    arguments = [
        mode,
        "--approved-data-root",
        str(tmp_path / "owner"),
        "--report-dir",
        str(tmp_path / "report"),
        "--artifact-dir",
        str(tmp_path / "bundle/artifacts"),
    ]
    if mode == "identity":
        arguments.extend(
            (
                "--artifact-manifest",
                str(tmp_path / "bundle/artifact-manifest.json"),
            )
        )
    with pytest.raises(SystemExit):
        module.main(arguments)


def test_nonartifact_modes_reject_an_artifact_directory_in_every_entrypoint(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_tool_module("verify")
    calls: list[object] = []
    monkeypatch.setattr(module, "_run_with_report", lambda *_args: calls.append(object()))

    with pytest.raises(SystemExit):
        module.main(("quality", "--artifact-dir", str(tmp_path / "artifacts")))
    assert not calls

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    with pytest.raises(ValueError, match="artifact directory"):
        module.run_mode(
            "quality",
            module.MODE_REGISTRY["quality"],
            repo_root=root,
            report_dir=tmp_path / "report",
            artifact_dir=tmp_path / "artifacts",
            runner=lambda *_args, **_kwargs: None,
        )
