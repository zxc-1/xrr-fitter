"""Immutable command registry for repository verification modes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

PYTHON = "{python}"
REPORT = "{report}"
ROOT = "{root}"
ARTIFACT = "{artifact}"
ARTIFACT_MANIFEST = "{artifact_manifest}"


@dataclass(frozen=True)
class Mode:
    commands: tuple[tuple[str, ...], ...]


PYTEST_PREFIX = (
    PYTHON,
    "-m",
    "pytest",
    "-o",
    "addopts=",
    "--import-mode=importlib",
    "--strict-config",
    "--strict-markers",
    "-p",
    "no:cacheprovider",
    "-p",
    "tests.outcome_gate",
    "--basetemp",
    f"{REPORT}/pytest-tmp",
)

MODE_REGISTRY: Mapping[str, Mode] = {
    "quality": Mode(
        (
            (PYTHON, "tools/check_radon.py", "--output", f"{REPORT}/radon.json"),
            PYTEST_PREFIX
            + (
                "tests/architecture/test_dependency_rules.py",
                "tests/architecture/test_naming_rules.py",
                "tests/architecture/test_public_api.py",
                "tests/architecture/test_distribution.py",
                "tests/architecture/test_windows_executable_workflow.py",
                "tests/architecture/test_quality_gate.py",
                "tests/architecture/test_removed_legacy_modules.py",
                "-q",
            ),
        )
    ),
    "tools": Mode((PYTEST_PREFIX + ("tests/unit/tools", "-q"),)),
    "unit": Mode(
        (
            PYTEST_PREFIX
            + (
                "tests/unit/model",
                "tests/unit/io",
                "tests/unit/physics",
                "tests/unit/test_evaluation.py",
                "tests/unit/fit",
                "tests/unit/analysis",
                "tests/unit/services",
                "-q",
            ),
        )
    ),
    "gui": Mode((PYTEST_PREFIX + ("tests/gui", "-q"),)),
    "integration": Mode(
        (
            PYTEST_PREFIX
            + (
                "tests/integration/test_entrypoints.py",
                "tests/integration/test_project_roundtrip.py",
                "tests/integration/test_single_fit_workflow.py",
                "tests/integration/test_joint_fit_workflow.py",
                "tests/integration/test_batch_resume.py",
                "tests/integration/test_export_workflow.py",
                "tests/integration/test_gui_project_workflow.py",
                "tests/integration/test_gui_automatic_workflow.py",
                "-q",
            ),
        )
    ),
    "spawn": Mode((PYTEST_PREFIX + ("tests/integration/test_process_workers.py", "-q"),)),
    "regression": Mode(
        (
            PYTEST_PREFIX
            + (
                "tests/regression/test_numerical_reference.py",
                "tests/regression/test_recovery_metrics.py",
                "tests/regression/test_profile_basin_regressions.py",
                "tests/regression/test_automatic_recovery.py",
                "-q",
            ),
        )
    ),
    "statistical": Mode(
        (PYTEST_PREFIX + ("tests/acceptance/test_synthetic_recovery_corpus.py", "-q"),)
    ),
    "approved-data": Mode(
        (
            PYTEST_PREFIX + ("tests/acceptance/test_real_data_workflows.py", "-q"),
            PYTEST_PREFIX + ("tests/acceptance/test_gui_real_data_workflows.py", "-q"),
        )
    ),
    "distribution": Mode(
        (
            (
                PYTHON,
                "tools/verify_distribution.py",
                "--repo-root",
                ROOT,
                "--report-dir",
                REPORT,
                "--artifact-dir",
                ARTIFACT,
            ),
        )
    ),
    "identity": Mode(
        (
            (
                PYTHON,
                "tools/release_identity.py",
                "build",
                "--repo-root",
                ROOT,
                "--report-dir",
                REPORT,
                "--artifact-dir",
                ARTIFACT,
                "--artifact-manifest",
                ARTIFACT_MANIFEST,
            ),
            (
                PYTHON,
                "tools/release_identity.py",
                "validate",
                "--repo-root",
                ROOT,
                "--release-identity",
                f"{REPORT}/release-identity.json",
                "--artifact-dir",
                ARTIFACT,
                "--artifact-manifest",
                ARTIFACT_MANIFEST,
            ),
        )
    ),
    "release": Mode(()),
}

RELEASE_ORDER = (
    "quality",
    "tools",
    "unit",
    "integration",
    "gui",
    "spawn",
    "regression",
    "statistical",
    "distribution",
    "identity",
)
