from __future__ import annotations


def _commands(module) -> tuple[tuple[str, ...], ...]:
    return tuple(
        command
        for mode in module.MODE_REGISTRY.values()
        for command in mode.commands
    )


def _expected_registry(module) -> dict[str, tuple[tuple[str, ...], ...]]:
    pytest_prefix = module.PYTEST_PREFIX
    return {
        "quality": (
            (
                module.PYTHON,
                "tools/check_radon.py",
                "--output",
                f"{module.REPORT}/radon.json",
            ),
            pytest_prefix
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
        ),
        "tools": (pytest_prefix + ("tests/unit/tools", "-q"),),
        "unit": (
            pytest_prefix
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
        ),
        "gui": (pytest_prefix + ("tests/gui", "-q"),),
        "integration": (
            pytest_prefix
            + (
                "tests/integration/test_entrypoints.py",
                "tests/integration/test_project_roundtrip.py",
                "tests/integration/test_single_fit_workflow.py",
                "tests/integration/test_joint_fit_workflow.py",
                "tests/integration/test_batch_resume.py",
                "tests/integration/test_export_workflow.py",
                "tests/integration/test_gui_project_workflow.py",
                "-q",
            ),
        ),
        "spawn": (
            pytest_prefix + ("tests/integration/test_process_workers.py", "-q"),
        ),
        "regression": (
            pytest_prefix
            + (
                "tests/regression/test_numerical_reference.py",
                "tests/regression/test_recovery_metrics.py",
                "tests/regression/test_profile_basin_regressions.py",
                "-q",
            ),
        ),
        "statistical": (
            pytest_prefix
            + ("tests/acceptance/test_synthetic_recovery_corpus.py", "-q"),
        ),
        "r22-reference": (
            pytest_prefix
            + ("tests/acceptance/test_r22_reference_equivalence.py", "-q"),
        ),
        "approved-data": (
            pytest_prefix
            + ("tests/acceptance/test_real_data_workflows.py", "-q"),
            pytest_prefix
            + ("tests/acceptance/test_gui_real_data_workflows.py", "-q"),
        ),
        "distribution": (
            (
                module.PYTHON,
                "tools/verify_distribution.py",
                "--repo-root",
                module.ROOT,
                "--report-dir",
                module.REPORT,
                "--artifact-dir",
                module.ARTIFACT,
            ),
        ),
        "identity": (
            (
                module.PYTHON,
                "tools/release_identity.py",
                "build",
                "--repo-root",
                module.ROOT,
                "--report-dir",
                module.REPORT,
                "--artifact-dir",
                module.ARTIFACT,
                "--artifact-manifest",
                module.ARTIFACT_MANIFEST,
            ),
            (
                module.PYTHON,
                "tools/release_identity.py",
                "validate",
                "--repo-root",
                module.ROOT,
                "--release-identity",
                f"{module.REPORT}/release-identity.json",
                "--artifact-dir",
                module.ARTIFACT,
                "--artifact-manifest",
                module.ARTIFACT_MANIFEST,
            ),
        ),
        "release": (),
    }


def test_registry_is_exact_for_completed_suites(load_tool_module) -> None:
    module = load_tool_module("verify")
    observed = {
        name: mode.commands for name, mode in module.MODE_REGISTRY.items()
    }
    assert observed == _expected_registry(module)
    assert module.RELEASE_ORDER == (
        "quality",
        "tools",
        "unit",
        "integration",
        "gui",
        "spawn",
        "regression",
        "statistical",
        "r22-reference",
        "distribution",
        "identity",
    )


def test_registry_uses_nonempty_argument_vectors(load_tool_module) -> None:
    module = load_tool_module("verify")
    for mode in module.MODE_REGISTRY.values():
        assert all(isinstance(command, tuple) and command for command in mode.commands)
        assert all(not isinstance(command, str) for command in mode.commands)


def test_every_pytest_command_uses_outcome_plugin(load_tool_module) -> None:
    module = load_tool_module("verify")
    pytest_commands = [command for command in _commands(module) if "pytest" in command]
    assert pytest_commands
    assert all("tests.outcome_gate" in command for command in pytest_commands)


def test_every_pytest_command_uses_importlib_mode(load_tool_module) -> None:
    module = load_tool_module("verify")
    pytest_commands = [command for command in _commands(module) if "pytest" in command]
    assert pytest_commands
    assert all("--import-mode=importlib" in command for command in pytest_commands)


def test_pytest_prefix_declares_importlib_mode_once(load_tool_module) -> None:
    module = load_tool_module("verify")
    assert module.PYTEST_PREFIX.count("--import-mode=importlib") == 1
