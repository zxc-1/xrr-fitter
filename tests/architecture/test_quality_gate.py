from __future__ import annotations

import ast
import copy
import re
from pathlib import Path

import pytest
import yaml
from tests.support.release_workflow_contract import (
    DOWNLOAD_ARTIFACT,
)
from tests.support.verify_workflow_contract import (
    JOB_TIMEOUTS,
    PYTHON_ENV,
    RUNNER,
    SETUP_MACOS_PYTHON,
    UPLOAD_ARTIFACT,
    expected_workflow,
    setup_step,
)

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "verify.yml"
SETUP_ACTION = ROOT / ".github" / "actions" / "setup-macos-python" / "action.yml"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    modules.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    return modules


def _payload() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _named_step(job: dict[str, object], name: str) -> dict[str, object]:
    steps = [step for step in job["steps"] if step.get("name") == name]
    assert len(steps) == 1, f"expected one step named {name!r}"
    return steps[0]


def _setup_action() -> dict[str, object]:
    assert SETUP_ACTION.is_file(), "missing shared macOS environment action"
    return yaml.safe_load(SETUP_ACTION.read_text(encoding="utf-8"))


def _assert_exact_workflow(payload: dict[str, object]) -> None:
    assert payload == expected_workflow()


def test_initial_workflow_has_exact_jobs_permissions_and_trigger() -> None:
    payload = _payload()
    assert payload["permissions"] == {"contents": "read"}
    assert payload["on"] == {
        "push": {
            "branches": ["main"],
            "tags": ["v*"],
        }
    }
    assert set(payload["jobs"]) == {
        "quality",
        "tools",
        "unit",
        "gui",
        "integration",
        "spawn",
        "regression",
        "statistical",
        "distribution",
        "candidate-readiness",
        "identity",
        "release",
        "windows",
        "draft-release",
        "checkpoint",
    }


def test_checkpoint_job_contract() -> None:
    payload = _payload()
    checkpoint = payload["jobs"]["checkpoint"]
    assert checkpoint["needs"] == [
        "quality",
        "tools",
        "unit",
        "gui",
        "integration",
        "spawn",
        "regression",
        "statistical",
        "distribution",
        "candidate-readiness",
        "identity",
        "release",
        "windows",
        "draft-release",
    ]
    assert checkpoint["if"] == "always()"
    assert checkpoint["runs-on"] == RUNNER
    assert checkpoint["timeout-minutes"] == 10
    assert len(checkpoint["steps"]) == 1


def test_checkpoint_step_requires_every_gate() -> None:
    checkpoint = _payload()["jobs"]["checkpoint"]
    step = checkpoint["steps"][0]
    assert step["env"] == {
        "QUALITY_RESULT": "${{ needs.quality.result }}",
        "TOOLS_RESULT": "${{ needs.tools.result }}",
        "UNIT_RESULT": "${{ needs.unit.result }}",
        "GUI_RESULT": "${{ needs.gui.result }}",
        "INTEGRATION_RESULT": "${{ needs.integration.result }}",
        "SPAWN_RESULT": "${{ needs.spawn.result }}",
        "REGRESSION_RESULT": "${{ needs.regression.result }}",
        "STATISTICAL_RESULT": "${{ needs.statistical.result }}",
        "DISTRIBUTION_RESULT": "${{ needs.distribution.result }}",
        "READINESS_RESULT": "${{ needs.candidate-readiness.result }}",
        "READY": "${{ needs.candidate-readiness.outputs.ready }}",
        "IDENTITY_RESULT": "${{ needs.identity.result }}",
        "RELEASE_RESULT": "${{ needs.release.result }}",
        "WINDOWS_RESULT": "${{ needs.windows.result }}",
        "DRAFT_RELEASE_RESULT": "${{ needs.draft-release.result }}",
        "REF": "${{ github.ref }}",
    }
    assert step["run"].splitlines() == [
        "set -euo pipefail",
        'test "$QUALITY_RESULT" = success',
        'test "$TOOLS_RESULT" = success',
        'test "$UNIT_RESULT" = success',
        'test "$GUI_RESULT" = success',
        'test "$INTEGRATION_RESULT" = success',
        'test "$SPAWN_RESULT" = success',
        'test "$REGRESSION_RESULT" = success',
        'case "$REF" in',
        '  refs/tags/*) test "$STATISTICAL_RESULT" = success ;;',
        '  *) test "$STATISTICAL_RESULT" = skipped ;;',
        "esac",
        'test "$DISTRIBUTION_RESULT" = success',
        'case "$REF" in',
        '  refs/tags/*) test "$READINESS_RESULT" = success ;;',
        '  *) test "$READINESS_RESULT" = skipped ;;',
        "esac",
        'case "$REF" in',
        "  refs/tags/*)",
        '    test "$READY" = true',
        '    test "$IDENTITY_RESULT" = success',
        '    test "$RELEASE_RESULT" = success',
        '    test "$WINDOWS_RESULT" = success',
        '    test "$DRAFT_RELEASE_RESULT" = success',
        "    ;;",
        "  *)",
        '    test "$IDENTITY_RESULT" = skipped',
        '    test "$RELEASE_RESULT" = skipped',
        '    test "$WINDOWS_RESULT" = skipped',
        '    test "$DRAFT_RELEASE_RESULT" = skipped',
        "    ;;",
        "esac",
    ]


def test_concurrency_never_cancels_an_exact_sha_run() -> None:
    payload = _payload()
    assert payload["concurrency"]["cancel-in-progress"] is False


def _action_steps(payload: dict[str, object]):
    for job in payload["jobs"].values():
        for step in job.get("steps", []):
            if "uses" in step:
                yield step


def test_actions_are_commit_pinned_and_checkout_drops_credentials() -> None:
    steps = tuple(_action_steps(_payload()))
    assert steps
    for step in steps:
        assert step["uses"] in {SETUP_MACOS_PYTHON, "./.github/actions/cleanup-macos-python"} or re.fullmatch(
            r"[^@]+@[0-9a-f]{40}", step["uses"]
        )


def test_release_tool_consumers_use_the_declared_public_owners() -> None:
    verify_imports = _imported_modules(ROOT / "tools/verify.py")
    assert "freeze_approved_data" in verify_imports
    assert "approved_data_evidence" not in verify_imports

    identity_paths = (
        ROOT / "tools/release_identity.py",
        ROOT / "tools/release_identity_model.py",
        ROOT / "tools/release_identity_schema.py",
    )
    identity_imports = set().union(*(_imported_modules(path) for path in identity_paths))
    assert "verify_distribution" in identity_imports
    assert identity_imports.isdisjoint({"distribution_manifest", "distribution_source"})


def _assert_standard_job(name: str, job: dict[str, object]) -> None:
    assert job["runs-on"] == RUNNER
    assert job["timeout-minutes"] == JOB_TIMEOUTS.get(name, 60)
    checkout = [step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")]
    assert len(checkout) == 1
    assert checkout[0]["with"] == {"persist-credentials": False, "fetch-depth": 0}
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert f"tools/verify.py {name}" in commands
    assert setup_step() in job["steps"]


def test_standard_jobs_use_required_runner_and_explicit_verifier_modes() -> None:
    jobs = _payload()["jobs"]
    for name in (
        "quality",
        "tools",
        "unit",
        "gui",
        "integration",
        "spawn",
        "regression",
        "distribution",
        "identity",
        "release",
    ):
        _assert_standard_job(name, jobs[name])
    run = next(step["run"] for step in _setup_action()["runs"]["steps"] if step.get("id") == "environment")
    assert "tools/check_hygiene.py --require-git-clean" in run


def test_standard_jobs_verify_locked_environment_metadata() -> None:
    jobs = _payload()["jobs"]
    for name in (
        "quality",
        "tools",
        "unit",
        "gui",
        "integration",
        "spawn",
        "regression",
        "distribution",
        "identity",
        "release",
    ):
        assert setup_step() in jobs[name]["steps"]
        command = next(step for step in jobs[name]["steps"] if step.get("name") == f"Verify {name}")
        assert command["env"] == PYTHON_ENV
    run = next(step["run"] for step in _setup_action()["runs"]["steps"] if step.get("id") == "environment")
    assert '"$PYTHON" -m pip check' in run


def test_release_job_runs_nested_gui_gates_offscreen() -> None:
    commands = next(
        step["run"] for step in _payload()["jobs"]["release"]["steps"] if step.get("name") == "Verify release"
    ).splitlines()
    assert (
        'QT_QPA_PLATFORM=offscreen "$PYTHON" tools/verify.py release '
        '--report-dir "$RUNNER_TEMP/release" '
        '--artifact-dir "$RUNNER_TEMP/release/artifacts" '
        '--statistical-results "$RUNNER_TEMP/statistical-inputs"'
    ) in commands


def test_candidate_readiness_is_static_and_owner_data_independent() -> None:
    job = _payload()["jobs"]["candidate-readiness"]
    assert job["outputs"] == {"ready": "${{ steps.readiness.outputs.ready || steps.readiness_inputs.outputs.ready }}"}
    _named_step(job, "Validate release version tag")
    step = _named_step(job, "Evaluate candidate readiness")
    assert step["id"] == "readiness"
    commands = step["run"]
    assert "verification/r23/tests.json" in commands
    assert "tools/collect_test_manifest.py" in commands
    assert "ready=false" in _named_step(job, "Check candidate readiness inputs")["run"]
    assert "ready=true" in commands
    assert "approved-data" not in commands
    assert "XRR_APPROVED_DATA_ROOT" not in commands


def test_release_jobs_are_readiness_gated_and_use_exact_bundles() -> None:
    jobs = _payload()["jobs"]
    identity = jobs["identity"]
    release = jobs["release"]
    assert identity["needs"] == ["candidate-readiness", "distribution"]
    assert release["needs"] == ["candidate-readiness", "statistical"]
    expected_condition = "startsWith(github.ref, 'refs/tags/') && needs.candidate-readiness.outputs.ready == 'true'"
    assert identity["if"] == expected_condition
    assert release["if"] == expected_condition

    distribution_upload = jobs["distribution"]["steps"][-1]
    identity_download = identity["steps"][1]
    release_upload = release["steps"][-1]
    assert distribution_upload == {
        "name": "Upload distribution bundle",
        "uses": UPLOAD_ARTIFACT,
        "with": {
            "name": "xrr-distribution-${{ github.sha }}",
            "path": "${{ runner.temp }}/distribution-bundle",
            "if-no-files-found": "error",
            "retention-days": 1,
            "compression-level": 0,
        },
    }
    assert identity_download == {
        "name": "Download distribution bundle",
        "uses": DOWNLOAD_ARTIFACT,
        "with": {
            "name": "xrr-distribution-${{ github.sha }}",
            "path": "${{ runner.temp }}/downloaded-distribution",
        },
    }
    assert release_upload == {
        "name": "Upload canonical release bundle",
        "uses": UPLOAD_ARTIFACT,
        "with": {
            "name": "xrr-release-${{ github.ref_name }}-${{ github.sha }}",
            "path": "${{ runner.temp }}/release",
            "if-no-files-found": "error",
            "retention-days": 1,
            "compression-level": 0,
        },
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["jobs"]["checkpoint"]["steps"][0].__setitem__(
            "run", 'set -euo pipefail\nexit 0\ntest "$QUALITY_RESULT" = success\ntest "$TOOLS_RESULT" = success\n'
        ),
        lambda payload: _named_step(payload["jobs"]["quality"], "Verify quality").__setitem__(
            "run", _named_step(payload["jobs"]["quality"], "Verify quality")["run"] + " || true"
        ),
        lambda payload: payload["jobs"]["tools"].__setitem__("continue-on-error", True),
        lambda payload: payload["jobs"]["unit"].__setitem__("if", "false"),
        lambda payload: payload["jobs"]["gui"].__setitem__("if", "false"),
        lambda payload: payload["jobs"]["integration"].__setitem__("if", "false"),
        lambda payload: payload["jobs"]["spawn"].__setitem__("continue-on-error", True),
        lambda payload: payload["jobs"]["statistical"].__setitem__("continue-on-error", True),
        lambda payload: payload["jobs"]["quality"].__setitem__("permissions", {"contents": "write"}),
        lambda payload: payload["jobs"]["quality"]["steps"].append({"uses": "actions/cache@" + "a" * 40}),
        lambda payload: payload["jobs"]["tools"].__setitem__("if", "false"),
        lambda payload: payload["jobs"]["distribution"].__setitem__("continue-on-error", True),
        lambda payload: _named_step(payload["jobs"]["candidate-readiness"], "Evaluate candidate readiness").__setitem__(
            "run", "printf 'ready=true\\n' >> \"$GITHUB_OUTPUT\"\n"
        ),
        lambda payload: payload["jobs"]["identity"].__setitem__("if", "true"),
        lambda payload: payload["jobs"]["release"].__setitem__("continue-on-error", True),
    ],
)
def test_exact_workflow_contract_rejects_success_bypasses(mutate) -> None:
    payload = copy.deepcopy(_payload())
    mutate(payload)
    with pytest.raises(AssertionError):
        _assert_exact_workflow(payload)


def test_exact_workflow_contract_accepts_committed_workflow() -> None:
    _assert_exact_workflow(_payload())


def test_software_delivery_workflow_never_requires_owner_data() -> None:
    payload = _payload()
    commands = "\n".join(step.get("run", "") for job in payload["jobs"].values() for step in job.get("steps", []))
    assert "approved-data" not in payload["jobs"]
    assert {"identity", "release"} < set(payload["jobs"])
    assert "--approved-data-root" not in commands
    assert "--capture-candidate" not in commands
