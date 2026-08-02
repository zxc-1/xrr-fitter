from __future__ import annotations

import ast
import copy
import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "verify.yml"
RUNNER = ["self-hosted", "macOS", "ARM64", "xrr-ci"]
CHECKOUT = "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683"
UPLOAD_ARTIFACT = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
DOWNLOAD_ARTIFACT = "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
JOB_TIMEOUTS = {"statistical": 720, "release": 720}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    return modules


def _payload() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _standard_run(mode: str) -> str:
    command = f'"$PYTHON" tools/verify.py {mode}'
    if mode == "gui":
        command = f"QT_QPA_PLATFORM=offscreen {command}"
    return "\n".join(
        (
            "set -euo pipefail",
            "python3.12 -c 'import platform, sys; assert sys.platform == \"darwin\" and platform.machine() == \"arm64\" and sys.version_info[:2] == (3, 12)'",
            'python3.12 -m venv "$RUNNER_TEMP/venv"',
            'PYTHON="$RUNNER_TEMP/venv/bin/python"',
            '"$PYTHON" -m pip install pip==26.1.2',
            '"$PYTHON" -m pip install -r requirements-macos-arm64-py312.lock',
            '"$PYTHON" tools/check_hygiene.py --require-git-clean',
            command,
            "",
        )
    )


def _standard_job(mode: str) -> dict[str, object]:
    return {
        "runs-on": RUNNER,
        "timeout-minutes": JOB_TIMEOUTS.get(mode, 60),
        "steps": [
            {
                "uses": CHECKOUT,
                "with": {"persist-credentials": False, "fetch-depth": 0},
            },
            {
                "name": f"Verify {mode}",
                "shell": "bash",
                "run": _standard_run(mode),
            },
        ],
    }


def _statistical_job() -> dict[str, object]:
    return {
        "if": "startsWith(github.ref, 'refs/tags/')",
        **_standard_job("statistical"),
    }


def _distribution_job() -> dict[str, object]:
    job = _standard_job("distribution")
    job["steps"][1]["run"] = _standard_run("distribution").replace(
        '"$PYTHON" tools/verify.py distribution',
        '"$PYTHON" tools/verify.py distribution '
        '--report-dir "$RUNNER_TEMP/distribution-bundle" '
        '--artifact-dir "$RUNNER_TEMP/distribution-bundle/artifacts"',
    )
    job["steps"].append(
        {
            "name": "Upload distribution bundle",
            "uses": UPLOAD_ARTIFACT,
            "with": {
                "name": "r23-distribution-${{ github.sha }}",
                "path": "${{ runner.temp }}/distribution-bundle",
                "if-no-files-found": "error",
                "retention-days": 1,
                "compression-level": 0,
            },
        }
    )
    return job


def _r22_reference_job() -> dict[str, object]:
    job = _standard_job("r22-reference")
    job["steps"][1]["run"] = _standard_run("r22-reference").replace(
        '"$PYTHON" tools/verify.py r22-reference',
        'QT_QPA_PLATFORM=offscreen "$PYTHON" tools/verify.py r22-reference '
        '--report-dir "$RUNNER_TEMP/r22-reference"',
    )
    return job


def _readiness_run() -> str:
    return "\n".join(
        (
            "set -euo pipefail",
            "python3.12 -c 'import platform, sys; assert sys.platform == \"darwin\" and platform.machine() == \"arm64\" and sys.version_info[:2] == (3, 12)'",
            'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
            "if test ! -f verification/r23/tests.json; then",
            "  printf 'ready=false\\n' >> \"$GITHUB_OUTPUT\"",
            "  exit 0",
            "fi",
            'python3.12 -m venv "$RUNNER_TEMP/venv"',
            'PYTHON="$RUNNER_TEMP/venv/bin/python"',
            '"$PYTHON" -m pip install pip==26.1.2',
            '"$PYTHON" -m pip install -r requirements-macos-arm64-py312.lock',
            '"$PYTHON" tools/check_hygiene.py --require-git-clean',
            'TEST_SOURCE_COMMIT=$("$PYTHON" -c \'import json; print(json.load(open("verification/r23/tests.json", encoding="utf-8"))["source_commit"])\')',
            'AUDIT_DIR="$RUNNER_TEMP/candidate-readiness"',
            'test ! -e "$AUDIT_DIR"',
            'mkdir "$AUDIT_DIR"',
            '"$PYTHON" tools/collect_test_manifest.py --repo-root "$GITHUB_WORKSPACE" --source-commit "$TEST_SOURCE_COMMIT" --lock-file "$GITHUB_WORKSPACE/requirements-macos-arm64-py312.lock" --suite tests --output "$AUDIT_DIR/tests.json"',
            'cmp verification/r23/tests.json "$AUDIT_DIR/tests.json"',
            'git diff --quiet "$TEST_SOURCE_COMMIT" HEAD -- tests',
            '"$PYTHON" tools/validate_test_ledger.py --phase final --active-manifest verification/r22/collections/tests-active.json --r21-manifest verification/r22/collections/tests-r21.json --target-manifest verification/r23/tests.json --ledger docs/architecture/r22-r23-test-ledger.csv',
            '"$PYTHON" tools/check_hygiene.py --require-git-clean',
            "printf 'ready=true\\n' >> \"$GITHUB_OUTPUT\"",
            "",
        )
    )


def _readiness_job() -> dict[str, object]:
    return {
        "runs-on": RUNNER,
        "timeout-minutes": 60,
        "outputs": {"ready": "${{ steps.readiness.outputs.ready }}"},
        "steps": [
            {
                "uses": CHECKOUT,
                "with": {"persist-credentials": False, "fetch-depth": 0},
            },
            {
                "name": "Evaluate candidate readiness",
                "id": "readiness",
                "shell": "bash",
                "run": _readiness_run(),
            },
        ],
    }


def _identity_job() -> dict[str, object]:
    job = _standard_job("identity")
    job["needs"] = ["candidate-readiness", "distribution"]
    job["if"] = "needs.candidate-readiness.outputs.ready == 'true'"
    job["steps"].insert(
        1,
        {
            "name": "Download distribution bundle",
            "uses": DOWNLOAD_ARTIFACT,
            "with": {
                "name": "r23-distribution-${{ github.sha }}",
                "path": "${{ runner.temp }}/downloaded-distribution",
            },
        },
    )
    job["steps"][2]["run"] = _standard_run("identity").replace(
        '"$PYTHON" tools/verify.py identity',
        '"$PYTHON" tools/verify.py identity '
        '--report-dir "$RUNNER_TEMP/identity" '
        '--artifact-dir "$RUNNER_TEMP/downloaded-distribution/artifacts" '
        '--artifact-manifest "$RUNNER_TEMP/downloaded-distribution/artifact-manifest.json"',
    )
    return job


def _release_job() -> dict[str, object]:
    job = _standard_job("release")
    job["timeout-minutes"] = 720
    job["needs"] = ["candidate-readiness"]
    job["if"] = "needs.candidate-readiness.outputs.ready == 'true'"
    job["steps"][1]["run"] = _standard_run("release").replace(
        '"$PYTHON" tools/verify.py release',
        'QT_QPA_PLATFORM=offscreen "$PYTHON" tools/verify.py release '
        '--report-dir "$RUNNER_TEMP/release" '
        '--artifact-dir "$RUNNER_TEMP/release/artifacts"',
    )
    job["steps"].append(
        {
            "name": "Upload canonical release bundle",
            "uses": UPLOAD_ARTIFACT,
            "with": {
                "name": "r23-release-${{ github.sha }}",
                "path": "${{ runner.temp }}/release",
                "if-no-files-found": "error",
                "retention-days": 1,
                "compression-level": 0,
            },
        }
    )
    return job


def _checkpoint_job() -> dict[str, object]:
    return {
        "needs": [
            "quality",
            "tools",
            "unit",
            "gui",
            "integration",
            "spawn",
            "regression",
            "statistical",
            "r22-reference",
            "distribution",
            "candidate-readiness",
            "identity",
            "release",
        ],
        "if": "always()",
        "runs-on": RUNNER,
        "timeout-minutes": 10,
        "steps": [
            {
                "name": "Require all gates",
                "env": {
                    "QUALITY_RESULT": "${{ needs.quality.result }}",
                    "TOOLS_RESULT": "${{ needs.tools.result }}",
                    "UNIT_RESULT": "${{ needs.unit.result }}",
                    "GUI_RESULT": "${{ needs.gui.result }}",
                    "INTEGRATION_RESULT": "${{ needs.integration.result }}",
                    "SPAWN_RESULT": "${{ needs.spawn.result }}",
                    "REGRESSION_RESULT": "${{ needs.regression.result }}",
                    "STATISTICAL_RESULT": "${{ needs.statistical.result }}",
                    "R22_REFERENCE_RESULT": "${{ needs.r22-reference.result }}",
                    "DISTRIBUTION_RESULT": "${{ needs.distribution.result }}",
                    "READINESS_RESULT": "${{ needs.candidate-readiness.result }}",
                    "READY": "${{ needs.candidate-readiness.outputs.ready }}",
                    "IDENTITY_RESULT": "${{ needs.identity.result }}",
                    "RELEASE_RESULT": "${{ needs.release.result }}",
                    "REF": "${{ github.ref }}",
                },
                "shell": "bash",
                "run": (
                    "set -euo pipefail\n"
                    'test "$QUALITY_RESULT" = success\n'
                    'test "$TOOLS_RESULT" = success\n'
                    'test "$UNIT_RESULT" = success\n'
                    'test "$GUI_RESULT" = success\n'
                    'test "$INTEGRATION_RESULT" = success\n'
                    'test "$SPAWN_RESULT" = success\n'
                    'test "$REGRESSION_RESULT" = success\n'
                    'case "$REF" in\n'
                    '  refs/tags/*) test "$STATISTICAL_RESULT" = success ;;\n'
                    '  *) test "$STATISTICAL_RESULT" = skipped ;;\n'
                    'esac\n'
                    'test "$R22_REFERENCE_RESULT" = success\n'
                    'test "$DISTRIBUTION_RESULT" = success\n'
                    'test "$READINESS_RESULT" = success\n'
                    'case "$READY" in\n'
                    '  true)\n'
                    '    test "$IDENTITY_RESULT" = success\n'
                    '    test "$RELEASE_RESULT" = success\n'
                    '    ;;\n'
                    '  false)\n'
                    '    test "$IDENTITY_RESULT" = skipped\n'
                    '    test "$RELEASE_RESULT" = skipped\n'
                    '    ;;\n'
                    '  *) exit 1 ;;\n'
                    'esac\n'
                ),
            }
        ],
    }


def _expected_workflow() -> dict[str, object]:
    return {
        "name": "verify",
        "on": {
            "push": {
                "branches": ["main"],
                "tags": ["R23-final"],
            }
        },
        "permissions": {"contents": "read"},
        "concurrency": {
            "group": "verify-${{ github.ref }}-${{ github.sha }}",
            "cancel-in-progress": False,
        },
        "jobs": {
            "quality": _standard_job("quality"),
            "tools": _standard_job("tools"),
            "unit": _standard_job("unit"),
            "gui": _standard_job("gui"),
            "integration": _standard_job("integration"),
            "spawn": _standard_job("spawn"),
            "regression": _standard_job("regression"),
            "statistical": _statistical_job(),
            "r22-reference": _r22_reference_job(),
            "distribution": _distribution_job(),
            "candidate-readiness": _readiness_job(),
            "identity": _identity_job(),
            "release": _release_job(),
            "checkpoint": _checkpoint_job(),
        },
    }


def _assert_exact_workflow(payload: dict[str, object]) -> None:
    assert payload == _expected_workflow()


def test_initial_workflow_has_exact_jobs_permissions_and_trigger() -> None:
    payload = _payload()
    assert payload["permissions"] == {"contents": "read"}
    assert payload["on"] == {
        "push": {
            "branches": ["main"],
            "tags": ["R23-final"],
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
        "r22-reference",
        "distribution",
        "candidate-readiness",
        "identity",
        "release",
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
        "r22-reference",
        "distribution",
        "candidate-readiness",
        "identity",
        "release",
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
        "R22_REFERENCE_RESULT": "${{ needs.r22-reference.result }}",
        "DISTRIBUTION_RESULT": "${{ needs.distribution.result }}",
        "READINESS_RESULT": "${{ needs.candidate-readiness.result }}",
        "READY": "${{ needs.candidate-readiness.outputs.ready }}",
        "IDENTITY_RESULT": "${{ needs.identity.result }}",
        "RELEASE_RESULT": "${{ needs.release.result }}",
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
        "  refs/tags/*) test \"$STATISTICAL_RESULT\" = success ;;",
        "  *) test \"$STATISTICAL_RESULT\" = skipped ;;",
        "esac",
        'test "$R22_REFERENCE_RESULT" = success',
        'test "$DISTRIBUTION_RESULT" = success',
        'test "$READINESS_RESULT" = success',
        'case "$READY" in',
        "  true)",
        '    test "$IDENTITY_RESULT" = success',
        '    test "$RELEASE_RESULT" = success',
        "    ;;",
        "  false)",
        '    test "$IDENTITY_RESULT" = skipped',
        '    test "$RELEASE_RESULT" = skipped',
        "    ;;",
        "  *) exit 1 ;;",
        "esac",
    ]


def test_checkpoint_step_requires_r22_reference_gate() -> None:
    step = _payload()["jobs"]["checkpoint"]["steps"][0]
    assert 'test "$R22_REFERENCE_RESULT" = success' in step["run"].splitlines()


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
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])


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
    checkout = [
        step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkout) == 1
    assert checkout[0]["with"] == {"persist-credentials": False, "fetch-depth": 0}
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert f"tools/verify.py {name}" in commands
    assert "tools/check_hygiene.py --require-git-clean" in commands


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
        "statistical",
        "r22-reference",
        "distribution",
        "identity",
        "release",
    ):
        _assert_standard_job(name, jobs[name])


def test_r22_reference_job_uses_offscreen_qt_and_external_report_directory() -> None:
    job = _payload()["jobs"]["r22-reference"]
    commands = job["steps"][1]["run"].splitlines()
    assert (
        'QT_QPA_PLATFORM=offscreen "$PYTHON" tools/verify.py r22-reference '
        '--report-dir "$RUNNER_TEMP/r22-reference"'
    ) in commands


def test_release_job_runs_nested_gui_gates_offscreen() -> None:
    commands = _payload()["jobs"]["release"]["steps"][1]["run"].splitlines()
    assert (
        'QT_QPA_PLATFORM=offscreen "$PYTHON" tools/verify.py release '
        '--report-dir "$RUNNER_TEMP/release" '
        '--artifact-dir "$RUNNER_TEMP/release/artifacts"'
    ) in commands


def test_candidate_readiness_is_static_and_owner_data_independent() -> None:
    job = _payload()["jobs"]["candidate-readiness"]
    assert job["outputs"] == {"ready": "${{ steps.readiness.outputs.ready }}"}
    step = job["steps"][1]
    assert step["id"] == "readiness"
    commands = step["run"]
    assert "verification/r23/tests.json" in commands
    assert "tools/collect_test_manifest.py" in commands
    assert "tools/validate_test_ledger.py --phase final" in commands
    assert "ready=false" in commands
    assert "ready=true" in commands
    assert "approved-data" not in commands
    assert "XRR_APPROVED_DATA_ROOT" not in commands


def test_release_jobs_are_readiness_gated_and_use_exact_bundles() -> None:
    jobs = _payload()["jobs"]
    identity = jobs["identity"]
    release = jobs["release"]
    assert identity["needs"] == ["candidate-readiness", "distribution"]
    assert release["needs"] == ["candidate-readiness"]
    expected_condition = "needs.candidate-readiness.outputs.ready == 'true'"
    assert identity["if"] == expected_condition
    assert release["if"] == expected_condition

    distribution_upload = jobs["distribution"]["steps"][-1]
    identity_download = identity["steps"][1]
    release_upload = release["steps"][-1]
    assert distribution_upload == {
        "name": "Upload distribution bundle",
        "uses": UPLOAD_ARTIFACT,
        "with": {
            "name": "r23-distribution-${{ github.sha }}",
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
            "name": "r23-distribution-${{ github.sha }}",
            "path": "${{ runner.temp }}/downloaded-distribution",
        },
    }
    assert release_upload == {
        "name": "Upload canonical release bundle",
        "uses": UPLOAD_ARTIFACT,
        "with": {
            "name": "r23-release-${{ github.sha }}",
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
            "run", "set -euo pipefail\nexit 0\ntest \"$QUALITY_RESULT\" = success\ntest \"$TOOLS_RESULT\" = success\n"
        ),
        lambda payload: payload["jobs"]["quality"]["steps"][1].__setitem__(
            "run", payload["jobs"]["quality"]["steps"][1]["run"] + " || true"
        ),
        lambda payload: payload["jobs"]["tools"].__setitem__("continue-on-error", True),
        lambda payload: payload["jobs"]["unit"].__setitem__("if", "false"),
        lambda payload: payload["jobs"]["gui"].__setitem__("if", "false"),
        lambda payload: payload["jobs"]["integration"].__setitem__("if", "false"),
        lambda payload: payload["jobs"]["spawn"].__setitem__("continue-on-error", True),
        lambda payload: payload["jobs"]["statistical"].__setitem__(
            "continue-on-error", True
        ),
        lambda payload: payload["jobs"]["r22-reference"].__setitem__(
            "continue-on-error", True
        ),
        lambda payload: payload["jobs"]["quality"].__setitem__(
            "permissions", {"contents": "write"}
        ),
        lambda payload: payload["jobs"]["quality"]["steps"].append(
            {"uses": "actions/cache@" + "a" * 40}
        ),
        lambda payload: payload["jobs"]["tools"].__setitem__("if", "false"),
        lambda payload: payload["jobs"]["distribution"].__setitem__(
            "continue-on-error", True
        ),
        lambda payload: payload["jobs"]["candidate-readiness"]["steps"][1].__setitem__(
            "run", "printf 'ready=true\\n' >> \"$GITHUB_OUTPUT\"\n"
        ),
        lambda payload: payload["jobs"]["identity"].__setitem__("if", "true"),
        lambda payload: payload["jobs"]["release"].__setitem__(
            "continue-on-error", True
        ),
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
    commands = "\n".join(
        step.get("run", "")
        for job in payload["jobs"].values()
        for step in job.get("steps", [])
    )
    assert "approved-data" not in payload["jobs"]
    assert {"identity", "release"} < set(payload["jobs"])
    assert "--approved-data-root" not in commands
    assert "--capture-candidate" not in commands
