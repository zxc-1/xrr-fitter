"""Exact main/tag workflow expectations, independent of the committed YAML."""

from __future__ import annotations

from tests.support.macos_action_contract import cleanup_step
from tests.support.release_workflow_contract import (
    CHECKOUT,
    DOWNLOAD_ARTIFACT,
    expected_draft_release_job,
    expected_windows_job,
)

RUNNER = "macos-15"
SETUP_MACOS_PYTHON = "./.github/actions/setup-macos-python"
PYTHON_ENV = {"PYTHON": "${{ steps.python.outputs.python }}"}
UPLOAD_ARTIFACT = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
JOB_TIMEOUTS = {"statistical": 360, "release": 360}


def setup_step(if_condition: str | None = None) -> dict[str, object]:
    step = {
        "name": "Set up locked macOS Python",
        "id": "python",
        "uses": SETUP_MACOS_PYTHON,
    }
    if if_condition is not None:
        step["if"] = if_condition
    return step


def _gate_run(*commands: str) -> str:
    return "\n".join(
        (
            "set -euo pipefail",
            'test "$PYTHON" = "${{ steps.python.outputs.python }}"',
            *commands,
            "",
        )
    )


def _standard_run(mode: str) -> str:
    command = f'"$PYTHON" tools/verify.py {mode}'
    if mode == "gui":
        command = f"QT_QPA_PLATFORM=offscreen {command}"
    return _gate_run(command)


def _standard_job(mode: str) -> dict[str, object]:
    return {
        "runs-on": RUNNER,
        "timeout-minutes": JOB_TIMEOUTS.get(mode, 60),
        "steps": [
            {
                "uses": CHECKOUT,
                "with": {"persist-credentials": False, "fetch-depth": 0},
            },
            setup_step(),
            {
                "name": f"Verify {mode}",
                "env": PYTHON_ENV,
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
    job["steps"][-1]["run"] = _standard_run("distribution").replace(
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
                "name": "xrr-distribution-${{ github.sha }}",
                "path": "${{ runner.temp }}/distribution-bundle",
                "if-no-files-found": "error",
                "retention-days": 1,
                "compression-level": 0,
            },
        }
    )
    return job


def _readiness_inputs_run() -> str:
    return "\n".join(
        (
            "set -euo pipefail",
            'python3.12 -c \'import platform, sys; assert sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3, 12)\'',
            'test -z "$(git status --porcelain=v1 --untracked-files=all)"',
            "if test ! -f verification/r23/tests.json; then",
            "  printf 'ready=false\\n' >> \"$GITHUB_OUTPUT\"",
            "  printf 'has_tests_manifest=false\\n' >> \"$GITHUB_OUTPUT\"",
            "  exit 0",
            "fi",
            "printf 'has_tests_manifest=true\\n' >> \"$GITHUB_OUTPUT\"",
            "",
        )
    )


def _readiness_run() -> str:
    return _gate_run(
        'TEST_SOURCE_COMMIT=$("$PYTHON" -c \'import json; print(json.load(open("verification/r23/tests.json", encoding="utf-8"))["source_commit"])\')',
        'AUDIT_DIR="$RUNNER_TEMP/candidate-readiness"',
        'test ! -e "$AUDIT_DIR"',
        'mkdir "$AUDIT_DIR"',
        '"$PYTHON" tools/collect_test_manifest.py --repo-root "$GITHUB_WORKSPACE" --source-commit "$TEST_SOURCE_COMMIT" --lock-file "$GITHUB_WORKSPACE/requirements-macos-arm64-py312.lock" --suite tests --output "$AUDIT_DIR/tests.json"',
        'cmp verification/r23/tests.json "$AUDIT_DIR/tests.json"',
        'git diff --quiet "$TEST_SOURCE_COMMIT" HEAD -- tests',
        '"$PYTHON" tools/check_hygiene.py --require-git-clean',
        "printf 'ready=true\\n' >> \"$GITHUB_OUTPUT\"",
    )


def _readiness_job() -> dict[str, object]:
    setup_condition = "steps.readiness_inputs.outputs.has_tests_manifest == 'true'"
    return {
        "if": "startsWith(github.ref, 'refs/tags/')",
        "runs-on": RUNNER,
        "timeout-minutes": 60,
        "outputs": {"ready": "${{ steps.readiness.outputs.ready || steps.readiness_inputs.outputs.ready }}"},
        "steps": [
            {
                "uses": CHECKOUT,
                "with": {"persist-credentials": False, "fetch-depth": 0},
            },
            {
                "name": "Validate release version tag",
                "if": "startsWith(github.ref, 'refs/tags/')",
                "shell": "bash",
                "run": "\n".join(
                    (
                        "set -euo pipefail",
                        'git fetch --force --no-tags origin "refs/tags/$GITHUB_REF_NAME:refs/tags/$GITHUB_REF_NAME"',
                        "python3.12 tools/release_version.py \\",
                        '  --repo-root "$GITHUB_WORKSPACE" \\',
                        '  --tag "$GITHUB_REF_NAME"',
                        "",
                    )
                ),
            },
            {
                "name": "Check candidate readiness inputs",
                "id": "readiness_inputs",
                "shell": "bash",
                "run": _readiness_inputs_run(),
            },
            setup_step(setup_condition),
            {
                "name": "Evaluate candidate readiness",
                "if": setup_condition,
                "id": "readiness",
                "env": PYTHON_ENV,
                "shell": "bash",
                "run": _readiness_run(),
            },
        ],
    }


def _identity_job() -> dict[str, object]:
    job = _standard_job("identity")
    job["needs"] = ["candidate-readiness", "distribution"]
    job["if"] = "startsWith(github.ref, 'refs/tags/') && needs.candidate-readiness.outputs.ready == 'true'"
    job["steps"].insert(
        1,
        {
            "name": "Download distribution bundle",
            "uses": DOWNLOAD_ARTIFACT,
            "with": {
                "name": "xrr-distribution-${{ github.sha }}",
                "path": "${{ runner.temp }}/downloaded-distribution",
            },
        },
    )
    job["steps"][-1]["run"] = _standard_run("identity").replace(
        '"$PYTHON" tools/verify.py identity',
        '"$PYTHON" tools/verify.py identity '
        '--report-dir "$RUNNER_TEMP/identity" '
        '--artifact-dir "$RUNNER_TEMP/downloaded-distribution/artifacts" '
        '--artifact-manifest "$RUNNER_TEMP/downloaded-distribution/artifact-manifest.json"',
    )
    return job


def _release_job() -> dict[str, object]:
    job = _standard_job("release")
    job["timeout-minutes"] = JOB_TIMEOUTS["release"]
    job["needs"] = ["candidate-readiness"]
    job["if"] = "startsWith(github.ref, 'refs/tags/') && needs.candidate-readiness.outputs.ready == 'true'"
    job["steps"][-1]["run"] = _standard_run("release").replace(
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
                "name": "xrr-release-${{ github.ref_name }}-${{ github.sha }}",
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
            "distribution",
            "candidate-readiness",
            "identity",
            "release",
            "windows",
            "draft-release",
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
                    "DISTRIBUTION_RESULT": "${{ needs.distribution.result }}",
                    "READINESS_RESULT": "${{ needs.candidate-readiness.result }}",
                    "READY": "${{ needs.candidate-readiness.outputs.ready }}",
                    "IDENTITY_RESULT": "${{ needs.identity.result }}",
                    "RELEASE_RESULT": "${{ needs.release.result }}",
                    "WINDOWS_RESULT": "${{ needs.windows.result }}",
                    "DRAFT_RELEASE_RESULT": "${{ needs.draft-release.result }}",
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
                    "esac\n"
                    'test "$DISTRIBUTION_RESULT" = success\n'
                    'case "$REF" in\n'
                    '  refs/tags/*) test "$READINESS_RESULT" = success ;;\n'
                    '  *) test "$READINESS_RESULT" = skipped ;;\n'
                    "esac\n"
                    'case "$REF" in\n'
                    "  refs/tags/*)\n"
                    '    test "$READY" = true\n'
                    '    test "$IDENTITY_RESULT" = success\n'
                    '    test "$RELEASE_RESULT" = success\n'
                    '    test "$WINDOWS_RESULT" = success\n'
                    '    test "$DRAFT_RELEASE_RESULT" = success\n'
                    "    ;;\n"
                    "  *)\n"
                    '    test "$IDENTITY_RESULT" = skipped\n'
                    '    test "$RELEASE_RESULT" = skipped\n'
                    '    test "$WINDOWS_RESULT" = skipped\n'
                    '    test "$DRAFT_RELEASE_RESULT" = skipped\n'
                    "    ;;\n"
                    "esac\n"
                ),
            }
        ],
    }


def expected_workflow() -> dict[str, object]:
    workflow = {
        "name": "verify",
        "on": {
            "push": {
                "branches": ["main"],
                "tags": ["v*"],
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
            "distribution": _distribution_job(),
            "candidate-readiness": _readiness_job(),
            "identity": _identity_job(),
            "release": _release_job(),
            "windows": expected_windows_job(),
            "draft-release": expected_draft_release_job(),
            "checkpoint": _checkpoint_job(),
        },
    }
    for job in workflow["jobs"].values():
        steps = job.get("steps", [])
        if any(step.get("uses") == SETUP_MACOS_PYTHON for step in steps):
            index = next((index for index, step in enumerate(steps) if step.get("uses") == UPLOAD_ARTIFACT), len(steps))
            steps.insert(index, cleanup_step())
    return workflow
