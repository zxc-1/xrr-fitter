from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/audit.yml"


def _workflow():
    assert WORKFLOW.is_file(), "missing independent audit workflow"
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _job():
    return _workflow()["jobs"]["audit"]


def test_audit_workflow_is_read_only_and_uses_hosted_runners():
    workflow = _workflow()
    assert workflow["on"] == {
        "pull_request": {},
        "push": {"branches": ["main", "audit-improvements"]},
        "workflow_dispatch": {},
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"audit"}
    assert _job()["runs-on"] == "macos-15"


def test_audit_matrix_is_non_gui_and_does_not_change_release_modes():
    job = _job()
    assert job["strategy"] == {"fail-fast": False, "matrix": {"audit": ["coverage", "typing", "advisories"]}}
    assert job["timeout-minutes"] == 60
    checkout, setup = job["steps"][:2]
    assert checkout == {
        "uses": "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",
        "with": {"persist-credentials": False, "fetch-depth": 0},
    }
    assert setup["uses"] == "./.github/actions/setup-macos-python"


def test_audit_tools_are_hash_checked_before_verification():
    install, verify = _job()["steps"][2:4]
    assert install["env"] == verify["env"] == {"PYTHON": "${{ steps.python.outputs.python }}"}
    assert install["run"].splitlines() == [
        "set -euo pipefail",
        '"$PYTHON" -m pip --isolated install --no-cache-dir --require-hashes --only-binary=:all: -r tools/audit-requirements.lock',
        '"$PYTHON" -m pip check',
    ]
    assert verify["run"].splitlines() == [
        "set -euo pipefail",
        '"$PYTHON" tools/verify.py "${{ matrix.audit }}" --report-dir "$RUNNER_TEMP/audit-reports"',
    ]
    assert "continue-on-error" not in verify


def test_audit_upload_keeps_failure_evidence_and_coverage_raw_data():
    upload = _job()["steps"][-1]
    assert upload["if"] == "${{ always() }}"
    assert upload["uses"] == "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
    assert upload["with"] == {
        "name": "audit-${{ matrix.audit }}-${{ github.run_id }}-${{ github.run_attempt }}",
        "path": "${{ runner.temp }}/audit-reports/",
        "include-hidden-files": True,
        "if-no-files-found": "error",
        "retention-days": 14,
    }
