from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _steps(filename, job):
    return yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())["jobs"][job]["steps"]


def _run_index(steps, fragment):
    return next((i for i, step in enumerate(steps) if fragment in step.get("run", "")), -1)


def _uses_index(steps, action):
    return next(i for i, step in enumerate(steps) if step.get("uses", "").startswith(action))


def _arguments(script, arguments):
    for argument in arguments:
        assert argument in script


def test_windows_auxiliary_inputs_are_retained_and_used_offline_separately():
    runs = "\n".join(step.get("run", "") for step in _steps("audit-windows.yml", "windows-headless"))
    _arguments(
        runs,
        [
            "-m pip --isolated download --require-hashes --no-deps --no-cache-dir",
            "-r tools/bootstrap-requirements.lock -r tools/windows-test-requirements.lock",
            '--no-index --find-links "$root/reports/auxiliary-wheels"',
            '--no-index --find-links "$env:AUDIT_ROOT/reports/auxiliary-wheels"',
        ],
    )


def test_windows_binding_follows_execution_and_precedes_cache_publication():
    steps = _steps("audit-windows.yml", "windows-headless")
    index = _run_index(steps, "tools/installed_sbom.py")
    assert 0 <= _run_index(steps, "--ort") < index
    binding = steps[index]["run"]
    _arguments(binding, ("--auxiliary-dir", "--executable-evidence", "--executable", "--pip-wheel", "--manifest"))
    assert 'reports/installed"' in binding
    assert index < _uses_index(steps, "actions/cache/save@")


def test_macos_inventory_is_only_an_audit_slice_and_precedes_audit_tool_installation():
    steps = _steps("audit.yml", "audit")
    index = _run_index(steps, "tools/installed_sbom.py")
    assert index >= 0
    binding = steps[index]
    assert binding["if"] == "${{ matrix.audit == 'advisories' }}"
    assert index < _run_index(steps, "-r tools/audit-requirements.lock")
    _arguments(
        binding["run"],
        ("tools/verify.py distribution", "--artifact-manifest", "--artifact-dir", "--refnx-build", "--pip-wheel"),
    )
    setup = (ROOT / ".github/actions/setup-macos-python/action.yml").read_text()
    assert "installed_sbom.py" not in setup


def test_advisory_binding_keeps_artifact_and_installed_evidence_even_on_failure():
    steps = _steps("audit.yml", "audit")
    binding = [step for step in steps if "tools/audit_bundle.py" in step.get("run", "")]
    assert len(binding) == 1
    assert binding[0]["if"] == "${{ matrix.audit == 'advisories' }}"
    assert '--advisory-report "$RUNNER_TEMP/audit-reports/audit"' in binding[0]["run"]
    upload = steps[-1]
    assert upload["if"] == "${{ always() }}"
    assert "reports/installed/" in upload["with"]["path"]
    assert "reports/distribution/" in upload["with"]["path"]
