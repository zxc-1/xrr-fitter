from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _workflow():
    path = ROOT / ".github/workflows/audit-windows.yml"
    assert path.is_file(), "missing non-GUI Windows audit workflow"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_windows_audit_uses_read_only_hosted_runner():
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["on"]["push"]["branches"] == ["audit-improvements"]
    job = workflow["jobs"]["windows-headless"]
    assert job["runs-on"] == "windows-2025"
    assert "cache" not in job["steps"][1]["with"]


def test_windows_audit_checks_out_exact_source_without_credentials():
    checkout = _workflow()["jobs"]["windows-headless"]["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["persist-credentials"] is False


def test_windows_audit_downloads_verifies_and_installs_without_global_cache():
    runs = "\n".join(step.get("run", "") for step in _workflow()["jobs"]["windows-headless"]["steps"])
    assert "tools/package_cache.py fetch" in runs
    assert "tools/package_downloads.py install" in runs
    assert "tools/package_sbom.py" in runs


def test_windows_audit_bootstraps_only_explicit_hash_locked_tools():
    runs = "\n".join(step.get("run", "") for step in _workflow()["jobs"]["windows-headless"]["steps"])
    assert "--require-hashes --no-deps --no-cache-dir" in runs
    assert "tools/bootstrap-requirements.lock" in runs
    assert "tools/windows-test-requirements.lock" in runs
    assert "requirements-windows-x64-py312.lock" not in runs


def test_windows_audit_checks_every_native_exit_code():
    steps = _workflow()["jobs"]["windows-headless"]["steps"]
    for step in steps:
        if "run" in step:
            assert step["shell"] == "pwsh"
            assert '$ErrorActionPreference = "Stop"' in step["run"]
            assert "$PSNativeCommandUseErrorActionPreference = $true" in step["run"]
            assert "continue-on-error" not in step


def test_windows_audit_reuses_existing_headless_test_selections():
    steps = _workflow()["jobs"]["windows-headless"]["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "tests/unit/cli" in runs
    assert "tests/integration/test_cli_workflow.py" in runs
    assert "tests/unit/io/test_export_run.py" in runs


def test_windows_audit_never_starts_gui_code():
    steps = _workflow()["jobs"]["windows-headless"]["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "--exclude-module PySide6" in runs
    assert "Start-Process" not in runs
    assert "tests/gui" not in runs
    assert "src/xrr_fitter/gui" not in runs


def test_windows_audit_retains_failure_evidence():
    steps = _workflow()["jobs"]["windows-headless"]["steps"]
    assert steps[-2]["if"] == "${{ always() }}"
    assert steps[-2]["uses"].startswith("actions/upload-artifact@")


def test_bootstrap_lock_is_the_existing_universal_pip_and_packaging_bytes():
    path = ROOT / "tools/bootstrap-requirements.lock"
    assert path.is_file(), "missing hash-verified tool bootstrap"
    expected = [
        line
        for line in (ROOT / "tools/audit-requirements.lock").read_text().splitlines()
        if line.startswith(("pip==", "packaging=="))
    ]
    assert path.read_text().splitlines() == expected


def test_windows_pytest_lock_reuses_verified_universal_test_wheels():
    path = ROOT / "tools/windows-test-requirements.lock"
    assert path.is_file(), "missing hash-verified Windows test dependencies"
    wheels = json.loads((ROOT / "tools/package-manifests/macos-arm64-py312.json").read_text())["wheels"]
    selected = [item for item in wheels if item["name"] in {"iniconfig", "pluggy", "pygments", "pytest"}]
    assert len(selected) == 4
    assert all(item["filename"].endswith("-py3-none-any.whl") for item in selected)
    assert path.read_text().splitlines() == [
        f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}" for item in selected
    ]


def _cache_steps():
    steps = _workflow()["jobs"]["windows-headless"]["steps"]
    cache_steps = [step for step in steps if step.get("uses", "").startswith("actions/cache/")]
    assert len(cache_steps) == 2
    return cache_steps


def test_windows_cache_actions_are_pinned():
    restore, save = _cache_steps()
    assert restore["uses"] == "actions/cache/restore@0400d5f644dc74513175e3cd8d07132dd4860809"
    assert save["uses"] == "actions/cache/save@0400d5f644dc74513175e3cd8d07132dd4860809"


def test_windows_cache_restores_only_an_exact_key_and_saves_only_verified_wheels():
    for step in _cache_steps():
        assert step["with"] == {
            "key": "${{ steps.cache-key.outputs.key }}",
            "path": "${{ runner.temp }}/xrr-wheel-cache/",
        }


def test_windows_cache_save_requires_success_and_a_missed_key():
    _restore, save = _cache_steps()
    assert "success()" in save["if"]
    assert "cache-hit" in save["if"]


def _cache_key_step():
    steps = _workflow()["jobs"]["windows-headless"]["steps"]
    key_steps = [step for step in steps if step.get("id") == "cache-key"]
    assert len(key_steps) == 1
    return key_steps[0]


def test_windows_cache_derives_trust_domain_from_event():
    key = _cache_key_step()
    domain = key["env"]["CACHE_TRUST_DOMAIN"]
    assert "github.event_name == 'pull_request'" in domain
    assert "format('pr-{0}', github.event.pull_request.number)" in domain
    assert "|| 'trusted'" in domain
    assert "tools/package_cache.py key" in key["run"]


def test_windows_cache_verifies_restored_bytes():
    steps = _workflow()["jobs"]["windows-headless"]["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "tools/package_cache.py fetch" in runs
    assert '--trust-domain "$env:CACHE_TRUST_DOMAIN"' in runs


def test_windows_cache_fetch_uses_the_stable_restore_directory():
    steps = _workflow()["jobs"]["windows-headless"]["steps"]
    fetch = next(step["run"] for step in steps if "tools/package_cache.py fetch" in step.get("run", ""))
    assert '--cache-dir "$env:RUNNER_TEMP/xrr-wheel-cache"' in fetch
    assert '--report-dir "$env:AUDIT_ROOT/reports/packages"' in fetch


def test_windows_cache_path_is_unchanged_across_runs_and_attempts():
    for step in _cache_steps():
        template = step["with"]["path"]
        paths = {
            template.replace("${{ runner.temp }}", "D:/a/_temp").replace(
                "${{ env.AUDIT_ROOT }}", f"D:/a/_temp/xrr-audit-{run_id}-{attempt}"
            )
            for run_id, attempt in ((123, 1), (123, 2), (456, 1))
        }
        assert len(paths) == 1, "actions/cache versions include the resolved path, not only the explicit key"
