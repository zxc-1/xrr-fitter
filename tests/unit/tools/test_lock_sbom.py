from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
COMMIT = "3d3808f66a14a8200eba020f8dff53f4d1e059bc"
REFNX = f"refnx @ git+https://github.com/refnx/refnx.git@{COMMIT}"


def _fixture(root: Path) -> Path:
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools==75.8.2"]\n'
        '[project]\nname = "xrr-fitter"\nversion = "1.2.3"\n'
        'dependencies = ["numpy>=2,<3"]\n'
        f'[project.optional-dependencies]\ntest = ["pytest==8.4.2", "{REFNX}"]\n'
        '[tool.xrr.windows-packaging]\nrequires = ["pyinstaller==6.21.0"]\n',
        encoding="utf-8",
    )
    locks = {
        "macos-arm64-py312": ("NumPy==2.5.2", "pytest==8.4.2", REFNX, "setuptools==75.8.2"),
        "windows-x64-py312": ("NumPy==2.5.2", "pyinstaller==6.21.0", "setuptools==75.8.2"),
    }
    for target, lines in locks.items():
        (root / f"requirements-{target}.lock").write_text(
            "\n".join(sorted(lines, key=str.casefold)) + "\n", encoding="utf-8"
        )
    return root


def test_sbom_reports_lock_inventory_without_inventing_artifact_evidence(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("lock_sbom")
    root = _fixture(tmp_path / "repo")
    report = module.build_lock_sbom(root, target="macos-arm64-py312")

    assert report["bomFormat"] == "CycloneDX"
    assert report["specVersion"] == "1.6"
    assert report["version"] == 1
    assert report["metadata"]["component"] == {
        "type": "application",
        "bom-ref": "pkg:pypi/xrr-fitter@1.2.3",
        "name": "xrr-fitter",
        "version": "1.2.3",
        "purl": "pkg:pypi/xrr-fitter@1.2.3",
    }
    components = report["components"]
    assert [item["name"] for item in components] == ["numpy", "pytest", "refnx", "setuptools"]
    assert components[0] == {
        "type": "library",
        "bom-ref": "pkg:pypi/numpy@2.5.2",
        "name": "numpy",
        "version": "2.5.2",
        "purl": "pkg:pypi/numpy@2.5.2",
    }
    assert all("hashes" not in item and "licenses" not in item for item in components)
    assert "dependencies" not in report
    assert report["compositions"] == [{"aggregate": "incomplete"}]


def test_sbom_binds_input_bytes(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("lock_sbom")
    root = _fixture(tmp_path / "repo")
    report = module.build_lock_sbom(root, target="macos-arm64-py312")
    properties = {item["name"]: item["value"] for item in report["metadata"]["properties"]}
    lock = root / "requirements-macos-arm64-py312.lock"
    assert properties["xrr:lock:sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert properties["xrr:pyproject:sha256"] == hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest()
    assert properties["xrr:lock:path"] == lock.name
    assert properties["xrr:inventory:target"] == "macos-arm64-py312"
    assert properties["xrr:inventory:scope"] == "locked-python-environment"


def test_sbom_preserves_vcs_commit_without_inventing_a_package_version(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("lock_sbom")
    root = _fixture(tmp_path / "repo")
    report = module.build_lock_sbom(root, target="macos-arm64-py312")
    refnx = next(item for item in report["components"] if item["name"] == "refnx")
    assert refnx["bom-ref"] == f"vcs:refnx:{COMMIT}"
    assert refnx["externalReferences"] == [{"type": "vcs", "url": "https://github.com/refnx/refnx.git"}]
    assert refnx["properties"] == [{"name": "xrr:vcs:commit", "value": COMMIT}]
    assert "version" not in refnx and "purl" not in refnx


def test_windows_inventory_uses_packaging_not_macos_test_closure(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("lock_sbom")
    root = _fixture(tmp_path / "repo")
    report = module.build_lock_sbom(root, target="windows-x64-py312")
    assert [item["name"] for item in report["components"]] == ["numpy", "pyinstaller", "setuptools"]


def test_sbom_is_byte_deterministic_across_repository_locations(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("lock_sbom")
    first = _fixture(tmp_path / "first")
    second = _fixture(tmp_path / "second")
    outputs = [
        module.canonical_sbom_bytes(module.build_lock_sbom(root, target="macos-arm64-py312"))
        for root in (first, first, second)
    ]
    assert outputs[0] == outputs[1] == outputs[2]
    assert str(tmp_path).encode() not in outputs[0]
    assert outputs[0].endswith(b"\n")
    assert "timestamp" not in json.loads(outputs[0])["metadata"]


@pytest.mark.parametrize(
    "old,new",
    [
        ("NumPy==2.5.2", "NumPy>=2.5.2"),
        ("NumPy==2.5.2", "NumPy==1.0.0"),
        ("pytest==8.4.2\n", ""),
        (COMMIT, "main"),
        ("\n", "\r\n"),
    ],
)
def test_sbom_rejects_unbound_or_noncanonical_locks(tmp_path: Path, load_tool_module, old: str, new: str) -> None:
    module = load_tool_module("lock_sbom")
    root = _fixture(tmp_path / "repo")
    lock = root / "requirements-macos-arm64-py312.lock"
    lock.write_bytes(lock.read_bytes().replace(old.encode(), new.encode()))
    with pytest.raises(ValueError):
        module.build_lock_sbom(root, target="macos-arm64-py312")


def test_sbom_rejects_symlink_inputs_and_unknown_target(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("lock_sbom")
    root = _fixture(tmp_path / "repo")
    with pytest.raises(ValueError, match="target"):
        module.build_lock_sbom(root, target="linux")
    lock = root / "requirements-macos-arm64-py312.lock"
    original = tmp_path / "original.lock"
    lock.rename(original)
    lock.symlink_to(original)
    with pytest.raises(ValueError, match="regular file"):
        module.build_lock_sbom(root, target="macos-arm64-py312")


def test_sbom_reads_dynamic_version_without_executing_source(tmp_path: Path, load_tool_module) -> None:
    module = load_tool_module("lock_sbom")
    root = _fixture(tmp_path / "repo")
    project = root / "pyproject.toml"
    project.write_text(
        project.read_text().replace('version = "1.2.3"', 'dynamic = ["version"]')
        + '\n[tool.setuptools.dynamic]\nversion = {attr = "xrr_fitter.version.__version__"}\n',
        encoding="utf-8",
    )
    source = root / "src/xrr_fitter/version.py"
    source.parent.mkdir(parents=True)
    source.write_text('__version__ = "1.2.3"\nraise AssertionError("must not execute")\n', encoding="utf-8")
    report = module.build_lock_sbom(root, target="macos-arm64-py312")
    assert report["metadata"]["component"]["version"] == "1.2.3"


def test_sbom_rejects_metadata_drift_during_validation(tmp_path: Path, load_tool_module, monkeypatch) -> None:
    module = load_tool_module("lock_sbom")
    root = _fixture(tmp_path / "repo")
    original = module.read_project_dependencies

    def change_after_read(path):
        dependencies = original(path)
        path.write_bytes(path.read_bytes() + b"\n# changed during generation\n")
        return dependencies

    monkeypatch.setattr(module, "read_project_dependencies", change_after_read)
    with pytest.raises(ValueError, match="inputs changed"):
        module.build_lock_sbom(root, target="macos-arm64-py312")


@pytest.mark.parametrize("target", ["macos-arm64-py312", "windows-x64-py312"])
def test_sbom_supports_both_committed_locks_without_platform_resolution(load_tool_module, target: str) -> None:
    module = load_tool_module("lock_sbom")
    report = module.build_lock_sbom(ROOT, target=target)
    assert len(report["components"]) == len((ROOT / f"requirements-{target}.lock").read_text().splitlines())


def test_cli_emits_json_from_other_cwd_and_fails_without_partial_report(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "repo")
    tool = ROOT / "tools/lock_sbom.py"
    assert tool.is_file(), "missing lock SBOM implementation"
    command = (sys.executable, str(tool), "--repo-root", str(root), "--target", "windows-x64-py312")
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["bomFormat"] == "CycloneDX"
    assert not result.stderr
    (root / "requirements-windows-x64-py312.lock").write_text("numpy>=1\n", encoding="utf-8")
    failed = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, check=False)
    assert failed.returncode == 2
    assert not failed.stdout
    assert b"Traceback" not in failed.stderr


def test_cli_preserves_canonical_bytes_under_windows_newline_translation(
    tmp_path: Path, load_tool_module, monkeypatch
) -> None:
    module = load_tool_module("lock_sbom")
    root = _fixture(tmp_path / "repo")
    expected = module.canonical_sbom_bytes(module.build_lock_sbom(root, target="windows-x64-py312"))
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="utf-8", newline="\r\n", write_through=True)
    monkeypatch.setattr(module.sys, "stdout", output)
    assert module.main(["--repo-root", str(root), "--target", "windows-x64-py312"]) == 0
    assert buffer.getvalue() == expected
