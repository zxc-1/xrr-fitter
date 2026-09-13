from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/locked_closure.py").is_file(), "missing frozen dependency closure verifier"
    return load_tool_module("locked_closure")


def _inventory(name, version, requirements=()):
    return {"name": name, "version": version, "requirements": list(requirements)}


def _fixture(tmp_path, load_tool_module, inventories=None, runtime="pandas==2.3.3"):
    inventories = inventories or [
        _inventory("builder", "1"),
        _inventory("pandas", "2.3.3", ['tzdata>=2022.7; sys_platform == "win32"']),
        _inventory("tzdata", "2026.3"),
    ]
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["builder==1"]\n'
        '[project]\nname = "fixture"\nversion = "1"\ndependencies = ' + json.dumps([runtime]) + "\n"
        "[tool.xrr.windows-packaging]\nrequires = []\n"
    )
    pins = sorted(f"{item['name']}=={item['version']}" for item in inventories)
    (root / "requirements-windows-x64-py312.lock").write_text("\n".join(pins) + "\n")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    records = [_wheel(wheels, item) for item in sorted(inventories, key=lambda item: item["name"])]
    package = load_tool_module("package_manifest")
    _, header = package.locked_inputs(root, "windows-x64-py312")
    manifest = {**header, "wheels": records}
    path = root / "tools/package-manifests/windows-x64-py312.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(package.manifest_bytes(manifest))
    return root, manifest, inventories, wheels


def _wheel(directory, item):
    filename = f"{item['name'].replace('-', '_')}-{item['version']}-py3-none-any.whl"
    path = directory / filename
    metadata = f"Metadata-Version: 2.1\nName: {item['name']}\nVersion: {item['version']}\n"
    metadata += "".join(f"Requires-Dist: {value}\n" for value in item["requirements"])
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{item['name']}-{item['version']}.dist-info/METADATA", metadata)
    return {
        "name": item["name"],
        "version": item["version"],
        "filename": filename,
        "url": f"https://files.pythonhosted.org/packages/{filename}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_frozen_transitive_pin_is_verified_without_live_resolution(tmp_path, monkeypatch, load_tool_module):
    root, manifest, inventories, wheels = _fixture(tmp_path, load_tool_module)
    module = _module(load_tool_module)
    resolver = load_tool_module("lock_windows_environment")
    monkeypatch.setattr(resolver, "resolve_windows_lock", lambda *_: pytest.fail("must not resolve latest tzdata"))
    monkeypatch.setattr(module, "download_packages", lambda *_: pytest.fail("offline inputs must not download"))

    result = module.verify_locked_closure(root, wheel_dir=wheels)

    assert result["state"] == "PASS"
    assert result["validation"] == "frozen-wheel-closure"
    assert result["dependencies"]["pandas"] == ["tzdata"]
    assert result["wheel_count"] == 3
    assert result["input_sha256"]["requirements-windows-x64-py312.lock"] == manifest["lock_sha256"]


@pytest.mark.parametrize("mutation", ["missing", "conflict", "extra", "duplicate"])
def test_closure_rejects_missing_conflicting_unreachable_or_duplicate_packages(tmp_path, load_tool_module, mutation):
    inventories = [
        _inventory("builder", "1"),
        _inventory("pandas", "2.3.3", ["tzdata>=2022.7"]),
        _inventory("tzdata", "2026.3"),
    ]
    if mutation == "missing":
        inventories.pop()
    elif mutation == "conflict":
        inventories[-1]["version"] = "2020.1"
    elif mutation == "extra":
        inventories.append(_inventory("unrelated", "1"))
    root, manifest, values, _ = _fixture(tmp_path, load_tool_module, inventories)
    if mutation == "duplicate":
        values.append(values[-1])

    with pytest.raises(ValueError, match="dependency|closure|metadata"):
        _module(load_tool_module).validate_closure(root, manifest, values)


def test_closure_follows_root_extras_with_windows_markers(tmp_path, load_tool_module):
    values = [
        _inventory("builder", "1"),
        _inventory("pandas", "2.3.3", ['tzdata>=2022.7; extra == "time" and sys_platform == "win32"']),
        _inventory("tzdata", "2026.3"),
    ]
    root, manifest, inventories, _ = _fixture(tmp_path, load_tool_module, values, "pandas[time]==2.3.3")
    assert _module(load_tool_module).validate_closure(root, manifest, inventories)["dependencies"]["pandas"] == [
        "tzdata"
    ]


def test_closure_rejects_changed_wheel_bytes(tmp_path, load_tool_module):
    root, manifest, _, wheels = _fixture(tmp_path, load_tool_module)
    (wheels / manifest["wheels"][0]["filename"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        _module(load_tool_module).verify_locked_closure(root, wheel_dir=wheels)


@pytest.mark.parametrize(
    "filename",
    ["pyproject.toml", "requirements-windows-x64-py312.lock", "tools/package-manifests/windows-x64-py312.json"],
)
def test_closure_guards_input_bytes_through_metadata_validation(tmp_path, monkeypatch, load_tool_module, filename):
    root, _, _, wheels = _fixture(tmp_path, load_tool_module)
    module = _module(load_tool_module)
    original = module.validate_closure

    def changed(*args):
        value = original(*args)
        path = root / filename
        path.write_bytes(path.read_bytes() + b"\n")
        return value

    monkeypatch.setattr(module, "validate_closure", changed)
    with pytest.raises(ValueError, match="changed"):
        module.verify_locked_closure(root, wheel_dir=wheels)


def test_online_path_downloads_only_the_fixed_manifest_to_external_scratch(tmp_path, monkeypatch, load_tool_module):
    root, _, _, wheels = _fixture(tmp_path, load_tool_module)
    module = _module(load_tool_module)
    calls = []

    def download(repo, manifest, report):
        assert repo == root
        assert manifest == root / "tools/package-manifests/windows-x64-py312.json"
        assert not report.is_relative_to(root)
        shutil.copytree(wheels, report / "wheels")
        calls.append(report)
        return 0

    monkeypatch.setattr(module, "download_packages", download)
    assert module.verify_locked_closure(root)["state"] == "PASS"
    assert len(calls) == 1
    assert not calls[0].exists()


def test_failed_fixed_download_does_not_fall_back_to_resolution(tmp_path, monkeypatch, load_tool_module):
    root, _, _, _ = _fixture(tmp_path, load_tool_module)
    module = _module(load_tool_module)
    monkeypatch.setattr(module, "download_packages", lambda *_: 23)
    with pytest.raises(ValueError, match="download"):
        module.verify_locked_closure(root)


def test_distribution_uses_frozen_closure_not_latest_index(load_tool_module):
    registry = load_tool_module("verify_registry")
    assert registry.MODE_REGISTRY["distribution"].commands[0] == (
        registry.PYTHON,
        "tools/locked_closure.py",
        "--repo-root",
        registry.ROOT,
    )
