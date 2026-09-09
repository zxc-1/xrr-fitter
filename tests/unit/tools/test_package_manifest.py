from __future__ import annotations

import hashlib
import json
import stat
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("target", ["macos-arm64-py312", "windows-x64-py312"])
def test_checked_in_package_manifests_bind_current_declarations(load_tool_module, target):
    path = ROOT / "tools/package-manifests" / f"{target}.json"
    assert path.is_file(), "missing checked-in package-byte identity"
    module = _module(load_tool_module)
    content = path.read_bytes()
    manifest = module.validate_manifest(ROOT, json.loads(content))
    assert content == module.manifest_bytes(manifest)


def _module(load_tool_module):
    assert (ROOT / "tools/package_manifest.py").is_file(), "missing target package-byte manifest"
    return load_tool_module("package_manifest")


def _resolution(load_tool_module, target):
    inventory = load_tool_module("lock_sbom").build_lock_sbom(ROOT, target=target)
    items = []
    for item in inventory["components"]:
        if "version" not in item:
            continue
        filename = f"{item['name'].replace('-', '_')}-{item['version']}-py3-none-any.whl"
        items.append(
            {
                "is_direct": False,
                "is_yanked": False,
                "metadata": {"name": item["name"], "version": item["version"]},
                "download_info": {
                    "url": f"https://files.pythonhosted.org/packages/{filename}",
                    "archive_info": {"hashes": {"sha256": hashlib.sha256(filename.encode()).hexdigest()}},
                },
            }
        )
    return {"version": "1", "pip_version": "26.1.2", "install": items}


@pytest.mark.parametrize("target,count", [("macos-arm64-py312", 44), ("windows-x64-py312", 38)])
def test_manifest_binds_every_exact_pin_to_one_target_wheel(load_tool_module, target, count):
    module = _module(load_tool_module)
    manifest = module.build_manifest(ROOT, target, _resolution(load_tool_module, target))
    assert len(manifest["wheels"]) == count
    assert manifest["target"] == target
    assert manifest["lock_sha256"] == hashlib.sha256((ROOT / f"requirements-{target}.lock").read_bytes()).hexdigest()
    assert module.validate_manifest(ROOT, manifest) == manifest


def test_manifest_keeps_git_source_outside_wheel_hash_claim(load_tool_module):
    module = _module(load_tool_module)
    manifest = module.build_manifest(ROOT, "macos-arm64-py312", _resolution(load_tool_module, "macos-arm64-py312"))
    assert [item["name"] for item in manifest["vcs"]] == ["refnx"]
    assert manifest["vcs"][0]["commit"] == "3d3808f66a14a8200eba020f8dff53f4d1e059bc"
    assert "sha256" not in manifest["vcs"][0]


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "hash", "yanked", "host", "platform", "python", "version"]
)
def test_resolution_rejects_missing_changed_or_untrusted_wheels(load_tool_module, mutation):
    module = _module(load_tool_module)
    report = _resolution(load_tool_module, "windows-x64-py312")
    first = report["install"][0]
    _mutate_resolution(report, first, mutation)
    with pytest.raises(ValueError):
        module.build_manifest(ROOT, "windows-x64-py312", report)


def _mutate_resolution(report, first, mutation):
    if mutation == "missing":
        report["install"].pop()
    elif mutation == "extra":
        report["install"].append(deepcopy(first))
        report["install"][-1]["metadata"]["name"] = "extra"
    elif mutation == "duplicate":
        report["install"].append(deepcopy(first))
    elif mutation == "hash":
        first["download_info"]["archive_info"]["hashes"]["sha256"] = "bad"
    elif mutation == "yanked":
        first["is_yanked"] = True
    elif mutation == "version":
        report["pip_version"] = "unlocked"
    else:
        substitutions = {
            "host": ("https://files.pythonhosted.org/", "https://untrusted.invalid/"),
            "platform": ("py3-none-any", "cp312-cp312-macosx_14_0_arm64"),
            "python": ("py3-none-any", "cp313-cp313-win_amd64"),
        }
        old, new = substitutions[mutation]
        first["download_info"]["url"] = first["download_info"]["url"].replace(old, new)


def test_manifest_serialization_replays_in_canonical_order(load_tool_module):
    module = _module(load_tool_module)
    report = _resolution(load_tool_module, "windows-x64-py312")
    first = module.build_manifest(ROOT, "windows-x64-py312", report)
    report["install"].reverse()
    second = module.build_manifest(ROOT, "windows-x64-py312", report)
    assert module.manifest_bytes(first) == module.manifest_bytes(second)
    assert json.loads(module.manifest_bytes(first)) == first


@pytest.mark.parametrize("field", ["target", "schema", "lock_sha256", "vcs", "extra"])
def test_consumption_rejects_changed_manifest_bindings(load_tool_module, field):
    module = _module(load_tool_module)
    manifest = module.build_manifest(ROOT, "macos-arm64-py312", _resolution(load_tool_module, "macos-arm64-py312"))
    manifest[field] = "altered"
    with pytest.raises(ValueError):
        module.validate_manifest(ROOT, manifest)


def test_hash_requirements_select_only_recorded_wheel_urls(load_tool_module):
    module = _module(load_tool_module)
    manifest = module.build_manifest(ROOT, "windows-x64-py312", _resolution(load_tool_module, "windows-x64-py312"))
    first = manifest["wheels"][0]
    expected = f"{first['name']} @ {first['url']} --hash=sha256:{first['sha256']}"
    assert module.download_requirements(manifest).decode().splitlines()[0] == expected


def _wheel_directory(tmp_path):
    content = b"recorded package bytes"
    filename = "sample-1.0-py3-none-any.whl"
    (tmp_path / filename).write_bytes(content)
    return [{"filename": filename, "sha256": hashlib.sha256(content).hexdigest()}]


def test_downloaded_bytes_are_verified_without_importing_packages(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    wheels = _wheel_directory(tmp_path)
    records = module.verify_wheels(tmp_path, wheels)
    assert records[0]["sha256"] == wheels[0]["sha256"]
    assert records[0]["size"] == len(b"recorded package bytes")


def test_windows_identity_accepts_missing_path_file_ids(load_tool_module, monkeypatch):
    module = _module(load_tool_module)
    monkeypatch.setattr(module.os, "name", "nt")
    path_stat = (0, 0, 12, 34, 56)
    handle_stat = (17, 29, 12, 34, 56)
    assert module._same_file_identity(path_stat, handle_stat)


def test_windows_file_identity_uses_creation_time(load_tool_module, monkeypatch):
    module = _module(load_tool_module)
    monkeypatch.setattr(module.os, "name", "nt")
    value = SimpleNamespace(
        st_mode=stat.S_IFREG,
        st_dev=17,
        st_ino=29,
        st_size=12,
        st_mtime_ns=34,
        st_ctime_ns=56,
        st_birthtime_ns=78,
    )
    assert module._file_identity(value) == (17, 29, 12, 34, 78)


@pytest.mark.parametrize("mutation", ["modified", "missing", "extra", "symlink"])
def test_wheel_cache_rejects_corruption_missing_extra_and_symlink_bytes(load_tool_module, tmp_path, mutation):
    module = _module(load_tool_module)
    wheels = _wheel_directory(tmp_path)
    path = tmp_path / wheels[0]["filename"]
    if mutation == "modified":
        path.write_bytes(b"modified")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        (tmp_path / "extra.whl").write_bytes(b"unrecorded")
    else:
        target = tmp_path.parent / f"{tmp_path.name}-target.whl"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError):
        module.verify_wheels(tmp_path, wheels)


def test_wheel_verification_rejects_changes_to_an_already_checked_file(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    wheels = _wheel_directory(tmp_path)
    second = tmp_path / "second-1.0-py3-none-any.whl"
    second.write_bytes(b"second")
    wheels.append({"filename": second.name, "sha256": hashlib.sha256(b"second").hexdigest()})
    original = module._verified_file

    def change_previous(path, digest):
        result = original(path, digest)
        if path == second:
            (tmp_path / wheels[0]["filename"]).write_bytes(b"changed after verification")
        return result

    monkeypatch.setattr(module, "_verified_file", change_previous)
    with pytest.raises(ValueError, match="changed"):
        module.verify_wheels(tmp_path, wheels)
