from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/package_cache.py").is_file(), "missing verified package cache"
    return load_tool_module("package_cache")


def _manifest():
    return json.loads((ROOT / "tools/package-manifests/windows-x64-py312.json").read_text())


def test_cache_key_separates_trust_domains_and_binds_full_manifest(load_tool_module):
    module = _module(load_tool_module)
    manifest = _manifest()
    trusted = module.cache_key(manifest, "trusted")
    assert "windows-x64-py312-python-3.12-pip-26.1.2" in trusted
    assert manifest["lock_sha256"] in trusted
    assert module.cache_key(deepcopy(manifest), "trusted") == trusted
    assert module.cache_key(manifest, "pr-12") != trusted
    assert module.cache_key(manifest, "pr-13") != module.cache_key(manifest, "pr-12")
    changed = deepcopy(manifest)
    changed["wheels"][0]["sha256"] = "0" * 64
    assert module.cache_key(changed, "trusted") != trusted


@pytest.mark.parametrize("domain", ["", "pr-0", "pr-01", "../trusted", "pr-12/trusted", "pull_request"])
def test_cache_key_rejects_ambiguous_trust_domains(load_tool_module, domain):
    with pytest.raises(ValueError, match="trust domain"):
        _module(load_tool_module).cache_key(_manifest(), domain)


def _fake_manifest():
    manifest = _manifest()
    wheel = manifest["wheels"][0]
    wheel["sha256"] = hashlib.sha256(b"verified wheel").hexdigest()
    manifest["wheels"] = [wheel]
    return manifest


def _setup_download(module, monkeypatch, manifest, calls, *, exit_code=0):
    monkeypatch.setattr(module, "read_manifest", lambda root, path: manifest)

    def download(root, path, report):
        calls.append(report)
        report.mkdir()
        (report / "download.stderr").write_text("download attempt\n")
        if exit_code == 0:
            (report / "wheels").mkdir()
            (report / "wheels" / manifest["wheels"][0]["filename"]).write_bytes(b"verified wheel")
        return exit_code

    monkeypatch.setattr(module, "download_packages", download)


def test_cold_and_warm_cache_reverify_bytes_without_redownloading(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    manifest, calls = _fake_manifest(), []
    _setup_download(module, monkeypatch, manifest, calls)
    cache = tmp_path / "cache"
    for label in ("cold", "warm"):
        report = tmp_path / label
        assert module.cached_download(ROOT, tmp_path / "manifest", report, cache, "trusted") == 0
        evidence = json.loads((report / "cache.json").read_text())
        assert evidence["hit"] is (label == "warm")
        assert evidence["state"] == "PASS"
        assert len(list((report / "wheels").iterdir())) == 1
    assert calls == [tmp_path / "cold"]
    key_hash = hashlib.sha256(module.cache_key(manifest, "trusted").encode("ascii")).hexdigest()
    assert {path.name for path in cache.iterdir()} == {key_hash}


def test_corrupt_warm_cache_fails_without_network_or_overwriting_evidence(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    manifest, calls = _fake_manifest(), []
    _setup_download(module, monkeypatch, manifest, calls)
    cache = tmp_path / "cache"
    module.cached_download(ROOT, tmp_path / "manifest", tmp_path / "cold", cache, "trusted")
    wheel = next(cache.glob("*/*.whl"))
    wheel.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="bytes do not match"):
        module.cached_download(ROOT, tmp_path / "manifest", tmp_path / "warm", cache, "trusted")
    assert calls == [tmp_path / "cold"]
    assert wheel.read_bytes() == b"corrupted"
    assert not (tmp_path / "warm/wheels").exists()


def test_failed_cold_download_does_not_publish_cache(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    manifest, calls = _fake_manifest(), []
    _setup_download(module, monkeypatch, manifest, calls, exit_code=23)
    report, cache = tmp_path / "report", tmp_path / "cache"
    assert module.cached_download(ROOT, tmp_path / "manifest", report, cache, "trusted") == 23
    assert not list(cache.iterdir())
    assert (report / "download.stderr").read_text() == "download attempt\n"


def test_cache_path_must_be_external_and_not_a_symlink(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    manifest = ROOT / "tools/package-manifests/windows-x64-py312.json"
    with pytest.raises(ValueError, match="external"):
        module.cached_download(ROOT, manifest, tmp_path / "report", ROOT / "cache", "trusted")
    (tmp_path / "actual").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "actual", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.cached_download(ROOT, manifest, tmp_path / "report", tmp_path / "link", "trusted")


def test_cache_copy_failure_removes_only_temporary_copy(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    manifest = _fake_manifest()
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / manifest["wheels"][0]["filename"]).write_bytes(b"verified wheel")

    def failed_copy(*args, **kwargs):
        raise OSError("copy interrupted")

    monkeypatch.setattr(module.shutil, "copytree", failed_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        module.copy_verified(source, destination, manifest["wheels"])
    assert {path.name for path in tmp_path.iterdir()} == {"source"}


def test_cache_copy_does_not_overwrite_existing_destination(load_tool_module, tmp_path):
    module = _module(load_tool_module)
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "existing").write_bytes(b"keep")
    with pytest.raises(ValueError, match="already exists"):
        module.copy_verified(tmp_path / "missing", destination, [])
    assert (destination / "existing").read_bytes() == b"keep"


def test_cache_rejects_unrelated_members_before_downloading(load_tool_module, tmp_path, monkeypatch):
    module = _module(load_tool_module)
    manifest, calls = _fake_manifest(), []
    _setup_download(module, monkeypatch, manifest, calls)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "unrelated").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="unrelated"):
        module.cached_download(ROOT, tmp_path / "manifest", tmp_path / "report", cache, "trusted")
    assert not calls
    assert (cache / "unrelated").read_bytes() == b"preserve"
