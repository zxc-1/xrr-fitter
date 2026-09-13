from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path("tools/package-manifests/refnx-build-macos-arm64-py312.json")
SOURCE = Path("tools/package-manifests/refnx-source.json")


def _module(load_tool_module):
    assert (ROOT / "tools/refnx_build_inputs.py").is_file(), "missing verified refnx build inputs"
    return load_tool_module("refnx_build_inputs")


def _wheel():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("sample/__init__.py", "")
        archive.writestr("sample-1.0.dist-info/METADATA", "Name: sample\nVersion: 1.0\nLicense-Expression: MIT\n")
    return stream.getvalue()


def _report(content):
    return {
        "version": "1",
        "pip_version": "26.1.2",
        "environment": {"python_version": "3.12", "sys_platform": "darwin", "platform_machine": "arm64"},
        "install": [
            {
                "is_direct": False,
                "is_yanked": False,
                "metadata": {"name": "sample", "version": "1.0", "requires_dist": []},
                "download_info": {
                    "url": "https://files.pythonhosted.org/packages/sample-1.0-py3-none-any.whl",
                    "archive_info": {"hashes": {"sha256": hashlib.sha256(content).hexdigest()}},
                },
            }
        ],
    }


@pytest.fixture
def input_case(tmp_path, load_tool_module):
    return lambda: _input_case(tmp_path, load_tool_module)


def _input_case(tmp_path, load_tool_module):
    module = _module(load_tool_module)
    root = tmp_path / "repo"
    (root / SOURCE.parent).mkdir(parents=True)
    (root / "src/xrr_fitter").mkdir(parents=True)
    shutil.copy2(ROOT / "src/xrr_fitter/version.py", root / "src/xrr_fitter/version.py")
    for name in ("pyproject.toml", "requirements-macos-arm64-py312.lock", "requirements-windows-x64-py312.lock"):
        shutil.copy2(ROOT / name, root / name)
    (root / "tools/refnx-build-requirements.in").write_text("sample==1.0\n")
    source = json.loads((ROOT / SOURCE).read_text())
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        item = tarfile.TarInfo(f"refnx-{source['vcs']['commit']}/code.py")
        item.size = 6
        archive.addfile(item, io.BytesIO(b"x = 1\n"))
    content = stream.getvalue()
    files = module.archive_files(content, f"refnx-{source['vcs']['commit']}")
    source.update(file_count=1, files_sha256=hashlib.sha256(module.manifest_bytes(files)).hexdigest())
    source["archive"].update(sha256=hashlib.sha256(content).hexdigest(), size=len(content))
    (root / SOURCE).write_bytes(module.manifest_bytes(source))
    wheel = _wheel()
    manifest, lock = module.build_inputs(root, _report(wheel))
    (root / "tools/refnx-build-requirements.lock").write_bytes(lock)
    (root / MANIFEST).write_bytes(module.manifest_bytes(manifest))
    return module, root, manifest, wheel, content


def test_checked_builder_lock_is_separate_and_exact(load_tool_module):
    module = _module(load_tool_module)
    manifest = module.read_inputs(ROOT)
    names = {item["name"] for item in manifest["wheels"]}
    assert names == {
        "cython",
        "meson",
        "meson-python",
        "ninja",
        "numpy",
        "packaging",
        "pip",
        "pyproject-metadata",
        "wheel",
    }
    assert {item["name"]: item["version"] for item in manifest["wheels"]}["pip"] == "26.1.2"
    assert manifest["source_manifest_sha256"] == hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest()
    assert "refnx-build" not in (ROOT / "pyproject.toml").read_text()


def test_builder_manifest_and_lock_are_deterministic_and_bound(input_case):
    module, root, manifest, wheel, _source = input_case()
    observed, lock = module.build_inputs(root, _report(wheel))
    assert observed == manifest == module.read_inputs(root)
    assert lock == f"sample==1.0 --hash=sha256:{hashlib.sha256(wheel).hexdigest()}\n".encode()
    assert manifest["lock_sha256"] == hashlib.sha256(lock).hexdigest()
    assert (
        manifest["application_lock_sha256"]
        == hashlib.sha256((root / "requirements-macos-arm64-py312.lock").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "relative", [SOURCE, "tools/refnx-build-requirements.in", "tools/refnx-build-requirements.lock"]
)
def test_input_changes_invalidate_builder_manifest(input_case, relative):
    module, root, *_rest = input_case()
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="binding|lock"):
        module.read_inputs(root)


@pytest.mark.parametrize("mutation", ["hash", "dependency", "extra", "pin"])
def test_resolution_must_cover_the_exact_builder_closure(input_case, mutation):
    module, root, _manifest, wheel, _source = input_case()
    report = _report(wheel)
    if mutation == "hash":
        report["install"][0]["download_info"]["archive_info"]["hashes"].clear()
    elif mutation == "dependency":
        report["install"][0]["metadata"]["requires_dist"] = ["missing>=1"]
    elif mutation == "extra":
        report["install"].append(deepcopy(report["install"][0]))
    else:
        (root / "tools/refnx-build-requirements.in").write_text("sample==2.0\n")
    with pytest.raises(ValueError):
        module.build_inputs(root, report)


def _downloads(input_case, monkeypatch):
    module, root, manifest, wheel, source = input_case()
    calls = []

    def download(_root, _manifest, stage, _environment):
        calls.append(stage)
        (stage / "wheels").mkdir()
        (stage / "wheels" / manifest["wheels"][0]["filename"]).write_bytes(wheel)
        return 0

    monkeypatch.setattr(module, "_download_stage", download)
    monkeypatch.setattr(module, "_download_source", lambda _source, path: path.write_bytes(source))
    return module, root, manifest, calls


def test_refnx_inputs_cache_cold_warm_rechecks_source_and_wheels(input_case, monkeypatch, tmp_path):
    module, root, manifest, calls = _downloads(input_case, monkeypatch)
    for label in ("cold", "warm"):
        report = tmp_path / label
        assert module.cached_inputs(root, report, tmp_path / "cache", "pr-26") == 0
        assert json.loads((report / "cache.json").read_text())["hit"] is (label == "warm")
        assert len(module.verify_inputs(root, manifest, report / "inputs")) == 2
    assert len(calls) == 1
    assert module.input_cache_key(manifest, "trusted") != module.input_cache_key(manifest, "pr-26")


@pytest.mark.parametrize("member", ["sample-1.0-py3-none-any.whl", "refnx-source.tar.gz"])
def test_corrupt_cached_input_never_falls_back_to_download(input_case, monkeypatch, tmp_path, member):
    module, root, _manifest, calls = _downloads(input_case, monkeypatch)
    module.cached_inputs(root, tmp_path / "cold", tmp_path / "cache", "trusted")
    target = next((tmp_path / "cache").glob(f"*/{member}"))
    target.write_bytes(b"modified")
    with pytest.raises(ValueError, match="bytes do not match"):
        module.cached_inputs(root, tmp_path / "warm", tmp_path / "cache", "trusted")
    assert len(calls) == 1
    assert target.read_bytes() == b"modified"
    assert not (tmp_path / "warm/inputs").exists()


def test_failed_download_publishes_no_inputs_or_cache(input_case, monkeypatch, tmp_path):
    module, root, *_rest = input_case()
    monkeypatch.setattr(module, "_download_stage", lambda *args: 29)
    assert module.cached_inputs(root, tmp_path / "report", tmp_path / "cache", "trusted") == 29
    assert not list((tmp_path / "cache").iterdir())
    assert not (tmp_path / "report/inputs").exists()
    assert json.loads((tmp_path / "report/summary.json").read_text())["state"] == "FAIL"


def test_cache_rejects_unexpected_derived_wheel(input_case, monkeypatch, tmp_path):
    module, root, _manifest, calls = _downloads(input_case, monkeypatch)
    module.cached_inputs(root, tmp_path / "cold", tmp_path / "cache", "trusted")
    directory = next((tmp_path / "cache").iterdir())
    (directory / "refnx-0.1.65.dev0-cp312-cp312-macosx_15_0_arm64.whl").write_bytes(b"self-attested wheel")
    with pytest.raises(ValueError, match="exactly"):
        module.cached_inputs(root, tmp_path / "warm", tmp_path / "cache", "trusted")
    assert len(calls) == 1
