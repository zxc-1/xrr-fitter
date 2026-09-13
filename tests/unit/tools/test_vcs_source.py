from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(load_tool_module):
    assert (ROOT / "tools/vcs_source.py").is_file(), "missing pinned VCS archive verification"
    return load_tool_module("vcs_source")


def _tar(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in entries:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
    return stream.getvalue()


def _git(root, *arguments):
    return subprocess.run(("git", "-C", str(root), *arguments), capture_output=True, check=True).stdout


@pytest.fixture
def vcs_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    (source / "code.py").write_bytes(b"value = 1\n")
    (source / "ignored.txt").write_bytes(b"not exported\n")
    (source / ".gitattributes").write_bytes(b"ignored.txt export-ignore\n")
    _git(source, "add", ".")
    _git(source, "-c", "user.name=Audit Test", "-c", "user.email=audit@example.invalid", "commit", "-qm", "fixture")
    commit = _git(source, "rev-parse", "HEAD").decode().strip()
    archive = tmp_path / "refnx.tar.gz"
    archive.write_bytes(
        _tar(
            [
                (f"refnx-{commit}/code.py", b"value = 1\n"),
                (f"refnx-{commit}/.gitattributes", b"ignored.txt export-ignore\n"),
            ]
        )
    )
    return source, archive, commit


def _bound(vcs_fixture, load_tool_module, monkeypatch):
    module = _module(load_tool_module)
    source, archive, commit = vcs_fixture
    binding = {"vcs": [{"name": "refnx", "url": "https://github.com/refnx/refnx.git", "commit": commit}]}
    monkeypatch.setattr(module, "locked_inputs", lambda root, target: ({}, binding))
    return module, source, archive, commit


def test_vcs_archive_matches_git_export_not_a_fabricated_wheel(vcs_fixture, load_tool_module, monkeypatch):
    module, source, archive, commit = _bound(vcs_fixture, load_tool_module, monkeypatch)
    result = module.record_source(ROOT, source, archive)
    assert result["vcs"]["commit"] == commit
    assert result["tree_oid"] == _git(source, "rev-parse", "HEAD^{tree}").decode().strip()
    assert result["scope"] == "git-archive"
    assert result["file_count"] == 2
    assert result["archive"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert "wheel" not in result
    module.verify_source(ROOT, result, archive)


def test_vcs_archive_rejects_content_not_in_the_locked_git_commit(vcs_fixture, load_tool_module, monkeypatch):
    module, source, archive, commit = _bound(vcs_fixture, load_tool_module, monkeypatch)
    archive.write_bytes(_tar([(f"refnx-{commit}/code.py", b"altered\n")]))
    with pytest.raises(ValueError, match="Git archive"):
        module.record_source(ROOT, source, archive)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "root/../outside", "root/code\\name"])
def test_source_archive_rejects_unsafe_members_without_extracting(load_tool_module, name):
    module = _module(load_tool_module)
    with pytest.raises(ValueError, match="archive"):
        module.archive_files(_tar([(name, b"bad")]), "root")


def test_source_verification_rejects_modified_bytes(vcs_fixture, load_tool_module, monkeypatch):
    module, source, archive, _commit = _bound(vcs_fixture, load_tool_module, monkeypatch)
    result = module.record_source(ROOT, source, archive)
    archive.write_bytes(b"modified")
    with pytest.raises(ValueError, match="bytes"):
        module.verify_source(ROOT, result, archive)


@pytest.mark.parametrize("field", ["vcs", "schema", "scope", "archive"])
def test_source_manifest_rejects_changed_bindings(vcs_fixture, field, load_tool_module, monkeypatch):
    module, source, archive, _commit = _bound(vcs_fixture, load_tool_module, monkeypatch)
    result = deepcopy(module.record_source(ROOT, source, archive))
    result[field] = "changed"
    with pytest.raises(ValueError):
        module.verify_source(ROOT, result, archive)
