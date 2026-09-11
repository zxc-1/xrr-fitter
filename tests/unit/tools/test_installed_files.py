from __future__ import annotations

import os
from pathlib import Path

import pytest
from tests.support.installed_wheels import make_wheel


def test_installed_snapshot_rejects_rewrite_and_restore(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("installed_files")
    path = tmp_path / "input"
    path.write_bytes(b"original")
    snapshot = module.InstalledSnapshot(tmp_path)
    original = module._verified_file

    def changed(file, digest):
        before = file.stat()
        result = original(file, digest)
        file.write_bytes(b"modified")
        file.write_bytes(b"original")
        os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(module, "_verified_file", changed)
    with pytest.raises(ValueError, match="identity"):
        snapshot.verify(path, __import__("hashlib").sha256(b"original").hexdigest())


def test_snapshot_checks_parent_identity_after_a_read(tmp_path, load_tool_module):
    module = load_tool_module("installed_files")
    root = tmp_path / "input"
    root.mkdir()
    path = root / "file"
    path.write_bytes(b"data")
    snapshot = module.InstalledSnapshot(root)
    assert snapshot.read(path) == b"data"
    root.rename(tmp_path / "original")
    root.mkdir()
    (root / "file").write_bytes(b"data")
    with pytest.raises(ValueError, match="directory.*changed"):
        snapshot.finish()


def test_snapshot_rejects_a_link_in_the_root_ancestor_chain(tmp_path, load_tool_module):
    module = load_tool_module("installed_files")
    real = tmp_path / "real"
    (real / "child").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|junction"):
        module.InstalledSnapshot(alias / "child")


def test_unowned_links_are_not_followed_but_remain_identity_bound(tmp_path, load_tool_module):
    module = load_tool_module("installed_files")
    path = tmp_path / "python"
    path.symlink_to("outside-interpreter")
    snapshot = module.InstalledSnapshot(tmp_path)
    assert snapshot.unowned(path)["kind"] == "link-not-followed"
    path.rename(tmp_path / "original-link")
    path.symlink_to("different-interpreter")
    with pytest.raises(ValueError, match="link.*changed|identity"):
        snapshot.finish()


def test_wheel_initialization_failure_closes_its_archive(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("installed_files")
    wheel = make_wheel(tmp_path)
    opened = []
    original = module.zipfile.ZipFile

    def archive(*args, **kwargs):
        value = original(*args, **kwargs)
        opened.append(value)
        return value

    def reject(self):
        raise ValueError("changed wheel")

    monkeypatch.setattr(module.zipfile, "ZipFile", archive)
    monkeypatch.setattr(module.VerifiedWheel, "guard", reject)
    with pytest.raises(ValueError, match="changed wheel"):
        module.VerifiedWheel(wheel["path"], wheel["record"])
    assert opened and all(value.fp is None for value in opened)


def test_bound_read_checks_the_open_file_not_a_later_reopening(tmp_path, load_tool_module, monkeypatch):
    module = load_tool_module("installed_inputs")
    path = tmp_path / "input.lock"
    path.write_bytes(b"exact input")

    def unexpected_read(self):
        pytest.fail("unanchored Path.read_bytes was used for an installation input")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    assert module.read_bound(path) == b"exact input"
