"""Anchored, read-only snapshots of installed files and verified wheel inputs."""

from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from package_manifest import _file_identity, _same_file_identity, _verified_file  # noqa: E402
from verify_publish import _directory_identity, _require_same_directory  # noqa: E402
from wheel_inventory import inspect_wheel  # noqa: E402

READ_LIMIT = 16 * 1024**2
MEMBER_LIMIT = 500_000


def _link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _same(left: tuple, right: tuple) -> None:
    if not _same_file_identity(left, right):
        raise ValueError("installed file identity changed")


class InstalledSnapshot:
    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self.directories: dict[Path, tuple] = {}
        self.files: dict[Path, tuple] = {}
        self.links: dict[Path, tuple] = {}
        for directory in reversed((self.root, *self.root.parents)):
            self._directory(directory)

    def _directory(self, path: Path) -> None:
        if _link(path):
            raise ValueError("installed directory contains a symlink or junction")
        if path in self.directories:
            _require_same_directory(path, self.directories[path], "installed directory")
        else:
            self.directories[path] = _directory_identity(path, "installed directory")

    def anchor(self, path: Path, *, missing: bool = False) -> Path:
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(self.root):
            raise ValueError("installed path escapes the environment")
        self._directory(self.root)
        for parent in reversed(path.parent.parents):
            if parent.is_relative_to(self.root):
                self._directory(parent)
        if path.parent != self.root:
            self._parent(path.parent, missing)
        if _link(path):
            raise ValueError("installed file is a symlink or junction")
        return path

    def _parent(self, parent: Path, missing: bool) -> None:
        if missing and not parent.exists():
            if _link(parent):
                raise ValueError("installed parent is a symlink")
            return
        self._directory(parent)

    def _remember(self, path: Path) -> tuple:
        identity = _file_identity(path.lstat())
        if path in self.files:
            _same(identity, self.files[path])
        self.files[path] = identity
        return identity

    def read(self, path: Path, *, limit: int = READ_LIMIT) -> bytes:
        path = self.anchor(path)
        before = self._remember(path)
        if before[2] > limit:
            raise ValueError("installed file exceeds the read limit")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags), "rb") as handle:
            _same(_file_identity(os.fstat(handle.fileno())), before)
            content = handle.read(limit + 1)
            _same(_file_identity(os.fstat(handle.fileno())), before)
        if len(content) != before[2]:
            raise ValueError("installed bytes changed while reading")
        _same(_file_identity(path.lstat()), before)
        return content

    def verify(self, path: Path, digest: str) -> dict:
        path = self.anchor(path)
        before = self._remember(path)
        record = _verified_file(path, digest)
        _same(_file_identity(path.lstat()), before)
        return {"path": path.relative_to(self.root).as_posix(), "sha256": digest, "size": record["size"]}

    def members(self) -> dict[str, Path]:
        pending = [self.root]
        result: dict[str, Path] = {}
        while pending:
            directory = pending.pop()
            self._directory(directory)
            for path in directory.iterdir():
                if not _link(path) and path.is_dir():
                    pending.append(path)
                else:
                    result[path.relative_to(self.root).as_posix()] = path
            if len(result) + len(self.directories) + len(pending) > MEMBER_LIMIT:
                raise ValueError("installed environment exceeds the member limit")
        return result

    def unowned(self, path: Path) -> dict:
        relative = path.relative_to(self.root).as_posix()
        if _link(path):
            value = self._link_value(path)
            self.links[path] = value
            return {"path": relative, "kind": "link-not-followed", "target_sha256": value[1]}
        path = self.anchor(path)
        before = self._remember(path)
        with path.open("rb") as handle:
            _same(_file_identity(os.fstat(handle.fileno())), before)
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            _same(_file_identity(os.fstat(handle.fileno())), before)
        return {**self.verify(path, digest), "kind": "outside-wheel-ownership"}

    @staticmethod
    def _link_value(path: Path) -> tuple:
        before = path.lstat()
        if not _link(path):
            raise ValueError("unowned installed link changed")
        target = os.fsencode(os.readlink(path))
        after = path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        identity = tuple(getattr(before, name) for name in fields)
        if identity != tuple(getattr(after, name) for name in fields):
            raise ValueError("unowned installed link changed while reading")
        return identity, hashlib.sha256(target).hexdigest()

    def finish(self) -> None:
        for path, identity in self.directories.items():
            _require_same_directory(path, identity, "installed directory")
        for path, identity in self.files.items():
            _same(_file_identity(self.anchor(path).lstat()), identity)
        for path, value in self.links.items():
            if self._link_value(path) != value:
                raise ValueError("unowned installed link identity changed")


class VerifiedWheel:
    def __init__(self, path: Path, record: dict, *, native_loaders: bool = False) -> None:
        self.path = path
        self.record = dict(record)
        self.identity = _file_identity(path.lstat())
        self.inventory = inspect_wheel(path, record, native_loaders=native_loaders)
        self.files = {item["path"]: item for item in self.inventory["files"]}
        self.archive = zipfile.ZipFile(path)
        try:
            self.guard()
        except BaseException:
            self.archive.close()
            raise

    def guard(self) -> None:
        _same(_file_identity(os.fstat(self.archive.fp.fileno())), self.identity)
        _same(_file_identity(self.path.lstat()), self.identity)

    def read(self, name: str) -> bytes:
        self.guard()
        if self.files[name]["size"] > READ_LIMIT:
            raise ValueError("wheel transformation input exceeds the read limit")
        result = self.archive.read(name)
        self.guard()
        if hashlib.sha256(result).hexdigest() != self.files[name]["sha256"]:
            raise ValueError("wheel member bytes changed during installation inspection")
        return result

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback) -> None:
        try:
            self.guard()
            _verified_file(self.path, self.record["sha256"])
            self.guard()
        finally:
            self.archive.close()
