"""Atomic publication helpers for symlink-resistant anchored directories."""

from __future__ import annotations

import os
import stat
import tempfile
import uuid
from pathlib import Path

DirectoryIdentity = tuple[int, int]

DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
DIRECTORY_FD_SUPPORTED = os.mkdir in os.supports_dir_fd and os.open in os.supports_dir_fd
ANCHORED_FILE_FD_SUPPORTED = (
    os.link in os.supports_dir_fd
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
)


def _directory_identity(path: Path, label: str) -> DirectoryIdentity:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} disappeared during validation") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")
    return metadata.st_dev, metadata.st_ino


def _descriptor_identity(descriptor: int, label: str) -> DirectoryIdentity:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")
    return metadata.st_dev, metadata.st_ino


def _open_directory(path: Path, label: str) -> tuple[int, DirectoryIdentity]:
    path_identity = _directory_identity(path, label)
    try:
        descriptor = os.open(path, DIRECTORY_OPEN_FLAGS)
    except OSError as error:
        raise ValueError(f"{label} changed during validation") from error
    try:
        descriptor_identity = _descriptor_identity(descriptor, label)
        if descriptor_identity != path_identity:
            raise ValueError(f"{label} changed during validation")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, descriptor_identity


def _open_child_directory(descriptor: int, name: str, label: str, *, create: bool) -> int:
    if create:
        try:
            os.mkdir(name, dir_fd=descriptor)
        except FileExistsError:
            pass
    try:
        child = os.open(name, DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
    except OSError as error:
        raise ValueError(f"{label} must be a directory") from error
    try:
        _descriptor_identity(child, label)
    except BaseException:
        os.close(child)
        raise
    return child


def _require_same_directory(path: Path, identity: DirectoryIdentity, label: str) -> None:
    if _directory_identity(path, label) != identity:
        raise ValueError(f"{label} changed during validation")


def _validate_child_filename(name: str, label: str) -> None:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"{label} must be a direct child file name")


def _reject_existing_child(descriptor: int, name: str, label: str) -> None:
    try:
        os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise ValueError(f"{label} must be a new regular-file path")


def _open_unique_temporary_child(descriptor: int, target_name: str) -> tuple[int, str]:
    prefix = f".{target_name}."
    for _ in range(1024):
        name = f"{prefix}{uuid.uuid4().hex}.tmp"
        try:
            temporary = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=descriptor,
            )
        except FileExistsError:
            continue
        return temporary, name
    raise ValueError("temporary file name generation failed")


def _unlink_child(descriptor: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=descriptor)
    except FileNotFoundError:
        pass


def _unlink_matching_regular_file(path: Path, content: bytes) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        return
    try:
        if path.read_bytes() == content:
            path.unlink()
    except (FileNotFoundError, OSError):
        pass


def _publish_new_file_by_directory_fd(
    directory: Path,
    identity: DirectoryIdentity,
    name: str,
    content: bytes,
    *,
    directory_label: str,
    file_label: str,
) -> None:
    _require_same_directory(directory, identity, directory_label)
    descriptor, observed = _open_directory(directory, directory_label)
    temporary_name: str | None = None
    try:
        if observed != identity:
            raise ValueError(f"{directory_label} changed during validation")
        _require_same_directory(directory, identity, directory_label)
        _reject_existing_child(descriptor, name, file_label)
        temporary, temporary_name = _open_unique_temporary_child(descriptor, name)
        try:
            with os.fdopen(temporary, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            _unlink_child(descriptor, temporary_name)
            temporary_name = None
            raise
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ValueError(f"{file_label} appeared during validation") from error
        _unlink_child(descriptor, temporary_name)
        temporary_name = None
        os.fsync(descriptor)
        try:
            _require_same_directory(directory, identity, directory_label)
        except BaseException:
            _unlink_child(descriptor, name)
            try:
                os.fsync(descriptor)
            except OSError:
                pass
            raise
    finally:
        if temporary_name is not None:
            _unlink_child(descriptor, temporary_name)
        os.close(descriptor)


def _stage_pathname_publish(
    directory: Path,
    identity: DirectoryIdentity,
    name: str,
    *,
    directory_label: str,
    file_label: str,
) -> tuple[int, Path, int]:
    _require_same_directory(directory, identity, directory_label)
    target = directory / name
    if directory.is_symlink() or not directory.is_dir() or os.path.lexists(target):
        raise ValueError(f"{file_label} must be a new regular-file path")
    directory_fd, observed = _open_directory(directory, directory_label)
    try:
        if observed != identity:
            raise ValueError(f"{directory_label} changed during validation")
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp",
            dir=directory,
        )
    except BaseException:
        os.close(directory_fd)
        raise
    return directory_fd, Path(temporary_name), temporary_descriptor


def _cleanup_pathname_publish(
    directory_fd: int | None,
    target: Path,
    content: bytes,
) -> None:
    if directory_fd is not None and os.unlink in os.supports_dir_fd:
        _unlink_child(directory_fd, target.name)
    _unlink_matching_regular_file(target, content)


def _publish_new_file_by_pathname(
    directory: Path,
    identity: DirectoryIdentity,
    name: str,
    content: bytes,
    *,
    directory_label: str,
    file_label: str,
) -> None:
    _require_same_directory(directory, identity, directory_label)
    target = directory / name
    directory_fd: int | None = None
    temporary: Path | None = None
    published = False
    try:
        directory_fd, temporary, temporary_descriptor = _stage_pathname_publish(
            directory,
            identity,
            name,
            directory_label=directory_label,
            file_label=file_label,
        )
        with os.fdopen(temporary_descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as error:
            raise ValueError(f"{file_label} appeared during validation") from error
        except (AttributeError, NotImplementedError, OSError, TypeError) as error:
            raise ValueError(f"{file_label} cannot be published atomically on this platform") from error
        published = True
        os.fsync(directory_fd)
        _require_same_directory(directory, identity, directory_label)
    except BaseException:
        if published:
            _cleanup_pathname_publish(directory_fd, target, content)
        raise
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        if directory_fd is not None:
            os.close(directory_fd)


def _write_new_file_in_anchored_directory(
    directory: Path,
    identity: DirectoryIdentity,
    name: str,
    content: bytes,
    *,
    directory_label: str,
    file_label: str,
) -> None:
    """Atomically publish a new child file bound to a captured directory inode."""
    _validate_child_filename(name, file_label)
    if DIRECTORY_FD_SUPPORTED and ANCHORED_FILE_FD_SUPPORTED:
        _publish_new_file_by_directory_fd(
            directory,
            identity,
            name,
            content,
            directory_label=directory_label,
            file_label=file_label,
        )
        return
    _publish_new_file_by_pathname(
        directory,
        identity,
        name,
        content,
        directory_label=directory_label,
        file_label=file_label,
    )
