"""Atomic publication for deterministic export artifact trees.

This module is the only directory-publication path used by result exports and
the fixed example bundle. Callers provide immutable bytes and normalized
relative names; project validation and numerical serialization remain outside
this filesystem boundary.

Run publication reserves a hidden sibling directory, writes every artifact,
re-reads it to construct size and digest records, flushes files and owned
directories, and finally performs one operating-system no-replace rename. A
concurrent final-name collision can therefore never replace an existing run.

Dataset IDs are preserved exactly in the returned manifest. Filesystem names
use a bounded display slug plus an exact-ID digest, so hostile or visually
equivalent identifiers remain distinct without becoming traversal paths.

Exact-tree publication applies the same write, validation, durability, rename,
and cleanup path to a fixed destination. An identical existing tree is an
idempotent success. Unknown members, links, special files, or byte conflicts
fail closed before any replacement is attempted.

Cleanup is restricted to the private path allocated by the current call. If
both the primary operation and cleanup fail, both exceptions are retained in
an exception group; cleanup errors are never hidden behind the original error.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import sys
import unicodedata

from xrr_fitter.model.export import (
    DatasetExportManifest,
    ExportFileRecord,
    ExportManifest,
)


TIMESTAMP_PATTERN = re.compile(r"\d{8}T\d{6}Z?")
UNSAFE_SLUG_PATTERN = re.compile(r"[^\w-]+", flags=re.UNICODE)
UNSUPPORTED_DIRECTORY_FSYNC = frozenset(
    {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP} - {None}
)


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("artifact path must be a normalized relative POSIX path")
    return value


def _validate_path_set(paths: tuple[str, ...], field: str) -> None:
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field} contain duplicate paths")
    pure_paths = tuple(map(PurePosixPath, paths))
    known = set(pure_paths)
    if any(known.intersection(path.parents) for path in pure_paths):
        raise ValueError(f"{field} contain file/ancestor conflicts")


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    """One nonempty immutable payload at a relative publication path."""

    path: str
    content: bytes

    def __post_init__(self) -> None:
        path = _relative_path(self.path)
        if not isinstance(self.content, bytes):
            raise TypeError("artifact content must be bytes")
        if not self.content:
            raise ValueError("artifact content must not be empty")
        object.__setattr__(self, "path", path)


def _payloads(values: object, field: str) -> tuple[ArtifactPayload, ...]:
    payloads = tuple(values)
    if any(not isinstance(value, ArtifactPayload) for value in payloads):
        raise TypeError(f"{field} must contain ArtifactPayload values")
    ordered = tuple(sorted(payloads, key=lambda value: value.path))
    paths = tuple(value.path for value in ordered)
    _validate_path_set(paths, field)
    return ordered


@dataclass(frozen=True, slots=True)
class DatasetArtifacts:
    """Serialized artifacts owned by one exact dataset identity."""

    dataset_id: str
    files: tuple[ArtifactPayload, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        files = _payloads(self.files, "dataset files")
        if not files:
            raise ValueError("dataset files must not be empty")
        object.__setattr__(self, "files", files)


def _datasets(values: object) -> tuple[DatasetArtifacts, ...]:
    datasets = tuple(values)
    if not datasets or any(not isinstance(value, DatasetArtifacts) for value in datasets):
        raise TypeError("datasets must contain DatasetArtifacts values")
    identifiers = tuple(value.dataset_id for value in datasets)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset IDs must be unique")
    return datasets


def _validate_timestamp(value: str) -> str:
    if not isinstance(value, str) or TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError("run_timestamp must match YYYYMMDDTHHMMSS[Z]")
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dataset_slug(dataset_id: str, limit: int = 48) -> str:
    normalized = unicodedata.normalize("NFKC", dataset_id).casefold()
    slug = UNSAFE_SLUG_PATTERN.sub("-", normalized).strip(".- ")
    slug = re.sub(r"-+", "-", slug)[:limit].strip(".- ")
    return slug or "dataset"


def _dataset_directory(order: int, dataset_id: str) -> str:
    identity = sha256(dataset_id.encode("utf-8")).hexdigest()[:8]
    return f"{order:03d}-{_dataset_slug(dataset_id)}-{identity}"


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _allocate_run(output: Path, timestamp: str) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    while True:
        token = secrets.token_hex(4)
        final = output / f"{timestamp}-{token}"
        partial = output / f".partial-{timestamp}-{token}"
        if _lexists(final) or _lexists(partial):
            continue
        try:
            partial.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return partial, final


def _write_payload(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def _record(root: Path, path: Path) -> ExportFileRecord:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"export artifact is not a regular file: {path.name}")
    content = path.read_bytes()
    if not content:
        raise ValueError(f"export artifact is empty: {path.name}")
    relative = path.relative_to(root).as_posix()
    return ExportFileRecord(relative, len(content), sha256(content).hexdigest())


def _sync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    try:
        _sync_file(path)
    except OSError as error:
        if error.errno not in UNSUPPORTED_DIRECTORY_FSYNC:
            raise


def _tree_directories(root: Path, files: tuple[Path, ...]) -> tuple[Path, ...]:
    directories = {root}
    for path in files:
        parent = path.parent
        while parent != root:
            directories.add(parent)
            parent = parent.parent
    return tuple(sorted(directories, key=lambda value: (-len(value.parts), value.as_posix())))


def _sync_tree(root: Path, files: tuple[Path, ...]) -> None:
    for path in files:
        _sync_file(path)
    for directory in _tree_directories(root, files):
        _sync_directory(directory)


def _raise_rename_error(target: Path) -> None:
    code = ctypes.get_errno()
    raise OSError(code, os.strerror(code), target)


def _darwin_rename(source: Path, target: Path) -> None:
    rename = ctypes.CDLL(None, use_errno=True).renamex_np
    rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(os.fsencode(source), os.fsencode(target), 0x00000004) != 0:
        _raise_rename_error(target)


def _linux_rename(source: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOTSUP, "renameat2 is unavailable") from error
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        _raise_rename_error(target)


def _windows_rename(source: Path, target: Path) -> None:
    rename = ctypes.windll.kernel32.MoveFileExW
    rename.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(str(source), str(target), 0) == 0:
        raise ctypes.WinError()


def _rename_exclusive(source: Path, target: Path) -> None:
    if sys.platform == "darwin":
        _darwin_rename(source, target)
        return
    if sys.platform.startswith("linux"):
        _linux_rename(source, target)
        return
    if os.name == "nt":
        _windows_rename(source, target)
        return
    raise OSError(errno.ENOTSUP, "exclusive directory rename is unsupported")


def _publish_directory(partial: Path, final: Path) -> None:
    _rename_exclusive(partial, final)


def _cleanup_partial(partial: Path) -> None:
    if _lexists(partial):
        shutil.rmtree(partial)


def _cleanup_after_failure(partial: Path, error: BaseException) -> None:
    try:
        _cleanup_partial(partial)
    except BaseException as cleanup_error:
        raise BaseExceptionGroup(
            "publication and cleanup both failed",
            [error, cleanup_error],
        ) from None


def _write_artifacts(root: Path, values: tuple[ArtifactPayload, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for value in values:
        path = root / PurePosixPath(value.path)
        _write_payload(path, value.content)
        paths.append(path)
    return tuple(paths)


def _stage_export(
    partial: Path,
    datasets: tuple[DatasetArtifacts, ...],
    root_files: tuple[ArtifactPayload, ...],
) -> tuple[ExportManifest, tuple[Path, ...]]:
    written = list(_write_artifacts(partial, root_files))
    root_records = tuple(_record(partial, path) for path in written)
    dataset_records: list[DatasetExportManifest] = []
    for order, dataset in enumerate(datasets, start=1):
        directory = _dataset_directory(order, dataset.dataset_id)
        paths = _write_artifacts(partial / directory, dataset.files)
        written.extend(paths)
        records = tuple(_record(partial, path) for path in paths)
        dataset_records.append(
            DatasetExportManifest(dataset.dataset_id, directory, records)
        )
    manifest = ExportManifest(partial, tuple(dataset_records), root_records)
    return manifest, tuple(written)


def _validate_export_layout(
    datasets: tuple[DatasetArtifacts, ...],
    root_files: tuple[ArtifactPayload, ...],
) -> None:
    paths = [value.path for value in root_files]
    for order, dataset in enumerate(datasets, start=1):
        directory = _dataset_directory(order, dataset.dataset_id)
        paths.extend(f"{directory}/{value.path}" for value in dataset.files)
    _validate_path_set(tuple(paths), "export files")


def publish_export_run(
    output_dir: str | Path,
    datasets: object,
    root_files: object = (),
    *,
    run_timestamp: str | None = None,
) -> ExportManifest:
    """Atomically publish one collision-safe export run."""
    dataset_values = _datasets(datasets)
    root_values = _payloads(root_files, "root files")
    _validate_export_layout(dataset_values, root_values)
    timestamp = _validate_timestamp(
        _utc_timestamp() if run_timestamp is None else run_timestamp
    )
    partial, final = _allocate_run(Path(output_dir), timestamp)
    try:
        staged, written = _stage_export(partial, dataset_values, root_values)
        _sync_tree(partial, written)
        _publish_directory(partial, final)
    except BaseException as error:
        _cleanup_after_failure(partial, error)
        raise
    return ExportManifest(final, staged.datasets, staged.root_files)


def _expected_tree(values: tuple[ArtifactPayload, ...]) -> tuple[set[str], set[str]]:
    files = {value.path for value in values}
    directories = {
        PurePosixPath(value.path).parent.as_posix()
        for value in values
        if PurePosixPath(value.path).parent != PurePosixPath(".")
    }
    expanded = {
        parent.as_posix()
        for directory in directories
        for parent in (PurePosixPath(directory), *PurePosixPath(directory).parents)
        if parent != PurePosixPath(".")
    }
    return files, expanded


def _validate_existing_target(destination: Path) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise FileExistsError(f"conflicting publication target: {destination}")


def _tree_members(destination: Path) -> tuple[Path, ...]:
    members = tuple(destination.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise FileExistsError(f"unexpected symlink in publication target: {destination}")
    return members


def _observed_files(destination: Path, members: tuple[Path, ...]) -> set[str]:
    return {
        path.relative_to(destination).as_posix()
        for path in members
        if path.is_file()
    }


def _observed_directories(destination: Path, members: tuple[Path, ...]) -> set[str]:
    return {
        path.relative_to(destination).as_posix()
        for path in members
        if path.is_dir()
    }


def _observed_tree(destination: Path) -> tuple[set[str], set[str]]:
    members = _tree_members(destination)
    unsupported = tuple(path for path in members if not path.is_file() and not path.is_dir())
    if unsupported:
        raise FileExistsError(f"unexpected special file in publication target: {destination}")
    return (
        _observed_files(destination, members),
        _observed_directories(destination, members),
    )


def _validate_observed_tree(
    destination: Path,
    observed: tuple[set[str], set[str]],
    expected: tuple[set[str], set[str]],
) -> None:
    if observed != expected:
        raise FileExistsError(f"unexpected members in publication target: {destination}")


def _tree_bytes_match(
    destination: Path,
    values: tuple[ArtifactPayload, ...],
) -> bool:
    expected = {value.path: value.content for value in values}
    return all((destination / path).read_bytes() == content for path, content in expected.items())


def _existing_tree_matches(
    destination: Path,
    values: tuple[ArtifactPayload, ...],
) -> bool:
    _validate_existing_target(destination)
    _validate_observed_tree(
        destination,
        _observed_tree(destination),
        _expected_tree(values),
    )
    return _tree_bytes_match(destination, values)


def _allocate_exact_partial(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    while True:
        partial = destination.parent / f".partial-{destination.name}-{secrets.token_hex(4)}"
        if _lexists(partial):
            continue
        try:
            partial.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return partial


def publish_exact_tree(
    destination: Path,
    files: object,
) -> tuple[Path, ...]:
    """Publish an exact immutable tree or accept an identical existing tree."""
    target = Path(destination)
    if not target.name or target in {Path("."), Path("..")}:
        raise ValueError("destination must identify a directory")
    values = _payloads(files, "files")
    if not values:
        raise ValueError("files must not be empty")
    if _lexists(target):
        if _existing_tree_matches(target, values):
            return tuple(target / value.path for value in values)
        raise FileExistsError(f"conflicting files in publication target: {target}")
    partial = _allocate_exact_partial(target)
    try:
        written = _write_artifacts(partial, values)
        for path in written:
            _record(partial, path)
        _sync_tree(partial, written)
        _publish_directory(partial, target)
    except BaseException as error:
        _cleanup_after_failure(partial, error)
        raise
    return tuple(target / value.path for value in values)
