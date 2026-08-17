"""Symlink-resistant report directory preparation for verification modes."""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

DirectoryIdentity = tuple[int, int]
DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
DIRECTORY_FD_SUPPORTED = os.mkdir in os.supports_dir_fd and os.open in os.supports_dir_fd

try:
    from verify_publish import _write_new_file_in_anchored_directory  # noqa: E402
except ModuleNotFoundError as error:
    if error.name != "verify_publish":
        raise

    def _write_new_file_in_anchored_directory(
        directory: Path,
        identity: DirectoryIdentity,
        name: str,
        content: bytes,
        *,
        directory_label: str,
        file_label: str,
    ) -> None:
        raise ModuleNotFoundError("verify_publish helper is required for anchored file publication")


@dataclass(frozen=True)
class DirectoryAnchor:
    path: Path
    identity: DirectoryIdentity
    missing: tuple[str, ...]


@dataclass(frozen=True)
class ReportAnchor:
    parent: DirectoryAnchor
    leaf_identity: DirectoryIdentity | None


class ReportGuard:
    """Callable identity guard for a report and its runtime cache directories."""

    def __init__(
        self,
        report: Path,
        identity: DirectoryIdentity | None,
        *,
        allow_missing: bool,
    ) -> None:
        self._report = report
        self._identity = identity
        self._allow_missing = allow_missing
        self._watched: tuple[tuple[Path, str, DirectoryIdentity], ...] = ()

    def watch_directory(self, path: Path, label: str) -> None:
        identity = _directory_identity(path, label)
        self._watched = (*self._watched, (path, label, identity))

    def __call__(self) -> None:
        if self._identity is not None:
            _require_same_directory(self._report, self._identity, "report directory")
        elif os.path.lexists(self._report):
            if not self._allow_missing:
                raise ValueError("report directory appeared unexpectedly")
            self._identity = _directory_identity(self._report, "report directory")
        for path, label, identity in self._watched:
            _require_same_directory(path, identity, label)


def _resolve_output_path(value: str | Path, label: str) -> Path:
    """Resolve an output path only after rejecting a symlink at its leaf."""
    path = Path(value).absolute()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    return path.resolve()


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


def _make_directory_anchor(path: Path, label: str) -> DirectoryAnchor:
    missing: list[str] = []
    current = path
    while not os.path.lexists(current):
        parent = current.parent
        if parent == current:
            raise ValueError(f"{label} parent cannot be created")
        missing.append(current.name)
        current = parent
    identity = _directory_identity(current, label)
    return DirectoryAnchor(current, identity, tuple(reversed(missing)))


def _make_report_anchor(report: Path) -> ReportAnchor:
    leaf_identity = _directory_identity(report, "report directory") if os.path.lexists(report) else None
    return ReportAnchor(_make_directory_anchor(report.parent, "report parent"), leaf_identity)


def _ensure_anchored_parent(anchor: DirectoryAnchor, label: str) -> DirectoryIdentity:
    _require_same_directory(anchor.path, anchor.identity, label)
    if not anchor.missing:
        return anchor.identity
    if DIRECTORY_FD_SUPPORTED:
        descriptor, observed = _open_directory(anchor.path, label)
        if observed != anchor.identity:
            os.close(descriptor)
            raise ValueError(f"{label} changed during validation")
        current = anchor.path
        try:
            for name in anchor.missing:
                child = _open_child_directory(descriptor, name, label, create=True)
                os.close(descriptor)
                descriptor = child
                current /= name
            identity = _descriptor_identity(descriptor, label)
        finally:
            os.close(descriptor)
        _require_same_directory(current, identity, label)
        return identity
    current = anchor.path
    for name in anchor.missing:
        current /= name
        try:
            current.mkdir()
        except FileExistsError:
            pass
        _directory_identity(current, label)
    return _directory_identity(current, label)


def _create_report_leaf(
    report: Path,
    parent_identity: DirectoryIdentity,
    label: str,
) -> DirectoryIdentity:
    parent = report.parent
    if DIRECTORY_FD_SUPPORTED:
        descriptor, observed = _open_directory(parent, "report parent")
        try:
            if observed != parent_identity:
                raise ValueError("report parent changed during validation")
            try:
                os.mkdir(report.name, dir_fd=descriptor)
            except FileExistsError as error:
                raise ValueError(f"{label} appeared during validation") from error
            child = _open_child_directory(descriptor, report.name, label, create=False)
            try:
                identity = _descriptor_identity(child, label)
            finally:
                os.close(child)
        finally:
            os.close(descriptor)
        _require_same_directory(report, identity, label)
        return identity
    try:
        report.mkdir()
    except FileExistsError as error:
        raise ValueError(f"{label} appeared during validation") from error
    identity = _directory_identity(report, label)
    _require_same_directory(parent, parent_identity, "report parent")
    return identity


def _directory_is_empty(path: Path) -> bool:
    return next(path.iterdir(), None) is None


def _prepare_report_directory(
    report: Path,
    anchor: ReportAnchor | None = None,
) -> DirectoryIdentity:
    observed = anchor or _make_report_anchor(report)
    parent_identity = _ensure_anchored_parent(observed.parent, "report parent")
    if observed.leaf_identity is not None:
        _require_same_directory(report, observed.leaf_identity, "report directory")
        return observed.leaf_identity
    if os.path.lexists(report):
        _directory_identity(report, "report directory")
        raise ValueError("report directory appeared during validation")
    return _create_report_leaf(report, parent_identity, "report directory")


def _prepare_regular_report(
    report: Path,
    anchor: ReportAnchor | None = None,
) -> DirectoryIdentity:
    observed = anchor or _make_report_anchor(report)
    identity = _prepare_report_directory(report, observed)
    if not _directory_is_empty(report):
        raise ValueError("report directory must be empty")
    return identity


def _ensure_cache_directory(
    report: Path,
    report_identity: DirectoryIdentity,
    name: str,
) -> DirectoryIdentity:
    _require_same_directory(report, report_identity, "report directory")
    path = report / name
    if DIRECTORY_FD_SUPPORTED:
        descriptor, observed = _open_directory(report, "report directory")
        try:
            if observed != report_identity:
                raise ValueError("report directory changed during validation")
            child = _open_child_directory(descriptor, name, f"{name} directory", create=True)
            try:
                identity = _descriptor_identity(child, f"{name} directory")
            finally:
                os.close(child)
        finally:
            os.close(descriptor)
    else:
        path.mkdir(exist_ok=True)
        identity = _directory_identity(path, f"{name} directory")
    _require_same_directory(report, report_identity, "report directory")
    _require_same_directory(path, identity, f"{name} directory")
    return identity


def _make_report_guard(
    report: Path,
    identity: DirectoryIdentity | None = None,
    *,
    allow_missing: bool = False,
    watched: tuple[tuple[Path, str], ...] = (),
) -> ReportGuard:
    observed = identity
    if observed is None and os.path.lexists(report):
        observed = _directory_identity(report, "report directory")
    if observed is None and not allow_missing:
        raise ValueError("report directory disappeared during validation")
    guard = ReportGuard(report, observed, allow_missing=allow_missing)
    for path, label in watched:
        guard.watch_directory(path, label)
    return guard


def build_environment(
    repo_root: str | Path,
    report_dir: str | Path,
    *,
    expected_identity: DirectoryIdentity | None = None,
) -> dict[str, str]:
    root = Path(repo_root).resolve()
    report = _resolve_output_path(report_dir, "report directory")
    anchor = _make_report_anchor(report)
    report_identity = _prepare_report_directory(report, anchor)
    if expected_identity is not None and report_identity != expected_identity:
        raise ValueError("report directory changed during validation")
    mpl = report / "mpl-cache"
    xdg = report / "xdg-cache"
    _ensure_cache_directory(report, report_identity, "mpl-cache")
    _ensure_cache_directory(report, report_identity, "xdg-cache")
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PYTEST_") or name == "PYTHONOPTIMIZE":
            environment.pop(name)
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(mpl)
    environment["XDG_CACHE_HOME"] = str(xdg)
    return environment
