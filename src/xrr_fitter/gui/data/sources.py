"""Shared deterministic folder discovery for import previews and commits."""

from pathlib import Path

SUPPORTED_SUFFIXES = {".xy", ".dat", ".txt"}


def folder_data_paths(folder: Path, recursive: bool) -> tuple[Path, ...]:
    entries = folder.rglob("*") if recursive else folder.glob("*")
    return tuple(
        sorted(
            (path for path in entries if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES),
            key=lambda path: path.relative_to(folder).as_posix().casefold(),
        )
    )
