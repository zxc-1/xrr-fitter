#!/usr/bin/env python3
"""Freeze the pinned ORSO validation suite for offline regression testing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence
import urllib.request


SUITE_COMMIT = "6a01b4a4febfc52cd3881d2147c732dd1701bc8e"
INDEX_SCHEMA = "xrr-r23-orso-suite-index-v1"
RAW_ROOT = "https://raw.githubusercontent.com/reflectivity/analysis"
SUITE_ROOT = "validation/test"
MANIFEST_COUNT = 8


def canonical_json_bytes(value: object) -> bytes:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def manifest_references(text: str) -> tuple[str, ...]:
    lines = tuple(
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    )
    if len(lines) != 2:
        return _reject(f"manifest must name exactly two references, found {len(lines)}")
    return lines


def _reject(detail: str) -> tuple[str, ...]:
    raise ValueError(detail)


def _manifests(unpolarised: Path) -> tuple[Path, ...]:
    return tuple(sorted(unpolarised.glob("test?.txt")))


def _referenced_paths(unpolarised: Path) -> tuple[Path, ...]:
    collected: list[Path] = []
    for manifest in _manifests(unpolarised):
        collected.append(manifest)
        text = manifest.read_text(encoding="utf-8")
        collected.extend(unpolarised / reference for reference in manifest_references(text))
    return tuple(sorted(set(collected)))


def _record(root: Path, path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing referenced suite file: {path.relative_to(root).as_posix()}")
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def frozen_index(root: Path) -> dict[str, object]:
    unpolarised = Path(root) / "unpolarised"
    records = [_record(Path(root), path) for path in _referenced_paths(unpolarised)]
    records.sort(key=lambda record: str(record["path"]))
    return {"schema": INDEX_SCHEMA, "suite_commit": SUITE_COMMIT, "files": records}


def _verify_record(root: Path, record: dict[str, object]) -> None:
    path = root / str(record["path"])
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing frozen suite file: {record['path']}")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != record["sha256"] or len(content) != record["size"]:
        raise ValueError(f"sha256 mismatch for {record['path']}; rerun tools/sync_orso_suite.py")


def verify_index(root: Path) -> None:
    root = Path(root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if index.get("schema") != INDEX_SCHEMA or index.get("suite_commit") != SUITE_COMMIT:
        raise ValueError("frozen ORSO index schema or suite commit does not match this tool")
    for record in index["files"]:
        _verify_record(root, record)


def _download(relative: str) -> bytes:
    url = f"{RAW_ROOT}/{SUITE_COMMIT}/{SUITE_ROOT}/{relative}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - pinned https literal
        return response.read()


def _write(root: Path, relative: str, content: bytes) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _fetch_suite(root: Path) -> None:
    for ordinal in range(MANIFEST_COUNT):
        relative = f"unpolarised/test{ordinal}.txt"
        content = _download(relative)
        _write(root, relative, content)
        for reference in manifest_references(content.decode("utf-8")):
            _write(root, f"unpolarised/{reference}", _download(f"unpolarised/{reference}"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", action="store_true")
    group.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.check:
        verify_index(args.root)
        print(json.dumps({"status": "PASS", "suite_commit": SUITE_COMMIT}, sort_keys=True))
        return 0
    _fetch_suite(args.root)
    index = frozen_index(args.root)
    (args.root / "index.json").write_bytes(canonical_json_bytes(index))
    print(json.dumps({"status": "PASS", "file_count": len(index["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
