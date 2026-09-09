#!/usr/bin/env python3
"""Trust-scoped wheel caches with exact keys and verification on every use."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from audit_reports import _write  # noqa: E402
from package_downloads import _summary, download_packages, prepare_report, read_manifest  # noqa: E402
from package_manifest import manifest_bytes, verify_wheels  # noqa: E402
from verify_report import (  # noqa: E402
    _directory_identity,
    _require_same_directory,
    _resolve_output_path,
    _write_new_file_in_anchored_directory,
)


def cache_key(manifest: dict, trust_domain: str) -> str:
    if re.fullmatch(r"trusted|pr-[1-9][0-9]*", trust_domain) is None:
        raise ValueError("cache trust domain must be trusted or pr-N")
    digest = hashlib.sha256(manifest_bytes(manifest)).hexdigest()
    return (
        f"xrr-wheels-v1-{trust_domain}-{manifest['target']}"
        f"-python-{manifest['python_version']}-pip-{manifest['pip_version']}"
        f"-{manifest['lock_sha256']}-{digest}"
    )


def _publish_copy(stage: Path, destination: Path, wheels: Sequence[dict]) -> None:
    destination.mkdir()
    identity = _directory_identity(destination, "cache copy")
    try:
        for item in wheels:
            name = item["filename"]
            _write_new_file_in_anchored_directory(
                destination,
                identity,
                name,
                (stage / name).read_bytes(),
                directory_label="cache copy",
                file_label="cached wheel",
            )
        verify_wheels(destination, wheels)
    except BaseException:
        try:
            _require_same_directory(destination, identity, "cache copy")
        except (OSError, ValueError):
            pass
        else:
            shutil.rmtree(destination)
        raise


def copy_verified(source: Path, destination: Path, wheels: Sequence[dict]) -> None:
    if os.path.lexists(destination):
        raise ValueError("cache copy destination already exists")
    parent = _directory_identity(destination.parent, "cache parent")
    before = verify_wheels(source, wheels)
    with tempfile.TemporaryDirectory(prefix="wheel-copy-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "wheels"
        shutil.copytree(source, stage, symlinks=True)
        if verify_wheels(stage, wheels) != before or verify_wheels(source, wheels) != before:
            raise ValueError("package bytes changed during cache copy")
        _require_same_directory(destination.parent, parent, "cache parent")
        _publish_copy(stage, destination, wheels)
    _require_same_directory(destination.parent, parent, "cache parent")


def _cache_directory(root: Path, cache: Path, report: Path) -> Path:
    cache = _resolve_output_path(cache, "package cache")
    report = _resolve_output_path(report, "package report")
    if cache.is_relative_to(root.resolve()):
        raise ValueError("package cache must be external")
    if cache.is_relative_to(report) or report.is_relative_to(cache):
        raise ValueError("package cache and report must be separate directories")
    cache.mkdir(parents=True, exist_ok=True)
    _directory_identity(cache, "package cache")
    return cache


def _restore(root: Path, cached: Path, report: Path, manifest: dict) -> None:
    verify_wheels(cached, manifest["wheels"])
    report, guard, _environment = prepare_report(root, report)
    _write(report, "manifest.json", manifest_bytes(manifest).decode("ascii"))
    copy_verified(cached, report / "wheels", manifest["wheels"])
    guard()
    _summary(report, "download", 0, target=manifest["target"], vcs_verified=False)


def _cache_members(cache: Path, expected: str) -> None:
    if {path.name for path in cache.iterdir()} - {expected}:
        raise ValueError("package cache contains unrelated members")


def cached_download(root: Path, manifest_path: Path, report: Path, cache: Path, trust_domain: str) -> int:
    manifest = read_manifest(root, manifest_path)
    key = cache_key(manifest, trust_domain)
    cache = _cache_directory(root, cache, report)
    identity = _directory_identity(cache, "package cache")
    cached = cache / hashlib.sha256(key.encode("ascii")).hexdigest()
    _cache_members(cache, cached.name)
    hit = os.path.lexists(cached)
    if hit:
        _restore(root, cached, report, manifest)
    else:
        code = download_packages(root, manifest_path, report)
        if code != 0:
            return code
        _require_same_directory(cache, identity, "package cache")
        copy_verified(report / "wheels", cached, manifest["wheels"])
    _require_same_directory(cache, identity, "package cache")
    _cache_members(cache, cached.name)
    if read_manifest(root, manifest_path) != manifest:
        raise ValueError("package manifest changed during cache operation")
    verify_wheels(cached, manifest["wheels"])
    verify_wheels(report / "wheels", manifest["wheels"])
    evidence = {"state": "PASS", "key": key, "hit": hit, "trust_domain": trust_domain}
    _write(report, "cache.json", manifest_bytes(evidence).decode("ascii"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("key", "fetch"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trust-domain", required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "fetch" and (args.cache_dir is None or args.report_dir is None):
        parser.error("fetch requires --cache-dir and --report-dir")
    root = args.repo_root.resolve()
    try:
        if args.mode == "key":
            print(cache_key(read_manifest(root, args.manifest), args.trust_domain))
            return 0
        return cached_download(root, args.manifest, args.report_dir, args.cache_dir, args.trust_domain)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
