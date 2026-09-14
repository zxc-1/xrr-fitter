#!/usr/bin/env python3
"""Fetch immutable refnx source/builder inputs, never self-attested built wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from audit_reports import _write  # noqa: E402
from lock_environment import _atomic_write  # noqa: E402
from package_cache import _cache_directory, _cache_members, cache_key, copy_verified  # noqa: E402
from package_downloads import _download_stage, _require_pip, _summary, prepare_report  # noqa: E402
from package_manifest import manifest_bytes  # noqa: E402
from refnx_build_manifest import (  # noqa: E402
    LOCK,
    MANIFEST,
    build_inputs,
    input_records,
    read_inputs,
    source_manifest,
    verify_inputs,
)
from vcs_source import MAX_ARCHIVE_SIZE, archive_files  # noqa: E402, F401
from verify_report import _directory_identity, _require_same_directory  # noqa: E402


def input_cache_key(manifest: dict, trust_domain: str) -> str:
    return cache_key(manifest, trust_domain).replace("xrr-wheels-v1-", "xrr-refnx-inputs-v1-", 1)


def _download_source(source: dict, destination: Path) -> None:
    request = urllib.request.Request(source["archive"]["url"], headers={"User-Agent": "xrr-refnx-inputs/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read(MAX_ARCHIVE_SIZE + 1)
    if len(content) != source["archive"]["size"] or hashlib.sha256(content).hexdigest() != source["archive"]["sha256"]:
        raise ValueError("downloaded refnx source differs from the trusted digest")
    with destination.open("xb") as stream:
        stream.write(content)


def _download(root: Path, manifest: dict, report: Path, environment: dict) -> int:
    with tempfile.TemporaryDirectory(prefix="refnx-input-stage-", dir=report) as temporary:
        stage = Path(temporary)
        try:
            code = _download_stage(root, manifest, stage, environment)
        finally:
            for path in stage.glob("download.*"):
                _write(report, path.name, path.read_text(encoding="utf-8"))
        if code != 0:
            return code
        _download_source(source_manifest(root), stage / "wheels/refnx-source.tar.gz")
        verify_inputs(root, manifest, stage / "wheels")
        copy_verified(stage / "wheels", report / "inputs", input_records(root, manifest))
    return 0


def cached_inputs(root: Path, report: Path, cache: Path, trust_domain: str) -> int:
    manifest = read_inputs(root)
    _require_pip()
    key = input_cache_key(manifest, trust_domain)
    cache = _cache_directory(root, cache, report)
    identity = _directory_identity(cache, "refnx input cache")
    cached = cache / hashlib.sha256(key.encode("ascii")).hexdigest()
    _cache_members(cache, cached.name)
    hit = os.path.lexists(cached)
    if hit:
        verify_inputs(root, manifest, cached)
    report, guard, environment = prepare_report(root, report)
    _write(report, "build-inputs.json", manifest_bytes(manifest).decode("ascii"))
    if hit:
        copy_verified(cached, report / "inputs", input_records(root, manifest))
    else:
        code = _download(root, manifest, report, environment)
        guard()
        if code != 0:
            _summary(report, "refnx-inputs", code)
            return code
        _require_same_directory(cache, identity, "refnx input cache")
        copy_verified(report / "inputs", cached, input_records(root, manifest))
    guard()
    _require_same_directory(cache, identity, "refnx input cache")
    _cache_members(cache, cached.name)
    verify_inputs(root, manifest, cached)
    verify_inputs(root, manifest, report / "inputs")
    _write(
        report,
        "cache.json",
        manifest_bytes({"state": "PASS", "key": key, "hit": hit, "trust_domain": trust_domain}).decode("ascii"),
    )
    _summary(report, "refnx-inputs", 0, derived_wheel_reused=False)
    return 0


def _record(args) -> int:
    manifest, lock = build_inputs(args.repo_root, json.loads(args.report.read_text(encoding="utf-8")))
    if args.check:
        if read_inputs(args.repo_root) != manifest or (args.repo_root / LOCK).read_bytes() != lock:
            raise ValueError("refnx builder manifest differs from the supplied resolution")
    else:
        _atomic_write(args.repo_root / LOCK, lock)
        _atomic_write(args.repo_root / MANIFEST, manifest_bytes(manifest))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "key", "fetch"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--trust-domain")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    required = {"record": ("report",), "key": ("trust_domain",), "fetch": ("trust_domain", "cache_dir", "report_dir")}
    if any(getattr(args, field) is None for field in required[args.mode]):
        parser.error(f"{args.mode} requires {required[args.mode]}")
    try:
        if args.mode == "record":
            return _record(args)
        if args.mode == "key":
            print(input_cache_key(read_inputs(args.repo_root), args.trust_domain))
            return 0
        return cached_inputs(args.repo_root, args.report_dir, args.cache_dir, args.trust_domain)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
