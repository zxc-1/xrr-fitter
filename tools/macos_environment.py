#!/usr/bin/env python3
"""Own, provision and narrowly clean a unique external macOS job environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from verify_report import (  # noqa: E402
    _directory_identity,
    _require_same_directory,
    _resolve_output_path,
    _write_new_file_in_anchored_directory,
)


def _write(directory: Path, name: str, value: dict) -> None:
    _write_new_file_in_anchored_directory(
        directory,
        _directory_identity(directory, "macOS job directory"),
        name,
        (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        directory_label="macOS job directory",
        file_label="macOS environment evidence",
    )


def _job_path(root: Path, job: Path, runner_temp: Path) -> Path:
    job = _resolve_output_path(job, "macOS job root")
    parent = _resolve_output_path(runner_temp, "runner temporary directory")
    if job.is_relative_to(root.resolve()):
        raise ValueError("macOS job environment must be external")
    if job.parent != parent or re.fullmatch(r"xrr-macos-[A-Za-z0-9_.-]+", job.name) is None:
        raise ValueError("macOS job root is not a direct runner-owned temporary directory")
    _directory_identity(job, "macOS job root")
    if job.stat().st_uid != os.getuid() or job.stat().st_mode & 0o077:
        raise ValueError("macOS job root must be private to the current user")
    return job


def own_environment(root: Path, job: Path, runner_temp: Path) -> None:
    job = _job_path(root, job, runner_temp)
    value = {
        "schema": "xrr-macos-job-owner-v1",
        "uid": os.getuid(),
        "root_identity": _directory_identity(job, "macOS job root"),
        "venv_identity": _directory_identity(job / "venv", "macOS job venv"),
    }
    _write(job, "owner.json", value)


def _owned_job(root: Path, job: Path, runner_temp: Path) -> Path:
    job = _job_path(root, job, runner_temp)
    path = job / "owner.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing regular macOS job ownership evidence")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "xrr-macos-job-owner-v1" or value.get("uid") != os.getuid():
        raise ValueError("macOS job ownership evidence differs")
    _require_same_directory(job, tuple(value["root_identity"]), "macOS job root")
    _require_same_directory(job / "venv", tuple(value["venv_identity"]), "macOS job venv")
    return job


def cleanup_environment(root: Path, job: Path, runner_temp: Path) -> None:
    job = _owned_job(root, job, runner_temp)
    identity = _directory_identity(job, "macOS job root")
    shutil.rmtree(job / "venv")
    _require_same_directory(job, identity, "macOS job root")
    _write(job, "cleanup.json", {"state": "PASS", "removed": "venv", "reports_retained": True})


def combined_key(root: Path, trust_domain: str) -> str:
    from package_cache import cache_key
    from package_downloads import read_manifest
    from refnx_build_inputs import input_cache_key, read_inputs

    ordinary = cache_key(read_manifest(root, root / "tools/package-manifests/macos-arm64-py312.json"), trust_domain)
    builder = input_cache_key(read_inputs(root), trust_domain)
    digest = hashlib.sha256(f"{ordinary}\n{builder}\n".encode("ascii")).hexdigest()
    return f"xrr-macos-inputs-v1-{trust_domain}-python-3.12-pip-26.1.2-{digest}"


def prepare_cache(root: Path, cache: Path, report: Path) -> Path:
    from package_cache import _cache_directory

    cache = _cache_directory(root, cache, report)
    if {item.name for item in cache.iterdir()} - {"wheels", "refnx"}:
        raise ValueError("macOS input cache contains unrelated members")
    return cache


def run_setup_steps(steps) -> tuple[str, int]:
    for label, operation in steps:
        code = operation()
        if code != 0:
            return label, code
    return "complete", 0


def setup_environment(root: Path, job: Path, runner_temp: Path, cache: Path, trust_domain: str) -> int:
    from package_cache import cached_download
    from package_downloads import install_packages, require_install_target
    from refnx_build import build_refnx
    from refnx_build_inputs import cached_inputs

    job = _owned_job(root, job, runner_temp)
    require_install_target("macos-arm64-py312")
    if Path(sys.prefix).resolve() != job / "venv":
        raise ValueError("macOS setup must run inside its owned environment")
    report = job / "reports"
    cache = prepare_cache(root, cache, report)
    report.mkdir()
    manifest = root / "tools/package-manifests/macos-arm64-py312.json"
    steps = (
        (
            "ordinary-inputs",
            lambda: cached_download(root, manifest, report / "packages", cache / "wheels", trust_domain),
        ),
        ("refnx-inputs", lambda: cached_inputs(root, report / "refnx-inputs", cache / "refnx", trust_domain)),
        (
            "ordinary-install",
            lambda: install_packages(root, manifest, report / "packages/wheels", report / "installation"),
        ),
        (
            "refnx-build",
            lambda: build_refnx(root, report / "refnx-inputs/inputs", report / "refnx-build", install=True),
        ),
    )
    label, code = run_setup_steps(steps)
    _owned_job(root, job, runner_temp)
    prepare_cache(root, cache, report)
    _write(report, "setup.json", {"state": "PASS" if code == 0 else "FAIL", "stage": label, "exit_code": code})
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("own", "key", "setup", "cleanup"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--runner-temp", type=Path, default=os.environ.get("RUNNER_TEMP"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--trust-domain")
    args = parser.parse_args(argv)
    required = {
        "own": ("job_root", "runner_temp"),
        "cleanup": ("job_root", "runner_temp"),
        "key": ("trust_domain",),
        "setup": ("job_root", "runner_temp", "cache_dir", "trust_domain"),
    }
    if any(getattr(args, field) is None for field in required[args.mode]):
        parser.error(f"{args.mode} requires {required[args.mode]}")
    try:
        if args.mode == "key":
            key = combined_key(args.repo_root, args.trust_domain)
            print(f"key={key}\ndigest={hashlib.sha256(key.encode('ascii')).hexdigest()}")
        elif args.mode == "setup":
            return setup_environment(args.repo_root, args.job_root, args.runner_temp, args.cache_dir, args.trust_domain)
        elif args.mode == "own":
            own_environment(args.repo_root, args.job_root, args.runner_temp)
        else:
            cleanup_environment(args.repo_root, args.job_root, args.runner_temp)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
