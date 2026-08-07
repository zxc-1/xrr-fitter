#!/usr/bin/env python3
"""Validate a stable release tag against the packaged project version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tomllib


VERSION_TAG = re.compile(r"^v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
HEX = frozenset("0123456789abcdef")


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *arguments),
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        detail = error.output.strip() if error.output else "git command failed"
        raise ValueError(detail) from error


def project_version(repo_root: str | Path) -> str:
    root = Path(repo_root).resolve()
    try:
        with (root / "pyproject.toml").open("rb") as handle:
            value = tomllib.load(handle)["project"]["version"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError("pyproject.toml does not declare a project version") from error
    if not isinstance(value, str) or not value:
        raise ValueError("project version must be a non-empty string")
    return value


def validate_release_tag(
    repo_root: str | Path,
    tag: str,
    *,
    expected_commit: str | None = None,
) -> str:
    root = Path(repo_root).resolve()
    if VERSION_TAG.fullmatch(tag) is None:
        raise ValueError("release tag must match vMAJOR.MINOR.PATCH")

    version = tag[1:]
    declared = project_version(root)
    if declared != version:
        raise ValueError(
            f"release tag version {version} does not match project version {declared}"
        )

    reference = f"refs/tags/{tag}"
    if _git(root, "cat-file", "-t", reference) != "tag":
        raise ValueError("release tag must be annotated")
    tagged_commit = _git(root, "rev-parse", f"{reference}^{{commit}}")
    target = expected_commit or _git(root, "rev-parse", "HEAD")
    if len(target) != 40 or any(char not in HEX for char in target):
        raise ValueError("expected commit must be a lowercase 40-character Git object ID")
    if tagged_commit != target:
        raise ValueError("release tag points to a different commit")
    return version


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    version = validate_release_tag(
        args.repo_root,
        args.tag,
        expected_commit=args.expected_commit,
    )
    print(json.dumps({"status": "PASS", "tag": args.tag, "version": version}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
