#!/usr/bin/env python3
"""Bind the locked refnx Git export to actual downloadable source archive bytes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from package_manifest import _file_identity, _sha256, _verified_file, locked_inputs, manifest_bytes  # noqa: E402

SCHEMA = "xrr-vcs-source-v1"
FIELDS = {"schema", "scope", "vcs", "tree_oid", "file_count", "files_sha256", "archive"}
MAX_ARCHIVE_SIZE = 64 * 1024 * 1024
MAX_SOURCE_SIZE = 256 * 1024 * 1024


def _source(root: Path) -> dict:
    _, inputs = locked_inputs(root, "macos-arm64-py312")
    sources = inputs["vcs"]
    if len(sources) != 1 or sources[0]["name"] != "refnx":
        raise ValueError("source verification requires the single locked refnx source")
    if sources[0]["url"] != "https://github.com/refnx/refnx.git":
        raise ValueError("unsupported VCS archive host")
    return sources[0]


def _archive_url(source: dict) -> str:
    return f"https://codeload.github.com/refnx/refnx/tar.gz/{source['commit']}"


def _member_name(member: tarfile.TarInfo, prefix: str) -> str:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or "\\" in member.name or ":" in member.name:
        raise ValueError("source archive contains an unsafe path")
    if not path.parts or path.parts[0] != prefix or path.as_posix() != member.name.rstrip("/"):
        raise ValueError("source archive root or member path is noncanonical")
    return path.relative_to(prefix).as_posix()


def _check_member_kind(member: tarfile.TarInfo) -> None:
    if not (member.isfile() or member.isdir()) or member.mode & 0o7000:
        raise ValueError("source archive contains a non-regular member")


def archive_files(content: bytes, prefix: str) -> dict:
    records = {}
    seen = set()
    size = 0
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as archive:
        for member in archive:
            _check_member_kind(member)
            name = _member_name(member, prefix)
            if name in seen:
                raise ValueError("source archive contains duplicate members")
            seen.add(name)
            size += member.size
            if size > MAX_SOURCE_SIZE or len(seen) > 10000:
                raise ValueError("source archive exceeds the inventory limit")
            if member.isfile():
                with archive.extractfile(member) as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                records[name] = {"sha256": digest, "size": member.size, "executable": bool(member.mode & 0o111)}
    if not records:
        raise ValueError("source archive contains no files")
    return records


def _git(repository: Path, *arguments: str) -> bytes:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_ATTR_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"})
    result = subprocess.run(
        ("git", "-C", str(repository), "-c", f"core.attributesFile={os.devnull}", *arguments),
        env=environment,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _git_export(repository: Path, commit: str, prefix: str) -> tuple[str, dict]:
    overrides = Path(_git(repository, "rev-parse", "--git-path", "info/attributes").decode().strip())
    if not overrides.is_absolute():
        overrides = repository / overrides
    if os.path.lexists(overrides):
        raise ValueError("Git archive must not use local attribute overrides")
    tree = _git(repository, "rev-parse", "--verify", f"{commit}^{{tree}}").decode().strip()
    content = _git(repository, "archive", "--format=tar", f"--prefix={prefix}/", commit)
    return tree, archive_files(content, prefix)


def _read_archive(path: Path) -> bytes:
    before = _file_identity(path.lstat())
    if before[2] > MAX_ARCHIVE_SIZE:
        raise ValueError("source archive exceeds the input limit")
    content = path.read_bytes()
    if _file_identity(path.lstat()) != before:
        raise ValueError("source archive changed during reading")
    return content


def record_source(root: Path, repository: Path, archive: Path) -> dict:
    source = _source(root)
    prefix = f"refnx-{source['commit']}"
    content = _read_archive(archive)
    files = archive_files(content, prefix)
    tree, expected = _git_export(repository, source["commit"], prefix)
    if files != expected:
        raise ValueError("downloaded source differs from the pinned Git archive")
    if _source(root) != source:
        raise ValueError("locked source changed during verification")
    digest = hashlib.sha256(content).hexdigest()
    _verified_file(archive, digest)
    return {
        "schema": SCHEMA,
        "scope": "git-archive",
        "vcs": source,
        "tree_oid": tree,
        "file_count": len(files),
        "files_sha256": hashlib.sha256(manifest_bytes(files)).hexdigest(),
        "archive": {"url": _archive_url(source), "sha256": digest, "size": len(content)},
    }


def _validate_manifest(root: Path, manifest: dict) -> None:
    if set(manifest) != FIELDS or manifest["schema"] != SCHEMA or manifest["scope"] != "git-archive":
        raise ValueError("source manifest schema mismatch")
    if manifest["vcs"] != _source(root) or re.fullmatch(r"[0-9a-f]{40}", str(manifest["tree_oid"])) is None:
        raise ValueError("source manifest Git identity mismatch")
    archive = manifest["archive"]
    if not isinstance(archive, dict) or set(archive) != {"url", "sha256", "size"}:
        raise ValueError("source manifest archive schema mismatch")
    if archive["url"] != _archive_url(manifest["vcs"]):
        raise ValueError("source manifest archive URL mismatch")
    _sha256(archive["sha256"])
    _sha256(manifest["files_sha256"])


def verify_source(root: Path, manifest: dict, archive: Path) -> None:
    _validate_manifest(root, manifest)
    observed = _verified_file(archive, manifest["archive"]["sha256"])
    if observed["size"] != manifest["archive"]["size"]:
        raise ValueError("source archive size mismatch")
    files = archive_files(_read_archive(archive), f"refnx-{manifest['vcs']['commit']}")
    if (
        len(files) != manifest["file_count"]
        or hashlib.sha256(manifest_bytes(files)).hexdigest() != manifest["files_sha256"]
    ):
        raise ValueError("source archive inventory mismatch")
    _verified_file(archive, manifest["archive"]["sha256"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "verify"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--git-repository", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.mode == "record" and args.git_repository is None) or (args.mode == "verify" and args.manifest is None):
        parser.error("record requires --git-repository; verify requires --manifest")
    try:
        if args.mode == "record":
            result = record_source(args.repo_root, args.git_repository, args.archive)
        else:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            verify_source(args.repo_root, manifest, args.archive)
            result = {"state": "PASS", "source_verified": True, "wheel_verified": False}
        sys.stdout.buffer.write(manifest_bytes(result))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
