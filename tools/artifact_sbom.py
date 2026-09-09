#!/usr/bin/env python3
"""Emit CycloneDX evidence bound to the built distribution artifact bytes."""

from __future__ import annotations

import argparse
import hashlib
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from packaging.utils import parse_wheel_filename  # noqa: E402

from distribution_manifest import (  # noqa: E402
    ArtifactManifest,
    ArtifactRecord,
    read_artifact_manifest,
    select_artifacts,
    validate_artifact_manifest,
)
from lock_sbom import canonical_sbom_bytes  # noqa: E402
from wheel_inventory import inspect_wheel  # noqa: E402


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sdist_path(member: tarfile.TarInfo) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in member.name
        or ":" in member.name
        or path.as_posix() != member.name.rstrip("/")
    ):
        raise ValueError("sdist contains an unsafe member path")
    if member.issym() or member.islnk() or stat.S_IFMT(member.mode) == stat.S_IFLNK:
        raise ValueError("sdist contains a symlink member")
    return path


def _sdist_inventory(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    names: set[str] = set()
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 100_000:
            raise ValueError("sdist exceeds the inventory limit")
        for member in members:
            relative = _sdist_path(member)
            if relative.as_posix() in names:
                raise ValueError("sdist contains duplicate members")
            names.add(relative.as_posix())
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("sdist file cannot be read")
            digest = hashlib.sha256()
            size = 0
            with handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
                    size += len(block)
            if size != member.size:
                raise ValueError("sdist member size changed during reading")
            records.append(
                {
                    "path": relative.as_posix(),
                    "sha256": digest.hexdigest(),
                    "size": size,
                    "kind": "metadata" if relative.name in {"PKG-INFO", "METADATA"} else "source",
                }
            )
    return records


def _wheel_record(path: Path, record: ArtifactRecord) -> dict[str, str]:
    name, version, _build, _tags = parse_wheel_filename(path.name)
    if str(name) != path.name.rsplit("-", 3)[0] and not path.name.startswith(f"{name}-{version}-"):
        raise ValueError("artifact wheel filename identity drift")
    return {"name": str(name), "version": str(version), "sha256": record.sha256}


def _properties(values: dict[str, object]) -> list[dict[str, str]]:
    return [{"name": key, "value": str(value)} for key, value in sorted(values.items())]


def _file_component(ref: str, item: dict[str, object]) -> dict[str, object]:
    return {
        "type": "file",
        "bom-ref": f"{ref}:file:{quote(str(item['path']), safe='')}",
        "name": str(item["path"]),
        "hashes": [{"alg": "SHA-256", "content": item["sha256"]}],
        "properties": _properties({"xrr:file:kind": item["kind"], "xrr:file:size": item["size"]}),
    }


def _artifact_component(record: ArtifactRecord, files: list[dict[str, object]]) -> dict[str, object]:
    ref = f"urn:xrr:artifact:{record.sha256}"
    return {
        "type": "application",
        "bom-ref": ref,
        "name": record.filename,
        "hashes": [{"alg": "SHA-256", "content": record.sha256}],
        "properties": _properties(
            {
                "xrr:artifact:kind": record.kind,
                "xrr:artifact:path": record.path,
                "xrr:artifact:size": record.size,
                "xrr:inventory:files": len(files),
                "xrr:inventory:native-files": sum(item.get("kind") == "native" for item in files),
            }
        ),
        "components": [_file_component(ref, item) for item in files],
    }


def _manifest_properties(manifest: ArtifactManifest, manifest_path: Path, artifacts: Path) -> list[dict[str, str]]:
    return _properties(
        {
            "xrr:artifact-manifest:sha256": _sha256(manifest_path),
            "xrr:artifact-scope": "built-distribution-artifacts",
            "xrr:artifact-directory": artifacts.name,
            "xrr:source:commit": manifest.head_commit,
            "xrr:source:tree": manifest.head_tree,
            "xrr:composition:status": "incomplete",
            "xrr:composition:missing": "executable/native-link-relationships/installed-closure",
        }
    )


def build_artifact_sbom(root: Path, manifest_path: Path, artifact_dir: Path) -> dict[str, object]:
    manifest = read_artifact_manifest(manifest_path)
    validate_artifact_manifest(manifest, artifact_dir, head_commit=manifest.head_commit, head_tree=manifest.head_tree)
    selected = select_artifacts(artifact_dir)
    wheel = next(record for record in manifest.artifacts if record.kind == "wheel")
    sdist = next(record for record in manifest.artifacts if record.kind == "sdist")
    wheel_inventory = inspect_wheel(selected["wheel"], _wheel_record(selected["wheel"], wheel))
    files = [*wheel_inventory["files"], *(_sdist_inventory(selected["sdist"]))]
    result = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {"properties": _manifest_properties(manifest, manifest_path, artifact_dir)},
        "components": [
            _artifact_component(wheel, wheel_inventory["files"]),
            _artifact_component(sdist, _sdist_inventory(selected["sdist"])),
        ],
        "dependencies": [],
        "compositions": [{"aggregate": "incomplete"}],
    }
    if len(files) != len(wheel_inventory["files"]) + len(result["components"][1]["components"]):
        raise ValueError("artifact inventory assembly drift")
    validate_artifact_manifest(manifest, artifact_dir, head_commit=manifest.head_commit, head_tree=manifest.head_tree)
    return result


def write_artifact_sbom(
    root: Path,
    manifest_path: Path,
    artifact_dir: Path,
    *,
    require_complete: bool = False,
) -> dict[str, object]:
    result = build_artifact_sbom(root, manifest_path, artifact_dir)
    if require_complete:
        raise ValueError("artifact SBOM cannot establish complete executable/native composition")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = write_artifact_sbom(
            args.repo_root,
            args.artifact_manifest,
            args.artifact_dir,
            require_complete=args.require_complete,
        )
        sys.stdout.buffer.write(canonical_sbom_bytes(result))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
