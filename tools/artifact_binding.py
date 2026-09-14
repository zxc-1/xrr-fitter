"""Bind distribution SBOMs to captured source and the verified runtime wheel closure."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TOOL_DIRECTORY = str(Path(__file__).resolve().parent)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

from artifact_sbom import _wheel_record, build_artifact_sbom  # noqa: E402
from distribution_archive import verify_archives  # noqa: E402
from distribution_manifest import parse_artifact_manifest, select_artifacts  # noqa: E402
from distribution_source import _cat_file_blobs, _regular_blob_oids, distribution_inputs, release_spec  # noqa: E402
from installed_files import InstalledSnapshot, VerifiedWheel  # noqa: E402
from installed_inputs import InputBindings  # noqa: E402
from lock_sbom import canonical_sbom_bytes  # noqa: E402
from package_sbom import dependency_edges  # noqa: E402


def require_source_identity(manifest, source: dict) -> None:
    if (manifest.head_commit, manifest.head_tree) != (source["source_commit"], source["source_tree"]):
        raise ValueError("distribution source identity differs from the installation evidence")


def _artifact_member_hashes(artifact: dict, wheel: bool) -> dict:
    return {
        item["name"] if wheel else item["name"].partition("/")[2]: item["hashes"][0]["content"]
        for item in artifact["components"]
    }


def _selected_source_hashes(expected: dict[str, bytes], wheel: bool) -> dict:
    return {
        path.removeprefix("src/") if wheel else path: hashlib.sha256(data).hexdigest()
        for path, data in expected.items()
        if not wheel or path.startswith("src/xrr_fitter/")
    }


def verify_artifact_sources(bom: dict, expected: dict[str, bytes]) -> dict:
    result = {}
    for artifact in bom["components"]:
        wheel = artifact["name"].endswith(".whl")
        files = _artifact_member_hashes(artifact, wheel)
        selected = _selected_source_hashes(expected, wheel)
        if any(files.get(path) != checksum for path, checksum in selected.items()):
            raise ValueError("distribution source member bytes differ from the captured Git objects")
        result["wheel" if wheel else "sdist"] = len(selected)
    return result


def runtime_closure(graph: dict[str, list[str]], application: str) -> set[str]:
    result, pending = set(), list(graph[application])
    while pending:
        name = pending.pop()
        if name not in result:
            result.add(name)
            pending.extend(graph[name])
    result.discard(application)
    return result


def _refs(components: list[dict]) -> set[str]:
    result = set()
    for component in components:
        result.add(component["bom-ref"])
        result.update(_refs(component.get("components", [])))
    return result


def _runtime_packages(names: set[str], installed_bom: dict) -> list[dict]:
    packages = [item for item in installed_bom["components"] if item["type"] == "library" and item["name"] in names]
    if {item["name"] for item in packages} != names:
        raise ValueError("artifact runtime closure is not bound to installed wheel evidence")
    return packages


def _provenance_edges(packages: list[dict], installed_bom: dict) -> list[dict]:
    references = _refs(packages)
    dependencies = [item for item in installed_bom["dependencies"] if item["ref"] in references]
    if any(not set(item["dependsOn"]) <= references for item in dependencies):
        raise ValueError("artifact runtime provenance graph has an unbound dependency")
    return dependencies


def _runtime_evidence(bom: dict, app: dict, installed: dict, installed_bom: dict) -> None:
    graph = dependency_edges([app, *(item["inventory"] for item in installed["packages"])], installed["target"])
    packages = _runtime_packages(runtime_closure(graph, app["name"]), installed_bom)
    dependencies = _provenance_edges(packages, installed_bom)
    by_name = {item["name"]: item["bom-ref"] for item in packages}
    wheel = next(item for item in bom["components"] if item["name"].endswith(".whl"))
    bom["components"].extend(packages)
    bom["dependencies"] = [
        *dependencies,
        {"ref": wheel["bom-ref"], "dependsOn": [by_name[name] for name in graph[app["name"]]]},
    ]


def bind_distribution(root, manifest_path, directory, installed, installed_bom, source):
    bound = InputBindings({"artifact-manifest": manifest_path})
    manifest = parse_artifact_manifest(bound.contents["artifact-manifest"])
    require_source_identity(manifest, source)
    snapshot = InstalledSnapshot(directory)
    for record in manifest.artifacts:
        snapshot.verify(directory / record.filename, record.sha256)
    spec = release_spec(root, source["source_commit"])
    inputs = distribution_inputs(root, spec["sdist_content_policy"], source["source_commit"])
    verify_archives(root, directory, inputs, spec)
    payloads = _cat_file_blobs(root, _regular_blob_oids(root, source["source_commit"], inputs))
    bom = build_artifact_sbom(root, manifest_path, directory)
    counts = verify_artifact_sources(bom, {path.as_posix(): data for path, data in payloads.items()})
    wheel_record = next(record for record in manifest.artifacts if record.kind == "wheel")
    path = select_artifacts(directory)["wheel"]
    with VerifiedWheel(path, _wheel_record(path, wheel_record)) as wheel:
        _runtime_evidence(bom, wheel.inventory, installed, installed_bom)
    values = {
        "xrr:installed:inventory-sha256": hashlib.sha256(canonical_sbom_bytes(installed)).hexdigest(),
        "xrr:installed:sbom-sha256": hashlib.sha256(canonical_sbom_bytes(installed_bom)).hexdigest(),
        "xrr:artifact:runtime-scope": "wheel-metadata-closure-not-observed-loader-closure",
        "xrr:artifact:verified-source-files": str(counts),
    }
    bom["metadata"]["properties"].extend({"name": key, "value": value} for key, value in sorted(values.items()))

    def guard():
        bound.guard()
        snapshot.finish()

    guard()
    return bom, guard
