from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest


GROUPS = (
    "model_project",
    "io",
    "physics",
    "fit_compile",
    "fit_search",
    "analysis",
    "services",
    "gui",
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _add(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = 0o644
    info.mtime = 0
    tar.addfile(info, io.BytesIO(content))


def _embedded(content: str) -> dict[str, object]:
    encoded = content.encode()
    return {
        "content": content,
        "size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _provenance(input_content: bytes) -> dict[str, object]:
    configurations = {}
    for group in GROUPS:
        value = {"case": group, "version": 1}
        configurations[group] = {
            "value": value,
            "sha256": hashlib.sha256(_canonical(value)).hexdigest(),
        }
    return {
        "schema": "xrr-r22-reference-provenance-v1",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "generator": _embedded("print('generator')\n"),
        "collection_lock": _embedded("numpy==2.3.2\n"),
        "python": {"implementation": "CPython", "version": "3.12.13"},
        "platform": {"system": "Darwin", "machine": "arm64"},
        "inputs": [
            {
                "input_id": "fixture",
                "input_class": "bundled-example-data",
                "path": "xrr_fitter/examples/input.xy",
                "size": len(input_content),
                "sha256": hashlib.sha256(input_content).hexdigest(),
            }
        ],
        "seeds": {group: ([17] if group in {"fit_search", "analysis"} else []) for group in GROUPS},
        "configurations": configurations,
        "real_data_acceptance": {
            "status": "NOT_RUN",
            "reason": "owner post-delivery acceptance",
        },
    }


def _tree_hash(root: Path) -> str:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        content = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    digest = hashlib.sha256()
    for record in records:
        for value in (record["path"], str(record["size"]), record["sha256"]):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    archive = tmp_path / "r22.tar.gz"
    identity = _canonical(
        {
            "schema": "xrr-r22-release-identity-v1",
            "algorithm_identity": {"source_sha256": "1" * 64, "corpus_sha256": "2" * 64},
            "canonical_acceptance": {"status": "PASS", "case_count": 220, "event_count": 224},
            "gui_task_10_status": "blocked: missing approved dataset",
        }
    )
    product = b"label\tpath\tsize\tsha256\n"
    input_content = b"0.1 1.0\n"
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    artifacts = []
    groups = {}
    for group in GROUPS:
        path = sidecar / f"{group}.json"
        path.write_bytes(_canonical({"group": group, "values": [group]}))
        relative = path.name
        artifacts.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        groups[group] = {
            "artifacts": [relative],
            "comparison_policy": {
                "kind": "mapping",
                "fields": {relative: "exact"},
            },
            "input_ids": ["fixture"],
        }
    (sidecar / "manifest.json").write_bytes(
        _canonical(
            {
                "schema": "xrr-r22-reference-sidecar-v1",
                "provenance": _provenance(input_content),
                "groups": groups,
                "artifacts": artifacts,
            }
        )
    )
    sidecar_lock = tmp_path / "reference-sidecar-lock.json"
    sidecar_lock.write_bytes(
        _canonical(
            {
                "schema": "xrr-r22-reference-sidecar-lock-v1",
                "status": "APPROVED",
                "reference_sidecar_manifest_sha256": hashlib.sha256(
                    (sidecar / "manifest.json").read_bytes()
                ).hexdigest(),
                "reference_sidecar_tree_sha256": _tree_hash(sidecar),
            }
        )
    )
    with tarfile.open(archive, "w:gz") as handle:
        _add(handle, "xrr-r22-final/.integration/release/release-identity.json", identity)
        _add(handle, "xrr-r22-final/.integration/release/product-manifest.tsv", product)
        _add(handle, "xrr-r22-final/xrr_fitter/examples/input.xy", input_content)
    freeze = tmp_path / "freeze.json"
    freeze.write_bytes(
        _canonical(
            {
                "schema": "xrr-r22-delivery-freeze-v1",
                "status": "PASS",
                "head_commit": "a" * 40,
                "head_tree": "b" * 40,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "release_identity_sha256": hashlib.sha256(identity).hexdigest(),
                "product_manifest_sha256": hashlib.sha256(product).hexdigest(),
                "post_delivery_real_data_acceptance_status": "NOT_RUN",
            }
        )
    )
    return archive, freeze, sidecar, sidecar_lock


def test_builder_normalizes_exact_sidecar_groups_without_running_r22(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("build_r22_reference")
    archive, freeze, sidecar, sidecar_lock = _fixture(tmp_path)
    output = tmp_path / "output"
    manifest = module.build_reference(
        r22_archive=archive,
        freeze_receipt=freeze,
        reference_sidecar=sidecar,
        sidecar_lock=sidecar_lock,
        output=output,
    )
    assert tuple(manifest["groups"]) == GROUPS
    assert manifest["real_data_acceptance_status"] == "NOT_RUN"
    assert manifest["real_data_acceptance_reason"] == "owner post-delivery acceptance"
    assert manifest["reference_sidecar_lock_sha256"] == hashlib.sha256(
        sidecar_lock.read_bytes()
    ).hexdigest()
    assert manifest["provenance"]["generator"]["content"] == "print('generator')\n"
    for group in GROUPS:
        entry = manifest["groups"][group]
        expected_path = f"golden/{group}.json"
        assert entry == {
            "artifacts": [expected_path],
            "comparison_policy": {
                "kind": "mapping",
                "fields": {expected_path: "exact"},
            },
            "input_ids": ["fixture"],
        }
    assert (output / "manifest.json").is_file()


def test_builder_is_atomic_rejects_drift_and_does_not_touch_siblings(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("build_r22_reference")
    archive, freeze, sidecar, sidecar_lock = _fixture(tmp_path)
    sibling = tmp_path / "collections"
    sibling.mkdir()
    marker = sibling / "keep"
    marker.write_text("keep\n", encoding="utf-8")
    output = tmp_path / "reference"
    module.build_reference(archive, freeze, sidecar, sidecar_lock, output)
    first = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="non-empty"):
        module.build_reference(archive, freeze, sidecar, sidecar_lock, output)
    assert marker.read_text() == "keep\n"
    assert {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()} == first
    archive.write_bytes(archive.read_bytes() + b"drift")
    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError, match="archive|hash"):
        module.build_reference(archive, freeze, sidecar, sidecar_lock, fresh)
    assert not fresh.exists()


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("source_commit", "c" * 40, "commit"),
        ("source_tree", "d" * 40, "tree"),
        ("generator", {"content": "drift\n", "size": 1, "sha256": "0" * 64}, "generator"),
        (
            "collection_lock",
            {"content": "numpy>=2\n", "size": 9, "sha256": "0" * 64},
            "lock",
        ),
        (
            "real_data_acceptance",
            {"status": "PASS", "reason": "synthetic"},
            "real-data",
        ),
    ],
)
def test_builder_rejects_provenance_identity_and_embedded_file_drift(
    tmp_path: Path, load_tool_module, field: str, replacement: object, match: str
) -> None:
    module = load_tool_module("build_r22_reference")
    archive, freeze, sidecar, sidecar_lock = _fixture(tmp_path)
    manifest = json.loads((sidecar / "manifest.json").read_bytes())
    manifest["provenance"][field] = replacement
    (sidecar / "manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(ValueError, match=match):
        module.build_reference(archive, freeze, sidecar, sidecar_lock, tmp_path / "output")


@pytest.mark.parametrize("mutation", ["hash", "path", "configuration"])
def test_builder_recomputes_archive_input_and_configuration_hashes(
    tmp_path: Path, load_tool_module, mutation: str
) -> None:
    module = load_tool_module("build_r22_reference")
    archive, freeze, sidecar, sidecar_lock = _fixture(tmp_path)
    manifest = json.loads((sidecar / "manifest.json").read_bytes())
    provenance = manifest["provenance"]
    if mutation == "hash":
        provenance["inputs"][0]["sha256"] = "0" * 64
    elif mutation == "path":
        provenance["inputs"][0]["path"] = "/private/input.xy"
    else:
        provenance["configurations"]["physics"]["value"] = {"drift": True}
    (sidecar / "manifest.json").write_bytes(_canonical(manifest))
    with pytest.raises(ValueError, match="input|path|configuration"):
        module.build_reference(archive, freeze, sidecar, sidecar_lock, tmp_path / "output")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda entry: entry.pop("comparison_policy"), "comparison policy"),
        (
            lambda entry: entry["comparison_policy"]["fields"].update({"other.json": "exact"}),
            "policy|artifact",
        ),
        (lambda entry: entry.pop("input_ids"), "input"),
        (lambda entry: entry.update(input_ids=["missing"]), "input"),
        (lambda entry: entry.update(input_ids=["fixture", "fixture"]), "input"),
        (lambda entry: entry.update(configuration={"duplicate": True}), "metadata"),
    ],
)
def test_builder_rejects_unbound_or_duplicated_group_replay_metadata(
    tmp_path: Path,
    load_tool_module,
    mutation,
    match: str,
) -> None:
    module = load_tool_module("build_r22_reference")
    archive, freeze, sidecar, sidecar_lock = _fixture(tmp_path)
    manifest = json.loads((sidecar / "manifest.json").read_bytes())
    mutation(manifest["groups"]["physics"])
    (sidecar / "manifest.json").write_bytes(_canonical(manifest))

    with pytest.raises(ValueError, match=match):
        module.build_reference(archive, freeze, sidecar, sidecar_lock, tmp_path / "output")


def test_builder_rejects_a_self_consistent_sidecar_not_bound_by_the_approved_lock(
    tmp_path: Path, load_tool_module
) -> None:
    module = load_tool_module("build_r22_reference")
    archive, freeze, sidecar, sidecar_lock = _fixture(tmp_path)
    artifact = sidecar / "physics.json"
    artifact.write_bytes(_canonical({"group": "physics", "values": ["substituted"]}))
    manifest = json.loads((sidecar / "manifest.json").read_bytes())
    record = next(item for item in manifest["artifacts"] if item["path"] == artifact.name)
    record["size"] = artifact.stat().st_size
    record["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (sidecar / "manifest.json").write_bytes(_canonical(manifest))

    with pytest.raises(ValueError, match="approved sidecar lock"):
        module.build_reference(
            archive,
            freeze,
            sidecar,
            sidecar_lock,
            tmp_path / "output",
        )
