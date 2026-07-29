from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


NOT_RUN = "NOT_RUN: owner post-delivery acceptance"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=root, text=True).strip()


def _commit(root: Path, message: str) -> str:
    subprocess.run(("git", "add", "."), cwd=root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            message,
        ),
        cwd=root,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2026-07-20T10:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-07-20T10:00:00+00:00",
        },
    )
    return _git(root, "rev-parse", "HEAD")


def _file(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.as_posix(),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _fixture_repo(tmp_path: Path, distribution_module, *, empty_test_file: bool = False):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "verification" / "r23").mkdir(parents=True)
    (root / "tests" / "test_sample.py").write_text("def test_sample(): pass\n", encoding="utf-8")
    lock = root / "requirements-macos-arm64-py312.lock"
    lock.write_text("numpy==2.3.1\n", encoding="utf-8")
    spec = root / "verification" / "release-spec.json"
    spec.write_bytes(
        _canonical(
            {
                "schema": "xrr-r23-release-spec-v1",
                "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "r22_oracle_tree_sha256": "8" * 64,
            }
        )
    )
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    source_commit = _commit(root, "test source")
    test_path = root / "tests" / "test_sample.py"
    test_paths = [test_path]
    if empty_test_file:
        empty_path = root / "tests" / "__init__.py"
        empty_path.write_bytes(b"")
        test_paths.append(empty_path)
        source_commit = _commit(root, "empty test package")
    base = {
        "schema": "xrr-test-manifest-v1",
        "source_commit": source_commit,
        "suite": "tests",
        "test_tree": sorted(
            (
                {
                    **_file(path),
                    "path": path.relative_to(root).as_posix(),
                }
                for path in test_paths
            ),
            key=lambda item: str(item["path"]),
        ),
        "node_count": 1,
        "nodes": [{"markers": [], "nodeid": "tests/test_sample.py::test_sample"}],
        "python_version": "3.12.13",
        "platform": "macOS-arm64",
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    }
    manifest = {**base, "collection_sha256": hashlib.sha256(_canonical(base)).hexdigest()}
    test_manifest = root / "verification" / "r23" / "tests.json"
    test_manifest.write_bytes(_canonical(manifest))
    head_commit = _commit(root, "test manifest")
    head_tree = _git(root, "rev-parse", "HEAD^{tree}")

    artifact_dir = tmp_path / "bundle" / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "xrr_fitter-0.2.0.tar.gz").write_bytes(b"sdist")
    (artifact_dir / "xrr_fitter-0.2.0-py3-none-any.whl").write_bytes(b"wheel")
    artifact_manifest = distribution_module.calculate_artifact_manifest(
        artifact_dir,
        head_commit=head_commit,
        head_tree=head_tree,
    )
    artifact_manifest_path = artifact_dir.parent / "artifact-manifest.json"
    distribution_module.write_artifact_manifest(artifact_manifest_path, artifact_manifest)
    return root, artifact_dir, artifact_manifest_path


def test_identity_accepts_zero_byte_test_tree_files(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(
        tmp_path,
        distribution,
        empty_test_file=True,
    )

    module.calculate_release_identity(root, artifacts, artifact_manifest)


def _identity(module, distribution_module, tmp_path: Path):
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution_module)
    value = module.calculate_release_identity(root, artifacts, artifact_manifest)
    return root, artifacts, artifact_manifest, value


def test_identity_composes_git_bindings(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest, identity = _identity(module, distribution, tmp_path)

    assert identity.schema == "xrr-r23-release-identity-v1"
    assert identity.status == "PASS"
    assert identity.head_commit == _git(root, "rev-parse", "HEAD")
    assert identity.head_tree == _git(root, "rev-parse", "HEAD^{tree}")


def test_identity_composes_static_bindings(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    _root, _artifacts, _artifact_manifest, identity = _identity(module, distribution, tmp_path)

    assert identity.release_spec.path == "verification/release-spec.json"
    assert identity.dependency_lock.path == "requirements-macos-arm64-py312.lock"
    assert identity.r22_oracle_tree_sha256 == "8" * 64
    assert identity.test_manifest.file.path == "verification/r23/tests.json"
    assert identity.test_manifest.source_commit != identity.head_commit
    assert identity.approved_data.status == NOT_RUN


def test_identity_composes_artifact_bindings_and_round_trips(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest, identity = _identity(module, distribution, tmp_path)

    assert identity.artifact_manifest.path == "artifact-manifest.json"
    assert identity.artifacts == distribution.read_artifact_manifest(artifact_manifest).artifacts
    assert module.parse_release_identity(module.canonical_identity_bytes(identity)) == identity
    module.validate_release_identity(identity, root, artifacts, artifact_manifest)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "schema",
        "status",
        "commit",
        "release-spec",
        "test-collection",
        "approved-status",
        "manifest-path",
        "artifact",
    ),
)
def test_parser_rejects_every_identity_layer(
    tmp_path: Path,
    load_tool_module,
    mutation: str,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    _root, _artifacts, _manifest, identity = _identity(module, distribution, tmp_path)
    value = module.identity_value(identity)
    if mutation == "missing":
        value.pop("head_tree")
    elif mutation == "extra":
        value["extra"] = True
    elif mutation in {"schema", "status"}:
        value[mutation] = "wrong"
    elif mutation == "commit":
        value["head_commit"] = "A" * 40
    elif mutation == "release-spec":
        value["release_spec"]["sha256"] = "F" * 64
    elif mutation == "test-collection":
        value["test_manifest"]["collection_sha256"] = "F" * 64
    elif mutation == "approved-status":
        value["approved_data"]["status"] = "PASS"
    elif mutation == "manifest-path":
        value["artifact_manifest"]["path"] = "renamed.json"
    else:
        value["artifacts"][0]["sha256"] = "F" * 64
    with pytest.raises(ValueError):
        module.parse_release_identity(_canonical(value))


def test_parser_rejects_duplicate_and_noncanonical_json(load_tool_module) -> None:
    module = load_tool_module("release_identity")
    for content in (b'{"schema":"x","schema":"y"}\n', b"{}", b"{}\r\n", b"{ }\n", b"[]\n", b"\xff"):
        with pytest.raises(ValueError):
            module.parse_release_identity(content)


def _replace_nested(value: object, path: tuple[object, ...], replacement: object) -> None:
    current = value
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = replacement


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("schema",), "wrong"),
        (("status",), "wrong"),
        (("head_commit",), "A" * 40),
        (("head_tree",), "A" * 40),
        (("release_spec", "path"), "wrong.json"),
        (("release_spec", "size"), 0),
        (("release_spec", "sha256"), "A" * 64),
        (("dependency_lock", "path"), "wrong.lock"),
        (("dependency_lock", "size"), 0),
        (("dependency_lock", "sha256"), "A" * 64),
        (("r22_oracle_tree_sha256",), "A" * 64),
        (("test_manifest", "file", "path"), "wrong.json"),
        (("test_manifest", "file", "size"), 0),
        (("test_manifest", "file", "sha256"), "A" * 64),
        (("test_manifest", "source_commit"), "A" * 40),
        (("test_manifest", "collection_sha256"), "A" * 64),
        (("approved_data", "status"), "PASS"),
        (("artifact_manifest", "path"), "wrong.json"),
        (("artifact_manifest", "size"), 0),
        (("artifact_manifest", "sha256"), "A" * 64),
        (("artifacts", 0, "kind"), "wheel"),
        (("artifacts", 0, "path"), "artifacts/other.tar.gz"),
        (("artifacts", 0, "filename"), "other.tar.gz"),
        (("artifacts", 0, "size"), 0),
        (("artifacts", 0, "sha256"), "A" * 64),
    ),
)
def test_parser_rejects_tamper_in_every_nested_identity_field(
    tmp_path: Path,
    load_tool_module,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    _root, _artifacts, _manifest, identity = _identity(module, distribution, tmp_path)
    value = module.identity_value(identity)
    _replace_nested(value, path, replacement)

    with pytest.raises(ValueError):
        module.parse_release_identity(_canonical(value))


@pytest.mark.parametrize(
    "path",
    (
        ("release_spec",),
        ("dependency_lock",),
        ("test_manifest",),
        ("test_manifest", "file"),
        ("approved_data",),
        ("artifact_manifest",),
        ("artifacts", 0),
    ),
)
@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_parser_rejects_missing_or_extra_nested_identity_fields(
    tmp_path: Path,
    load_tool_module,
    path: tuple[object, ...],
    mutation: str,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    _root, _artifacts, _manifest, identity = _identity(module, distribution, tmp_path)
    value = module.identity_value(identity)
    current = value
    for part in path:
        current = current[part]
    if mutation == "missing":
        current.pop(next(iter(current)))
    else:
        current["extra"] = True

    with pytest.raises(ValueError):
        module.parse_release_identity(_canonical(value))


@pytest.mark.parametrize(
    "drift",
    ("release-spec", "lock", "test-tree", "artifact", "artifact-manifest"),
)
def test_validation_recomputes_every_owned_input(
    tmp_path: Path,
    load_tool_module,
    drift: str,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest, identity = _identity(module, distribution, tmp_path)
    if drift == "release-spec":
        (root / "verification/release-spec.json").write_bytes(b"drift")
    elif drift == "lock":
        (root / "requirements-macos-arm64-py312.lock").write_bytes(b"drift")
    elif drift == "test-tree":
        (root / "tests/test_sample.py").write_text("def changed(): pass\n", encoding="utf-8")
    elif drift == "artifact":
        next(artifacts.glob("*.whl")).write_bytes(b"drift")
    else:
        artifact_manifest.write_bytes(b"drift")

    with pytest.raises(ValueError):
        module.validate_release_identity(identity, root, artifacts, artifact_manifest)


def test_build_is_atomic_and_records_not_run_status(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    report = tmp_path / "identity-report"

    target = module.build_release_identity(root, report, artifacts, artifact_manifest)
    assert target == report / "release-identity.json"
    assert json.loads(target.read_text(encoding="utf-8"))["approved_data"] == {"status": NOT_RUN}

    failed = tmp_path / "failed-report"
    monkeypatch.setattr(
        module.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        module.build_release_identity(root, failed, artifacts, artifact_manifest)
    assert not failed.exists()


def test_build_adds_identity_to_an_existing_distribution_bundle(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    report = artifacts.parent

    target = module.build_release_identity(root, report, artifacts, artifact_manifest)

    assert target == report / "release-identity.json"
    assert {path.name for path in report.iterdir()} == {
        "artifact-manifest.json",
        "artifacts",
        "release-identity.json",
    }
    module.validate_identity_file(root, target, artifacts, artifact_manifest)


def test_annotated_tag_validation_writes_bound_freeze_receipt(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    report = tmp_path / "identity"
    identity_path = module.build_release_identity(root, report, artifacts, artifact_manifest)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "tag",
            "-a",
            "R23-final",
            "-m",
            "fixture final",
        ),
        cwd=root,
        check=True,
    )
    receipt = tmp_path / "r23-final-freeze.json"

    module.validate_identity_file(
        root,
        identity_path,
        artifacts,
        artifact_manifest,
        expected_tag="R23-final",
        write_freeze_receipt=receipt,
    )

    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["schema"] == "xrr-r23-final-freeze-v1"
    assert value["status"] == "PASS"
    assert value["tag"] == "R23-final"
    assert value["head_commit"] == _git(root, "rev-parse", "HEAD")
    assert value["release_identity"]["sha256"] == hashlib.sha256(identity_path.read_bytes()).hexdigest()


def test_freeze_receipt_publish_is_atomic_on_replace_failure(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    identity_path = module.build_release_identity(
        root,
        tmp_path / "identity",
        artifacts,
        artifact_manifest,
    )
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "tag",
            "-a",
            "R23-final",
            "-m",
            "fixture final",
        ),
        cwd=root,
        check=True,
    )
    receipt = tmp_path / "r23-final-freeze.json"
    monkeypatch.setattr(
        module.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        module.validate_identity_file(
            root,
            identity_path,
            artifacts,
            artifact_manifest,
            expected_tag="R23-final",
            write_freeze_receipt=receipt,
        )

    assert not receipt.exists()
    assert not tuple(tmp_path.glob(".r23-final-freeze.json.*.tmp"))


def test_lightweight_tag_is_rejected(tmp_path: Path, load_tool_module) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest, identity = _identity(module, distribution, tmp_path)
    subprocess.run(("git", "tag", "R23-final"), cwd=root, check=True)

    with pytest.raises(ValueError, match="annotated"):
        module.validate_release_identity(
            identity,
            root,
            artifacts,
            artifact_manifest,
            expected_tag="R23-final",
        )


def test_annotated_tag_pointing_to_another_commit_is_rejected(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest, identity = _identity(module, distribution, tmp_path)
    source_commit = identity.test_manifest.source_commit
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "tag",
            "-a",
            "R23-final",
            "-m",
            "wrong target",
            source_commit,
        ),
        cwd=root,
        check=True,
    )

    with pytest.raises(ValueError, match="different commit"):
        module.validate_release_identity(
            identity,
            root,
            artifacts,
            artifact_manifest,
            expected_tag="R23-final",
        )
