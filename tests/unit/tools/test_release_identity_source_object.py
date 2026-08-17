from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from tests.unit.tools.test_release_identity import (
    _canonical,
    _commit,
    _file,
    _fixture_repo,
    _git,
    _identity,
)


def _committed_bytes(root: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(("git", "show", f"{commit}:{path}"), cwd=root)


def _write_committed_test_manifest(root: Path, value: dict[str, object], message: str) -> str:
    value["collection_sha256"] = hashlib.sha256(
        _canonical({key: item for key, item in value.items() if key != "collection_sha256"})
    ).hexdigest()
    path = root / "verification/r23/tests.json"
    path.write_bytes(_canonical(value))
    return _commit(root, message)


def test_identity_binds_repo_records_to_the_captured_commit(
    tmp_path: Path,
    load_tool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, artifacts, artifact_manifest = _fixture_repo(tmp_path, distribution)
    captured_commit = _git(root, "rev-parse", "HEAD")
    expected = {
        path: hashlib.sha256(_committed_bytes(root, captured_commit, path)).hexdigest()
        for path in (module.RELEASE_SPEC_PATH, module.LOCK_PATH, module.TEST_MANIFEST_PATH)
    }
    original_clean_head_identity = module.clean_head_identity

    def mutate_after_capture(path: Path):
        identity = original_clean_head_identity(path)
        lock = root / module.LOCK_PATH
        lock.write_text("numpy==9.9.9\n", encoding="utf-8")
        release_spec_path = root / module.RELEASE_SPEC_PATH
        release_spec = json.loads(release_spec_path.read_text(encoding="utf-8"))
        release_spec["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
        release_spec["raced"] = True
        release_spec_path.write_bytes(_canonical(release_spec))
        test_path = root / "tests/test_sample.py"
        test_path.write_text("def test_raced(): pass\n", encoding="utf-8")
        manifest_path = root / module.TEST_MANIFEST_PATH
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
        manifest["test_tree"] = [{**_file(test_path), "path": "tests/test_sample.py"}]
        manifest["python_version"] = "raced"
        manifest["collection_sha256"] = hashlib.sha256(
            _canonical({key: item for key, item in manifest.items() if key != "collection_sha256"})
        ).hexdigest()
        manifest_path.write_bytes(_canonical(manifest))
        return identity

    monkeypatch.setattr(module, "clean_head_identity", mutate_after_capture)

    identity = module.calculate_release_identity(root, artifacts, artifact_manifest)

    assert identity.head_commit == captured_commit
    assert identity.release_spec.sha256 == expected[module.RELEASE_SPEC_PATH]
    assert identity.dependency_lock.sha256 == expected[module.LOCK_PATH]
    assert identity.test_manifest.file.sha256 == expected[module.TEST_MANIFEST_PATH]


def test_test_manifest_rejects_a_git_revision_expression_as_source_commit(
    tmp_path: Path,
    load_tool_module,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, _artifacts, _artifact_manifest, _identity_value = _identity(module, distribution, tmp_path)
    path = root / "verification/r23/tests.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source_commit"] = "HEAD~1"
    head = _write_committed_test_manifest(root, value, "invalid manifest source expression")
    lock = module._repo_file(root, head, module.LOCK_PATH)

    with pytest.raises(ValueError, match="test source commit"):
        module._test_manifest(root, head, lock)


@pytest.mark.parametrize("object_kind", ("annotated-tag", "tree"))
def test_test_manifest_rejects_non_commit_object_oid_as_source_commit(
    tmp_path: Path,
    load_tool_module,
    object_kind: str,
) -> None:
    distribution = load_tool_module("verify_distribution")
    module = load_tool_module("release_identity")
    root, _artifacts, _artifact_manifest, identity = _identity(module, distribution, tmp_path)
    if object_kind == "annotated-tag":
        subprocess.run(
            (
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "tag",
                "-a",
                "test-source",
                "-m",
                "fixture source",
                identity.test_manifest.source_commit,
            ),
            cwd=root,
            check=True,
        )
        source_commit = _git(root, "rev-parse", "refs/tags/test-source")
    else:
        source_commit = _git(root, "rev-parse", f"{identity.test_manifest.source_commit}^{{tree}}")
    path = root / "verification/r23/tests.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source_commit"] = source_commit
    head = _write_committed_test_manifest(root, value, f"invalid manifest {object_kind} source")
    lock = module._repo_file(root, head, module.LOCK_PATH)

    with pytest.raises(ValueError, match="test source commit"):
        module._test_manifest(root, head, lock)
