"""Create-once preregistration binds versions, sources, and fixed shard coverage."""

from __future__ import annotations

import hashlib

import pytest
from tests.unit.tools.poisson_diagnostic_fixtures import _test_registration
from tests.unit.tools.poisson_diagnostic_fixtures import tool as tool

import xrr_fitter.api as api


def test_finished_rms_protocol_rejects_the_revised_joint_runtime(tool):
    project = api.new_project()
    config = api.FitConfig.fast(0)
    actual = {
        "schema_version": project.schema_version,
        "algorithm_version": project.algorithm_version,
        "objective_version": config.objective_version,
        "diagnostic_version": config.diagnostic_version,
    }

    assert actual["objective_version"] == "2"
    assert tool.protocol.version_identity() == actual
    assert actual == {
        "schema_version": 5,
        "algorithm_version": "xrr-fit-v2-poisson-5",
        "objective_version": "2",
        "diagnostic_version": "poisson-refit-null-v4",
    }
    assert actual != tool.protocol.EXPECTED_VERSIONS
    with pytest.raises(ValueError, match="integrated version"):
        tool._check_versions()


def test_real_public_api_version_gate_does_not_coerce_numeric_expectations(tool, monkeypatch):
    expected = dict(tool.protocol.version_identity())
    monkeypatch.setattr(tool.protocol, "EXPECTED_VERSIONS", expected)
    assert tool._check_versions() == expected
    monkeypatch.setattr(tool.protocol, "EXPECTED_VERSIONS", dict(expected, objective_version=2))

    with pytest.raises(ValueError, match="integrated version required"):
        tool._check_versions()


def test_registration_is_create_once_hashed_and_fixed(tool, monkeypatch, tmp_path):
    output, manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)
    encoded = (output / "preregistration.json").read_bytes()

    assert hashlib.sha256(encoded).hexdigest() == (output / "preregistration.sha256").read_text().strip()
    assert manifest["protocol"] == tool.protocol.definition()
    assert tool.load_manifest(output) == manifest
    with pytest.raises(FileExistsError):
        tool.register(output, tmp_path / "toy-review.md", shard_count=3)


def test_registration_requires_integrated_current_versions(tool, monkeypatch, tmp_path):
    review = tmp_path / "review.md"
    review.write_bytes(b"review")
    monkeypatch.setattr(tool.protocol, "REVIEW_SHA256", hashlib.sha256(b"review").hexdigest())
    monkeypatch.setattr(tool.protocol, "version_identity", lambda: {"schema_version": 3})

    with pytest.raises(ValueError, match="version"):
        tool.register(tmp_path / "must-not-exist", review, shard_count=1)
    assert not (tmp_path / "must-not-exist").exists()


def test_registration_rejects_a_different_statistical_review(tool, tmp_path):
    review = tmp_path / "review.md"
    review.write_text("not the approved recipe")

    with pytest.raises(ValueError, match="review"):
        tool.register(tmp_path / "no-register", review, shard_count=1)


def test_manifest_changes_and_even_rehashed_threshold_changes_are_rejected(tool, monkeypatch, tmp_path):
    output, manifest, _identity = _test_registration(tool, monkeypatch, tmp_path)
    manifest["protocol"]["diagnostic_alpha"] = 0.05
    target = output / "preregistration.json"
    target.write_bytes(tool.protocol.canonical(manifest))
    with pytest.raises(ValueError, match="hash"):
        tool.load_manifest(output)
    (output / "preregistration.sha256").write_text(hashlib.sha256(target.read_bytes()).hexdigest() + "\n")
    with pytest.raises(ValueError, match="protocol"):
        tool.load_manifest(output)


def test_source_change_blocks_new_attempts_without_overwriting_evidence(tool, monkeypatch, tmp_path):
    output, _manifest, identity = _test_registration(tool, monkeypatch, tmp_path)
    identity["sha256"] = "b" * 64

    with pytest.raises(ValueError, match="source"):
        tool.run_shard(output, 0, max_cases=1, case_runner=lambda *_args: pytest.fail("fit started"))
    assert not (output / "cases").exists()


def test_fixed_shards_partition_the_complete_protocol_without_duplicate_seeds(tool):
    manifest = {"protocol": tool.protocol.definition(), "shard_count": 7}
    shards = [tool.assigned_cases(manifest, index) for index in range(7)]
    flat = [task for shard in shards for task in shard]

    assert len(flat) == len(set(flat)) == 4600
    assert ("replay_low", 199) in flat
    with pytest.raises(ValueError, match="shard"):
        tool.assigned_cases(manifest, 7)
