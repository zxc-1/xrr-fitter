from __future__ import annotations

from pathlib import Path

import pytest

from xrr_fitter.model.analysis import ConfidenceClass
from xrr_fitter.model.export import (
    DatasetExportManifest,
    ExportFileRecord,
    ExportManifest,
    ExportPlan,
)


def test_export_file_record_requires_relative_path_size_and_sha256() -> None:
    record = ExportFileRecord("curve/result.json", 12, "a" * 64)

    assert record.path == "curve/result.json"
    with pytest.raises(ValueError, match="relative"):
        ExportFileRecord("/tmp/result.json", 12, "a" * 64)
    with pytest.raises(ValueError, match="size"):
        ExportFileRecord("result.json", 0, "a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        ExportFileRecord("result.json", 12, "bad")


def test_export_manifest_copies_records_and_flattens_deterministically() -> None:
    root = ExportFileRecord("manifest.json", 10, "a" * 64)
    curve = ExportFileRecord("curve/result.json", 12, "b" * 64)
    records = [curve]
    dataset = DatasetExportManifest("curve", "curve", records)
    manifest = ExportManifest(Path("run-1"), (dataset,), (root,))

    records.clear()

    assert dataset.files == (curve,)
    assert manifest.files == (root, curve)
    assert manifest.run_directory == Path("run-1")


def test_export_manifest_rejects_duplicate_paths_across_groups() -> None:
    record = ExportFileRecord("same.json", 10, "a" * 64)
    dataset = DatasetExportManifest("curve", "curve", (record,))

    with pytest.raises(ValueError, match="duplicate"):
        ExportManifest(Path("run-1"), (dataset,), (record,))


def test_export_plan_validates_the_same_descriptor_fields_as_a_manifest() -> None:
    """A plan is the pre-publication half of a manifest and must be just as strict.

    The export dialog shows the plan before any byte is written, so a malformed
    verdict has to be rejected here rather than surface as a preview that
    disagrees with the manifest the run finally returns.
    """
    plan = ExportPlan(
        dataset_ids=("curve",),
        mode="joint",
        stages=("A", "E"),
        confidence=ConfidenceClass.CORRELATED,
        reproducible=True,
    )

    assert (plan.dataset_ids, plan.mode, plan.stages) == (("curve",), "joint", ("A", "E"))
    assert plan.confidence is ConfidenceClass.CORRELATED
    assert plan.reproducible is True
    with pytest.raises(ValueError, match="unsupported export mode"):
        ExportPlan(dataset_ids=("curve",), mode="batch")
    with pytest.raises(ValueError, match="stages must not repeat"):
        ExportPlan(dataset_ids=("curve",), stages=("E", "E"))
    with pytest.raises(TypeError, match="confidence must be ConfidenceClass"):
        ExportPlan(dataset_ids=("curve",), confidence="可信")
    with pytest.raises(TypeError, match="reproducible must be a bool"):
        ExportPlan(dataset_ids=("curve",), reproducible=1)


def test_export_plan_requires_the_datasets_it_claims_to_describe() -> None:
    """An empty or duplicated dataset list would misreport the preview's count."""
    with pytest.raises(ValueError, match="dataset"):
        ExportPlan(dataset_ids=())
    with pytest.raises(ValueError, match="unique"):
        ExportPlan(dataset_ids=("curve", "curve"))
    with pytest.raises(ValueError, match="dataset"):
        ExportPlan(dataset_ids=("curve", " "))


def test_export_plan_defaults_to_the_weakest_claim_and_owns_no_files() -> None:
    """Nothing is published yet, so the plan may not carry publication records."""
    plan = ExportPlan(dataset_ids=("curve",))

    assert plan.mode == "independent"
    assert plan.stages == ()
    assert plan.confidence is ConfidenceClass.UNTRUSTED
    assert plan.reproducible is False
    assert not hasattr(plan, "run_directory")
    assert not hasattr(plan, "files")
