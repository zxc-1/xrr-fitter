from __future__ import annotations

from pathlib import Path

import pytest

from xrr_fitter.model.export import (
    DatasetExportManifest,
    ExportFileRecord,
    ExportManifest,
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
