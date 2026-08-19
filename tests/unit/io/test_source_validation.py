from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from tests.support.model_cases import dataset_project, project

from xrr_fitter.io.source import (
    dataset_by_id,
    dataset_index,
    resolve_source_path,
    validate_source,
    validate_sources,
)
from xrr_fitter.model.project import SourceStatus


def _project_for_source(path: Path, *, declared_path: str | None = None):
    digest = sha256(path.read_bytes()).hexdigest()
    dataset = replace(
        dataset_project("sample-1"),
        source_path=str(path) if declared_path is None else declared_path,
        source_sha256=digest,
    )
    return replace(project(dataset), base_directory=str(path.parent))


def test_dataset_lookup_is_centralized_and_identity_stable(tmp_path: Path) -> None:
    source = tmp_path / "curve.xy"
    source.write_bytes(b"curve\n")
    current = _project_for_source(source)

    assert dataset_index(current, "sample-1") == 0
    assert dataset_by_id(current, "sample-1") is current.datasets[0]
    with pytest.raises(KeyError, match="missing"):
        dataset_by_id(current, "missing")


def test_relative_source_resolves_against_base_directory_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "curve.xy"
    source.write_bytes(b"curve\n")
    current = _project_for_source(source, declared_path="curve.xy")
    monkeypatch.chdir(tmp_path.parent)

    resolved = resolve_source_path(current, current.datasets[0])
    validation = validate_sources(current)

    assert resolved == source
    assert validation.datasets[0].status is SourceStatus.OK
    assert current.datasets[0].source_path == "curve.xy"
    assert current.base_directory == str(tmp_path)


def test_revalidation_hashes_raw_bytes_without_parsing(tmp_path: Path) -> None:
    source = tmp_path / "binary.xy"
    raw = b"\xff\x00not a text curve\n"
    source.write_bytes(raw)
    current = _project_for_source(source)

    record = validate_source(current, current.datasets[0])

    assert record.status is SourceStatus.OK
    assert record.actual_sha256 == sha256(raw).hexdigest()


def test_relative_source_requires_base_directory(tmp_path: Path) -> None:
    source = tmp_path / "curve.xy"
    source.write_bytes(b"curve\n")
    current = replace(
        _project_for_source(source, declared_path="curve.xy"),
        base_directory=None,
    )

    with pytest.raises(ValueError, match="base_directory"):
        validate_sources(current)


def test_empty_base_directory_is_not_a_relative_source_root(tmp_path: Path) -> None:
    source = tmp_path / "curve.xy"
    source.write_bytes(b"curve\n")
    current = replace(
        _project_for_source(source, declared_path="curve.xy"),
        base_directory="",
    )

    with pytest.raises(ValueError, match="base_directory"):
        validate_sources(current)


def test_missing_source_has_typed_status_and_message(tmp_path: Path) -> None:
    source = tmp_path / "curve.xy"
    source.write_bytes(b"curve\n")
    current = _project_for_source(source)
    source.unlink()

    record = validate_sources(current).datasets[0]

    assert record.status is SourceStatus.MISSING
    assert record.actual_sha256 is None
    assert record.expected_sha256 == current.datasets[0].source_sha256
    assert current.datasets[0].dataset_id in record.user_message
    assert record.user_message


def test_unreadable_source_has_distinct_status_and_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "curve.xy"
    source.write_bytes(b"curve\n")
    current = _project_for_source(source)

    def denied(_path: Path) -> bytes:
        raise PermissionError("denied for test")

    monkeypatch.setattr(Path, "read_bytes", denied)
    record = validate_sources(current).datasets[0]

    assert record.status is SourceStatus.UNREADABLE
    assert record.actual_sha256 is None
    assert current.datasets[0].dataset_id in record.user_message
    assert "denied for test" in record.user_message


def test_unexpected_source_error_is_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "curve.xy"
    source.write_bytes(b"curve\n")
    current = _project_for_source(source)

    def explode(_path: Path) -> bytes:
        raise RuntimeError("programming failure")

    monkeypatch.setattr(Path, "read_bytes", explode)
    with pytest.raises(RuntimeError, match="programming failure"):
        validate_sources(current)


def test_unchanged_source_preserves_derived_state_and_identity(tmp_path: Path) -> None:
    source = tmp_path / "curve.xy"
    source.write_bytes(b"curve\n")
    current = _project_for_source(source)
    original_dataset = current.datasets[0]

    validation = validate_sources(current)

    assert validation.valid
    assert current.datasets[0] is original_dataset
    assert current.datasets[0].source_sha256 == original_dataset.source_sha256
