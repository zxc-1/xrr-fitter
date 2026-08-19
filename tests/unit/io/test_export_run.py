from __future__ import annotations

import errno
import os
import re
from hashlib import sha256
from pathlib import Path

import pytest

from xrr_fitter.io import export_run
from xrr_fitter.io.export_run import (
    ArtifactProducer,
    DatasetArtifacts,
    publish_export_run,
)


def _producer(path: str, content: bytes) -> ArtifactProducer:
    return ArtifactProducer(path, lambda: content)


def _root(*values: ArtifactProducer) -> tuple[ArtifactProducer, ...]:
    return (_producer("project_snapshot.xrrproj.json", b"project"), *values)


def _dataset(
    dataset_id: str = "sample-a",
    *,
    name: str = "fit_result.json",
    content: bytes = b"{}\n",
) -> DatasetArtifacts:
    return DatasetArtifacts(dataset_id, (_producer(name, content),))


def test_export_run_records_complete_relative_size_and_digest_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(export_run.secrets, "token_hex", lambda _size: "11111111")

    manifest = publish_export_run(
        tmp_path,
        (_dataset(),),
        _root(_producer("compatibility_summary.xlsx", b"workbook")),
        run_timestamp="20260715T120000",
    )

    assert manifest.run_directory == tmp_path / "20260715T120000-11111111"
    assert tuple(record.path for record in manifest.root_files) == (
        "compatibility_summary.xlsx",
        "export_manifest.json",
        "project_snapshot.xrrproj.json",
    )
    dataset = manifest.datasets[0]
    assert dataset.dataset_id == "sample-a"
    assert dataset.directory.startswith("001-sample-a-")
    assert tuple(record.path for record in dataset.files) == (f"{dataset.directory}/fit_result.json",)
    for record in manifest.files:
        published = manifest.run_directory / record.path
        content = published.read_bytes()
        assert record.size == len(content)
        assert record.sha256 == sha256(content).hexdigest()
    assert not tuple(tmp_path.glob(".partial-*"))


def test_export_retries_existing_final_and_partial_run_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = "20260715T120000"
    existing_final = tmp_path / f"{timestamp}-11111111"
    existing_final.mkdir()
    marker = existing_final / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    existing_partial = tmp_path / f".partial-{timestamp}-22222222"
    existing_partial.mkdir()
    tokens = iter(("11111111", "22222222", "33333333"))
    monkeypatch.setattr(export_run.secrets, "token_hex", lambda _size: next(tokens))

    manifest = publish_export_run(
        tmp_path,
        (_dataset(),),
        _root(),
        run_timestamp=timestamp,
    )

    assert manifest.run_directory.name == f"{timestamp}-33333333"
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert existing_partial.is_dir()


def test_export_allocation_treats_dangling_final_symlink_as_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = "20260715T120000"
    dangling = tmp_path / f"{timestamp}-11111111"
    dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
    tokens = iter(("11111111", "22222222"))
    monkeypatch.setattr(export_run.secrets, "token_hex", lambda _size: next(tokens))

    manifest = publish_export_run(
        tmp_path,
        (_dataset(),),
        _root(),
        run_timestamp=timestamp,
    )

    assert manifest.run_directory.name == f"{timestamp}-22222222"
    assert dangling.is_symlink()


def _assert_safe_dataset_directory(
    order: int,
    identifier: str,
    directory: str,
    run_directory: Path,
) -> None:
    digest = sha256(identifier.encode("utf-8")).hexdigest()[:8]
    assert directory.startswith(f"{order:03d}-")
    assert directory.endswith(f"-{digest}")
    assert not re.search(r"[/\\\x00-\x1f]", directory)
    assert not directory.startswith(".")
    assert (run_directory / directory).resolve().parent == run_directory.resolve()


def test_export_dataset_directories_are_traversal_safe_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifiers = ("../a/b\x00", "..\\a\\b\x1f", "Ｓａｍｐｌｅ／Ａ")
    tokens = iter(("aaaaaaaa", "bbbbbbbb"))
    monkeypatch.setattr(export_run.secrets, "token_hex", lambda _size: next(tokens))

    first = publish_export_run(
        tmp_path,
        tuple(_dataset(value) for value in identifiers),
        _root(),
        run_timestamp="20260715T120000",
    )
    second = publish_export_run(
        tmp_path,
        tuple(_dataset(value) for value in identifiers),
        _root(),
        run_timestamp="20260715T120000",
    )

    first_names = tuple(item.directory for item in first.datasets)
    assert first_names == tuple(item.directory for item in second.datasets)
    assert len(set(first_names)) == len(identifiers)
    for order, identifier, directory in zip(
        range(1, len(identifiers) + 1),
        identifiers,
        first_names,
        strict=True,
    ):
        _assert_safe_dataset_directory(
            order,
            identifier,
            directory,
            first.run_directory,
        )


@pytest.mark.parametrize(
    "timestamp",
    (
        "2026-07-15T120000",
        "20260715T120000/../escape",
        ".partial-20260715T120000",
        "20260715T120000ZZ",
    ),
)
def test_export_rejects_invalid_timestamp_before_writing(
    tmp_path: Path,
    timestamp: str,
) -> None:
    destination = tmp_path / "exports"

    with pytest.raises(ValueError, match="run_timestamp"):
        publish_export_run(
            destination,
            (_dataset(),),
            _root(),
            run_timestamp=timestamp,
        )

    assert not destination.exists()


@pytest.mark.parametrize(
    "path",
    (
        "/absolute.json",
        "../escape.json",
        "a/./b.json",
        "a\\b.json",
        "bad\x00name.json",
        "",
    ),
)
def test_export_rejects_unsafe_artifact_paths_before_writing(
    tmp_path: Path,
    path: str,
) -> None:
    with pytest.raises(ValueError, match="relative POSIX path"):
        ArtifactProducer(path, lambda: b"payload")

    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize(
    ("root_files", "message"),
    (
        (
            (
                _producer("a", b"file"),
                _producer("a/b", b"descendant"),
            ),
            "file/ancestor",
        ),
        (
            (
                _producer(
                    export_run._dataset_directory(1, "sample-a"),
                    b"blocks dataset directory",
                ),
            ),
            "file/ancestor",
        ),
        (
            (
                _producer(
                    f"{export_run._dataset_directory(1, 'sample-a')}/fit_result.json",
                    b"duplicates dataset file",
                ),
            ),
            "duplicate",
        ),
    ),
    ids=(
        "root-file-ancestor",
        "root-file-dataset-ancestor",
        "root-dataset-duplicate",
    ),
)
def test_export_rejects_artifact_tree_conflicts_before_writing(
    tmp_path: Path,
    root_files: tuple[ArtifactProducer, ...],
    message: str,
) -> None:
    destination = tmp_path / "exports"

    with pytest.raises(ValueError, match=message):
        publish_export_run(
            destination,
            (_dataset(),),
            _root(*root_files),
            run_timestamp="20260715T120000",
        )

    assert not destination.exists()


def test_export_writer_failure_exposes_no_final_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "exports"
    monkeypatch.setattr(export_run.secrets, "token_hex", lambda _size: "44444444")
    monkeypatch.setattr(
        export_run,
        "_write_payload",
        lambda *_args: (_ for _ in ()).throw(OSError("writer failed")),
    )

    with pytest.raises(OSError, match="writer failed"):
        publish_export_run(
            destination,
            (_dataset(),),
            _root(),
            run_timestamp="20260715T120000",
        )

    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()


def test_export_fsyncs_written_files_and_directories_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced_files: list[Path] = []
    synced_directories: list[Path] = []
    real_sync_file = export_run._sync_file
    real_sync_directory = export_run._sync_directory

    def record_file(path: Path) -> None:
        synced_files.append(Path(path))
        real_sync_file(path)

    def record_directory(path: Path) -> None:
        synced_directories.append(Path(path))
        real_sync_directory(path)

    monkeypatch.setattr(export_run, "_sync_file", record_file)
    monkeypatch.setattr(export_run, "_sync_directory", record_directory)

    manifest = publish_export_run(
        tmp_path,
        (_dataset(),),
        _root(_producer("root.txt", b"root")),
        run_timestamp="20260715T120000",
    )

    synced_regular_files = set(synced_files) - set(synced_directories)
    assert {path.name for path in synced_regular_files} == {Path(item.path).name for item in manifest.files}
    assert any(path.name.startswith("001-") for path in synced_directories)
    assert any(path.name.startswith(".partial-") for path in synced_directories)


def test_export_fsync_failure_cleans_owned_partial_and_preserves_existing_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "keep-run"
    existing.mkdir()
    marker = existing / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        export_run,
        "_sync_file",
        lambda _path: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    with pytest.raises(OSError, match="fsync failed"):
        publish_export_run(
            tmp_path,
            (_dataset(),),
            _root(),
            run_timestamp="20260715T120000",
        )

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not tuple(tmp_path.glob(".partial-*"))


@pytest.mark.parametrize("error_number", (errno.EINVAL, errno.ENOTSUP))
def test_export_directory_fsync_suppresses_only_unsupported_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    monkeypatch.setattr(
        export_run.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError(error_number, "unsupported")),
    )

    export_run._sync_directory(tmp_path)


def test_export_directory_fsync_propagates_io_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        export_run.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError(errno.EIO, "fsync failed")),
    )

    with pytest.raises(OSError, match="fsync failed"):
        export_run._sync_directory(tmp_path)


def test_export_publication_uses_exclusive_rename_and_preserves_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / ".partial-run"
    partial.mkdir()
    (partial / "payload").write_bytes(b"payload")
    final = tmp_path / "run"

    def collide(_source: Path, target: Path) -> None:
        target.mkdir()
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), target)

    monkeypatch.setattr(export_run, "_rename_exclusive", collide)

    with pytest.raises(FileExistsError):
        export_run._publish_directory(partial, final)
    assert partial.is_dir()
    assert final.is_dir()


def test_export_publication_failure_cleans_only_owned_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "keep-run"
    existing.mkdir()
    marker = existing / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        export_run,
        "_publish_directory",
        lambda *_args: (_ for _ in ()).throw(OSError("publish failed")),
    )

    with pytest.raises(OSError, match="publish failed"):
        publish_export_run(
            tmp_path,
            (_dataset(),),
            _root(),
            run_timestamp="20260715T120000",
        )

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not tuple(tmp_path.glob(".partial-*"))


def test_export_cleanup_failure_is_never_silenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(export_run.secrets, "token_hex", lambda _size: "55555555")
    monkeypatch.setattr(
        export_run,
        "_write_payload",
        lambda *_args: (_ for _ in ()).throw(OSError("write failed")),
    )
    monkeypatch.setattr(
        export_run,
        "_cleanup_partial",
        lambda *_args: (_ for _ in ()).throw(OSError("cleanup failed")),
    )

    with pytest.raises(BaseExceptionGroup) as captured:
        publish_export_run(
            tmp_path,
            (_dataset(),),
            _root(),
            run_timestamp="20260715T120000",
        )

    assert [str(error) for error in captured.value.exceptions] == [
        "write failed",
        "cleanup failed",
    ]
