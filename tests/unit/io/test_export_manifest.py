from __future__ import annotations

import json
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


def _dataset(*producers: ArtifactProducer) -> DatasetArtifacts:
    values = producers or (_producer("fit_result.json", b"{}\n"),)
    return DatasetArtifacts("sample", values)


def _assert_file_records(run_directory: Path, listed: dict[str, dict[str, object]]) -> None:
    for relative, record in listed.items():
        body = (run_directory / relative).read_bytes()
        assert record == {
            "path": relative,
            "size": len(body),
            "sha256": sha256(body).hexdigest(),
        }


def _canonical_manifest_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _listed_records(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {item["path"]: item for item in payload["files"]}


def _published_paths(manifest) -> set[str]:
    return {item.path for item in manifest.files if item.path != "export_manifest.json"}


def test_export_producers_render_lazily_in_sorted_path_order(tmp_path: Path) -> None:
    calls: list[str] = []
    producers = (
        ArtifactProducer("z.txt", lambda: (calls.append("z"), b"z")[1]),
        ArtifactProducer("a.txt", lambda: (calls.append("a"), b"a")[1]),
    )
    assert calls == []

    manifest = publish_export_run(
        tmp_path,
        (_dataset(*producers),),
        _root(),
        run_timestamp="20260715T120000",
    )

    assert calls == ["a", "z"]
    assert all((manifest.run_directory / item.path).is_file() for item in manifest.files)


def test_export_persists_canonical_manifest_covering_every_other_file(tmp_path: Path) -> None:
    manifest = publish_export_run(
        tmp_path,
        (_dataset(),),
        _root(),
        run_timestamp="20260715T120000",
    )
    content = (manifest.run_directory / "export_manifest.json").read_bytes()
    payload = json.loads(content)
    listed = _listed_records(payload)

    assert content == _canonical_manifest_bytes(payload)
    assert tuple(payload) == (
        "schema",
        "export_schema_version",
        "project_snapshot",
        "datasets",
        "files",
    )
    assert payload["schema"] == "xrr-fitter-export-manifest-v2"
    assert payload["project_snapshot"] == listed["project_snapshot.xrrproj.json"]
    assert set(listed) == _published_paths(manifest)
    assert tuple(listed) == tuple(sorted(listed))
    assert "export_manifest.json" in {item.path for item in manifest.root_files}
    _assert_file_records(manifest.run_directory, listed)


def test_export_record_hashes_files_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "large.bin"
    artifact.write_bytes(b"x" * (1024 * 1024 + 1))
    read_sizes: list[int] = []
    original_open = Path.open

    class RecordingStream:
        def __init__(self, stream) -> None:
            self._stream = stream

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *args):
            return self._stream.__exit__(*args)

        def read(self, size: int = -1):
            read_sizes.append(size)
            return self._stream.read(size)

    def recording_open(path: Path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return RecordingStream(stream) if path == artifact else stream

    monkeypatch.setattr(Path, "open", recording_open)

    record = export_run._record(tmp_path, artifact)

    assert record.size == 1024 * 1024 + 1
    assert read_sizes and all(size == 1024 * 1024 for size in read_sizes)


@pytest.mark.parametrize(
    ("renderer", "error", "message"),
    (
        (lambda: "not-bytes", TypeError, "must return bytes"),
        (lambda: b"", ValueError, "must not be empty"),
    ),
)
def test_export_rejects_invalid_renderer_output_during_publication(
    tmp_path: Path,
    renderer,
    error: type[Exception],
    message: str,
) -> None:
    destination = tmp_path / "exports"
    dataset = _dataset(ArtifactProducer("bad.bin", renderer))

    with pytest.raises(error, match=message):
        publish_export_run(
            destination,
            (dataset,),
            _root(),
            run_timestamp="20260715T120000",
        )

    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()


def test_export_manifest_failure_cleans_owned_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "exports"
    monkeypatch.setattr(
        export_run,
        "_manifest_bytes",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("manifest failed")),
    )

    with pytest.raises(RuntimeError, match="manifest failed"):
        publish_export_run(
            destination,
            (_dataset(),),
            _root(),
            run_timestamp="20260715T120000",
        )

    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()


def test_export_rejects_caller_owned_manifest_path_before_writing(tmp_path: Path) -> None:
    destination = tmp_path / "exports"

    with pytest.raises(ValueError, match="duplicate"):
        publish_export_run(
            destination,
            (_dataset(),),
            _root(_producer("export_manifest.json", b"caller manifest")),
            run_timestamp="20260715T120000",
        )

    assert not destination.exists()
