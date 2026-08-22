from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from tests.support.model_cases import dataset_project, project

from xrr_fitter.io.project_codec import save_project


def test_atomic_save_fsyncs_parent_directory_after_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "durable.xrrproj.json"
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(f"fsync-{kind}")
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    monkeypatch.setattr(os, "replace", tracked_replace)

    save_project(project(dataset_project("sample-1")), target)

    assert events == ["fsync-file", "replace", "fsync-directory"]
