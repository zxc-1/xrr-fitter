"""GUI round-trip verification for published owner-data candidate projects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Callable

from tests.support.approved_workflow_model import (
    CANDIDATE_SCHEMA,
    CASE_SPECS,
    GuiResult,
    canonical_json_bytes,
    file_record,
    file_value,
    owner_source,
)


GuiVerifier = Callable[[str, Path, Path], GuiResult]


def _object_pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate candidate key: {key}")
        result[key] = value
    return result


def _candidate_cases(value: dict[str, object]) -> list[dict[str, object]]:
    cases = value.get("cases")
    if value.get("schema") != CANDIDATE_SCHEMA or not isinstance(cases, list):
        raise ValueError("approved-data candidate schema drift")
    _validate_case_registry(cases)
    return cases


def _validate_case_registry(cases: list[dict[str, object]]) -> None:
    if any(not isinstance(item, dict) for item in cases):
        raise ValueError("approved-data candidate cases must be objects")
    identifiers = tuple(item.get("case_id") for item in cases)
    if identifiers != tuple(spec.case_id for spec in CASE_SPECS):
        raise ValueError("approved-data candidate case registry drift")


def _read_candidate(report: Path) -> list[dict[str, object]]:
    path = report / "approved-data-candidate.json"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("approved-data candidate report is missing")
    content = path.read_bytes()
    value = json.loads(content.decode("utf-8"), object_pairs_hook=_object_pairs)
    if not isinstance(value, dict) or content != canonical_json_bytes(value):
        raise ValueError("approved-data candidate report is not canonical")
    return _candidate_cases(value)


def _verify_record(report: Path, value: object, expected_path: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "size", "sha256"}:
        raise ValueError("candidate project record drift")
    if value["path"] != expected_path:
        raise ValueError("candidate project path drift")
    path = report.joinpath(*PurePosixPath(expected_path).parts)
    record = file_record(path, expected_path)
    if (record.size, record.sha256) != (value["size"], value["sha256"]):
        raise ValueError("candidate project content drift")
    return path


def _manifest_files(manifest) -> tuple[object, ...]:
    return manifest.root_files + tuple(
        record for dataset in manifest.datasets for record in dataset.files
    )


def _manifest_is_current(manifest) -> bool:
    for record in _manifest_files(manifest):
        path = manifest.run_directory / PurePosixPath(record.path)
        if path.is_symlink() or not path.is_file():
            return False
        content = path.read_bytes()
        if (len(content), hashlib.sha256(content).hexdigest()) != (record.size, record.sha256):
            return False
    return True


def verify_gui_case(case_id: str, project: Path, output: Path) -> GuiResult:
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from xrr_fitter.gui.main_window import MainWindow

    application = QApplication.instance() or QApplication([])
    output.mkdir(parents=True, exist_ok=False)
    window = MainWindow()
    reopened = MainWindow()
    try:
        window.open_project(project)
        window.show()
        application.processEvents()
        saved = output / "gui-round-trip.xrrproj.json"
        window.save_project(saved)
        manifest = window.export_results(output / "exports")
        reopened.open_project(saved)
        reopened.show()
        application.processEvents()
        reopened_ok = reopened.document.project.ui_state.selected_candidate_ids == (
            window.document.project.ui_state.selected_candidate_ids
        )
        current = _manifest_is_current(manifest)
        plots = tuple(item for item in _manifest_files(manifest) if item.path.endswith(".png"))
        return GuiResult(case_id, reopened_ok, current, current and bool(plots))
    finally:
        for item in (reopened, window):
            item.close()
            item.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def verify_gui_acceptance(
    owner_root: str | Path,
    report_dir: str | Path,
    *,
    case_verifier: GuiVerifier = verify_gui_case,
) -> tuple[GuiResult, ...]:
    root = Path(owner_root)
    report = Path(report_dir)
    cases = _read_candidate(report)
    results = []
    for spec, case in zip(CASE_SPECS, cases, strict=True):
        source = owner_source(root, spec)
        if case.get("source") != file_value(file_record(source, spec.source_name)):
            raise ValueError(f"approved source drift: {spec.case_id}")
        runs = case.get("runs")
        if not isinstance(runs, list) or len(runs) != 4:
            raise ValueError("candidate run registry drift")
        first = runs[0]
        if not isinstance(first, dict):
            raise ValueError("candidate run must be an object")
        expected = f"runs/{spec.case_id}/run-1/project.xrrproj.json"
        project = _verify_record(report, first.get("project"), expected)
        results.append(case_verifier(spec.case_id, project, report / "gui" / spec.case_id))
    return tuple(results)
