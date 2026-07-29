from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QMessageBox, QPlainTextEdit

import xrr_fitter.api as api
from tests.support.model_cases import final_fit_result, fit_candidate, simple_structure


def _exports():
    try:
        return importlib.import_module("xrr_fitter.gui.export.dialog")
    except ModuleNotFoundError as error:
        pytest.fail(f"missing Slice 9 export implementation: {error}", pytrace=False)


def _write_curve(path: Path, scale: float = 1000.0) -> Path:
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {scale / (index + 1):.12g}"
            for index in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _fitted_project(tmp_path: Path, *, datasets: int = 1) -> api.XrrProject:
    source = _write_curve(tmp_path / "curve.xy")
    beam = api.BeamSpec("monochromatic")
    value = api.add_dataset(
        api.new_project(),
        source,
        api.InstrumentSpec(instrument_id="gui-export", footprint_mode="none"),
        beam=beam,
    )
    value = api.set_structure(value, "curve", simple_structure())
    data = api.import_data(source, beam)
    candidate = replace(
        fit_candidate("selected"),
        qz_a_inv=data.qz_a_inv,
        model_normalized=data.intensity_normalized,
        log_residuals_decades=np.zeros(data.qz_a_inv.size),
        weighted_residuals=np.zeros(data.qz_a_inv.size),
    )
    result = final_fit_result(candidate)
    first = replace(value.datasets[0], last_valid_result=result)
    rows = [first]
    if datasets == 2:
        rows.append(replace(first, dataset_id="second", display_name="second"))
    value = replace(value, datasets=tuple(rows))
    for dataset in value.datasets:
        value = api.select_candidate(value, dataset.dataset_id, candidate.candidate_id)
    return value


def _workflow(project: api.XrrProject):
    from xrr_fitter.gui.document import ProjectDocument

    module = _exports()
    return module.ExportWorkflow(ProjectDocument(project), is_running=lambda: False)


def _manifest(directory: Path):
    records = (
        SimpleNamespace(path="compatibility_summary.xlsx"),
        SimpleNamespace(path="01-curve/fit_result.json"),
    )
    return SimpleNamespace(
        run_directory=directory / "run",
        datasets=(SimpleNamespace(dataset_id="curve"),),
        files=records,
    )


def test_one_click_fit_save_export_reopen(tmp_path: Path) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    module = _exports()
    document = ProjectDocument(_fitted_project(tmp_path))
    project_path = tmp_path / "one-click.xrrproj.json"
    document.save(project_path)
    workflow = module.ExportWorkflow(document, is_running=lambda: False)

    manifest = workflow.export_results(tmp_path / "exports")
    reopened = ProjectDocument()
    reopened.open(project_path)

    assert manifest.run_directory.is_dir()
    assert reopened.project.datasets[0].last_valid_result is not None
    assert reopened.project.ui_state.selected_candidate_ids == (("curve", "selected"),)
    assert str(manifest.run_directory) in workflow.summary_text


def test_export_dialog_failure_names_destination_exception_type_and_message(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _exports()
    workflow = _workflow(api.new_project())
    destination = tmp_path / "exports"
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(destination),
    )
    monkeypatch.setattr(
        api,
        "export_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: messages.append((title, message)),
    )

    assert workflow.export_results_dialog(None) is None

    assert messages == [
        (
            "导出失败",
            f"目标目录：{destination}\nOSError: disk full\n"
            "请检查目标目录的写入权限和可用空间后重试。",
        )
    ]


def test_export_dialog_uses_scrollable_summary_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _exports()
    workflow = _workflow(api.new_project())
    destination = tmp_path / "exports"
    manifest = _manifest(destination)
    shown: list[object] = []

    class CapturingSummaryDialog:
        def __init__(self, summary: str, parent=None) -> None:
            shown.append((summary, parent))

        def exec(self) -> None:
            shown.append("exec")

    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(destination),
    )
    monkeypatch.setattr(api, "export_result", lambda *_args: manifest)
    monkeypatch.setattr(module, "ExportSummaryDialog", CapturingSummaryDialog)

    assert workflow.export_results_dialog(None) is manifest

    assert shown == [(workflow.summary_text, None), "exec"]
    assert str(manifest.run_directory / manifest.files[1].path) in workflow.summary_text


def test_export_failure_preserves_previous_summary_and_published_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = _workflow(_fitted_project(tmp_path))
    manifest = workflow.export_results(tmp_path / "exports")
    summary = workflow.summary_text
    published = {
        record.path: sha256((manifest.run_directory / record.path).read_bytes()).hexdigest()
        for record in manifest.files
    }
    monkeypatch.setattr(
        api,
        "export_result",
        lambda *_args: (_ for _ in ()).throw(OSError("cannot publish")),
    )

    with pytest.raises(OSError, match="cannot publish"):
        workflow.export_results(tmp_path / "exports")

    assert workflow.summary_text == summary
    assert {
        record.path: sha256((manifest.run_directory / record.path).read_bytes()).hexdigest()
        for record in manifest.files
    } == published


def test_export_repeated_multi_dataset_runs_preserve_manifest_and_curve_units(
    tmp_path: Path,
) -> None:
    workflow = _workflow(_fitted_project(tmp_path, datasets=2))
    root = tmp_path / "exports"

    first = workflow.export_results(root)
    first_summary = workflow.summary_text
    second = workflow.export_results(root)
    second_summary = workflow.summary_text

    assert first.run_directory != second.run_directory
    for manifest, summary in ((first, first_summary), (second, second_summary)):
        assert tuple(item.dataset_id for item in manifest.datasets) == ("curve", "second")
        assert all((manifest.run_directory / record.path).is_file() for record in manifest.files)
        assert all(str(manifest.run_directory / record.path) in summary for record in manifest.files)
        assert any(record.path == "compatibility_summary.xlsx" for record in manifest.root_files)


def test_export_summary_dialog_uses_read_only_scrollable_text(qtbot) -> None:
    module = _exports()
    summary = "\n".join(f"artifact-{index}" for index in range(100))
    dialog = module.ExportSummaryDialog(summary)
    qtbot.addWidget(dialog)

    text = dialog.findChild(QPlainTextEdit, "exportSummaryText")
    buttons = dialog.findChild(QDialogButtonBox, "exportSummaryButtons")

    assert isinstance(text, QPlainTextEdit)
    assert isinstance(buttons, QDialogButtonBox)
    dialog.show()
    dialog.resize(dialog.minimumSize())
    qtbot.wait(1)
    assert (
        dialog.windowTitle(),
        dialog.accessibleName(),
        text.isReadOnly(),
        text.toPlainText(),
        text.accessibleName(),
        text.verticalScrollBar().maximum() > 0,
        dialog.height() <= dialog.screen().availableGeometry().height(),
        bool(buttons.button(QDialogButtonBox.StandardButton.Close).accessibleName()),
    ) == ("导出完成", "导出完成", True, summary, "导出文件清单", True, True, True)
