from __future__ import annotations

import importlib
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QMessageBox, QPlainTextEdit
from tests.support.model_cases import final_fit_result, fit_candidate, simple_structure

import xrr_fitter.api as api


def _exports():
    try:
        return importlib.import_module("xrr_fitter.gui.export.dialog")
    except ModuleNotFoundError as error:
        pytest.fail(f"missing Slice 9 export implementation: {error}", pytrace=False)


def _write_curve(path: Path, scale: float = 1000.0) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {scale / (index + 1):.12g}" for index in range(32)) + "\n",
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
    result = replace(
        final_fit_result(candidate),
        parameter_definitions=(
            api.ParameterDefinition(
                "scale",
                "Scale",
                "",
                "instrument",
                1.0,
                0.5,
                1.5,
                "linear",
                False,
            ),
        ),
    )
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
        SimpleNamespace(path="compatibility_summary.xlsx", size=101, sha256="a" * 64),
        SimpleNamespace(path="export_manifest.json", size=202, sha256="b" * 64),
        SimpleNamespace(path="01-curve/fit_result.json", size=303, sha256="c" * 64),
    )
    return SimpleNamespace(
        run_directory=directory / "run",
        datasets=(SimpleNamespace(dataset_id="curve"),),
        files=records,
        root_files=records[:2],
    )


def _option_double(*, include_ort: bool = True, accepted: bool = True):
    """Stand in for the pre-export ORT option dialog without a modal event loop."""
    from PySide6.QtWidgets import QDialog

    code = QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
    return lambda *_args, **_kwargs: SimpleNamespace(exec=lambda: code, include_ort=include_ort)


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
    monkeypatch.setattr(module, "OrtOptionDialog", _option_double(include_ort=True))
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
            f"目标目录：{destination}\nOSError: disk full\n请检查目标目录的写入权限和可用空间后重试。",
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
    monkeypatch.setattr(module, "OrtOptionDialog", _option_double(include_ort=True))
    monkeypatch.setattr(api, "export_result", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(module, "ExportSummaryDialog", CapturingSummaryDialog)

    assert workflow.export_results_dialog(None) is manifest

    assert shown == [(workflow.summary_text, None), "exec"]
    record = manifest.files[1]
    assert (
        f"{manifest.run_directory / record.path} ({record.size} bytes, sha256 {record.sha256})"
    ) in workflow.summary_text


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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot publish")),
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


def _assert_published_ort_paths(manifest, summary: str) -> None:
    records = tuple(record for record in manifest.files if str(record.path).endswith(".ort"))
    assert records
    for record in records:
        published = manifest.run_directory / record.path
        assert (published.is_file(), str(published) in summary) == (True, True)


def test_export_results_include_ort_publishes_ort_with_extension_disclosure(
    tmp_path: Path,
) -> None:
    workflow = _workflow(_fitted_project(tmp_path))

    manifest = workflow.export_results(tmp_path / "exports", include_ort=True)
    summary = workflow.summary_text

    _assert_published_ort_paths(manifest, summary)
    # 扩展字段说明：三个扩展命名空间都要在摘要里点名
    assert "xrr_fitter.confidence" in summary
    assert "xrr_fitter.reduction" in summary
    assert "xrr_fitter.model" in summary
    # final_fit_result 的 uncertainty=None -> 协方差缺席，摘要写出缺席原因字段
    assert "covariance_absent_reason" in summary


def test_export_results_discloses_covariance_absence_for_selected_candidate_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _fitted_project(tmp_path)
    dataset = project.datasets[0]
    result = dataset.last_valid_result
    assert result is not None
    selected = result.candidates[0]
    owner = replace(selected, candidate_id="owner")
    uncertainty = api.UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.eye(1),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=owner.candidate_id,
        parameter_sigma=np.array([0.1]),
    )
    updated_result = replace(
        result,
        candidates=(owner, selected),
        best_index=0,
        uncertainty=uncertainty,
    )
    project = replace(
        project,
        datasets=(replace(dataset, last_valid_result=updated_result),),
    )
    manifest = SimpleNamespace(
        run_directory=tmp_path / "exports" / "run",
        datasets=(SimpleNamespace(dataset_id="curve"),),
        files=(
            SimpleNamespace(
                path="01-curve/fit_result.ort",
                size=123,
                sha256="d" * 64,
            ),
        ),
    )
    monkeypatch.setattr(api, "export_result", lambda *_args, **_kwargs: manifest)
    workflow = _workflow(project)

    workflow.export_results(tmp_path / "exports", include_ort=True)

    assert "covariance_absent_reason" in workflow.summary_text


def test_export_results_without_ort_publishes_no_ort_artifact(
    tmp_path: Path,
) -> None:
    workflow = _workflow(_fitted_project(tmp_path))

    manifest = workflow.export_results(tmp_path / "exports", include_ort=False)
    summary = workflow.summary_text

    assert not any(str(record.path).endswith(".ort") for record in manifest.files)
    assert not list(manifest.run_directory.rglob("*.ort"))
    assert ".ort" not in summary
    assert "xrr_fitter.confidence" not in summary


def test_ort_option_dialog_defaults_checked_and_accessible(qtbot) -> None:
    from PySide6.QtWidgets import QCheckBox

    module = _exports()
    dialog = module.OrtOptionDialog()
    qtbot.addWidget(dialog)

    checkbox = dialog.findChild(QCheckBox, "ortOptionCheckbox")
    assert isinstance(checkbox, QCheckBox)
    assert checkbox.isChecked()
    assert checkbox.accessibleName()
    assert checkbox.toolTip()
    assert dialog.objectName() == "ortOptionDialog"
    assert dialog.accessibleName()
    assert dialog.include_ort is True
    checkbox.setChecked(False)
    assert dialog.include_ort is False


def test_export_dialog_forwards_ort_choice_and_cancel_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _exports()
    workflow = _workflow(api.new_project())
    destination = tmp_path / "exports"
    manifest = _manifest(destination)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(destination),
    )
    monkeypatch.setattr(
        api,
        "export_result",
        lambda *_args, **_kwargs: (captured.update(kwargs=_kwargs), manifest)[1],
    )
    monkeypatch.setattr(
        module,
        "ExportSummaryDialog",
        lambda *_args, **_kwargs: SimpleNamespace(exec=lambda: None),
    )

    monkeypatch.setattr(module, "OrtOptionDialog", _option_double(include_ort=True))
    assert workflow.export_results_dialog(None) is manifest
    assert captured["kwargs"] == {"include_ort": True}

    captured.clear()
    monkeypatch.setattr(module, "OrtOptionDialog", _option_double(accepted=False))
    assert workflow.export_results_dialog(None) is None
    assert captured == {}
