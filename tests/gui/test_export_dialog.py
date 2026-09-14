from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtGui import QFontInfo
from PySide6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QWidget,
)
from tests.support.model_cases import final_fit_result, fit_candidate, simple_structure

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.model.fitting import FitStageSummary


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


STAGE_HISTORY = ("A", "B", "E")


def _stage_summaries(candidate_id: str, stages: tuple[str, ...]) -> tuple[FitStageSummary, ...]:
    """Mirror the staged search built by ``tests/unit/services/test_export_run_description``.

    Stage A is exempt from candidate-reference validation, and the last Stage E
    scope is the only scope allowed to publish a winner, so every injected stage
    must name the retained candidate for ``best_index=0`` to stay selectable.
    """
    return tuple(
        FitStageSummary(
            stage=stage,
            candidate_ids=(candidate_id,),
            best_objective=1.0,
            total_nfev=12,
            stop_reasons=("converged",),
        )
        for stage in stages
    )


def _fitted_project(
    tmp_path: Path,
    *,
    datasets: int = 1,
    stages: tuple[str, ...] = (),
    seed_index: int = 0,
    batch_mode: str = "independent",
) -> api.XrrProject:
    """Build a published-looking project, switching batch mode before results land.

    ``set_batch_mode`` clears every ``last_valid_result`` (joint and independent runs
    do not share a projection), so a caller asking for ``"joint"`` has to get the mode
    change in first or the export helpers would see an unfitted project.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        seed_index=seed_index,
        qz_a_inv=data.qz_a_inv,
        model_normalized=data.intensity_normalized,
        log_residuals_decades=np.zeros(data.qz_a_inv.size),
        residuals=np.zeros(data.qz_a_inv.size),
        weighted_residuals=np.zeros(data.qz_a_inv.size),
        # A joint projection must claim the mean of the aligned local objectives;
        # every dataset here shares one candidate, so the mean is its own objective.
        ranking_objective=None if batch_mode == "independent" else fit_candidate("selected").objective,
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
        stage_summaries=_stage_summaries(candidate.candidate_id, stages),
    )
    rows = [value.datasets[0]]
    if datasets == 2:
        rows.append(replace(rows[0], dataset_id="second", display_name="second"))
    value = api.set_batch_mode(replace(value, datasets=tuple(rows)), batch_mode)
    value = replace(
        value,
        datasets=tuple(replace(dataset, last_valid_result=result) for dataset in value.datasets),
    )
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


def _option_double(
    *,
    optional: tuple[api.ExportFormat, ...] = (api.ExportFormat.ORT,),
    accepted: bool = True,
):
    """Stand in for the pre-export format dialog without a modal event loop.

    The double answers ``selected_formats``, which is the whole of what the workflow
    reads off the dialog; a stand-in that still spelled the three checkbox properties
    would turn a forwarding regression into an ``AttributeError`` instead of a failed
    assertion. It composes that set the way the real dialog does -- the locked default
    set plus whichever optional formats were ticked -- so call sites name only their
    opt-ins and a format added to :data:`api.DEFAULT_FORMATS` is not restated here.
    """
    from PySide6.QtWidgets import QDialog

    code = QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected
    return lambda *_args, **_kwargs: SimpleNamespace(
        exec=lambda: code,
        selected_formats=(*api.DEFAULT_FORMATS, *optional),
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
    monkeypatch.setattr(module, "OrtOptionDialog", _option_double(optional=(api.ExportFormat.ORT,)))
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
    monkeypatch.setattr(module, "OrtOptionDialog", _option_double(optional=(api.ExportFormat.ORT,)))
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


def test_export_results_with_the_ort_format_publishes_ort_with_extension_disclosure(
    tmp_path: Path,
) -> None:
    workflow = _workflow(_fitted_project(tmp_path))

    manifest = workflow.export_results(
        tmp_path / "exports",
        formats=(*api.DEFAULT_FORMATS, api.ExportFormat.ORT),
    )
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

    workflow.export_results(tmp_path / "exports", formats=(*api.DEFAULT_FORMATS, api.ExportFormat.ORT))

    assert "covariance_absent_reason" in workflow.summary_text


def test_export_results_without_ort_publishes_no_ort_artifact(
    tmp_path: Path,
) -> None:
    workflow = _workflow(_fitted_project(tmp_path))

    manifest = workflow.export_results(tmp_path / "exports", formats=api.DEFAULT_FORMATS)
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


def test_export_dialog_forwards_the_chosen_format_set_and_cancel_aborts(
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

    monkeypatch.setattr(module, "OrtOptionDialog", _option_double(optional=(api.ExportFormat.ORT,)))
    assert workflow.export_results_dialog(None) is manifest
    assert captured["kwargs"] == {"formats": (*api.DEFAULT_FORMATS, api.ExportFormat.ORT)}

    captured.clear()
    monkeypatch.setattr(
        module,
        "OrtOptionDialog",
        _option_double(optional=(api.ExportFormat.CSV, api.ExportFormat.SVG)),
    )
    assert workflow.export_results_dialog(None) is manifest
    assert captured["kwargs"] == {"formats": (*api.DEFAULT_FORMATS, api.ExportFormat.CSV, api.ExportFormat.SVG)}

    captured.clear()
    monkeypatch.setattr(module, "OrtOptionDialog", _option_double(accepted=False))
    assert workflow.export_results_dialog(None) is None
    assert captured == {}


DESIGN_SECTIONS = (
    ("exportFormatsGroup", "数据格式"),
    ("exportPlotsGroup", "图件"),
    ("exportExtrasGroup", "随附内容"),
)

ALWAYS_PUBLISHED_CHECKBOXES = (
    "xlsxOptionCheckbox",
    "overviewPngCheckbox",
    "residualsPngCheckbox",
    "parameterTableCheckbox",
    "correlationCheckbox",
    "manifestCheckbox",
)

# The three artefacts ``services.exports`` really switches. Every row is enabled,
# carries an accessible name, and lacks the locked tooltip; only the opening
# default differs, so the table states all four facts instead of just the default.
OPTIONAL_CHECKBOX_STATES = {
    "ortOptionCheckbox": {"checked": True, "enabled": True, "named": True, "always": False},
    "csvOptionCheckbox": {"checked": False, "enabled": True, "named": True, "always": False},
    "sldSvgCheckbox": {"checked": True, "enabled": True, "named": True, "always": False},
}


def _group_title(dialog: QWidget, name: str) -> str | None:
    """The caption of one design section, or ``None`` where the group is absent."""
    group = dialog.findChild(QGroupBox, name)
    return None if group is None else group.title()


def _group_titles(dialog: QWidget) -> dict[str, str | None]:
    return {name: _group_title(dialog, name) for name, _title in DESIGN_SECTIONS}


def _checkbox_state(dialog: QWidget, name: str) -> dict[str, bool] | None:
    """Everything the design pins on one checkbox, or ``None`` if it is missing.

    Reporting a mapping rather than asserting attribute by attribute lets a single
    comparison print the whole table, so a missing box and a wrong default surface
    together instead of the test stopping at the first surprise.
    """
    box = dialog.findChild(QCheckBox, name)
    if box is None:
        return None
    return {
        "checked": box.isChecked(),
        "enabled": box.isEnabled(),
        "named": bool(box.accessibleName()),
        "always": "始终" in box.toolTip(),
    }


def _checkbox_states(dialog: QWidget, names: Iterable[str]) -> dict[str, dict[str, bool] | None]:
    return {name: _checkbox_state(dialog, name) for name in names}


def _expected_states(names: Iterable[str], **fields: bool) -> dict[str, dict[str, bool]]:
    """One identical expected row per name, for the group whose boxes all behave alike."""
    return {name: dict(fields) for name in names}


def _preview_payload(dialog: QWidget) -> dict[str, object]:
    """The manifest preview parsed back out of the label the dialog renders it into."""
    preview = dialog.findChild(QLabel, "exportManifestPreview")
    assert isinstance(preview, QLabel)
    return json.loads(preview.text())


def test_export_option_dialog_covers_design_format_sections(qtbot) -> None:
    """Frame ⑥ groups the nine artefacts into 数据格式 / 图件 / 随附内容.

    Only ``.ort``/CSV/SVG are real switches in ``services.exports``; the six
    artefacts that every publication writes stay visible but locked, so the
    dialog never promises a toggle the export contract cannot honour.
    """
    module = _exports()
    dialog = module.OrtOptionDialog()
    qtbot.addWidget(dialog)

    assert _group_titles(dialog) == dict(DESIGN_SECTIONS)
    assert _checkbox_states(dialog, OPTIONAL_CHECKBOX_STATES) == OPTIONAL_CHECKBOX_STATES
    assert _checkbox_states(dialog, ALWAYS_PUBLISHED_CHECKBOXES) == _expected_states(
        ALWAYS_PUBLISHED_CHECKBOXES, checked=True, enabled=False, named=True, always=True
    )

    assert (dialog.include_csv, dialog.include_svg) == (False, True)
    dialog.findChild(QCheckBox, "csvOptionCheckbox").setChecked(True)
    dialog.findChild(QCheckBox, "sldSvgCheckbox").setChecked(False)
    assert (dialog.include_csv, dialog.include_svg) == (True, False)


def test_export_option_dialog_manifest_preview_describes_run(qtbot, tmp_path: Path) -> None:
    """The preview must read the project instead of the mock-up's placeholders.

    ``services.exports`` derives ``mode``/``stages``/``confidence``/``reproducible``
    while publishing; the dialog states the same conclusions beforehand, so the
    design's English ``"correlated"`` cannot stand in for the real verdict.

    The payload is compared whole rather than key by key. The preview is the last
    place the run's identity is stated before publication, so a key that quietly
    appears or vanishes is as much a defect as a wrong value.
    """
    module = _exports()
    project = _fitted_project(tmp_path, stages=STAGE_HISTORY)
    dialog = module.OrtOptionDialog(None, project=project)
    qtbot.addWidget(dialog)

    caption = dialog.findChild(QLabel, "exportManifestCaption")
    assert isinstance(caption, QLabel)
    assert "export_manifest.json" in caption.text()
    assert "run_manifest.json" not in caption.text()

    assert api.ConfidenceClass.TRUSTED.value == "可信"
    default_formats = [value.value for value in api.DEFAULT_FORMATS]
    assert _preview_payload(dialog) == {
        "app": "xrr-fitter",
        "master_seed": project.master_seed,
        "datasets": 1,
        "mode": "independent",
        "stages": list(STAGE_HISTORY),
        "confidence": api.ConfidenceClass.TRUSTED.value,
        "reproducible": True,
        # 预览报的是 ``export_result`` 会收到的整个格式集合，所以锁定的默认集合也在里面；
        # 只有 ``ort``/``svg`` 是开箱勾选的可选项，``csv`` 未勾选就不该出现。
        "formats": [*default_formats, "ort", "svg"],
    }

    # 取消一个可选格式要真的把它从请求里拿掉，而不是只翻转一个没人读的布尔。
    dialog.findChild(QCheckBox, "ortOptionCheckbox").setChecked(False)
    assert _preview_payload(dialog)["formats"] == [*default_formats, "svg"]


def test_manifest_preview_is_quoted_as_a_file_not_written_as_prose(qtbot, tmp_path: Path) -> None:
    """帧⑥ 把清单预览画成下沉的等宽块，不是对话框自己的一段灰正文。

    预览的内容是 ``indent=2`` 的 JSON。比例字体会把缩进排歪，而缩进是它 pretty-print 的
    全部理由——排歪之后，读者反而不如去看压成一行的原文。等宽在这里是内容的正确性要求。

    这里断言的是「实际落到控件上的字体是等宽」，不是「样式表里写了 monospace」：Qt 只按已安装
    字体族匹配 ``font-family``，没有哪个平台装了名叫 monospace 的族，所以那样写会静默退回比例
    字体——评审时看着像等宽，发货时不是。
    """
    module = _exports()
    dialog = module.OrtOptionDialog(None, project=_fitted_project(tmp_path, stages=STAGE_HISTORY))
    qtbot.addWidget(dialog)
    dialog.setStyleSheet(theme.build_stylesheet(dialog.palette()))

    preview = dialog.findChild(QLabel, "exportManifestPreview")
    preview.ensurePolished()
    assert QFontInfo(preview.font()).fixedPitch(), "清单预览没落到等宽字体，JSON 缩进会排歪"
    assert preview.property("codeBlock") is True, "清单预览没有下沉框，会被读成对话框自己的说明"


def test_export_option_dialog_preview_tracks_joint_mode_and_missing_seed(qtbot, tmp_path: Path) -> None:
    """``mode`` follows ``batch_mode`` and ``reproducible`` follows ``seed_index``."""
    module = _exports()
    joint = _fitted_project(tmp_path / "joint", datasets=2, stages=STAGE_HISTORY, batch_mode="joint")
    dialog = module.OrtOptionDialog(None, project=joint)
    qtbot.addWidget(dialog)
    payload = _preview_payload(dialog)
    assert (payload["mode"], payload["datasets"]) == ("joint", 2)

    archived = _fitted_project(tmp_path / "archived", stages=STAGE_HISTORY, seed_index=-1)
    replay = module.OrtOptionDialog(None, project=archived)
    qtbot.addWidget(replay)
    assert _preview_payload(replay)["reproducible"] is False


def test_export_summary_close_button_is_labelled_in_chinese(qtbot) -> None:
    """关闭按钮要写中文，和同一条导出链路上的另一个对话框一致。

    ``QDialogButtonBox`` 的标准按钮文字来自 Qt 自带翻译，而应用没有装中文
    translator，所以它发出去是 "Close"；紧挨着它的 ``OrtOptionDialog`` 用的是自己
    设的「导出到文件夹…」「取消」。截图里同一步骤的两个窗口一个中文一个英文。
    """
    module = _exports()
    dialog = module.ExportSummaryDialog("artifact")
    qtbot.addWidget(dialog)

    buttons = dialog.findChild(QDialogButtonBox, "exportSummaryButtons")

    assert buttons.button(QDialogButtonBox.StandardButton.Close).text() == "关闭"
