from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTreeWidget,
    QWidget,
)

import xrr_fitter.api as api


def _accessibility():
    try:
        return importlib.import_module("xrr_fitter.gui.accessibility")
    except ModuleNotFoundError as error:
        pytest.fail(f"missing Slice 9 accessibility implementation: {error}", pytrace=False)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(
            f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}"
            for index in range(32)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _document(tmp_path: Path | None = None):
    from xrr_fitter.gui.document import ProjectDocument

    if tmp_path is None:
        return ProjectDocument()
    source = _write_curve(tmp_path / "sample.xy")
    project = api.add_dataset(
        api.new_project(),
        source,
        api.InstrumentSpec(instrument_id="accessible-instrument"),
    )
    return ProjectDocument(project)


def _named(root: QWidget, object_name: str) -> QWidget:
    widget = root if root.objectName() == object_name else root.findChild(QWidget, object_name)
    assert widget is not None, object_name
    return widget


def _assert_named(root: QWidget, object_names: tuple[str, ...]) -> None:
    for object_name in object_names:
        widget = _named(root, object_name)
        assert widget.accessibleName(), object_name
        assert widget.toolTip(), object_name


def test_column_mapping_dialog_names_all_actionable_controls(qtbot) -> None:
    from xrr_fitter.gui.data.import_dialog import ColumnMappingDialog

    module = _accessibility()
    dialog = ColumnMappingDialog()
    qtbot.addWidget(dialog)
    module.configure_accessibility(dialog)

    _assert_named(
        dialog,
        (
            "twoThetaColumnEditor",
            "intensityColumnEditor",
            "intensitySigmaEnabled",
            "intensitySigmaColumnEditor",
            "resolutionEnabled",
            "resolutionColumnEditor",
            "resolutionKindEditor",
        ),
    )
    for standard in (QDialogButtonBox.StandardButton.Ok, QDialogButtonBox.StandardButton.Cancel):
        assert dialog.buttons.button(standard).accessibleName()


def test_data_panel_shows_source_and_instrument_summaries_with_full_tooltips(
    qtbot,
    tmp_path: Path,
) -> None:
    from xrr_fitter.gui.data.panel import DataPanel

    module = _accessibility()
    panel = DataPanel(_document(tmp_path))
    qtbot.addWidget(panel)
    module.configure_accessibility(panel)
    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None
    item = tree.topLevelItem(0)
    dataset = panel.document.project.datasets[0]

    assert item.toolTip(1) == dataset.source_path
    assert item.toolTip(2) == panel.beam_text(dataset.dataset_id)
    assert item.toolTip(3) == panel.instrument_text(dataset.dataset_id)
    assert item.toolTip(5) == dataset.source_sha256
    assert "accessible-instrument" in item.text(3)


def test_import_action_is_keyboard_accessible_and_shows_path_error(
    qtbot,
    tmp_path: Path,
) -> None:
    from xrr_fitter.gui.data.panel import DataPanel

    module = _accessibility()
    panel = DataPanel(_document())
    qtbot.addWidget(panel)
    module.configure_accessibility(panel)
    bad_path = tmp_path / "broken.xy"
    button = panel.import_files_button

    assert button.accessibleName() == "导入文件"
    assert panel.import_files_shortcut.key() == QKeySequence("Ctrl+I")
    assert module.accessible_error_text(bad_path, ValueError("not an XRR dataset")) == (
        f"{bad_path}\nValueError: not an XRR dataset"
    )


def test_import_dialog_names_all_actionable_controls(qtbot, tmp_path: Path) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    module = _accessibility()
    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),), folder_mode=True)
    qtbot.addWidget(dialog)
    module.configure_accessibility(dialog)

    _assert_named(
        dialog,
        (
            "recursiveFolderImportCheck",
            "monochromaticWavelengthEditor",
            "mixedWavelength1Editor",
            "mixedWavelength2Editor",
            "mixedIntensityRatioEditor",
            "instrumentIdEditor",
            "footprintModeEditor",
            "sampleLengthEditor",
            "beamWidthEditor",
            "backgroundModelEditor",
            "resolutionDomainEditor",
            "columnMappingButton",
        ),
    )
    assert dialog.mono_button.accessibleName() == "单色光路"
    assert dialog.mixed_button.accessibleName() == "混合 Kα 光路"
    assert dialog.import_button().accessibleName() == "确认导入"


def test_main_window_has_three_accessible_columns(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    module = _accessibility()
    window = MainWindow()
    qtbot.addWidget(window)
    module.configure_accessibility(window)

    expected = {
        "projectColumn": "项目与数据列",
        "plotColumn": "反射率与 SLD 列",
        "analysisColumn": "参数与结果列",
    }
    assert window.workspace_splitter.count() == 3
    for object_name, name in expected.items():
        widget = _named(window, object_name)
        assert widget.accessibleName() == name
        assert widget.accessibleDescription()


def test_panels_have_stable_accessible_identity_and_titles(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    module = _accessibility()
    window = MainWindow()
    qtbot.addWidget(window)
    module.configure_accessibility(window)

    expected = {
        "dataPanel": "数据与掩膜",
        "structurePanel": "样品结构",
        "plotPanel": "反射率、SLD 与拟合诊断",
        "parametersPanel": "参数与共享",
    }
    for object_name, name in expected.items():
        panel = _named(window, object_name)
        assert panel.accessibleName() == name
        assert panel.accessibleDescription()


def test_project_and_fit_commands_are_visible_and_accessible_at_minimum_size(qtbot) -> None:
    from xrr_fitter.gui.fitting.panel import FitPanel
    from xrr_fitter.gui.main_window import MainWindow

    module = _accessibility()
    window = MainWindow()
    fit_panel = FitPanel(window.document)
    qtbot.addWidget(window)
    qtbot.addWidget(fit_panel)
    module.configure_accessibility(window)
    module.configure_accessibility(fit_panel)
    window.resize(1280, 760)
    window.show()
    fit_panel.show()
    qtbot.wait(1)

    for root, names in (
        (window, ("newProjectButton", "openProjectButton", "saveProjectButton", "saveAsProjectButton")),
        (fit_panel, ("startFitButton", "cancelFitButton", "forceStopFitButton")),
    ):
        for name in names:
            button = _named(root, name)
            assert button.isVisible()
            assert button.accessibleName() and button.toolTip()


def test_result_panel_mcmc_controls_have_descriptive_accessible_names(qtbot) -> None:
    from xrr_fitter.gui.results.panel import ResultsPanel

    module = _accessibility()
    panel = ResultsPanel(_document())
    qtbot.addWidget(panel)
    module.configure_accessibility(panel)

    _assert_named(
        panel,
        (
            "mcmcWalkers",
            "mcmcBurnIn",
            "mcmcProduction",
            "mcmcThin",
            "mcmcButton",
            "cancelMcmcButton",
            "forceStopMcmcButton",
        ),
    )


def test_result_panel_wraps_long_evidence_and_preserves_copy_accessibility(qtbot) -> None:
    from xrr_fitter.gui.results.panel import ResultsPanel

    module = _accessibility()
    panel = ResultsPanel(_document())
    qtbot.addWidget(panel)
    module.configure_accessibility(panel)
    text = "\n".join(f"evidence {index}: " + "x" * 80 for index in range(80))
    panel.uncertainty.evidence.setPlainText(text)
    button = module.create_copy_button(panel.uncertainty.evidence, panel)

    button.click()

    assert panel.uncertainty.evidence.isReadOnly()
    assert panel.uncertainty.evidence.lineWrapMode() != QPlainTextEdit.NoWrap
    assert button.objectName() == "copyEvidenceButton"
    assert button.accessibleName() == "复制拟合证据"
    assert button.toolTip()
    assert QApplication.clipboard().text() == text


def test_structure_component_actions_are_accessible_and_dataset_gated(qtbot) -> None:
    from xrr_fitter.gui.structure.editor import StructureEditor
    from tests.support.model_cases import simple_structure

    module = _accessibility()
    editor = StructureEditor(lambda _value: None, lambda: None, lambda: None)
    qtbot.addWidget(editor)
    module.configure_accessibility(editor)

    buttons = (
        editor.add_layer_button,
        editor.add_periodic_button,
        editor.remove_button,
        editor.up_button,
        editor.down_button,
    )
    assert all(button.accessibleName() and button.toolTip() for button in buttons)
    assert all(not button.isEnabled() for button in buttons)

    editor.load(simple_structure())

    assert editor.add_layer_button.isEnabled()
    assert editor.add_periodic_button.isEnabled()
    assert not editor.remove_button.isEnabled()


def test_save_as_button_and_primary_controls_are_accessible_at_minimum_size(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    module = _accessibility()
    window = MainWindow()
    qtbot.addWidget(window)
    module.configure_accessibility(window)
    window.resize(1280, 760)
    window.show()
    qtbot.wait(1)

    names = (
        "newProjectButton",
        "openProjectButton",
        "saveProjectButton",
        "saveAsProjectButton",
        "reloadSourceButton",
        "relinkSourceButton",
        "importFilesButton",
        "importFolderButton",
    )
    for name in names:
        button = _named(window, name)
        assert button.isVisible()
        assert button.accessibleName() and button.toolTip()
        assert button.width() >= button.minimumSizeHint().width()


def test_parameter_grid_has_specific_accessible_identity_and_tooltip(qtbot) -> None:
    from xrr_fitter.gui.parameters.table import ParameterTable

    module = _accessibility()
    table = ParameterTable()
    qtbot.addWidget(table)
    module.configure_accessibility(table)

    assert table.accessibleName() == "拟合参数与边界表"
    assert table.toolTip() == "逐行查看和编辑参数初值、边界、拟合状态与共享关系"


def test_primary_commands_have_precise_accessible_names_and_tooltips(qtbot) -> None:
    from xrr_fitter.gui.fitting.panel import FitPanel
    from xrr_fitter.gui.main_window import MainWindow
    from xrr_fitter.gui.results.panel import ResultsPanel

    module = _accessibility()
    window = MainWindow()
    fit_panel = FitPanel(window.document)
    results = ResultsPanel(window.document)
    for widget in (window, fit_panel, results):
        qtbot.addWidget(widget)
        module.configure_accessibility(widget)

    expected = {
        "importFilesButton": ("导入文件", "选择一个或多个 XRR 数据文件并确认导入设置"),
        "importFolderButton": ("导入文件夹", "选择 XRR 数据文件夹并确认批量导入设置"),
        "reloadSourceButton": ("重新加载活动数据源", "从当前路径重新读取活动数据集并核对哈希"),
        "relinkSourceButton": ("重新链接活动数据源", "为活动数据集选择新源文件并核对哈希"),
        "startFitButton": ("开始一键拟合", "运行当前项目的拟合工作流"),
        "mcmcButton": ("运行专家 MCMC", "对当前候选解运行显式专家 MCMC"),
    }
    roots = (window, fit_panel, results)
    for object_name, values in expected.items():
        matches = [root.findChild(QPushButton, object_name) for root in roots]
        button = next(value for value in matches if value is not None)
        assert (button.accessibleName(), button.toolTip()) == values
