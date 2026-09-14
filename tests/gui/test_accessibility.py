from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
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
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
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


def test_the_workspace_columns_and_sections_carry_accessible_names(qtbot) -> None:
    """Losing the dock title bars must not leave the columns unnamed.

    A dock announced itself; a plain QWidget column announces nothing, so a
    screen reader met three unlabelled containers where it used to be told which
    panel it had entered.  The inspector's cards are named from the caption they
    already draw, which is the only label those sections have now.
    """
    from xrr_fitter.gui.main_window import MainWindow
    from xrr_fitter.gui.window_layout import INSPECTOR_SECTIONS

    module = _accessibility()
    window = MainWindow()
    qtbot.addWidget(window)
    module.configure_accessibility(window)

    columns = {
        "navigationColumn": "导航栏",
        "canvasColumn": "画布",
        "inspectorColumn": "上下文检查器",
    }
    for object_name, name in columns.items():
        column = window.findChild(QWidget, object_name)
        assert column is not None, object_name
        assert column.accessibleName() == name, object_name

    for object_name, title, _subtitle, _attribute in INSPECTOR_SECTIONS:
        card = window.inspector_column.findChild(QWidget, object_name)
        assert card is not None, object_name
        if title is None:
            # 抬头是 ``None`` 的那一段是个纯占位框（``_plain_section``）：它自己不画抬头，
            # 因为里面那张面板（帧① 的 ``resultsPanel``）已经自带三段带抬头的卡。给外层
            # 再补一个名字，读屏进出这两层容器时会把同一句话念两遍。
            assert card.accessibleName() == "", object_name
            continue
        assert card.accessibleName() == title, object_name


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
        (window, ("newProjectButton", "openProjectButton", "saveProjectButton")),
        (fit_panel, ("startAutomaticFitButton", "cancelFitButton")),
    ):
        for name in names:
            button = _named(root, name)
            assert button.isVisible()
            assert button.accessibleName() and button.toolTip()

    # 「强制停止」不在常驻命令里：它只在按过「停止并保留最优」之后才露出来，因为在没有
    # 待停的拟合时它没有可执行的语义。没露出来的命令仍要带得住读屏的名字与提示，否则它
    # 露出来的那一刻是个无名按钮。
    for name in ("startFitButton", "forceStopFitButton"):
        hidden = _named(fit_panel, name)
        assert hidden.isVisible() is False
        assert hidden.accessibleName() and hidden.toolTip()
    expert_start = _named(fit_panel, "startFitButton")

    window.document.replace_project(api.set_expert_mode(window.document.project, True))

    assert expert_start.isVisible() is True


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
    from tests.support.model_cases import simple_structure

    from xrr_fitter.gui.structure.editor import StructureEditor

    module = _accessibility()
    editor = StructureEditor(lambda _value: None, lambda: None, lambda: None)
    qtbot.addWidget(editor)
    module.configure_accessibility(editor)

    buttons = (
        editor.add_layer_button,
        editor.add_periodic_button,
    )
    assert all(button.accessibleName() and button.toolTip() for button in buttons)
    assert all(not button.isEnabled() for button in buttons)
    # 删除 / 上移 / 下移 / 编辑基底搬到了层行的右键菜单上，读屏读的是 ``QAction`` 的
    # text；它们同样在没有结构时全灰。
    actions = (
        editor.edit_backing_action,
        editor.remove_action,
        editor.up_action,
        editor.down_action,
    )
    assert all(action.text() and action.toolTip() for action in actions)
    assert all(not action.isEnabled() for action in actions)

    editor.load(simple_structure())

    assert editor.add_layer_button.isEnabled()
    assert editor.add_periodic_button.isEnabled()
    assert editor.edit_backing_action.isEnabled()
    assert not editor.remove_action.isEnabled()


def test_save_as_button_and_primary_controls_are_accessible_at_minimum_size(qtbot) -> None:
    from xrr_fitter.gui.main_window import MainWindow

    module = _accessibility()
    window = MainWindow()
    qtbot.addWidget(window)
    module.configure_accessibility(window)
    window.resize(1280, 760)
    window.show()
    # The import buttons live in the data dock, which the guided opening surface
    # hides; this contract is about the expert surface at its minimum size.
    window.set_guidance_visible(False)
    qtbot.wait(1)

    names = (
        "newProjectButton",
        "openProjectButton",
        "saveProjectButton",
        # 另存为 / 重载源 / 重链接源 曾经也摊在这条命令栏上，占掉三个位子。设计稿六张帧
        # 的 cmdbar 都只有前三颗，它们已经收回 文件 菜单——菜单项按文本自动排宽，没有
        # 被裁窄这回事，所以这份「量宽度」的名单里不再有它们。
        # 数据集抬头右侧的 ＋。导入的两颗按钮此前也在这份名单里，直接摊在抬头行；左栏
        # 实测 264 px，两颗就吃掉 144 px，专家模式再加「更换测量预设」共 242 px——第三
        # 颗那时量出来 0 宽，本该由这条契约拦住的「按钮被挤到画不出来」已经发生了。
        # 它们现在挂在 ＋ 的菜单里，由下一条契约走弹出后的真实路径来量。
        "datasetAddButton",
    )
    for name in names:
        _assert_named_control_fits(window, name)


def _assert_named_control_fits(window, name: str) -> None:
    """一颗按钮：看得见、有可及名与提示、没被裁窄。"""
    button = _named(window, name)
    assert button.isVisible()
    assert button.accessibleName() and button.toolTip()
    assert button.width() >= button.minimumSizeHint().width()


def test_dataset_import_menu_keeps_its_commands_accessible_at_minimum_size(qtbot) -> None:
    """展开 ＋ 之后，菜单里托管的导入命令仍然可见、有名字、没被裁窄。

    托管的是同一批 ``QPushButton``，所以可及名和提示还在它们身上；菜单是另一个窗口，
    弹出前它们 ``isVisible()`` 为假，因此这里先点开再量。
    """
    from xrr_fitter.gui.main_window import MainWindow

    module = _accessibility()
    window = MainWindow()
    qtbot.addWidget(window)
    module.configure_accessibility(window)
    window.resize(1280, 760)
    window.show()
    window.set_guidance_visible(False)
    qtbot.wait(1)

    add = window.findChild(QToolButton, "datasetAddButton")
    menu = add.menu()
    menu.popup(window.mapToGlobal(add.pos()))
    qtbot.addWidget(menu)
    qtbot.wait(1)
    try:
        for name in ("importFilesButton", "importFolderButton"):
            _assert_named_control_fits(window, name)
    finally:
        menu.close()


def test_parameter_grid_has_specific_accessible_identity_and_tooltip(qtbot) -> None:
    from xrr_fitter.gui.parameters.table import ParameterTable

    module = _accessibility()
    table = ParameterTable()
    qtbot.addWidget(table)
    module.configure_accessibility(table)

    assert table.accessibleName() == "拟合参数与边界表"
    assert table.toolTip() == "逐行查看和编辑参数初值、边界、锁定状态与先验"


def test_parameter_alignment_accessibility_does_not_commit_parameter_settings(
    qtbot,
    tmp_path: Path,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.parameters.panel import ParametersPanel

    module = _accessibility()
    source = _write_curve(tmp_path / "parameter-accessibility.xy")
    project = api.add_dataset(
        api.new_project(),
        source,
        api.InstrumentSpec(instrument_id="parameter-accessibility"),
    )
    project = api.set_structure(
        project,
        "parameter-accessibility",
        api.StructureSpec(
            api.MaterialSpec("Air", None, None, 0.0j),
            (
                api.LayerSpec(
                    "film",
                    api.MaterialSpec("SiO2", "SiO2", 2.20),
                    40.0,
                    roughness_a=3.0,
                ),
            ),
            api.MaterialSpec("Si", "Si", 2.329),
        ),
    )
    document = ProjectDocument(project)
    panel = ParametersPanel(document)
    qtbot.addWidget(panel)
    events: list[object] = []
    panel.settings_changed.connect(lambda dataset_id, settings: events.append((dataset_id, settings)))

    module.configure_accessibility(panel)

    assert document.project is project
    assert document.project.datasets[0].parameter_settings == ()
    assert events == []


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
        "startFitButton": ("开始一键拟合", "运行当前项目的拟合工作流"),
        "mcmcButton": ("运行专家 MCMC", "对当前候选解运行显式专家 MCMC"),
    }
    roots = (window, fit_panel, results)
    for object_name, values in expected.items():
        matches = [root.findChild(QPushButton, object_name) for root in roots]
        button = next(value for value in matches if value is not None)
        assert (button.accessibleName(), button.toolTip()) == values

    # 重载源 / 重链接源 只在 文件 菜单里（命令栏按设计稿只摆 新建/打开/保存），所以它们
    # 要念的名字由 QAction 的 text/toolTip 承担，而不是某颗按钮的 accessibleName。
    for object_name, text in (
        ("reloadSourceAction", "重新加载数据源"),
        ("relinkSourceAction", "重新链接数据源…"),
    ):
        action = window.chrome_actions[object_name]
        assert action.text() == text
        assert action.toolTip()
