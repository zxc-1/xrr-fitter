"""What the export flow tells the reader: stage names, interval promises, busy and done states.

Fixtures come from ``test_export_dialog`` so both files build the same published-looking project.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QProgressDialog,
    QPushButton,
    QScrollArea,
)
from tests.gui.test_export_dialog import (
    STAGE_HISTORY,
    _exports,
    _fitted_project,
    _option_double,
    _preview_payload,
    _workflow,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.fitting.progress import stage_text
from xrr_fitter.gui.window_layout import MINIMUM_SHELL_HEIGHT


def test_export_preview_names_stages_the_way_the_fit_card_does(qtbot, tmp_path: Path) -> None:
    """F62：预览把阶段写成拟合卡上的名字，而不是裸码 A/B/E。

    清单预览是发布前最后一处说明「这次跑了什么」的地方。裸码只对读过源码的人有意义；
    拟合卡已经给每个阶段起了中文名（名字末尾保留原码，方便对照进度和日志），这里沿用同一张表。
    """
    module = _exports()
    dialog = module.OrtOptionDialog(None, project=_fitted_project(tmp_path, stages=STAGE_HISTORY))
    qtbot.addWidget(dialog)

    stages = _preview_payload(dialog)["stages"]

    assert stages == [stage_text(stage) for stage in STAGE_HISTORY]
    assert not set(stages) & set(STAGE_HISTORY), f"预览仍在用裸阶段码：{stages}"


def _with_owned_uncertainty(project: api.XrrProject, dataset_id: str) -> api.XrrProject:
    """Give one dataset an uncertainty report owned by the candidate it selects."""
    datasets = []
    for dataset in project.datasets:
        result = dataset.last_valid_result
        if dataset.dataset_id == dataset_id and result is not None:
            report = api.UncertaintyReport(
                correlation_names=("scale",),
                correlation_matrix=np.eye(1),
                profiles=(),
                bootstrap_intervals=(),
                bootstrap_failure_rate=0.0,
                boundary_hits=(),
                strong_correlations=(),
                systematic_residual=False,
                diagnostics=(),
                candidate_id=result.candidates[0].candidate_id,
                parameter_sigma=np.array([0.1]),
            )
            dataset = replace(dataset, last_valid_result=replace(result, uncertainty=report))
        datasets.append(dataset)
    return replace(project, datasets=tuple(datasets))


def test_locked_interval_row_names_datasets_without_an_interval(qtbot, tmp_path: Path) -> None:
    """F62：「参数表与 16–84% 置信区间」锁定行不能对没有不确定度报告的数据集也承诺区间。

    参数表每次都发布，区间却只在选中候选自己拥有不确定度报告时才有；缺报告的数据集
    在参数表里只剩 ``sigma_unavailable_reason``。锁定行要点名这些数据集。
    """
    module = _exports()
    project = _with_owned_uncertainty(_fitted_project(tmp_path, datasets=2), "curve")
    dialog = module.OrtOptionDialog(None, project=project)
    qtbot.addWidget(dialog)

    row = dialog.findChild(QCheckBox, "parameterTableCheckbox")

    assert (row.isChecked(), row.isEnabled()) == (True, False)
    assert "second" in row.text(), f"缺区间的数据集没被点名：{row.text()!r}"
    assert "curve" not in row.text()
    assert "始终" in row.toolTip() and "second" in row.toolTip()


def test_locked_interval_row_keeps_its_promise_when_every_dataset_has_one(qtbot, tmp_path: Path) -> None:
    module = _exports()
    project = _fitted_project(tmp_path, datasets=2)
    for dataset_id in ("curve", "second"):
        project = _with_owned_uncertainty(project, dataset_id)
    dialog = module.OrtOptionDialog(None, project=project)
    qtbot.addWidget(dialog)

    assert dialog.findChild(QCheckBox, "parameterTableCheckbox").text() == "参数表与 16–84% 置信区间"


def _visible_busy_dialogs() -> list[QProgressDialog]:
    return [
        widget
        for widget in QApplication.topLevelWidgets()
        if isinstance(widget, QProgressDialog) and widget.objectName() == "exportBusyDialog" and widget.isVisible()
    ]


def test_export_shows_a_busy_indicator_while_it_writes(qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F21：导出在主线程同步写盘（约 1 s/数据集）；写盘期间要有忙碌提示，而不是看起来卡死。"""
    module = _exports()
    workflow = _workflow(_fitted_project(tmp_path, datasets=2))
    destination = tmp_path / "exports"
    real_export = api.export_result
    seen: list[tuple[str, object]] = []

    def observed_export(*args, **kwargs):
        busy = _visible_busy_dialogs()
        cursor = QApplication.overrideCursor()
        seen.append((busy[0].labelText() if busy else "", None if cursor is None else cursor.shape()))
        return real_export(*args, **kwargs)

    monkeypatch.setattr(module, "OrtOptionDialog", _option_double(optional=()))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: str(destination))
    monkeypatch.setattr(api, "export_result", observed_export)
    monkeypatch.setattr(module, "ExportSummaryDialog", lambda *_args, **_kwargs: SimpleNamespace(exec=lambda: None))

    assert workflow.export_results_dialog(None) is not None

    assert len(seen) == 1
    label, cursor = seen[0]
    assert "正在导出" in label and "2 个数据集" in label, f"写盘期间没有忙碌提示：{label!r}"
    assert cursor == Qt.CursorShape.WaitCursor
    QApplication.processEvents()
    assert not _visible_busy_dialogs()
    assert QApplication.overrideCursor() is None


def test_export_summary_opens_the_run_folder(qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """F21：完成框带「打开文件夹」，直达这次运行的目录，而不是让读者复制路径。"""
    module = _exports()
    run = tmp_path / "exports" / "run"
    run.mkdir(parents=True)
    opened: list[str] = []
    monkeypatch.setattr(
        module, "QDesktopServices", SimpleNamespace(openUrl=lambda url: opened.append(url.toLocalFile()) or True)
    )
    dialog = module.ExportSummaryDialog("artifact", run_directory=run)
    qtbot.addWidget(dialog)
    with qtbot.waitExposed(dialog):
        dialog.show()

    button = dialog.findChild(QPushButton, "exportOpenFolderButton")
    assert button is not None and button.text() == "打开文件夹"
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert opened == [str(run)]
    assert dialog.isVisible(), "打开文件夹后不该顺手关掉清单"


def _half_fitted_project(tmp_path: Path) -> api.XrrProject:
    """Two datasets where only ``curve`` carries a result; ``second`` was never fitted."""
    project = api.select_candidate(_fitted_project(tmp_path, datasets=2), "second", None)
    datasets = tuple(
        replace(dataset, last_valid_result=None) if dataset.dataset_id == "second" else dataset
        for dataset in project.datasets
    )
    return replace(project, datasets=datasets)


def test_export_unavailable_reason_names_the_datasets_without_results(tmp_path: Path) -> None:
    module = _exports()

    assert module.export_unavailable_reason(_fitted_project(tmp_path / "full", datasets=2), running=False) is None
    assert "运行" in module.export_unavailable_reason(_fitted_project(tmp_path / "busy"), running=True)
    assert "数据" in module.export_unavailable_reason(api.new_project(), running=False)
    reason = module.export_unavailable_reason(_half_fitted_project(tmp_path / "half"), running=False)
    assert "second" in reason and "curve" not in reason, reason


def test_greyed_export_entries_say_which_dataset_is_missing_a_result(qtbot, tmp_path: Path) -> None:
    """F21：导出灰着时，按钮与菜单项的提示点名还没有拟合结果的数据集。"""
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_half_fitted_project(tmp_path / "half")))
    qtbot.addWidget(window)
    button = window.export_button
    action = window._workflow_actions["exportResultsAction"]

    assert not button.isEnabled() and not action.isEnabled()
    assert "second" in button.toolTip() and "second" in action.toolTip(), (button.toolTip(), action.toolTip())
    assert "second" in action.statusTip()

    window.document.replace_project(_fitted_project(tmp_path / "full", datasets=2))

    assert button.isEnabled() and action.isEnabled()
    assert "second" not in button.toolTip() and "second" not in action.toolTip(), "可用之后还挂着旧的灰原因"


def test_export_options_fit_a_small_screen_and_scroll_the_rest(qtbot, tmp_path: Path) -> None:
    """F22：选项框原本最小 821 px 高、不能滚动，小屏上「导出到文件夹…」落到屏幕外。

    Qt 首次显示时把对话框夹到屏幕可用高度的 2/3，但布局最小高度压不下去。1366×768 的笔记本
    可用约 680 px，2/3 约 455 px：最小高度要低于它，多出来的内容进滚动区，页脚按钮留在滚动区外。
    """
    module = _exports()
    dialog = module.OrtOptionDialog(None, project=_fitted_project(tmp_path, stages=STAGE_HISTORY))
    qtbot.addWidget(dialog)
    floor = dialog.minimumSizeHint().height()

    assert floor <= MINIMUM_SHELL_HEIGHT * 2 // 3, f"选项框最小高 {floor}px，小屏放不下"
    dialog.resize(dialog.sizeHint().width(), floor)
    with qtbot.waitExposed(dialog):
        dialog.show()
    buttons = dialog.findChild(QDialogButtonBox, "ortOptionButtons")
    scroll = dialog.findChild(QScrollArea, "exportOptionsScroll")
    preview = dialog.findChild(QLabel, "exportManifestPreview")

    assert scroll is not None and scroll.isAncestorOf(preview) and not scroll.isAncestorOf(buttons)
    confirm = buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert dialog.rect().contains(QRect(confirm.mapTo(dialog, QPoint(0, 0)), confirm.size())), "确认按钮被挤出对话框"
    scroll.ensureWidgetVisible(preview)
    QApplication.processEvents()
    assert scroll.viewport().rect().intersects(QRect(preview.mapTo(scroll.viewport(), QPoint(0, 0)), preview.size()))
    # 大屏不该无故出滚动条：自然高度仍按全部内容请求。
    assert dialog.sizeHint().height() > floor


def test_export_options_need_no_scrolling_at_their_natural_size(qtbot, tmp_path: Path) -> None:
    """F22 的另一头：大屏按自然尺寸摆开时一次铺开，最窄时也不裁掉清单预览。

    清单是 ``indent=2`` 的 JSON。折行会把 ``master_seed`` 这种 20 位的数折成两行，自然高度差出
    一行，大屏也平白多一道滚动条；种子取 64 位上限，让这一行是最长的那种。
    """
    module = _exports()
    project = _fitted_project(tmp_path, datasets=2, stages=STAGE_HISTORY)
    dialog = module.OrtOptionDialog(None, master_seed=2**64 - 1, project=project)
    qtbot.addWidget(dialog)
    dialog.setStyleSheet(theme.build_stylesheet(dialog.palette()))
    scroll = dialog.findChild(QScrollArea, "exportOptionsScroll")
    preview = dialog.findChild(QLabel, "exportManifestPreview")
    with qtbot.waitExposed(dialog):
        dialog.show()

    dialog.resize(dialog.sizeHint())
    QApplication.processEvents()
    assert scroll.verticalScrollBar().maximum() == 0, "自然尺寸下仍要滚动：清单预览折行撑高了自己"
    dialog.resize(dialog.minimumSizeHint())
    QApplication.processEvents()
    assert preview.width() >= preview.sizeHint().width(), "最窄时清单预览被裁掉右半截"
