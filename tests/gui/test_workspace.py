from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api


def _workspace():
    try:
        return importlib.import_module("xrr_fitter.gui.workspace")
    except ModuleNotFoundError as error:
        pytest.fail(f"missing Slice 9 workspace implementation: {error}", pytrace=False)


class _RejectingTabs(QTabWidget):
    def __init__(self) -> None:
        super().__init__()
        self.rejected_index: int | None = None

    def setCurrentIndex(self, index: int) -> None:
        if index == self.rejected_index:
            raise RuntimeError("simulated workspace projection failure")
        super().setCurrentIndex(index)


def _workspace_root(qtbot, *, rejecting: bool = False):
    module = _workspace()
    root = QWidget()
    root.setObjectName("workspaceRoot")
    outer = QVBoxLayout(root)
    workspace = QSplitter(Qt.Orientation.Horizontal)
    workspace.setObjectName("workspaceSplitter")
    left = QSplitter(Qt.Orientation.Vertical)
    left.setObjectName("leftSplitter")
    left.addWidget(QWidget())
    left.addWidget(QWidget())
    plot_column = QWidget()
    plot_layout = QVBoxLayout(plot_column)
    tabs = _RejectingTabs() if rejecting else QTabWidget()
    tabs.setObjectName("diagnosticTabs")
    tabs.addTab(QWidget(), "曲线")
    tabs.addTab(QWidget(), "SLD")
    plot_layout.addWidget(tabs)
    analysis = QWidget()
    analysis_layout = QVBoxLayout(analysis)
    expert = QCheckBox("专家模式")
    expert.setObjectName("expertModeToggle")
    analysis_layout.addWidget(expert)
    workspace.addWidget(left)
    workspace.addWidget(plot_column)
    workspace.addWidget(analysis)
    workspace.setSizes([320, 580, 380])
    left.setSizes([280, 480])
    outer.addWidget(workspace)
    qtbot.addWidget(root)
    root.resize(1280, 760)
    root.show()
    qtbot.wait(1)
    return root, module.WorkspaceView.from_root(root)


def test_invalid_footprint_geometry_does_not_mutate_state(qtbot) -> None:
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    module = _workspace()
    dialog = ImportDialog((Path("curve.xy"),))
    qtbot.addWidget(dialog)
    dialog.select_beam_kind("monochromatic")
    dialog.footprint.setCurrentIndex(dialog.footprint.findData("geometry"))
    dialog.sample_length.setValue(1.0)
    dialog.beam_width.setValue(2.0)
    project = api.new_project()
    state = project.ui_state

    with pytest.raises(ValueError, match="0 < beam_width_mm <= sample_length_mm"):
        dialog.instrument_spec()

    assert project.ui_state is state
    assert module.workspace_values(project.ui_state) == module.workspace_values(state)


def test_main_window_save_and_reopen_restores_workspace_state(
    qtbot,
    tmp_path: Path,
) -> None:
    module = _workspace()
    _root, view = _workspace_root(qtbot)
    view.workspace_splitter.setSizes([300, 640, 340])
    assert view.left_splitter is not None
    view.left_splitter.setSizes([250, 510])
    assert view.plot_tabs is not None and view.expert_toggle is not None
    view.plot_tabs.setCurrentIndex(1)
    view.expert_toggle.setChecked(True)
    project = module.capture_project(api.new_project(), view)
    target = tmp_path / "workspace.xrrproj.json"
    api.save_project(project, target)
    reopened = api.load_project(target)
    _new_root, new_view = _workspace_root(qtbot)

    module.restore_project(new_view, reopened)

    assert module.workspace_values(reopened.ui_state) == module.workspace_values(project.ui_state)
    assert tuple(new_view.workspace_splitter.sizes()) == reopened.ui_state.workspace_splitter_sizes
    assert new_view.left_splitter is not None
    assert tuple(new_view.left_splitter.sizes()) == reopened.ui_state.left_splitter_sizes
    assert new_view.plot_tabs is not None and new_view.plot_tabs.currentIndex() == 1
    assert new_view.expert_toggle is not None and new_view.expert_toggle.isChecked()


def test_minimum_and_large_layout_keep_panels_visible_and_nonoverlapping(qtbot) -> None:
    module = _workspace()
    root, view = _workspace_root(qtbot)
    module.configure_splitters(view)

    for size in ((1280, 760), (1600, 900)):
        root.resize(*size)
        qtbot.wait(1)
        rectangles = []
        for index in range(view.workspace_splitter.count()):
            widget = view.workspace_splitter.widget(index)
            rectangle = QRect(widget.mapTo(root, QPoint(0, 0)), widget.size())
            assert widget.isVisible()
            assert root.rect().contains(rectangle)
            rectangles.append(rectangle)
        for index, rectangle in enumerate(rectangles):
            assert all(not rectangle.intersects(other) for other in rectangles[index + 1 :])


def test_open_failure_restores_complete_previous_workspace(qtbot) -> None:
    module = _workspace()
    _root, view = _workspace_root(qtbot, rejecting=True)
    before = view.snapshot()
    assert isinstance(view.plot_tabs, _RejectingTabs)
    view.plot_tabs.rejected_index = 1
    project = api.set_workspace_state(
        api.new_project(),
        replace(
            api.new_project().ui_state,
            workspace_splitter_sizes=(250, 700, 330),
            left_splitter_sizes=(330, 430),
            plot_tab_index=1,
            expert_mode=True,
        ),
    )

    with pytest.raises(RuntimeError, match="simulated workspace projection failure"):
        module.restore_project(view, project)

    assert view.snapshot() == before


def test_splitters_preserve_approved_layout_contract(qtbot) -> None:
    module = _workspace()
    _root, view = _workspace_root(qtbot)

    module.configure_splitters(view)

    workspace = view.workspace_splitter
    assert workspace.orientation() == Qt.Orientation.Horizontal
    assert workspace.count() == 3
    assert not workspace.childrenCollapsible()
    assert all(not workspace.isCollapsible(index) for index in range(3))
    assert view.left_splitter is not None
    assert view.left_splitter.orientation() == Qt.Orientation.Vertical
    assert view.left_splitter.count() == 2
    assert not view.left_splitter.childrenCollapsible()
    assert all(not view.left_splitter.isCollapsible(index) for index in range(2))
