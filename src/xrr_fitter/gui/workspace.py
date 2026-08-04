"""Transactional capture and projection of the persisted GUI workspace."""

from __future__ import annotations

from dataclasses import dataclass, replace

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QAbstractButton, QSplitter, QTabWidget, QWidget

import xrr_fitter.api as api


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    workspace_splitter_sizes: tuple[int, int, int]
    left_splitter_sizes: tuple[int, int] | None
    plot_tab_index: int | None
    expert_mode: bool | None


@dataclass(frozen=True, slots=True)
class WorkspaceView:
    root: QWidget
    workspace_splitter: QSplitter
    left_splitter: QSplitter | None
    plot_tabs: QTabWidget | None
    expert_toggle: QAbstractButton | None

    @classmethod
    def from_root(cls, root: QWidget) -> WorkspaceView:
        workspace = root.findChild(QSplitter, "workspaceSplitter")
        if workspace is None:
            raise LookupError("workspaceSplitter is required")
        return cls(
            root,
            workspace,
            root.findChild(QSplitter, "leftSplitter"),
            root.findChild(QTabWidget, "diagnosticTabs"),
            root.findChild(QAbstractButton, "expertModeToggle"),
        )

    def snapshot(self) -> WorkspaceSnapshot:
        return WorkspaceSnapshot(
            tuple(self.workspace_splitter.sizes()),
            None if self.left_splitter is None else tuple(self.left_splitter.sizes()),
            None if self.plot_tabs is None else self.plot_tabs.currentIndex(),
            None if self.expert_toggle is None else self.expert_toggle.isChecked(),
        )

    def apply(self, snapshot: WorkspaceSnapshot) -> None:
        previous = self.snapshot()
        try:
            self._apply(snapshot)
        except Exception:
            self._apply(previous)
            raise

    def _apply(self, snapshot: WorkspaceSnapshot) -> None:
        widgets = [self.workspace_splitter]
        widgets.extend(
            widget
            for widget in (self.left_splitter, self.plot_tabs, self.expert_toggle)
            if widget is not None
        )
        blockers = [QSignalBlocker(widget) for widget in widgets]
        self.workspace_splitter.setSizes(list(snapshot.workspace_splitter_sizes))
        if self.left_splitter is not None and snapshot.left_splitter_sizes is not None:
            self.left_splitter.setSizes(list(snapshot.left_splitter_sizes))
        if self.plot_tabs is not None and snapshot.plot_tab_index is not None:
            self.plot_tabs.setCurrentIndex(snapshot.plot_tab_index)
        if self.expert_toggle is not None and snapshot.expert_mode is not None:
            self.expert_toggle.setChecked(snapshot.expert_mode)
        del blockers


def _snapshot_from_state(state: api.ProjectUiState) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        state.workspace_splitter_sizes,
        state.left_splitter_sizes,
        state.plot_tab_index,
        state.expert_mode,
    )


def workspace_values(state: api.ProjectUiState) -> tuple[object, ...]:
    return (
        state.workspace_splitter_sizes,
        state.left_splitter_sizes,
        state.plot_tab_index,
        state.expert_mode,
    )


def capture_project(project: api.XrrProject, view: WorkspaceView) -> api.XrrProject:
    snapshot = view.snapshot()
    current = project.ui_state
    state = replace(
        current,
        workspace_splitter_sizes=snapshot.workspace_splitter_sizes,
        left_splitter_sizes=(
            current.left_splitter_sizes
            if snapshot.left_splitter_sizes is None
            else snapshot.left_splitter_sizes
        ),
        plot_tab_index=(
            current.plot_tab_index
            if snapshot.plot_tab_index is None
            else snapshot.plot_tab_index
        ),
        expert_mode=(
            current.expert_mode
            if snapshot.expert_mode is None
            else snapshot.expert_mode
        ),
    )
    return api.set_workspace_state(project, state)


def restore_project(view: WorkspaceView, project: api.XrrProject) -> None:
    view.apply(_snapshot_from_state(project.ui_state))


def configure_splitters(view: WorkspaceView) -> None:
    _connect_automatic_workflow(view.root)
    workspace = view.workspace_splitter
    workspace.setOrientation(Qt.Orientation.Horizontal)
    workspace.setChildrenCollapsible(False)
    for index in range(workspace.count()):
        workspace.setCollapsible(index, False)
        workspace.setStretchFactor(index, 1 if index == 1 else 0)
    left = view.left_splitter
    if left is None:
        return
    left.setOrientation(Qt.Orientation.Vertical)
    left.setChildrenCollapsible(False)
    for index in range(left.count()):
        left.setCollapsible(index, False)
        left.setStretchFactor(index, 1)


def _connect_automatic_workflow(root: QWidget) -> None:
    if root.property("automaticWorkflowConnected"):
        return
    data_panel = getattr(root, "data_panel", None)
    fit_panel = getattr(root, "fit_panel", None)
    if data_panel is None or fit_panel is None:
        return
    data_panel.automatic_fit_requested.connect(fit_panel.start_automatic_fit)
    root.setProperty("automaticWorkflowConnected", True)
