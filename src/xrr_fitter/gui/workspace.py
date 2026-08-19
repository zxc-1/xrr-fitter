"""Transactional capture and projection of the persisted GUI workspace."""

from __future__ import annotations

from dataclasses import dataclass, replace

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QAbstractButton, QSplitter, QTabWidget, QWidget

import xrr_fitter.api as api


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    workspace_splitter_sizes: tuple[int, int, int] | None
    left_splitter_sizes: tuple[int, int] | None
    plot_tab_index: int | None
    expert_mode: bool | None


@dataclass(frozen=True, slots=True)
class WorkspaceView:
    root: QWidget
    workspace_splitter: QSplitter | None
    left_splitter: QSplitter | None
    plot_tabs: QTabWidget | None
    expert_toggle: QAbstractButton | None

    @classmethod
    def from_root(cls, root: QWidget) -> WorkspaceView:
        """Bind to whichever workspace widgets the root actually has.

        Panel geometry moved to the dock layout, which persists separately as
        an opaque ``dock_state``. Splitters are therefore optional here; this
        view now carries only the tab selection and expert mode.
        """
        return cls(
            root,
            root.findChild(QSplitter, "workspaceSplitter"),
            root.findChild(QSplitter, "leftSplitter"),
            root.findChild(QTabWidget, "diagnosticTabs"),
            root.findChild(QAbstractButton, "expertModeToggle"),
        )

    def snapshot(self) -> WorkspaceSnapshot:
        return WorkspaceSnapshot(
            None if self.workspace_splitter is None else tuple(self.workspace_splitter.sizes()),
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

    def _assignments(
        self,
        snapshot: WorkspaceSnapshot,
    ) -> tuple[tuple[QWidget | None, object, object], ...]:
        """Pair each optional widget with its setter and the value to apply."""
        return (
            (
                self.workspace_splitter,
                snapshot.workspace_splitter_sizes,
                lambda widget, value: widget.setSizes(list(value)),
            ),
            (
                self.left_splitter,
                snapshot.left_splitter_sizes,
                lambda widget, value: widget.setSizes(list(value)),
            ),
            (
                self.plot_tabs,
                snapshot.plot_tab_index,
                lambda widget, value: widget.setCurrentIndex(value),
            ),
            (
                self.expert_toggle,
                snapshot.expert_mode,
                lambda widget, value: widget.setChecked(value),
            ),
        )

    def _apply(self, snapshot: WorkspaceSnapshot) -> None:
        pending = [
            (widget, value, setter)
            for widget, value, setter in self._assignments(snapshot)
            if widget is not None and value is not None
        ]
        blockers = [QSignalBlocker(widget) for widget, _value, _setter in pending]
        for widget, value, setter in pending:
            setter(widget, value)
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
        workspace_splitter_sizes=(
            current.workspace_splitter_sizes
            if snapshot.workspace_splitter_sizes is None
            else snapshot.workspace_splitter_sizes
        ),
        left_splitter_sizes=(
            current.left_splitter_sizes if snapshot.left_splitter_sizes is None else snapshot.left_splitter_sizes
        ),
        plot_tab_index=(current.plot_tab_index if snapshot.plot_tab_index is None else snapshot.plot_tab_index),
        expert_mode=(current.expert_mode if snapshot.expert_mode is None else snapshot.expert_mode),
    )
    return api.set_workspace_state(project, state)


def restore_project(view: WorkspaceView, project: api.XrrProject) -> None:
    view.apply(_snapshot_from_state(project.ui_state))


def configure_splitters(view: WorkspaceView) -> None:
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
