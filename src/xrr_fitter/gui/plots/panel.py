"""Transactional Qt projection of prepared data and fit diagnostics.

Every public mutation renders a complete Agg scratch projection before it
touches the live Qt canvases.  Python state is committed only after the live
draw succeeds; a live failure redraws the previous projection.  Matplotlib
event ownership is delegated to one composed interaction controller so panel
teardown has one explicit callback and figure release boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui.plots.diagnostics import (
    TAB_SPECS,
    DiagnosticView,
    build_scratch_views,
    build_tabs,
    draw_batch_trends,
    draw_candidate_comparison,
    draw_empty,
    release_scratch_views,
    validate_batch_trends,
)
from xrr_fitter.gui.plots.interactions import (
    PlotInteractionController,
    PlotInteractionToolbar,
    ordered_finite_range,
)
from xrr_fitter.gui.plots.reflectivity import (
    draw_log,
    draw_qz4,
    draw_raw,
    draw_residual,
    prepare_project_plots,
    preview_display_values,
    validate_plot_data,
    validate_result,
)
from xrr_fitter.gui.plots.sld import draw_sld, draw_uncertainty


BatchTrends = tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class _Projection:
    data: api.PreparedData | None
    mask: np.ndarray | None
    result: object | None
    candidate_id: str | None
    trends: BatchTrends | None
    visible_range: tuple[float, float] | None


def _empty_state_widget(panel: PlotPanel) -> QWidget:
    widget = QWidget(panel)
    widget.setObjectName("plotEmptyState")
    title = QLabel("尚未导入数据")
    title.setObjectName("emptyStateTitle")
    title.setProperty("emptyTitle", True)
    title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    body = QLabel(
        "导入 .xy / .dat / .txt 反射率数据后，这里将显示曲线、SLD 剖面与拟合诊断。"
    )
    body.setProperty("mutedText", True)
    body.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    body.setWordWrap(True)
    button = QPushButton("导入数据文件…")
    button.setObjectName("emptyStateImportButton")
    button.setProperty("primary", True)
    button.clicked.connect(panel.import_requested.emit)
    hint = QLabel("也可以使用菜单「文件 ▸ 导入数据文件…」或快捷键 Ctrl+I")
    hint.setProperty("mutedText", True)
    hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(button)
    row.addStretch(1)
    layout = QVBoxLayout(widget)
    layout.setSpacing(12)
    layout.addStretch(2)
    layout.addWidget(title)
    layout.addWidget(body)
    layout.addLayout(row)
    layout.addWidget(hint)
    layout.addStretch(3)
    return widget


class PlotPanel(QWidget):
    """Render one selected dataset only after a complete drawing preflight."""

    fit_range_requested = Signal(float, float)
    point_mask_requested = Signal(int)
    view_changed = Signal(int)
    import_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("plotPanel")
        self.setAccessibleName("反射率、SLD 与拟合诊断")
        self._datasets: dict[str, api.PreparedData] = {}
        self._masks: dict[str, np.ndarray] = {}
        self._dataset_id: str | None = None
        self._result: object | None = None
        self._candidate_id: str | None = None
        self._trends: BatchTrends | None = None
        self._visible_range: tuple[float, float] | None = None
        self._preview_line: object | None = None
        self._released = False
        self.toolbar = PlotInteractionToolbar(self)
        self.tabs, self._views = build_tabs()
        content = QWidget(self)
        content.setObjectName("plotContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        content_layout.addWidget(self.toolbar)
        content_layout.addWidget(self.tabs, 1)
        self._pages = QStackedLayout(self)
        self._pages.addWidget(_empty_state_widget(self))
        self._pages.addWidget(content)
        self._sync_pages()
        self._interactions = PlotInteractionController(self, self.toolbar)
        self._install_view_shortcuts()

    def _install_view_shortcuts(self) -> None:
        """Bind Alt+1..Alt+8 to the diagnostic tabs by visible position.

        Users switch among eight diagnostic plots constantly; clicking or cycling
        with Ctrl+Tab is slow. Numbering by visible position (not fixed view key)
        keeps the keys contiguous when the expert-only SLD tab is hidden, so the
        same key never lands on a hidden tab or skips a number.
        """
        self.view_shortcuts: list[QShortcut] = []
        for position in range(len(self.view_keys())):
            shortcut = QShortcut(QKeySequence(f"Alt+{position + 1}"), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.setProperty("viewPosition", position)
            # Bind the position through the sender rather than a lambda closing
            # over self; a self-capturing closure held by the shortcut's signal
            # forms a cycle PySide cannot break, leaking the panel on teardown.
            shortcut.activated.connect(self._view_shortcut_activated)
            self.view_shortcuts.append(shortcut)

    def _view_shortcut_activated(self) -> None:
        shortcut = self.sender()
        if shortcut is not None:
            self.select_visible_view(int(shortcut.property("viewPosition")))

    def select_visible_view(self, position: int) -> bool:
        """Select the Nth (0-based) currently-visible diagnostic view."""
        visible = [
            key
            for index, key in enumerate(self.view_keys())
            if self.tabs.isTabVisible(index)
        ]
        if not 0 <= position < len(visible):
            return False
        self.select_view(visible[position])
        return True

    def _sync_pages(self) -> None:
        self._pages.setCurrentIndex(0 if self._dataset_id is None else 1)

    def tab_titles(self) -> tuple[str, ...]:
        return tuple(self.tabs.tabText(index) for index in range(self.tabs.count()))

    def view_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _title, _description in TAB_SPECS)

    def view(self, key: str) -> DiagnosticView:
        try:
            return self._views[key]
        except KeyError as error:
            raise KeyError(f"unknown diagnostic view: {key}") from error

    def selected_dataset_id(self) -> str | None:
        return self._dataset_id

    def selected_candidate_id(self) -> str | None:
        return self._candidate_id

    def current_view_key(self) -> str:
        return self._interactions.current_view_key()

    def select_view(self, key: str) -> None:
        self._interactions.select_view(key)

    def set_expert_mode(self, enabled: bool) -> None:
        self._interactions.set_expert_mode(enabled)

    def apply_workspace(self, *, expert_mode: bool, tab_index: int) -> None:
        self._interactions.apply_workspace(expert_mode, tab_index)

    def mode_buttons(self) -> dict[str, object]:
        return self.toolbar.buttons()

    def interaction_mode(self) -> str:
        return self.toolbar.mode()

    def set_interaction_mode(self, mode: str) -> None:
        self.toolbar.set_mode(mode)

    def set_dataset(self, dataset_id: str, data: api.PreparedData) -> None:
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset id must be nonempty")
        if not isinstance(data, api.PreparedData):
            raise TypeError("data must be PreparedData")
        mask = validate_plot_data(data, data.fit_mask)
        projection = _Projection(data, mask, None, None, self._trends, self._visible_range)
        self._transact(projection)
        self._datasets[dataset_id] = data
        self._masks[dataset_id] = mask
        self._dataset_id = dataset_id
        self._result = None
        self._candidate_id = None
        self._sync_pages()

    def select_dataset(self, dataset_id: str) -> None:
        if dataset_id not in self._datasets:
            raise KeyError(f"unknown dataset: {dataset_id}")
        projection = _Projection(
            self._datasets[dataset_id],
            self._masks[dataset_id],
            None,
            None,
            self._trends,
            self._visible_range,
        )
        self._transact(projection)
        self._dataset_id = dataset_id
        self._result = None
        self._candidate_id = None
        self._sync_pages()

    def update_mask(self, dataset_id: str, mask: object) -> None:
        if dataset_id not in self._datasets:
            raise KeyError(f"unknown dataset: {dataset_id}")
        converted = validate_plot_data(self._datasets[dataset_id], mask)
        if dataset_id != self._dataset_id:
            self._masks[dataset_id] = converted
            return
        projection = _Projection(
            self._datasets[dataset_id],
            converted,
            self._result,
            self._candidate_id,
            self._trends,
            self._visible_range,
        )
        self._transact(projection)
        self._masks[dataset_id] = converted

    def set_result(self, result: object, candidate_id: str | None) -> None:
        data = self._active_data()
        validate_result(data, result)
        self._candidate(result, candidate_id)
        projection = _Projection(
            data,
            self._active_mask(),
            result,
            candidate_id,
            self._trends,
            self._visible_range,
        )
        self._transact(projection)
        self._result = result
        self._candidate_id = candidate_id

    def set_batch_trends(
        self,
        dataset_ids: tuple[str, ...],
        thickness_a: tuple[float, ...],
        period_a: tuple[float, ...],
    ) -> None:
        trends = (tuple(dataset_ids), tuple(thickness_a), tuple(period_a))
        validate_batch_trends(*trends)
        projection = self._current_projection(trends=trends)
        self._transact(projection)
        self._trends = trends

    def select_fit_range(self, first: float, second: float) -> bool:
        return self._interactions.select_fit_range(first, second)

    def request_point_mask(self, index: int) -> bool:
        return self._interactions.request_point_mask(index)

    def show_range(self, lower: float, upper: float) -> None:
        visible = ordered_finite_range(lower, upper)
        projection = self._current_projection(visible_range=visible)
        self._transact(projection)
        self._visible_range = visible

    def visible_range(self) -> tuple[float, float] | None:
        return self._visible_range

    def zoom_to_range(self) -> bool:
        """Focus the angle-domain views on the active fit range.

        The fit range is often a small window of a wide scan, so keeping the
        full sweep on screen buries the region the user is actually judging.
        This clamps the raw and log x-axes to the highlighted range; it is a
        pure view operation that leaves the committed projection untouched, so
        the next redraw restores the full sweep on its own.
        """
        visible = self._visible_range
        if visible is None or self._released or self._dataset_id is None:
            return False
        for key in ("raw", "log"):
            view = self._views[key]
            view.axes.set_xlim(*visible)
            view.canvas.draw_idle()
        return True

    def reset_zoom(self) -> bool:
        """Return the angle-domain views to their data-driven autoscale."""
        if self._released or self._dataset_id is None:
            return False
        for key in ("raw", "log"):
            view = self._views[key]
            view.axes.autoscale(enable=True, axis="x")
            view.axes.relim()
            view.axes.autoscale_view(scalex=True, scaley=False)
            view.canvas.draw_idle()
        return True

    def cancel_interaction(self) -> None:
        self._interactions.cancel()

    def _clear_visible_range(self) -> None:
        projection = self._current_projection(visible_range=None)
        self._transact(projection)
        self._visible_range = None

    def displayed_prepared_indices(self) -> tuple[int, ...]:
        data = self._active_data()
        finite = np.isfinite(data.two_theta_deg) & np.isfinite(data.intensity_raw)
        return tuple(int(index) for index in np.flatnonzero(finite))

    def callback_counts(self) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        return self._interactions.callback_counts()

    def resources_released(self) -> bool:
        return self._released

    def project_project(self, project: api.XrrProject) -> None:
        prepared = prepare_project_plots(project)
        self._candidate(prepared.result, prepared.candidate_id)
        projection = _Projection(
            prepared.data,
            prepared.mask,
            prepared.result,
            prepared.candidate_id,
            self._trends,
            self._visible_range,
        )
        self._transact(projection)
        self._datasets = prepared.datasets
        self._masks = prepared.masks
        self._dataset_id = prepared.dataset_id
        self._result = prepared.result
        self._candidate_id = prepared.candidate_id
        self._sync_pages()
        self.apply_workspace(
            expert_mode=project.ui_state.expert_mode,
            tab_index=project.ui_state.plot_tab_index,
        )

    def _active_data(self) -> api.PreparedData:
        if self._dataset_id is None:
            raise RuntimeError("no active plot dataset")
        return self._datasets[self._dataset_id]

    def _active_mask(self) -> np.ndarray:
        if self._dataset_id is None:
            raise RuntimeError("no active plot dataset")
        return self._masks[self._dataset_id]

    def _candidate(self, result: object | None, candidate_id: str | None) -> object | None:
        if result is None or candidate_id is None:
            return None
        matches = tuple(
            candidate
            for candidate in result.candidates
            if candidate.candidate_id == candidate_id
        )
        if len(matches) != 1:
            raise KeyError(f"unknown candidate: {candidate_id}")
        return matches[0]

    def _comparison_candidates(self, projection: _Projection) -> tuple[object, ...]:
        """Return the other candidates whose SLD profiles overlay the selected one."""
        result = projection.result
        if result is None or projection.candidate_id is None:
            return ()
        return tuple(
            candidate
            for candidate in result.candidates
            if candidate.candidate_id != projection.candidate_id
        )

    def _current_projection(self, **changes: object) -> _Projection:
        values = {
            "data": None if self._dataset_id is None else self._active_data(),
            "mask": None if self._dataset_id is None else self._active_mask(),
            "result": self._result,
            "candidate_id": self._candidate_id,
            "trends": self._trends,
            "visible_range": self._visible_range,
        }
        values.update(changes)
        return _Projection(**values)

    def set_preview_curve(
        self,
        qz_a_inv: object,
        model_normalized: object,
    ) -> bool:
        """Overlay the searching model on the log view without a projection.

        A live preview updates many times per search, so it mutates one owned
        line artist instead of running the transactional redraw. It carries no
        committed evidence and is discarded whenever a real projection lands.
        """
        if self._dataset_id is None or self._released:
            return False
        angles, values = preview_display_values(
            self._active_data(),
            qz_a_inv,
            model_normalized,
        )
        view = self._views["log"]
        if self._preview_line is None:
            self._preview_line = view.axes.plot(
                angles,
                values,
                "-",
                color="#009E73",
                linewidth=1.4,
                label="搜索中模型",
            )[0]
            view.axes.legend()
        else:
            self._preview_line.set_data(angles, values)
        view.canvas.draw_idle()
        return True

    def clear_preview_curve(self) -> None:
        """Drop the live overlay so committed evidence renders on its own."""
        line = self._preview_line
        self._preview_line = None
        if line is None or self._released:
            return
        line.remove()
        view = self._views["log"]
        view.axes.legend()
        view.canvas.draw_idle()

    def _transact(self, projection: _Projection) -> None:
        if self._released:
            raise RuntimeError("plot panel resources have been released")
        # A full projection clears every axes, so the preview artist it owned
        # is already gone; dropping the reference avoids reusing a dead line.
        self._preview_line = None
        scratch = build_scratch_views()
        try:
            self._draw(scratch, projection)
        finally:
            release_scratch_views(scratch)
        previous = self._current_projection()
        try:
            self._draw(self._views, projection)
        except Exception:
            self._draw(self._views, previous)
            raise

    def _draw(self, views: dict[str, DiagnosticView], projection: _Projection) -> None:
        data = projection.data
        candidate = self._candidate(projection.result, projection.candidate_id)
        if data is None or projection.mask is None:
            for key in ("raw", "log", "qz4", "residual", "sld"):
                title = next(title for name, title, _description in TAB_SPECS if name == key)
                draw_empty(views[key], title)
        else:
            draw_raw(views["raw"], data, projection.mask, candidate)
            draw_log(views["log"], data, candidate)
            draw_qz4(views["qz4"], data, candidate)
            draw_residual(views["residual"], candidate)
            draw_sld(views["sld"], candidate, self._comparison_candidates(projection))
            self._draw_range(views, projection.visible_range)
        draw_candidate_comparison(views["candidates"], projection.result, projection.candidate_id)
        draw_uncertainty(views["uncertainty"], projection.result, projection.candidate_id)
        draw_batch_trends(views["trend"], projection.trends)

    def _draw_range(
        self,
        views: dict[str, DiagnosticView],
        visible_range: tuple[float, float] | None,
    ) -> None:
        if visible_range is None:
            return
        for key in ("raw", "log"):
            views[key].axes.axvspan(*visible_range, color="#E69F00", alpha=0.16, label="拟合范围")
            views[key].canvas.draw_idle()

    def release_resources(self) -> None:
        if self._released:
            return
        self._released = True
        self._interactions.release()
        for view in self._views.values():
            view.canvas.release()
            view.figure.clear()
            view.figure.set_canvas(None)
        self._datasets.clear()
        self._masks.clear()
        self._result = None
        self._candidate_id = None

    def closeEvent(self, event: object) -> None:
        self.release_resources()
        super().closeEvent(event)

    def event(self, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.DeferredDelete, QEvent.Type.Destroy):
            self.release_resources()
        elif event.type() == QEvent.Type.ParentChange:
            controller = getattr(self, "_interactions", None)
            if controller is not None:
                controller.watch_parent()
        return super().event(event)
