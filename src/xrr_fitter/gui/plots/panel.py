"""Transactional Qt projection of prepared data and fit diagnostics.

Every mutation validates a complete scratch projection before changing live
artists. A failed live draw restores the committed projection and all SLD view
state. SLD bands are persisted evidence; alternate alignments are view-only
cache entries owned by one dataset and one MCMC report. Dataset or report
changes invalidate that cache, while ordinary redraws reuse it.

The panel publishes Python state only after a successful draw. Project and
legacy dataset transitions therefore restore their dictionaries, structure,
selector, toggle, and cache together. Preview artists remain outside this
transaction and are discarded by the next full projection; teardown owns the
figures and callbacks idempotently.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.diagnostics import (
    TAB_SPECS,
    VIEW_SPECS,
    DiagnosticView,
    build_scratch_views,
    build_tabs,
    draw_batch_trends,
    draw_candidate_comparison,
    draw_empty,
    release_scratch_views,
    validate_batch_trends,
)
from xrr_fitter.gui.plots.heatmaps import draw_parameter_heatmap, draw_residual_heatmap
from xrr_fitter.gui.plots.interactions import (
    PlotInteractionController,
    PlotInteractionToolbar,
    ordered_finite_range,
)
from xrr_fitter.gui.plots.live import LiveReflectivityPlot
from xrr_fitter.gui.plots.reflectivity import (
    draw_log,
    draw_qz4,
    draw_raw,
    draw_residual,
    prepare_project_plots,
    preview_display_values,
    reflectivity_pane_arrays,
    validate_plot_data,
    validate_result,
)
from xrr_fitter.gui.plots.sld import draw_sld, draw_uncertainty
from xrr_fitter.gui.plots.sld_state import (
    ALIGN_KEYS,
    BatchTrends,
    Projection,
    SldBandReplay,
    SldViewState,
    alignment_index_from_cache,
    build_sld_companion_pane,
    cache_matches,
    candidate_for_result,
    capture_sld_view_state,
    committed_projection,
    comparison_candidates,
    current_projection,
    project_structure,
    projection_bands,
    projection_mcmc,
    reset_band_view,
    restore_sld_view_state,
    set_alignment_index,
    sync_band_controls,
    visible_bands,
)


def _empty_state_widget(panel: PlotPanel) -> QWidget:
    widget = QWidget(panel)
    widget.setObjectName("plotEmptyState")
    title = QLabel("尚未导入数据")
    title.setObjectName("emptyStateTitle")
    title.setProperty("emptyTitle", True)
    title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    body = QLabel("导入 .xy / .dat / .txt 反射率数据后，这里将显示曲线、SLD 剖面与拟合诊断。")
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
    layout.setSpacing(theme.SPACE_MD)
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
    # Emitted when an interface handle on the SLD pane is dragged: the payload is
    # the edited StructureSpec, committed by the window through the structure
    # editor so a hand edit and a typed edit take exactly the same path.
    structure_edit_requested = Signal(object)

    # The standing hint shown when the pointer is not over any curve; it names
    # the gesture rather than leaving the coordinate strip blank, so an empty
    # readout never reads as a broken control.
    CURSOR_IDLE_HINT = "将指针移到曲线上可读取坐标"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("plotPanel")
        self.setAccessibleName("反射率、SLD 与拟合诊断")
        self._datasets: dict[str, api.PreparedData] = {}
        self._masks: dict[str, np.ndarray] = {}
        self._dataset_id: str | None = None
        self._result: object | None = None
        self._candidate_id: str | None = None
        self._structure: object | None = None
        self._sld_band_cache: SldBandReplay | None = None
        self._trends: BatchTrends | None = None
        self._visible_range: tuple[float, float] | None = None
        self._released = False
        self.toolbar = PlotInteractionToolbar(self)
        self.tabs, self._views = build_tabs()
        self.sld_pane, self.sld_bands_toggle, self.sld_align_selector = build_sld_companion_pane(
            self,
            self._views,
            self._on_bands_toggled,
            self._on_align_changed,
        )
        # Reflectivity above, depth profile below: a fit is judged on curve
        # agreement and structural plausibility at once, so neither may hide
        # the other behind a tab.
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.plot_splitter.setObjectName("plotSplitter")
        self.plot_splitter.setChildrenCollapsible(False)
        self.plot_splitter.addWidget(self.tabs)
        self.plot_splitter.addWidget(self.sld_pane)
        # The diagnostic tabs carry the axis labels, legends and tick text that
        # a 3:2 split squeezed; the depth profile stays legible at a third of
        # the height because it plots two smooth curves against one axis.
        self.plot_splitter.setStretchFactor(0, 2)
        self.plot_splitter.setStretchFactor(1, 1)
        self._float_toolbar_over_plot()
        content = QWidget(self)
        content.setObjectName("plotContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(theme.SPACE_SM)
        content_layout.addWidget(self.plot_splitter, 1)
        # A graphical plot should say where the pointer is, like every peer
        # tool.  The controller feeds this label from the navigator's cursor
        # read-out, so it has to exist before the controller is built below.
        self.cursor_readout = QLabel(self.CURSOR_IDLE_HINT, content)
        self.cursor_readout.setObjectName("plotCursorReadout")
        self.cursor_readout.setProperty("mutedText", True)
        content_layout.addWidget(self.cursor_readout)
        self._pages = QStackedLayout(self)
        self._pages.addWidget(_empty_state_widget(self))
        self._pages.addWidget(content)
        self._sync_pages()
        self._interactions = PlotInteractionController(self, self.toolbar)
        self._install_view_shortcuts()

    def _float_toolbar_over_plot(self) -> None:
        """Lift the interaction bar out of the layout and onto the plot itself.

        Peer charting tools put their mode bar over the chart rather than in a
        labelled row above it: the controls belong to the picture they act on,
        and a row of their own spent a band of vertical space on chrome.  The bar
        parents to the tab stack, not to a tab page, so it survives a view switch
        instead of being rebuilt once per diagnostic.
        """
        self.toolbar.setParent(self.tabs)
        self.tabs.installEventFilter(self)
        self.tabs.currentChanged.connect(self._raise_toolbar_overlay)
        self._position_toolbar_overlay()

    def _position_toolbar_overlay(self) -> None:
        """Pin the bar inside the plot's top-right corner, clear of the tab bar.

        Sized to its own content: the bar has no stretch, so it covers only the
        strip of plot it needs.  The left edge is clamped to the margin so a
        narrow panel slides the bar leftward instead of pushing it off-screen.
        """
        if self._released:
            return
        tab_bar = self.tabs.tabBar()
        top = tab_bar.height() if tab_bar.isVisible() else 0
        hint = self.toolbar.sizeHint()
        margin = theme.SPACE_SM
        left = max(margin, self.tabs.width() - hint.width() - margin)
        self.toolbar.setGeometry(left, top + margin, hint.width(), hint.height())
        self.toolbar.raise_()

    def _raise_toolbar_overlay(self, *_args: object) -> None:
        """Re-assert the bar's place after a view switch restacks the children."""
        self._position_toolbar_overlay()

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        """Hold the floating bar in the corner as the plot stack is resized.

        An overlay is outside the layout, so nothing else moves it: without this
        the bar would keep its first geometry and drift away from the corner the
        moment the window, the splitter or the dock changed the plot's width.
        """
        if watched is self.tabs and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.LayoutRequest,
        ):
            self._position_toolbar_overlay()
        return super().eventFilter(watched, event)

    def _install_view_shortcuts(self) -> None:
        """Bind Alt+1..Alt+8 to the diagnostic tabs by visible position.

        Users switch among eight diagnostic plots constantly; clicking or cycling
        with Ctrl+Tab is slow. Numbering by visible position (not fixed view key)
        keeps the keys contiguous when the expert-only SLD tab is hidden, so the
        same key never lands on a hidden tab or skips a number.
        """
        self.view_shortcuts: list[QShortcut] = []
        for position in range(len(self.tab_keys())):
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
        visible = [key for index, key in enumerate(self.tab_keys()) if self.tabs.isTabVisible(index)]
        if not 0 <= position < len(visible):
            return False
        self.select_view(visible[position])
        return True

    def _sync_pages(self) -> None:
        self._pages.setCurrentIndex(0 if self._dataset_id is None else 1)

    def tab_titles(self) -> tuple[str, ...]:
        return tuple(self.tabs.tabText(index) for index in range(self.tabs.count()))

    def tab_keys(self) -> tuple[str, ...]:
        """The switchable diagnostic tabs, in tab-bar order."""
        return tuple(key for key, _title, _description in TAB_SPECS)

    def view_keys(self) -> tuple[str, ...]:
        """Every owned view, including the companion pane outside the tab bar."""
        return tuple(key for key, _title, _description in VIEW_SPECS)

    def view(self, key: str) -> DiagnosticView | LiveReflectivityPlot:
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

    def navigation_buttons(self) -> dict[str, object]:
        return self.toolbar.navigation_buttons()

    def navigation_mode(self) -> str:
        return self._interactions.navigation_mode()

    def navigators(self) -> dict[str, object]:
        return self._interactions.navigators()

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
        projection = Projection(
            data,
            mask,
            None,
            None,
            self._trends,
            self._visible_range,
            dataset_id,
            None,
        )
        previous_context = (
            self._datasets,
            self._masks,
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._structure,
            self._sld_band_cache,
            self.sld_align_selector.currentIndex(),
        )
        self._reset_sld_band_view(projection)
        try:
            self._transact(projection)
        except Exception:
            (
                self._datasets,
                self._masks,
                self._dataset_id,
                self._result,
                self._candidate_id,
                self._structure,
                self._sld_band_cache,
                alignment_index,
            ) = previous_context
            set_alignment_index(self.sld_align_selector, alignment_index)
            raise
        self._datasets = {**self._datasets, dataset_id: data}
        self._masks = {**self._masks, dataset_id: mask}
        self._dataset_id = dataset_id
        self._result = None
        self._candidate_id = None
        self._structure = None
        self._sync_pages()

    def select_dataset(self, dataset_id: str) -> None:
        if dataset_id not in self._datasets:
            raise KeyError(f"unknown dataset: {dataset_id}")
        projection = Projection(
            self._datasets[dataset_id],
            self._masks[dataset_id],
            None,
            None,
            self._trends,
            self._visible_range,
            dataset_id,
            None,
        )
        previous = (
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._structure,
            self._sld_band_cache,
            self.sld_align_selector.currentIndex(),
        )
        self._reset_sld_band_view(projection)
        try:
            self._transact(projection)
        except Exception:
            (
                self._dataset_id,
                self._result,
                self._candidate_id,
                self._structure,
                self._sld_band_cache,
                alignment_index,
            ) = previous
            set_alignment_index(self.sld_align_selector, alignment_index)
            raise
        self._dataset_id = dataset_id
        self._result = None
        self._candidate_id = None
        self._structure = None
        self._sync_pages()

    def update_mask(self, dataset_id: str, mask: object) -> None:
        if dataset_id not in self._datasets:
            raise KeyError(f"unknown dataset: {dataset_id}")
        converted = validate_plot_data(self._datasets[dataset_id], mask)
        if dataset_id != self._dataset_id:
            self._masks[dataset_id] = converted
            return
        projection = Projection(
            self._datasets[dataset_id],
            converted,
            self._result,
            self._candidate_id,
            self._trends,
            self._visible_range,
            dataset_id,
            self._structure,
        )
        self._transact(projection)
        self._masks[dataset_id] = converted

    def set_result(self, result: object, candidate_id: str | None) -> None:
        data = self._active_data()
        validate_result(data, result)
        candidate_for_result(result, candidate_id)
        previous_projection = self._current_projection()
        projection = Projection(
            data,
            self._active_mask(),
            result,
            candidate_id,
            self._trends,
            self._visible_range,
            self._dataset_id,
            self._structure,
        )
        current_report = projection_mcmc(self._result)
        report_changed = projection_mcmc(projection.result) is not current_report
        previous_sld_state = capture_sld_view_state(
            self._sld_band_cache,
            self.sld_bands_toggle,
            self.sld_align_selector,
        )
        if report_changed:
            self._reset_sld_band_view(projection)
        self._transact(
            projection,
            rollback_projection=previous_projection,
            rollback_sld_state=previous_sld_state,
        )
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
            if isinstance(view, LiveReflectivityPlot):
                view.set_view_xrange(*visible)
        return True

    def reset_zoom(self) -> bool:
        """Return the angle-domain views to their data-driven autoscale."""
        if self._released or self._dataset_id is None:
            return False
        for key in ("raw", "log"):
            view = self._views[key]
            if isinstance(view, LiveReflectivityPlot):
                view.autoscale_view()
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

    def set_cursor_readout(self, message: str) -> None:
        """Show the pointer's coordinates, or the standing hint when it leaves.

        Matplotlib clears the read-out with an empty string as the pointer
        leaves the axes, so an empty message restores the idle hint rather than
        letting a stale reading linger as if the pointer were still on the curve.
        """
        if self._released:
            return
        self.cursor_readout.setText(message if message else self.CURSOR_IDLE_HINT)

    def project_project(self, project: api.XrrProject) -> None:
        prepared = prepare_project_plots(project)
        candidate_for_result(prepared.result, prepared.candidate_id)
        projection = Projection(
            prepared.data,
            prepared.mask,
            prepared.result,
            prepared.candidate_id,
            self._trends,
            self._visible_range,
            prepared.dataset_id,
            project_structure(project, prepared.dataset_id),
        )
        previous_dataset_id = self._dataset_id
        previous_report = projection_mcmc(self._result)
        previous_context = (
            self._datasets,
            self._masks,
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._structure,
            self._sld_band_cache,
            self.sld_align_selector.currentIndex(),
        )
        previous_projection = self._committed_projection()
        previous_sld_state = capture_sld_view_state(
            self._sld_band_cache,
            self.sld_bands_toggle,
            self.sld_align_selector,
        )
        self._datasets, self._masks = prepared.datasets, prepared.masks
        self._dataset_id = prepared.dataset_id
        self._structure = projection.structure
        if prepared.dataset_id != previous_dataset_id or projection_mcmc(projection.result) is not previous_report:
            self._reset_sld_band_view(projection)
        try:
            self._transact(
                projection,
                rollback_projection=previous_projection,
                rollback_sld_state=previous_sld_state,
            )
        except Exception:
            (
                self._datasets,
                self._masks,
                self._dataset_id,
                self._result,
                self._candidate_id,
                self._structure,
                self._sld_band_cache,
                alignment_index,
            ) = previous_context
            set_alignment_index(self.sld_align_selector, alignment_index)
            raise
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

    def _reset_sld_band_view(self, projection: Projection) -> None:
        """Drop a view-only replay when its dataset or MCMC owner changes."""
        self._sld_band_cache = None
        reset_band_view(projection_bands(projection.result), self.sld_align_selector)

    def _on_bands_toggled(self) -> None:
        if self._released or self._dataset_id is None:
            return
        if self.sld_bands_toggle.isChecked():
            self._on_align_changed(self.sld_align_selector.currentIndex())
            return
        self._transact(self._current_projection())

    def _on_align_changed(self, index: int) -> None:
        """Recompute the bands for the picked alignment as a view-only overlay."""
        if self._released or self._dataset_id is None or self._structure is None or not 0 <= index < len(ALIGN_KEYS):
            return
        projection = self._current_projection()
        report = projection_mcmc(projection.result)
        if report is None or not self.sld_bands_toggle.isChecked():
            return
        alignment = ALIGN_KEYS[index]
        if cache_matches(self._sld_band_cache, self._dataset_id, report, alignment):
            self._transact(projection)
            return
        previous_index = alignment_index_from_cache(
            self._sld_band_cache,
            projection_bands(projection.result),
            self.sld_align_selector.itemText(1),
        )
        previous_sld_state = capture_sld_view_state(
            self._sld_band_cache,
            self.sld_bands_toggle,
            self.sld_align_selector,
        )
        rollback_sld_state = replace(previous_sld_state, alignment_index=previous_index)
        try:
            bands = api.sld_uncertainty_bands(
                self._structure,
                report,
                wavelength_a=self._active_data().beam.effective_wavelength_a,
                align=alignment,
            )
            self._sld_band_cache = SldBandReplay(self._dataset_id, report, alignment, bands)
            self._transact(
                projection,
                rollback_projection=self._committed_projection(),
                rollback_sld_state=rollback_sld_state,
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError):
            self._sld_band_cache = restore_sld_view_state(
                rollback_sld_state,
                self.sld_bands_toggle,
                self.sld_align_selector,
            )

    def _current_projection(self, **changes: object) -> Projection:
        return current_projection(
            self._datasets,
            self._masks,
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._trends,
            self._visible_range,
            self._structure,
            changes,
        )

    def set_preview_curve(
        self,
        qz_a_inv: object,
        model_normalized: object,
    ) -> bool:
        """Overlay the searching model on the log view without a projection.

        A live preview updates many times per search, so it pushes into one
        owned pyqtgraph curve instead of running the transactional redraw. It
        carries no committed evidence and is discarded whenever a real
        projection lands.
        """
        if self._dataset_id is None or self._released:
            return False
        angles, values = preview_display_values(
            self._active_data(),
            qz_a_inv,
            model_normalized,
        )
        view = self._views["log"]
        if not isinstance(view, LiveReflectivityPlot):
            return False
        return view.set_preview(angles, values)

    def clear_preview_curve(self) -> None:
        """Drop the live overlay so committed evidence renders on its own."""
        if self._released:
            return
        view = self._views["log"]
        if isinstance(view, LiveReflectivityPlot):
            view.clear_preview()

    def _transact(
        self,
        projection: Projection,
        *,
        rollback_projection: Projection | None = None,
        rollback_sld_state: SldViewState | None = None,
    ) -> None:
        if self._released:
            raise RuntimeError("plot panel resources have been released")
        # A full projection supersedes the live preview overlay. The pyqtgraph
        # log pane keeps its preview curve across show_log_reflectivity (unlike
        # the old axes.clear() path), so the overlay is dropped explicitly here.
        self.clear_preview_curve()
        scratch = build_scratch_views()
        try:
            self._draw(scratch, projection)
        finally:
            release_scratch_views(scratch)
        previous = self._committed_projection() if rollback_projection is None else rollback_projection
        try:
            self._draw(self._views, projection)
        except Exception:
            if rollback_sld_state is not None:
                self._sld_band_cache = restore_sld_view_state(
                    rollback_sld_state,
                    self.sld_bands_toggle,
                    self.sld_align_selector,
                )
            self._draw(self._views, previous)
            raise
        finally:
            # Whatever ended up on screen, the new projection or the rolled back
            # one, is the view the reset button has to return to.
            self._interactions.refresh_navigation_baselines()

    def _committed_projection(self) -> Projection:
        return committed_projection(
            self._datasets,
            self._masks,
            self._dataset_id,
            self._result,
            self._candidate_id,
            self._trends,
            self._visible_range,
            self._structure,
        )

    def _draw(
        self,
        views: dict[str, DiagnosticView | LiveReflectivityPlot],
        projection: Projection,
    ) -> None:
        data = projection.data
        candidate = candidate_for_result(projection.result, projection.candidate_id)
        bands = projection_bands(projection.result)
        sync_band_controls(
            self.sld_bands_toggle,
            self.sld_align_selector,
            bands=bands,
            has_structure=projection.structure is not None,
        )
        if data is None or projection.mask is None:
            for key in ("raw", "log", "qz4", "residual", "sld"):
                view = views[key]
                title = next(title for name, title, _description in VIEW_SPECS if name == key)
                if isinstance(view, LiveReflectivityPlot):
                    view.show_placeholder("暂无可用数据")
                    view.clear_fit_range()
                else:
                    draw_empty(view, title)
        else:
            self._draw_reflectivity_panes(views, data, projection.mask, candidate)
            shown = visible_bands(
                checked=self.sld_bands_toggle.isChecked(),
                cache=self._sld_band_cache,
                persisted=projection_bands(projection.result),
                dataset_id=projection.dataset_id,
                report=projection_mcmc(projection.result),
                alignment=ALIGN_KEYS[self.sld_align_selector.currentIndex()],
                surface_label=self.sld_align_selector.itemText(1),
            )
            draw_sld(
                views["sld"],
                candidate,
                comparison_candidates(projection.result, projection.candidate_id),
                shown,
                structure=projection.structure,
                wavelength_a=data.beam.effective_wavelength_a,
            )
            self._draw_range(views, projection.visible_range)
        draw_candidate_comparison(views["candidates"], projection.result, projection.candidate_id)
        draw_residual_heatmap(views["residual_map"], projection.result, projection.candidate_id)
        draw_parameter_heatmap(views["parameter_map"], projection.result, projection.candidate_id)
        draw_uncertainty(views["uncertainty"], projection.result, projection.candidate_id)
        draw_batch_trends(views["trend"], projection.trends)

    def _draw_reflectivity_panes(
        self,
        views: dict[str, DiagnosticView | LiveReflectivityPlot],
        data: api.PreparedData,
        mask: np.ndarray,
        candidate: object | None,
    ) -> None:
        """Draw the four reflectivity panes through whichever backend holds them.

        The scratch preflight dict is all-matplotlib, while the live dict holds
        pyqtgraph widgets for these four keys, so one projection reaches two
        rendering paths. The four panes always share a backend within a dict, so
        the log pane's type decides for all of them. The matplotlib path keeps
        the byte-identical draw_* functions; the live path computes the pane
        arrays once and pushes them into the pyqtgraph curves.
        """
        log = views["log"]
        if not isinstance(log, LiveReflectivityPlot):
            draw_raw(views["raw"], data, mask, candidate)
            draw_log(views["log"], data, candidate)
            draw_qz4(views["qz4"], data, candidate)
            draw_residual(views["residual"], candidate)
            return
        arrays = reflectivity_pane_arrays(data, mask, candidate)
        log.show_log_reflectivity(
            arrays.log_angles,
            arrays.log_observed,
            arrays.log_model,
            r_floor=arrays.r_floor,
        )
        log.set_quality_caption(arrays.quality_caption)
        raw = views["raw"]
        raw.show_raw_reflectivity(arrays.raw_angles, arrays.raw_intensity, arrays.raw_mask, arrays.raw_model)
        raw.set_quality_caption(arrays.quality_caption)
        qz4 = views["qz4"]
        if arrays.qz4 is None:
            qz4.show_placeholder("暂无当前候选")
        else:
            qz4.show_qz4(*arrays.qz4, ylabel=arrays.qz4_ylabel or "qz⁴R")
        residual = views["residual"]
        if arrays.residual is None:
            residual.show_placeholder("暂无当前候选")
        else:
            residual.show_residual(*arrays.residual)

    def _draw_range(
        self,
        views: dict[str, DiagnosticView | LiveReflectivityPlot],
        visible_range: tuple[float, float] | None,
    ) -> None:
        for key in ("raw", "log"):
            view = views[key]
            if isinstance(view, LiveReflectivityPlot):
                # pyqtgraph range items persist across show_*, so an absent
                # range must be cleared explicitly rather than relying on a
                # cleared axes.
                if visible_range is None:
                    view.clear_fit_range()
                else:
                    view.show_fit_range(*visible_range)
            elif visible_range is not None:
                view.axes.axvspan(*visible_range, color=theme.DATA_RANGE, alpha=0.16, label="拟合范围")
                view.canvas.draw_idle()

    def release_resources(self) -> None:
        if self._released:
            return
        self._released = True
        # The floating bar's keepers outlive the layout, so they have to be undone
        # by hand: a filter left on the tab stack would still be called while the
        # panel is being torn down.
        self.tabs.removeEventFilter(self)
        self.tabs.currentChanged.disconnect(self._raise_toolbar_overlay)
        self._interactions.release()
        for view in self._views.values():
            if isinstance(view, LiveReflectivityPlot):
                view.release()
            else:
                view.canvas.release()
                view.figure.clear()
                view.figure.set_canvas(None)
        self._datasets.clear()
        self._masks.clear()
        self._result = None
        self._candidate_id = None
        self._structure = None
        self._sld_band_cache = None

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
