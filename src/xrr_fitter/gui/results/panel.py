"""Active-dataset result, candidate, and asynchronous MCMC coordinator."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import messages, theme
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.fitting.controller import FitController
from xrr_fitter.gui.results.automatic import (
    AutomaticPointLayerTable,
    AutomaticUniformityTable,
)
from xrr_fitter.gui.results.candidates import (
    CandidateList,
    active_dataset,
    candidate_is_archived,
    candidate_is_mcmc_ready,
    candidate_is_selectable,
    persisted_candidate_id,
)
from xrr_fitter.gui.results.uncertainty import (
    McmcControls,
    UncertaintyView,
    classification_summary,
)

CONFIDENCE_VISUALS = {
    "可信": ("●", "#2E7D32"),
    "可用但相关": ("◆", "#1565C0"),
    "多解": ("▲", "#B36B00"),
    "不可信": ("■", "#B3261E"),
}


class ResultsPanel(QWidget):
    """Project immutable result state through explicit public API mutations."""

    candidate_selected = Signal(str)
    candidate_inspected = Signal(str)
    results_cleared = Signal(object)
    mcmc_completed = Signal(object)
    operation_failed = Signal(object)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("resultsPanel")
        self.controller = FitController(self)
        self._mcmc_source_project: api.XrrProject | None = None
        self._build_widgets()
        self._connect_events()
        self._refresh()

    def _build_widgets(self) -> None:
        confidence_row = QWidget()
        confidence_layout = QHBoxLayout(confidence_row)
        confidence_layout.setContentsMargins(0, 0, 0, 0)
        confidence_layout.setSpacing(theme.SPACE_SM)
        self.confidence_marker = QLabel("○")
        self.confidence_marker.setObjectName("confidenceMarker")
        self.confidence_marker.setAccessibleName("可信度状态标记")
        self.confidence_label = QLabel("不可用")
        self.confidence_label.setObjectName("confidenceBadge")
        self.confidence_label.setAccessibleName("拟合可信度")
        confidence_layout.addWidget(self.confidence_marker)
        confidence_layout.addWidget(self.confidence_label, 1)
        self.candidates = CandidateList()
        self.automatic_points = AutomaticPointLayerTable()
        self.automatic_uniformity = AutomaticUniformityTable()
        self.clear_button = QPushButton("清除结果")
        self.clear_button.setObjectName("clearResultsButton")
        self.uncertainty = UncertaintyView()
        # MCMC is an opt-in deep dive whose seven inputs would otherwise sit on
        # screen for every project. The panel keeps ownership so candidate
        # configuration and operation state still track the selection, but the
        # widgets live in a dialog that is only built when the user asks.
        # Parented to the panel but deliberately left out of its layout, so
        # accessibility configuration and lookups still reach the controls while
        # they occupy no space until the dialog adopts them.
        self.mcmc_group = McmcControls(self)
        self.mcmc_group.hide()
        self.uncertainty_button = QPushButton("不确定度分析…")
        self.uncertainty_button.setObjectName("openUncertaintyDialogButton")
        self.uncertainty_button.setAccessibleName("打开不确定度分析")
        self.uncertainty_button.setToolTip("对当前候选解运行专家 MCMC 采样")
        self.uncertainty_button.clicked.connect(self._show_uncertainty_dialog)
        self._uncertainty_dialog: QDialog | None = None
        self.walkers = self.mcmc_group.walkers
        self.burn_in = self.mcmc_group.burn_in
        self.production = self.mcmc_group.production
        self.thin = self.mcmc_group.thin
        self.mcmc_button = self.mcmc_group.run_button
        self.cancel_button = self.mcmc_group.cancel_button
        self.force_button = self.mcmc_group.force_button
        self.status_label = QLabel()
        self.status_label.setObjectName("resultStatus")
        self.status_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(confidence_row)
        layout.addWidget(self.automatic_points)
        layout.addWidget(self.automatic_uniformity)
        layout.addWidget(self.candidates)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.uncertainty)
        layout.addWidget(self.uncertainty_button)
        layout.addWidget(self.status_label)

    def open_uncertainty_dialog(self) -> QDialog:
        """Reparent the owned MCMC controls into a reusable modeless dialog.

        The dialog is built once and reused so the controls keep their identity
        across openings; a user's hand-edited sampling config therefore survives
        closing and reopening the window.
        """
        dialog = self._uncertainty_dialog
        if dialog is None:
            dialog = QDialog(self)
            dialog.setObjectName("uncertaintyDialog")
            dialog.setWindowTitle("不确定度分析")
            dialog.setAccessibleName("不确定度分析")
            layout = QVBoxLayout(dialog)
            layout.addWidget(self.mcmc_group)
            self._uncertainty_dialog = dialog
        self.mcmc_group.show()
        return dialog

    def _show_uncertainty_dialog(self) -> None:
        self.open_uncertainty_dialog().show()

    def _connect_events(self) -> None:
        self.document.project_changed.connect(self._refresh)
        self.candidates.candidate_requested.connect(self._candidate_requested)
        self.clear_button.clicked.connect(lambda: self.clear_results())
        self.mcmc_button.clicked.connect(self._start_mcmc_clicked)
        self.cancel_button.clicked.connect(self.controller.cancel)
        self.force_button.clicked.connect(self.controller.force_stop)
        self.controller.running_changed.connect(self._running_changed)
        self.controller.mcmc_finished.connect(self._publish_mcmc)
        self.controller.cancelled.connect(self._show_cancelled)
        self.controller.failed.connect(self._show_failure)
        self.controller.stopped.connect(self._operation_stopped)

    def candidate_count(self) -> int:
        return self.candidates.candidate_count()

    def candidate_text(self, row: int) -> str:
        return self.candidates.candidate_text(row)

    def selected_candidate_id(self) -> str | None:
        return self.candidates.selected_candidate_id()

    def recommended_candidate_id(self) -> str | None:
        return self.candidates.recommended_id

    def confidence_text(self) -> str:
        return self.confidence_label.text()

    def uncertainty_text(self) -> str:
        return self.uncertainty.text()

    def warning_texts(self) -> tuple[str, ...]:
        return self.uncertainty.warning_texts()

    def status_text(self) -> str:
        return self.status_label.text()

    def select_candidate(self, candidate_id: str) -> bool:
        self._require_idle("select a candidate")
        candidate = self._candidate(candidate_id)
        if not candidate_is_selectable(candidate):
            raise ValueError("candidate is inspection-only")
        return self._persist_candidate(candidate_id)

    def _persist_candidate(self, candidate_id: str) -> bool:
        dataset_id = self._require_active_dataset_id()
        current = self.document.project
        updated = api.select_candidate(current, dataset_id, candidate_id)
        if updated is current:
            return False
        self.candidates.clear_inspection()
        self.document.replace_project(updated)
        self.candidate_selected.emit(candidate_id)
        return True

    def clear_results(self, dataset_ids=None) -> bool:
        self._require_idle("clear results")
        requested = (self._require_active_dataset_id(),) if dataset_ids is None else tuple(dataset_ids)
        current = self.document.project
        updated = api.clear_fit_results(current, requested)
        if updated is current:
            return False
        self.document.replace_project(updated)
        self.results_cleared.emit(requested)
        return True

    def mcmc_config(self) -> api.McmcConfig:
        return self.mcmc_group.config()

    def start_mcmc(self) -> bool:
        self._require_idle("start MCMC")
        dataset_id = self._require_active_dataset_id()
        candidate = self._selected_candidate()
        if not self._mcmc_ready(candidate):
            raise ValueError("MCMC requires current candidate-owned uncertainty evidence")
        config = self.mcmc_group.validated_config(len(candidate.unit_vector))
        source = self.document.project
        self._mcmc_source_project = source
        started = self.controller.start_mcmc(
            source,
            dataset_id,
            candidate.candidate_id,
            config,
        )
        if started:
            self._show_status("MCMC 已启动", kind="ok")
        else:
            self._mcmc_source_project = None
        return started

    def _start_mcmc_clicked(self) -> None:
        try:
            self.start_mcmc()
        except (RuntimeError, ValueError) as error:
            self._show_status(str(error), kind="error")

    def _candidate_requested(self, candidate_id: str) -> None:
        candidate = self._candidate(candidate_id)
        if candidate is None:
            self._refresh()
            self._show_status("拟合结果已过期，候选列表已刷新", kind="warn")
            return
        try:
            if candidate_is_selectable(candidate):
                self.select_candidate(candidate_id)
                return
            self._inspect_candidate(candidate_id)
            if candidate_is_archived(candidate) and self._confirm_archived_candidate(candidate):
                self._persist_candidate(candidate_id)
        except (KeyError, ValueError) as error:
            self._refresh()
            self._show_status(str(error), kind="error")

    def _inspect_candidate(self, candidate_id: str) -> None:
        self._require_active_dataset_id()
        candidate = self.candidates.inspect(candidate_id)
        result = self.candidates.result
        self.uncertainty.set_result(result, candidate_id)
        self._configure_mcmc(candidate)
        self.candidate_inspected.emit(candidate_id)

    def _confirm_archived_candidate(self, candidate: object) -> bool:
        response = QMessageBox.question(
            self,
            "选择已归档候选",
            f"候选 {candidate.candidate_id} 已在早期淘汰。仍将其设为持久候选吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return response == QMessageBox.StandardButton.Yes

    def _refresh(self, *_args) -> None:
        summary = api.summarize_automatic_results(self.document.project)
        self.automatic_points.project_summary(summary)
        self.automatic_uniformity.project_summary(summary)
        dataset = active_dataset(self.document.project)
        if dataset is None or dataset.last_valid_result is None:
            self._clear_projection("当前数据集尚无拟合结果")
            return
        result = dataset.last_valid_result
        selected_id = persisted_candidate_id(self.document.project, dataset.dataset_id)
        visible_id = self.candidates.project_result(
            dataset.dataset_id,
            result,
            selected_id,
        )
        self._set_confidence(result.confidence.value, classification_summary(result))
        self.uncertainty.set_result(result, visible_id)
        self.uncertainty_button.setVisible(self.document.project.ui_state.expert_mode)
        self._configure_mcmc(self._candidate(visible_id))
        # The candidate list directly above already shows how many there are, so
        # this line stays free for MCMC outcomes and failures.
        self._show_status("", kind="")

    def _clear_projection(self, message: str) -> None:
        self._set_confidence("不可用")
        self.candidates.clear_projection()
        self.uncertainty.clear_result(message)
        self.uncertainty_button.setVisible(self.document.project.ui_state.expert_mode)
        self._configure_mcmc(None)
        self._show_status("", kind="")

    def _show_status(self, text: str, *, kind: str) -> None:
        self.status_label.setText(text)
        theme.set_status_kind(self.status_label, kind)

    def _set_confidence(self, text: str, detail: str = "") -> None:
        marker, color = CONFIDENCE_VISUALS.get(text, ("○", "#666666"))
        self.confidence_label.setText(text)
        self.confidence_label.setProperty("semanticColor", color)
        self.confidence_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.confidence_marker.setText(marker)
        # Bind the reasons to the badge itself so hovering it answers "why";
        # the accessible description carries the same text for screen readers.
        described = f"{text}：{detail}" if detail else text
        self.confidence_marker.setAccessibleDescription(described)
        self.confidence_marker.setStyleSheet(f"color: {color};")
        tooltip = described if detail else ""
        self.confidence_label.setToolTip(tooltip)
        self.confidence_marker.setToolTip(tooltip)

    def _configure_mcmc(self, candidate: object | None) -> None:
        candidate_id = None if candidate is None else candidate.candidate_id
        dimension = 0 if candidate is None else len(candidate.unit_vector)
        self.mcmc_group.configure(candidate_id, dimension)
        self._refresh_mcmc_buttons()

    def _refresh_mcmc_buttons(self, running: bool | None = None) -> None:
        active = self.controller.is_running if running is None else running
        self.candidates.setEnabled(not active)
        self.clear_button.setEnabled(not active and self.candidates.result is not None)
        self.mcmc_group.set_operation_state(
            running=active,
            ready=self._mcmc_ready(self._selected_candidate()),
        )

    def _mcmc_ready(self, candidate: object | None) -> bool:
        return candidate_is_mcmc_ready(candidate, self.candidates.result)

    def _running_changed(self, running: bool) -> None:
        self._refresh_mcmc_buttons(running)

    def _publish_mcmc(self, project: api.XrrProject) -> None:
        source = self._mcmc_source_project
        self._mcmc_source_project = None
        if source is None or self.document.project is not source:
            self._show_failure(
                api.OperationError(
                    "RuntimeError",
                    "stale MCMC result rejected because the project changed",
                    "MCMC source project identity changed before completion",
                )
            )
            return
        self.document.replace_project(project)
        self._show_status("MCMC 完成", kind="ok")
        self.mcmc_completed.emit(project)

    def _show_cancelled(self, reason: str) -> None:
        self._mcmc_source_project = None
        self._show_status(f"MCMC 已取消：{reason}", kind="warn")

    def _show_failure(self, error: api.OperationError) -> None:
        self._mcmc_source_project = None
        self._show_status(messages.operation_error_text(error), kind="error")
        self.operation_failed.emit(error)

    def _operation_stopped(self) -> None:
        self._mcmc_source_project = None

    def _require_active_dataset_id(self) -> str:
        dataset_id = self.document.active_dataset_id
        if dataset_id is None:
            raise ValueError("an active dataset is required")
        return dataset_id

    def _require_idle(self, operation: str) -> None:
        if self.controller.is_running:
            raise RuntimeError(f"cannot {operation} while MCMC is running")

    def _candidate(self, candidate_id: str | None) -> object | None:
        return self.candidates.candidate(candidate_id)

    def _selected_candidate(self) -> object | None:
        return self._candidate(self.selected_candidate_id())
