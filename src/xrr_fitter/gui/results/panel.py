"""Active-dataset result, candidate, and asynchronous MCMC coordinator."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
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
from xrr_fitter.gui.parameters.grouping import bulk_densities, group_captions
from xrr_fitter.gui.results.automatic import (
    AutomaticPointLayerTable,
    AutomaticUniformityTable,
)
from xrr_fitter.gui.results.candidate_row import candidate_is_archived, candidate_is_selectable
from xrr_fitter.gui.results.candidates import (
    CandidateList,
    active_dataset,
    candidate_is_mcmc_ready,
    persisted_candidate_id,
)
from xrr_fitter.gui.results.inspector import (
    BootstrapPanel,
    ConvergencePanel,
    QuantilePanel,
    walkers_rule_summary,
)
from xrr_fitter.gui.results.uncertainty import (
    McmcControls,
    UncertaintyView,
    classification_summary,
)
from xrr_fitter.gui.results.values import ResultValueTable
from xrr_fitter.gui.results.verdict import VerdictEvidence, free_parameter_count, reduced_chi_squared

# Both halves of the double-encoding now come from theme: the shape from
# CONFIDENCE_GLYPHS, the colour from CONFIDENCE_COMPARISON_KINDS.  The badge is one
# of two surfaces that show several verdicts at once (the dataset rail is the
# other), and both need the same four-distinct-colours rule — 可用但相关 stays
# "info" instead of folding onto 多解's "warn", the way the window chrome folds it
# when showing one verdict alone.  Sharing the map means that rule is stated once;
# the local name stays because it reads as this surface's contract.  The colour is
# named, not spelled: the theme resolves each kind against the active appearance,
# which is what the four hardcoded light-theme hex values here used to get wrong on
# a dark desktop.
CONFIDENCE_STATUS_KINDS = theme.CONFIDENCE_COMPARISON_KINDS

# Stands in for the candidate count before a fit has run.  "0 个" would read as a
# search that came back empty, which is a different report from one not yet made.
CANDIDATES_EMPTY_SUBTITLE = "尚未拟合"
# 结果读数那一段的空态。与候选解同一种写法：没有结果时说没有，而不是报「0 自由」——
# 零个自由参数读起来像一次全锁死的求解，那是另一回事。
VALUES_EMPTY_SUBTITLE = "尚未拟合"

# Stands in for the dataset name when nothing is selected.  Left empty the caption
# would keep a blank line where a curve name belongs.
VERDICT_EMPTY_SUBTITLE = "未选择数据集"


class ResultsPanel(QWidget):
    """Project immutable result state through explicit public API mutations."""

    candidate_selected = Signal(str)
    candidate_inspected = Signal(str)
    # 这一节的抬头是外层卡片，而卡片比面板后构造，所以计数只能广播出去让抬头跟上。
    candidate_summary_changed = Signal(str)
    results_cleared = Signal(object)
    mcmc_completed = Signal(object)
    operation_failed = Signal(object)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("resultsPanel")
        self.controller = FitController(self)
        self._mcmc_source_project: api.XrrProject | None = None
        # 起始态就是空态那句，所以第一次刷新不会白广播一次相同的措辞。
        self._candidate_summary = CANDIDATES_EMPTY_SUBTITLE
        self._build_widgets()
        self._connect_events()
        self._refresh()

    def _build_widgets(self) -> None:
        # 设计稿 ``.confbadge``：判定是一只淡染的圆角框，字形、判定、理由三者同在框内。三者
        # 平铺的代价不在好看——理由和它下面那排判读徽章、再下面那三行键值指标一样都只是卡里
        # 的一行，读者没有线索知道「参数唯一，置信区间收敛」是在解释上面那两个字，而不是又一
        # 条并列的证据。框把这层归属画了出来。
        self.confidence_box = QFrame()
        self.confidence_box.setObjectName("confidenceVerdictBox")
        self.confidence_box.setProperty("verdictBox", True)
        verdict_layout = QHBoxLayout(self.confidence_box)
        # 内距走布局而不是 QSS：这只框自己带布局，两处都写会叠加成设计稿的两倍。
        verdict_layout.setContentsMargins(
            theme.VERDICT_PAD_H_PX,
            theme.VERDICT_PAD_V_PX,
            theme.VERDICT_PAD_H_PX,
            theme.VERDICT_PAD_V_PX,
        )
        verdict_layout.setSpacing(theme.SPACE_SM)
        self.confidence_marker = QLabel(theme.CONFIDENCE_FALLBACK_GLYPH)
        self.confidence_marker.setObjectName("confidenceMarker")
        self.confidence_marker.setAccessibleName("可信度状态标记")
        self.confidence_marker.setProperty("mono", True)
        glyph_font = self.confidence_marker.font()
        glyph_font.setPointSizeF(theme.VERDICT_GLYPH_PT)
        self.confidence_marker.setFont(glyph_font)
        self.confidence_label = QLabel("不可用")
        self.confidence_label.setObjectName("confidenceBadge")
        self.confidence_label.setAccessibleName("拟合可信度")
        # A printed screenshot and a touch pad have no hover, which is where a
        # downgraded verdict most needs to say why it was downgraded, so the
        # reasons take a line of their own instead of living under the pointer.
        self.confidence_reason = QLabel()
        self.confidence_reason.setObjectName("confidenceReason")
        self.confidence_reason.setProperty("mutedText", True)
        self.confidence_reason.setWordWrap(True)
        verdict_text = QVBoxLayout()
        verdict_text.setContentsMargins(0, 0, 0, 0)
        verdict_text.setSpacing(0)
        verdict_text.addWidget(self.confidence_label)
        verdict_text.addWidget(self.confidence_reason)
        verdict_layout.addWidget(self.confidence_marker, 0, Qt.AlignmentFlag.AlignVCenter)
        verdict_layout.addLayout(verdict_text, 1)
        # 副标题是这份判定属于哪条曲线，跟着当前数据集走：设计稿写的是「拟合判定
        # aSi_ML_25C」。多数据集的项目里这一栏只投影当前那条，不写出来的话「可信」读
        # 不出是对谁可信。这张卡自己就在面板里，所以副标题直接握在手上刷新。
        confidence_card, confidence_layout = theme.titled_card(
            self, "resultConfidenceCard", "拟合判定", VERDICT_EMPTY_SUBTITLE, flat=True
        )
        self.verdict_caption = confidence_card.findChild(QLabel, "resultConfidenceCardSubtitle")
        confidence_layout.addWidget(self.confidence_box)
        # 判读徽章与指标行属于判定，所以住在判定卡里而不是自成一节：设计稿的
        # ``.insp-sec`` 把大徽标、徽章、kv 三层收在同一个标题下面。
        self.verdict_evidence = VerdictEvidence(confidence_card)
        confidence_layout.addWidget(self.verdict_evidence)
        # 设计稿帧① 右栏是三段，每段一句抬头：拟合判定 / 参数 · 结果值〈12 自由〉/ 候选解
        # 〈3 个〉。判定那句本来就在，另两句此前没有落处——读数表裸挂在面板里，而「参数 ·
        # 结果值」挂在了设置区那张边界表上，那张表一个结果值都不写。所以这两段也各自成卡，
        # 抬头右侧带这一段自己的计数。
        self.result_values = ResultValueTable()
        values_card, values_layout = theme.titled_card(
            self, "resultValuesCard", "参数 · 结果值", VALUES_EMPTY_SUBTITLE, flat=True
        )
        self.values_caption = values_card.findChild(QLabel, "resultValuesCardSubtitle")
        values_layout.addWidget(self.result_values)
        self.candidates = CandidateList()
        candidates_card, candidates_layout = theme.titled_card(
            self, "resultCandidatesCard", "候选解", CANDIDATES_EMPTY_SUBTITLE, flat=True
        )
        self.candidates_caption = candidates_card.findChild(QLabel, "resultCandidatesCardSubtitle")
        candidates_layout.addWidget(self.candidates)
        self.automatic_points = AutomaticPointLayerTable()
        self.automatic_uniformity = AutomaticUniformityTable()
        self.clear_button = QPushButton("清除结果")
        self.clear_button.setObjectName("clearResultsButton")
        self.uncertainty = UncertaintyView()
        # 帧⑤ 右栏那三段。读的是同一份 ``UncertaintyReport``，换了一套读法：``self.uncertainty``
        # 把它念成散文（专家对话框里那一版），这三段把它摆成读数。归属留在这里，因为投影这件事
        # 已经在这里做了；摆在哪一栏是版式，归 ``window_layout``——所以不进本面板的布局，也不
        # 给父，且构造路径上一次都不 ``hide()``（显式隐藏过的控件被加进布局后不会随窗口现身）。
        self.convergence_panel = ConvergencePanel()
        self.quantile_panel = QuantilePanel()
        self.bootstrap_panel = BootstrapPanel()
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
        # 设计稿帧① 的右栏到候选解为止：自动模式那两张表、清除按钮、证据散文、打开分析的
        # 按钮、状态行都不在那一栏里画。它们仍然要存在——面板单独立起来时（不确定度对话框
        # 与无障碍走查走的就是这条路）这些是唯一入口——所以收进一个可以整体关掉的容器，由
        # 检视器那一栏在装配时关一次，而不是把六个 ``hide()`` 散在另一个模块里。
        #
        # 顺带把三张设计稿画着的卡排成判定 → 结果值 → 候选解：自动模式那两张表此前插在
        # 结果值和候选解之间，空态下高度为 0 所以顺序看着是对的，一有内容就把候选解顶下去。
        self.secondary = QWidget(self)
        self.secondary.setObjectName("resultSecondary")
        secondary_layout = QVBoxLayout(self.secondary)
        secondary_layout.setContentsMargins(0, 0, 0, 0)
        secondary_layout.setSpacing(theme.SPACE_SM)
        secondary_layout.addWidget(self.automatic_points)
        secondary_layout.addWidget(self.automatic_uniformity)
        secondary_layout.addWidget(self.clear_button)
        secondary_layout.addWidget(self.uncertainty)
        secondary_layout.addWidget(self.uncertainty_button)
        secondary_layout.addWidget(self.status_label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # 段间距归零：这三段各带一道 ``.insp-sec`` 的下边线，段与段之间留空隙时那道线离两边
        # 都远，读起来是装饰性的分隔线；紧贴着上下两段时它才是「这一节到此为止」。
        layout.setSpacing(0)
        layout.addWidget(confidence_card)
        layout.addWidget(values_card)
        layout.addWidget(candidates_card)
        layout.addWidget(self.secondary)
        self._sections = (confidence_card, values_card, candidates_card)
        self._mark_sections()

    def _mark_sections(self) -> None:
        """Re-decide which of the three sections is the column's last one.

        次要内容露着的时候候选解不是末段，收起来之后才是——所以这件事读的是 ``secondary``
        此刻的可见性，而不是构造期的排布。次要区自己不是 ``.insp-sec``（里面是表和按钮，
        不带抬头），进这串只为把「它还在下面」这件事说给末段判定听。
        """
        sections = (*self._sections, self.secondary)
        theme.mark_last_section(sections)

    def set_secondary_visible(self, visible: bool) -> None:
        """Show or hide everything frame ①'s inspector does not draw.

        默认是露着的：面板单独用起来时这几件就是清除结果、读证据、打开 MCMC 的唯一入口。
        关掉它的是检视器那一栏——那一栏照设计稿只画三段，而这些命令改从拟合菜单进。
        """
        self.secondary.setVisible(visible)
        # 候选解那一段的横线跟着这件事变：下面还有东西时它是节界，没有了就成了整栏的封边。
        self._mark_sections()

    def open_uncertainty_dialog(self) -> QDialog:
        """Reparent the owned MCMC controls and evidence into a reusable dialog.

        The dialog is built once and reused so the controls keep their identity
        across openings; a user's hand-edited sampling config therefore survives
        closing and reopening the window.

        The evidence view comes along because the inspector column does not carry
        it: ``window_layout`` draws the three designed result sections and leaves
        the panel's ``secondary`` container out, so a dialog holding only the
        sampling inputs would offer a place to start a chain and nowhere to read
        what it said.  Evidence takes the stretch so a taller dialog grows the
        figures rather than the seven spin boxes.
        """
        dialog = self._uncertainty_dialog
        if dialog is None:
            dialog = QDialog(self)
            dialog.setObjectName("uncertaintyDialog")
            dialog.setWindowTitle("不确定度分析")
            dialog.setAccessibleName("不确定度分析")
            layout = QVBoxLayout(dialog)
            layout.addWidget(self.uncertainty, 1)
            layout.addWidget(self.mcmc_group)
            self._uncertainty_dialog = dialog
        self.uncertainty.show()
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

    def verdict_subtitle(self) -> str:
        """Which curve the verdict on screen belongs to."""
        dataset = active_dataset(self.document.project)
        return VERDICT_EMPTY_SUBTITLE if dataset is None else str(dataset.dataset_id)

    def candidate_summary(self) -> str:
        """How many candidates this search returned, for the section's caption.

        The mockup writes the count faintly beside the 候选解 heading, so the
        reader knows how many solutions there are without counting rows.  An
        empty list says so in words rather than reporting "0 个": a count of zero
        reads as a search that came back empty, which is a different report from
        one that has not been made.
        """
        count = self.candidates.candidate_count()
        return f"{count} 个" if count else CANDIDATES_EMPTY_SUBTITLE

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

    def sampling_rule_summary(self) -> str | None:
        """帧⑤ 左栏页脚那句 walkers 下界，没有候选解可数时是 ``None``。

        两个数一个来自当前候选解（自由参数个数），一个来自 spin box 此刻的值——与
        ``start_mcmc`` 递给 ``validated_config`` 的那两个是同一对，所以这一句预告的正是「按下
        ``▶ 运行 MCMC`` 会不会被拦」。读报告里那条跑完的链的话，「已满足」就成了恒真的话：报告
        存在本身就说明它当初过了校验。
        """
        candidate = self._selected_candidate()
        if candidate is None:
            return None
        return walkers_rule_summary(len(candidate.unit_vector), self.walkers.value())

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
        self.result_values.project_result(
            result,
            candidate_id,
            self._group_captions(),
            self._bulk_densities(),
        )
        self._project_verdict(result, candidate_id)
        self.uncertainty.set_result(result, candidate_id)
        self._project_sampling_evidence(result, candidate_id)
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
        self.result_values.project_result(
            result,
            visible_id,
            group_captions(dataset.structure),
            bulk_densities(dataset.structure),
        )
        self._project_verdict(result, visible_id)
        self.uncertainty.set_result(result, visible_id)
        self._project_sampling_evidence(result, visible_id)
        self.uncertainty_button.setVisible(self.document.project.ui_state.expert_mode)
        self._configure_mcmc(self._candidate(visible_id))
        self._refresh_captions()
        # The candidate list directly above already shows how many there are, so
        # this line stays free for MCMC outcomes and failures.
        self._show_status("", kind="")

    def _group_captions(self) -> dict[str, str]:
        dataset = active_dataset(self.document.project)
        return group_captions(None if dataset is None else dataset.structure)

    def _bulk_densities(self) -> dict[str, float]:
        dataset = active_dataset(self.document.project)
        return bulk_densities(None if dataset is None else dataset.structure)

    def _refresh_captions(self) -> None:
        """Re-read all three section subtitles; the count now rides its own card."""
        caption = self.verdict_subtitle()
        self.verdict_caption.setText(caption)
        self.verdict_caption.setToolTip(caption)
        self._write_caption(self.values_caption, self.free_parameter_summary())
        summary = self.candidate_summary()
        self._write_caption(self.candidates_caption, summary)
        # 计数已经写在自己那张卡上，这个信号留着是因为它是面板对外的播报口，外面还有别的
        # 接收者可以跟着它刷新，去掉等于让「候选数变了」这件事只在面板内部可见。
        if summary != self._candidate_summary:
            self._candidate_summary = summary
            self.candidate_summary_changed.emit(summary)

    @staticmethod
    def _write_caption(label: QLabel | None, text: str) -> None:
        if label is None:  # pragma: no cover - titled_card always names its subtitle
            return
        label.setText(text)
        label.setToolTip(text)

    def free_parameter_summary(self) -> str:
        """How many parameters this search actually solved for, for the caption.

        设计稿在「参数 · 结果值」右侧写「12 自由」。数的是自由参数而不是表里的行数：锁定
        与被约束的参数在表里各占一行，但它们没有参与求解，数进去会把「求解了几个」报成
        「表里有几行」。声明随结果一起存档，所以这个数不必回去读源文件。
        """
        dataset = active_dataset(self.document.project)
        result = None if dataset is None else dataset.last_valid_result
        if result is None:
            return VALUES_EMPTY_SUBTITLE
        return f"{free_parameter_count(result)} 自由"

    def _project_verdict(self, result: object | None, candidate_id: str | None) -> None:
        """Read the badges off the report that names this candidate.

        A report belongs to the candidate it was computed for, so a report whose
        ``candidate_id`` names another candidate is withheld rather than shown
        against the one on screen -- the same rule the evidence panel applies.
        """
        report = None if result is None else result.uncertainty
        if report is not None and report.candidate_id not in (None, candidate_id):
            report = None
        self.verdict_evidence.set_report(report)
        candidate = self._candidate(candidate_id)
        self.verdict_evidence.set_objective(None if candidate is None else candidate.objective)
        # χ²ᵥ 与 J 同源：同一个候选解的残差，除以同一份声明数出来的自由度。
        self.verdict_evidence.set_reduced_chi_squared(
            None
            if candidate is None or result is None
            else reduced_chi_squared(candidate.weighted_residuals, free_parameter_count(result))
        )

    def _project_sampling_evidence(self, result: object | None, candidate_id: str | None) -> None:
        """帧⑤ 右栏那三段一起投影。

        三段读的是同一条链，分头刷新会让它们各停在一拍上——收敛诊断报着这个候选解，分位表还
        画着上一个的后验。归属由各段自己判（``mcmc_evidence`` / ``_owned_report``），所以这里
        把结果原样递下去，不在这一层先筛一遍。
        """
        for panel in (self.convergence_panel, self.quantile_panel, self.bootstrap_panel):
            panel.set_result(result, candidate_id)

    def _clear_sampling_evidence(self, message: str) -> None:
        """三段一起退回空态；那句话由各段自行决定念不念（只有分位段的图会念）。"""
        for panel in (self.convergence_panel, self.quantile_panel, self.bootstrap_panel):
            panel.clear_result(message)

    def _clear_projection(self, message: str) -> None:
        self._set_confidence("不可用")
        self.verdict_evidence.clear_report()
        self.result_values.clear_projection()
        self.candidates.clear_projection()
        self._refresh_captions()
        self.uncertainty.clear_result(message)
        self._clear_sampling_evidence(message)
        self.uncertainty_button.setVisible(self.document.project.ui_state.expert_mode)
        self._configure_mcmc(None)
        self._show_status("", kind="")

    def _show_status(self, text: str, *, kind: str) -> None:
        self.status_label.setText(text)
        theme.set_status_kind(self.status_label, kind)

    def _set_confidence(self, text: str, detail: str = "") -> None:
        # An unclassified badge falls back to the inherited text colour rather
        # than a grey of its own, so "no result yet" cannot be mistaken for a
        # fourth state.
        marker = theme.CONFIDENCE_GLYPHS.get(text, theme.CONFIDENCE_FALLBACK_GLYPH)
        kind = CONFIDENCE_STATUS_KINDS.get(text, "")
        self.confidence_label.setText(text)
        theme.set_status_kind(self.confidence_label, kind)
        self.confidence_marker.setText(marker)
        # The reasons read on screen; the row collapses when a verdict carries
        # none rather than leaving a blank line under it.
        self.confidence_reason.setText(detail)
        self.confidence_reason.setVisible(bool(detail))
        # Hovering either half of the badge answers "why" as well, and the
        # accessible description carries the same text for screen readers.
        described = f"{text}：{detail}" if detail else text
        self.confidence_marker.setAccessibleDescription(described)
        theme.set_status_kind(self.confidence_marker, kind)
        # 框自己也上这把键：设计稿把整块底色按判定分四档，只染字形和判定那两个控件的话，
        # 框会永远是那条中性边框——「可信」和「不可信」在屏幕上长得一样。
        theme.set_status_kind(self.confidence_box, kind)
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
