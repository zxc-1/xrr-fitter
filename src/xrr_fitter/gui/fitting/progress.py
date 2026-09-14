"""Compact visible projection of immutable fit-progress values."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.canvas_top import canvas_top
from xrr_fitter.gui.plots.live import LiveReflectivityPlot, ObjectiveTracePlot

PROGRESS_RESOLUTION = 1000

# 设计稿帧④ 画布顶栏的三段：进度 / 实时反射率 / 目标值轨迹。运行中这一列有三样可看，
# 而屏幕一次只放得下一样；没有这一行，读者看不见往下还有另外两段。
CANVAS_TAB_TITLES: tuple[str, ...] = ("进度", "实时反射率", "目标值轨迹")

# ``FitController.poll_interval_ms`` 的活跃取值，与设计稿帧④ 的徽标同一个数。徽标默认
# 报它，退到空闲间隔时由控制器改写——所以这个数是一处默认，不是一句写死的承诺。
DEFAULT_POLL_INTERVAL_MS = 250

# 目标值轨迹两轴的名字。横轴用完成比例而不是帧序号：帧的疏密取决于轮询间隔。
# 部件本身住在 ``gui.plots.live``（门禁只允许绘图模块直接依赖 pyqtgraph），这里只是
# 把两个标签名转出来，读者与既有测试不必换地方找。
OBJECTIVE_TRACE_Y_LABEL = ObjectiveTracePlot.Y_LABEL
OBJECTIVE_TRACE_X_LABEL = ObjectiveTracePlot.X_LABEL

# Below this completed fraction the elapsed time is too short a lever to
# extrapolate an honest remaining estimate, so the view shows a placeholder
# instead of a wild guess during the fast opening stage.
ETA_MIN_FRACTION = 0.02

# Fractional share of one search that each stage occupies, measured from real
# single-dataset runs. The bar advances monotonically across stage changes
# because every stage maps into its own slice of one fixed global range.
STAGE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("A", 0.06),
    ("B", 0.34),
    ("C", 0.16),
    ("D", 0.16),
    ("E", 0.16),
    ("basin-recovery", 0.02),
    ("bootstrap", 0.05),
    ("profile", 0.04),
    ("finalizing", 0.01),
)

# Each name carries the key the fit itself reports, so a row can be matched
# against a progress event, a resume checkpoint or a worker log line. Only A-E
# need the suffix: the last four keys already read as words, and repeating them
# would put English into the row for nothing. Two stages also name their method,
# which is what explains why B alone takes a third of the run.
STAGE_LABELS = {
    "A": "初筛候选 A",
    "B": "全局搜索 B（差分进化）",
    "C": "密度精修 C",
    "D": "粗糙度与仪器精修 D",
    "E": "最终种子搜索 E",
    "basin-recovery": "基态复核",
    "bootstrap": "自助抽样（Bootstrap）",
    "profile": "置信区间",
    # This stage classifies the result as well as assembling it, so naming only
    # the assembly would hide where the confidence verdict is decided.
    "finalizing": "汇总与判定",
}


def _stage_offsets() -> dict[str, tuple[float, float]]:
    offsets: dict[str, tuple[float, float]] = {}
    cursor = 0.0
    for name, weight in STAGE_WEIGHTS:
        offsets[name] = (cursor, weight)
        cursor += weight
    return offsets


STAGE_OFFSETS = _stage_offsets()


def stage_position(stage: str, completed: int, total: int) -> int | None:
    """Map one stage-local count into the fixed global progress range."""
    placement = STAGE_OFFSETS.get(stage)
    if placement is None:
        return None
    start, weight = placement
    fraction = 0.0 if total <= 0 else min(1.0, max(0.0, completed / total))
    return round((start + weight * fraction) * PROGRESS_RESOLUTION)


def stage_text(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage)


def joint_layout_text(layout: object) -> str:
    """Name the coupled datasets, the shared parameters, and what stays separate.

    A joint fit optimises shared parameters across several datasets at once, so
    its progress frames carry no single owning dataset. Stating the participants
    and the shared-parameter groups up front turns an anonymous "联合拟合" into a
    run the user can actually attribute.

    Both sides of the split get said, because a shared parameter shows one value
    that is a compromise across the members while an unshared one keeps a value
    per dataset, where a difference between members is real. Naming only what is
    coupled leaves the reader unable to tell which kind of number they are
    looking at. The separate side is stated as the complement rather than by
    listing scale and background: those are only the usual leftovers, and a
    project that shares them too would make a fixed sentence false.
    """
    datasets = "、".join(layout.dataset_ids)
    if layout.shared_parameters:
        keys = "、".join(rule.sharing_key for rule in layout.shared_parameters)
        shared = f"共享参数：{keys}；其余参数各数据集各自独立"
    else:
        shared = "无显式共享参数，各数据集参数彼此独立"
    return f"🔗 <b>联合拟合。</b>数据集：{datasets} · {shared}"


def _stage_percent_text(completed: int, total: int) -> str:
    """Render the stage-local count with its own percentage for a live feel.

    The global bar deliberately cannot rewind across stages, so a slow stage
    with many seeds can look stalled even while it advances. Showing the
    stage-local percent next to the raw count gives that motion a visible home.
    """
    if total <= 0:
        return f"{completed}/{total}"
    percent = min(100, max(0, round(100 * completed / total)))
    return f"{completed}/{total} ({percent}%)"


def _format_duration(seconds: float) -> str:
    """Render a non-negative duration as MM:SS, or H:MM:SS past an hour."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# Stage costs span orders of magnitude -- the opening screen can finish inside a
# second while the global search runs for tens of them -- so a stage measure picks
# its precision from the size of the number. The MM:SS the elapsed row uses would
# render a 0.8s stage as 00:00, which says "took no time" about a stage that took
# most of a second; past a minute MM:SS is the form the reader already knows.
#
# The tenth is only kept below a second, where it still buys resolution: it is an
# eighth of a 0.8s reading, but a hundredth of a 9s one, and nobody re-budgets a
# run on 9.0 against 9.1.
STAGE_SECONDS_DECIMAL_BELOW = 1.0


def _format_stage_duration(seconds: float) -> str:
    """Render one stage's cost at a precision its own magnitude deserves."""
    if seconds < STAGE_SECONDS_DECIMAL_BELOW:
        return f"{seconds:.1f}s"
    if seconds < 60.0:
        return f"{round(seconds)}s"
    return _format_duration(seconds)


# Refresh cadence of the elapsed/remaining clock while the worker is silent. A
# long fit spends most stages emitting no progress frames, so an idle screen
# reads as frozen; ticking the clock once a second proves the run is alive.
HEARTBEAT_INTERVAL_MS = 1000

# Shown in place of the position while the worker winds the current seed down.
# The card's meta row and the status bar both say it, so it lives here rather
# than being typed twice and drifting.
CANCELLING_TEXT = "正在取消，等待当前种子结束…"

# Stands in for the remaining time before the run has produced enough progress to
# extrapolate one honestly.
ETA_UNKNOWN_TEXT = "--:--"


@dataclass(frozen=True, slots=True)
class RunSummary:
    """状态栏那三段的取值：阶段、进度、预计剩余。

    设计稿帧④ 把一次运行拆成三段读——``阶段 4 / 9``、``进度 620 / 1000``、
    ``预计剩余 00:29``——段间只靠间距分开，两段在弹簧左边一段在右边。所以这里报三个
    取值而不是一句拼好的话：拼好的话没法分段加粗，也没法让「预计剩余」独自落到右边去。

    空串表示这一段无话可报，由状态栏据此把它藏起来。
    """

    stage: str = ""
    position: str = ""
    remaining: str = ""


# The card's caption. Before a run there is no stage to name, so it can only state
# the scale the bar moves on; once a stage is live the caption names that stage,
# because the header is the one line that stays in place when the nine-row ladder
# below it is scrolled out of view.
PROGRESS_SCALE_TEXT = "单调递增 0–1000"
PROGRESS_SUBTITLE_IDLE = f"共九阶段 · {PROGRESS_SCALE_TEXT}"


def progress_subtitle(ordinal: int, count: int) -> str:
    return f"阶段 {ordinal} / {count} · {PROGRESS_SCALE_TEXT}"


class _RemainingEstimator:
    """Project the remaining time from the recent completion rate.

    The cumulative average rate lags badly when stages carry uneven real cost:
    the fast opening stage makes a slow later stage look nearly done, so a naive
    ``elapsed * (1 - frac) / frac`` estimate lurches at every stage boundary.
    Smoothing the instantaneous rate with an exponential moving average tracks
    the stage actually running, and freezing the projected finish in elapsed
    terms lets the estimate count down honestly during the silent gaps between
    frames instead of standing still.
    """

    def __init__(self, *, smoothing: float = 0.35) -> None:
        self._smoothing = smoothing
        self._rate: float | None = None
        self._previous: tuple[float, float] | None = None
        self._finish_elapsed: float | None = None

    def reset(self) -> None:
        self._rate = None
        self._previous = None
        self._finish_elapsed = None

    def update(self, elapsed: float, fraction: float) -> None:
        previous = self._previous
        self._previous = (elapsed, fraction)
        if previous is None:
            return
        delta_elapsed = elapsed - previous[0]
        delta_fraction = fraction - previous[1]
        if delta_elapsed <= 0.0 or delta_fraction <= 0.0:
            return
        rate = delta_fraction / delta_elapsed
        if self._rate is None:
            self._rate = rate
        else:
            self._rate += self._smoothing * (rate - self._rate)
        self._finish_elapsed = elapsed + (1.0 - fraction) / self._rate

    def remaining(self, elapsed: float) -> float | None:
        if self._finish_elapsed is None:
            return None
        return max(0.0, self._finish_elapsed - elapsed)


# A marker cell wide enough for a two-digit ordinal, matching the pipeline
# navigator's cell so the two ladders a user reads on one screen share a left edge.
STAGE_MARKER_PX = 18
# Only the running stage has a live count; the others just say where they stand.
STAGE_METRIC_DONE = "已完成"
STAGE_METRIC_PENDING = "待运行"


@dataclass(frozen=True, slots=True)
class _StageRow:
    """The three labels one ladder row is made of, handed back for wiring.

    A row is a marker, a name and a metric, and every progress frame repaints all
    three, so the builder returns them together instead of leaving the caller to
    re-find them by object name on each frame.
    """

    marker: QLabel
    name: QLabel
    metric: QLabel


def _paint_row(row: _StageRow, state: str, glyph: str, metric: str) -> None:
    """Repaint one row's marker glyph, step state and metric in one place.

    Marker and name carry the same state so shape and weight ride on top of colour,
    the same double encoding the pipeline navigator uses.  Both also carry
    ``ladderStep``, which is what entitles this map -- and only this map -- to the
    done and pending shades.
    """
    row.marker.setText(glyph)
    row.metric.setText(metric)
    theme.set_step_state(row.marker, state)
    theme.set_step_state(row.name, state)


def _stage_row(parent: QWidget, key: str, ordinal: int, label: str) -> tuple[QWidget, _StageRow]:
    """Build one ladder row: ordinal marker, stage name, right-aligned metric.

    The metric sits right-aligned in its own cell so the nine measures line up as
    a column: this build of Qt exposes no tabular-figure font setting, so column
    alignment is what keeps the counts readable as they change width.
    """
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_SM)
    marker = QLabel("", row)
    marker.setObjectName(f"fitStageMarker_{key}")
    marker.setFixedWidth(STAGE_MARKER_PX)
    marker.setFixedHeight(STAGE_MARKER_PX)
    marker.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
    marker.setProperty("ladderStep", True)
    marker.setProperty("stageMarker", True)
    name = QLabel(label, row)
    name.setObjectName(f"fitStageName_{key}")
    # 这张梯子是设计稿三张步骤地图里唯一给完成项和未开始项也上色的（``.stg.done .ic`` 变绿、
    # ``.stg.pending`` 整行 ``opacity:.5``），所以那两档颜色由它自己声明；左栏的六步管线和
    # 向导抬头只给当前那一步上色，共用规则里不能替它们把三档都涂了。
    name.setProperty("ladderStep", True)
    metric = QLabel("", row)
    metric.setObjectName(f"fitStageMetric_{key}")
    metric.setProperty("mutedText", True)
    metric.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(marker)
    layout.addWidget(name, 1)
    layout.addWidget(metric)
    parts = _StageRow(marker=marker, name=name, metric=metric)
    _paint_row(parts, "pending", str(ordinal), STAGE_METRIC_PENDING)
    return row, parts


def _build_ladder(parent: QWidget, layout: QVBoxLayout) -> dict[str, _StageRow]:
    """Stack one row per stage under ``layout``, in ``STAGE_WEIGHTS`` order.

    Showing all nine at once is the point: one stage line says where a run is but
    never how far it still has to reach.
    """
    rows: dict[str, _StageRow] = {}
    for index, (key, _weight) in enumerate(STAGE_WEIGHTS):
        row, parts = _stage_row(parent, key, index + 1, stage_text(key))
        layout.addWidget(row)
        rows[key] = parts
    return rows


def _scrolled(card: QWidget) -> QScrollArea:
    """把「总进度」卡放进一只只上下滚的容器。

    这张卡装的是联合横幅、四行文字和九行阶梯，加起来比中栏高：1400×900 下卡自己要 364px，
    中栏给 350px。没有容器接住这段亏空时，Qt 只能把卡里每一件都压到它要的高度以下——而且
    连 ``minimumSizeHint`` 都不保，联合横幅实测被压到 13px，那句「共享了什么」于是在屏上
    永远读不到，而不是"滚一下就看见"。

    横向不滚：一栏的宽度是给定的预算，卡片该在这个宽度里排版，而不是靠左右拖动去读。容器
    自己不接焦点，Tab 序仍从卡里的控件走过去。
    """
    scroll = QScrollArea()
    scroll.setObjectName("fitProgressScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(card)
    return scroll


class ProgressView(QWidget):
    """Show stage identity, completion, objective, and worker message."""

    # Republished on every frame and every heartbeat so a surface outside this
    # card -- the status bar, which is all a guided user has -- can show the same
    # position without running an estimator of its own.
    summary_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        heartbeat_interval_ms: int = HEARTBEAT_INTERVAL_MS,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("fitProgressView")
        self._clock = clock
        self._floor = 0
        self._started: float | None = None
        self._cancelling = False
        self._stage: str | None = None
        # The remaining-time text as the card last rendered it. The summary reads
        # this instead of re-deriving it, so the bar cannot show an estimate the
        # card has not shown -- and so a query never consumes a clock reading.
        self._eta_display = ETA_UNKNOWN_TEXT
        # When each stage was first seen, and how long the ones already displaced
        # ran for. A stage's end is only observable when a later stage's frame
        # arrives, so a duration is fixed at that handover and never revised.
        self._stage_started: dict[str, float] = {}
        self._stage_seconds: dict[str, float] = {}
        self._estimator = _RemainingEstimator()
        self._heartbeat = QTimer(self)
        self._heartbeat.setObjectName("fitProgressHeartbeat")
        self._heartbeat.setInterval(heartbeat_interval_ms)
        self._heartbeat.timeout.connect(self._tick)
        self.joint_label = QLabel("")
        self.joint_label.setObjectName("fitProgressJointLayout")
        theme.set_status_kind(self.joint_label, "warn")
        theme.mark_banner(self.joint_label)
        self.joint_label.setWordWrap(True)
        self.joint_label.hide()
        self.stage_label = QLabel("等待开始")
        self.stage_label.setObjectName("fitProgressStage")
        self.bar = QProgressBar()
        self.bar.setObjectName("fitProgressBar")
        self.bar.setRange(0, PROGRESS_RESOLUTION)
        self.bar.setValue(0)
        # The meta row states the position in the 0-1000 scale the card's subtitle
        # promises; the bar's own centred percentage would state the same position
        # in a second unit, so the trough stays wordless as it is in the design.
        self.bar.setTextVisible(False)
        self._meta_row = QHBoxLayout()
        self._meta_row.setContentsMargins(0, 0, 0, 0)
        self._meta_row.setSpacing(0)
        self.meta_elapsed = QLabel("")
        self.meta_elapsed.setObjectName("fitProgressMetaElapsed")
        self.meta_elapsed.setProperty("mono", True)
        self.meta_position = QLabel("")
        self.meta_position.setObjectName("fitProgressMetaPosition")
        self.meta_position.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.meta_position.setProperty("mono", True)
        self.meta_remaining = QLabel("")
        self.meta_remaining.setObjectName("fitProgressMetaRemaining")
        self.meta_remaining.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.meta_remaining.setProperty("mono", True)
        self._meta_row.addWidget(self.meta_elapsed)
        self._meta_row.addStretch(1)
        self._meta_row.addWidget(self.meta_position)
        self._meta_row.addStretch(1)
        self._meta_row.addWidget(self.meta_remaining)
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("fitProgressDetail")
        self.detail_label.setWordWrap(True)
        self.canvas_row, self.canvas_tabs, _actions = canvas_top(
            self,
            name="fitCanvasTop",
            tabs_name="fitCanvasTabs",
            titles=CANVAS_TAB_TITLES,
        )
        self.refresh_badge = QLabel("")
        self.refresh_badge.setObjectName("fitRefreshBadge")
        theme.set_status_kind(self.refresh_badge, "info")
        self.canvas_row.layout().addWidget(self.refresh_badge)
        self.set_refresh_interval_ms(DEFAULT_POLL_INTERVAL_MS)
        card, card_layout = theme.titled_card(self, "fitProgressCard", "总进度", PROGRESS_SUBTITLE_IDLE)
        self._card_subtitle: QLabel | None = card.findChild(QLabel, "fitProgressCardSubtitle")
        card_layout.addWidget(self.joint_label)
        card_layout.addWidget(self.stage_label)
        card_layout.addWidget(self.bar)
        card_layout.addLayout(self._meta_row)
        card_layout.addWidget(self.detail_label)
        self._stage_rows = _build_ladder(card, card_layout)
        self.card_scroll = _scrolled(card)
        # 三段是三页：卡片、这一帧的预览曲线、逐帧攒出来的 J 轨迹。三样都从同一个
        # ``FitProgress`` 里来，所以换页换的是同一次运行的三种读法，不是三份数据。
        self.live_plot = LiveReflectivityPlot(self)
        self.trace_plot = ObjectiveTracePlot(self)
        self.trace_item = self.trace_plot.trace_item
        self.pages = QStackedWidget(self)
        self.pages.setObjectName("fitCanvasPages")
        self.pages.addWidget(self.card_scroll)
        self.pages.addWidget(self.live_plot)
        self.pages.addWidget(self.trace_plot)
        self.canvas_tabs.currentChanged.connect(self.pages.setCurrentIndex)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas_row)
        layout.addWidget(self.pages)

    def _record_frame(self, progress: api.FitProgress) -> None:
        """把这一帧摊到两张图上：预览曲线换一条，J 往轨迹后面接一个点。

        轨迹的横轴用完成比例而不是帧序号：帧的疏密取决于轮询间隔，而完成比例是这次
        运行自己的进度，两次运行摆在一起也能对读。
        """
        qz = progress.preview_qz_a_inv
        model = progress.preview_model_normalized
        if qz is not None and model is not None:
            self.live_plot.set_preview(qz, model)
        total = progress.total or 1
        self.trace_plot.append(float(progress.completed) / float(total), float(progress.best_objective))

    def set_refresh_interval_ms(self, interval_ms: int) -> None:
        """徽标改写成轮询器此刻真正用的那个间隔。

        轮询活跃时用设计稿写的 250 ms，长时间无事件后退到 1000 ms；徽标报的一律是此刻
        真正在用的那个数。写死一个好看的数就成了一句与画面对不上的保证：一条不动的曲线，
        究竟是收敛了还是界面卡住了，读者是拿这个周期去分的。
        """
        self.refresh_badge.setText(f"实时刷新 · 每 {int(interval_ms)} ms")

    def _set_card_subtitle(self, text: str) -> None:
        """Repaint the card caption and the tooltip that carries it when elided."""
        if self._card_subtitle is None:
            return
        self._card_subtitle.setText(text)
        self._card_subtitle.setToolTip(text)

    def set_joint_layout(self, layout: object | None) -> None:
        """Show a joint run's datasets and shared parameters, or hide the banner.

        Passing ``None`` (an independent fit) clears the banner so stale member
        names never linger from a previous joint run.
        """
        if layout is None:
            self.joint_label.clear()
            self.joint_label.hide()
            return
        self.joint_label.setText(joint_layout_text(layout))
        self.joint_label.show()

    def set_progress(self, progress: api.FitProgress) -> None:
        now = self._clock()
        if self._started is None:
            self._started = now
        elapsed = now - self._started
        owner = progress.dataset_id or "联合拟合"
        stage_index, stage_count = self._stage_ordinal(progress.stage)
        self.stage_label.setText(f"{owner} · 阶段 {stage_index}/{stage_count} · {stage_text(progress.stage)}")
        self._set_card_subtitle(progress_subtitle(stage_index, stage_count))
        self._mark_stage_clock(progress.stage, now)
        self._stage = progress.stage
        self._advance(progress)
        self._record_frame(progress)
        self._paint_ladder(progress)
        self._estimator.update(elapsed, self._floor / PROGRESS_RESOLUTION)
        self._render_meta(elapsed)
        self.detail_label.setText(
            f"{progress.stage} {_stage_percent_text(progress.completed, progress.total)}"
            f" · {progress.message} · best={progress.best_objective:.4g}"
        )
        if not self._heartbeat.isActive():
            self._heartbeat.start()
        self._publish_summary()

    def overall_percent(self) -> int:
        """总进度的百分数，读的是进度条自己的值。

        另一处要显示同一个位置的是左栏「拟合」那一行（设计稿帧④「进行中 62%」）。它从这里取数
        而不是自己按 ``STAGE_WEIGHTS`` 再算一遍，这样两处不可能报出不同的进度。
        """
        return round(100 * self.bar.value() / PROGRESS_RESOLUTION)

    def _set_meta_text(self, text: str) -> None:
        """Set all three meta labels to the same text (for cancelling state)."""
        self.meta_elapsed.setText(text)
        self.meta_position.clear()
        self.meta_remaining.clear()

    def _clear_meta(self) -> None:
        self.meta_elapsed.clear()
        self.meta_position.clear()
        self.meta_remaining.clear()

    def mark_cancelling(self) -> None:
        """Acknowledge a cancel request while the worker winds a seed down."""
        self._cancelling = True
        self._set_meta_text(CANCELLING_TEXT)
        self._publish_summary()

    def freeze(self) -> None:
        """Stop the live clock without clearing the last rendered values."""
        self._heartbeat.stop()

    def _tick(self) -> None:
        """Advance the elapsed/remaining clock between silent progress frames."""
        if self._started is None:
            return
        self._render_meta(self._clock() - self._started)
        self._publish_summary()

    def _publish_summary(self) -> None:
        self.summary_changed.emit(self.status_summary())

    def status_summary(self) -> RunSummary:
        """State where the run stands in this card's own numbers,段与段分开报。

        Guided mode hides the inspector, and this card with it, so the status bar
        becomes the only surface a run is visible on -- which is what the design's
        frame ④ bar shows with 阶段 4 / 9 and 进度 620 / 1000. The values are taken
        from what the card last rendered rather than recomputed, because a second
        remaining-time estimator would count down towards a different finish than
        the one on screen.

        Empty before the first frame: there is no position to report yet, and a
        placeholder ladder would claim a run that has not started.

        取消在飞行中时只有一件事可报，它落在进度那一段——那一段本来就是整句白文，
        而阶段与剩余时间在等最后一颗种子收尾时都已不再是真话。
        """
        if self._cancelling:
            return RunSummary(position=CANCELLING_TEXT)
        if self._stage is None:
            return RunSummary()
        ordinal, count = self._stage_ordinal(self._stage)
        return RunSummary(
            f"{ordinal} / {count}",
            f"进度 {self._floor} / {PROGRESS_RESOLUTION}",
            self._eta_display,
        )

    def _eta_text(self, elapsed: float) -> str:
        """Say how much longer, or admit the evidence is too thin to say."""
        remaining = self._estimator.remaining(elapsed)
        if self._floor / PROGRESS_RESOLUTION >= ETA_MIN_FRACTION and remaining is not None:
            return _format_duration(remaining)
        return ETA_UNKNOWN_TEXT

    def _render_meta(self, elapsed: float) -> None:
        if self._cancelling:
            self._set_meta_text(CANCELLING_TEXT)
            return
        self._eta_display = self._eta_text(elapsed)
        self.meta_elapsed.setText(f"已用 {_format_duration(elapsed)}")
        self.meta_position.setText(f"进度 {self._floor} / {PROGRESS_RESOLUTION}")
        self.meta_remaining.setText(f"预计剩余 {self._eta_display}")

    def _mark_stage_clock(self, stage: str, now: float) -> None:
        """Open this stage's clock, and close the one it just displaced.

        A worker reports no stage-finished event, so the only observable end of a
        stage is the arrival of a different stage's frame. Measuring across that
        handover keeps the nine measures adding up to the elapsed time on screen.
        """
        previous = self._stage
        if previous is not None and previous != stage:
            started = self._stage_started.get(previous)
            if started is not None:
                self._stage_seconds[previous] = max(0.0, now - started)
        self._stage_started.setdefault(stage, now)

    def _stage_metric_done(self, stage: str) -> str:
        """Say how long a finished stage ran, or only that it is finished.

        A stage that never emitted a frame has no opening reading to measure from.
        Printing 0.0s for it would report "too fast to measure" about a stage that
        was never measured at all.
        """
        seconds = self._stage_seconds.get(stage)
        if seconds is None:
            return STAGE_METRIC_DONE
        return _format_stage_duration(seconds)

    def _stage_ordinal(self, stage: str) -> tuple[int, int]:
        names = tuple(name for name, _weight in STAGE_WEIGHTS)
        count = len(names)
        if stage not in names:
            return count, count
        return names.index(stage) + 1, count

    def _advance(self, progress: api.FitProgress) -> None:
        position = stage_position(progress.stage, progress.completed, progress.total)
        if position is None:
            return
        # An unknown or out-of-order stage must never move the bar backwards.
        self._floor = max(self._floor, position)
        self.bar.setValue(self._floor)

    def _paint_ladder(self, progress: api.FitProgress) -> None:
        """Mark the stages before the live one done, it current, and the rest pending.

        An unrecognised stage leaves the ladder as it stands instead of declaring every
        row done, mirroring how ``_advance`` refuses to move the bar for one.
        """
        names = tuple(name for name, _weight in STAGE_WEIGHTS)
        if progress.stage not in names:
            return
        live = names.index(progress.stage)
        for index, key in enumerate(names):
            if index < live:
                state, glyph, metric = "done", "✓", self._stage_metric_done(key)
            elif index == live:
                metric = _stage_percent_text(progress.completed, progress.total)
                state, glyph = "current", str(index + 1)
            else:
                state, glyph, metric = "pending", str(index + 1), STAGE_METRIC_PENDING
            _paint_row(self._stage_rows[key], state, glyph, metric)

    def _reset_ladder(self) -> None:
        """Return every row to its pending presentation for the next run."""
        for index, (key, _weight) in enumerate(STAGE_WEIGHTS):
            _paint_row(self._stage_rows[key], "pending", str(index + 1), STAGE_METRIC_PENDING)

    def reset(self) -> None:
        self._heartbeat.stop()
        self._estimator.reset()
        self._floor = 0
        self._started = None
        self._cancelling = False
        self._stage = None
        self._eta_display = ETA_UNKNOWN_TEXT
        self._stage_started.clear()
        self._stage_seconds.clear()
        self.trace_plot.reset()
        self.stage_label.setText("等待开始")
        self._set_card_subtitle(PROGRESS_SUBTITLE_IDLE)
        self.bar.setRange(0, PROGRESS_RESOLUTION)
        self.bar.setValue(0)
        self.detail_label.clear()
        self._clear_meta()
        self._reset_ladder()
        self._publish_summary()
