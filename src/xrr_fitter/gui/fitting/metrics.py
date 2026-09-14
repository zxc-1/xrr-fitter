"""运行中右栏的两段读数：实时指标，以及各数据集目标值。

设计稿帧④ 的右栏是三段。第三段「控制」由 ``FitPanel`` 自己画，前两段在这里：一组
键值行报此刻跑到哪儿、收敛到多少，一张表报每个数据集各自的目标值。

数字全部取自 ``FitProgress``，没有别的来源。遥测那几行（迭代次数、函数评估、接受率、
步长）在事件里各自可以是 ``None``，因为没有哪个阶段同时数得出全部：一次性池扫描没有
代数，``least_squares`` 没有接受率，MCMC 没有独立的函数评估计数。缺的那行显示破折号，
不显示 0——把缺量画成 0 就是一个看着像读数的编造值。

趋势是两次读数之间比出来的，不是求解器给的：目标函数单调下降是拟合正常的标志，
停住则是收敛或卡住的信号，而单看一个数看不出是哪一种。第一次读数没有可比对象，
所以它的趋势是破折号而不是箭头。

联合批量下 ``dataset_id`` 是 ``None``，全局 J 只有一个，但每条曲线各自的残差是求解器
算过的，随进度事件带在 ``dataset_objectives`` 里。表因此优先报各成员自己的 J：让三行
显示同一个数会把"哪条曲线拖后腿"藏起来。分解缺席时才退回那个共享的数。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.fitting.progress import stage_text

OBJECTIVE_HEADERS = ("数据集", "J", "趋势")

# 目标值的显示位数。四位小数够看清一次迭代带来的改善，而进度事件每秒来若干次，
# 位数再多末位就只是在跳。
OBJECTIVE_DECIMALS = 4

FALLING = "↓ 下降"
PLATEAU = "≈ 平台"
RISING = "↑ 上升"
UNKNOWN = "—"

# 目标值算「没动」的相对阈值。求解器在平台期仍会在末位上抖，逐位比较会把平台读成
# 一串上下交替的箭头。
PLATEAU_RELATIVE = 1e-4

METRIC_STAGE = "当前阶段"
METRIC_OBJECTIVE = "全局目标 J"
METRIC_STEP = "阶段进度"
METRIC_ITERATION = "迭代次数"
METRIC_NFEV = "函数评估"
METRIC_ACCEPTANCE = "接受率"
METRIC_STEP_SIZE = "步长"

METRIC_KEYS = (
    METRIC_STAGE,
    METRIC_OBJECTIVE,
    METRIC_STEP,
    METRIC_ITERATION,
    METRIC_NFEV,
    METRIC_ACCEPTANCE,
    METRIC_STEP_SIZE,
)


def _number(value: float) -> str:
    return f"{value:.{OBJECTIVE_DECIMALS}f}"


def _count(value: int | None) -> str:
    """计数类读数。千分位分组，因为函数评估数动辄五位。"""
    return UNKNOWN if value is None else f"{value:,}"


def _rate(value: float | None) -> str:
    return UNKNOWN if value is None else f"{value:.2f}"


def _step(value: float | None) -> str:
    """步长跨若干个数量级——收敛时缩到 1e-6 量级，所以用有效位而非定点小数。"""
    return UNKNOWN if value is None else f"{value:.6g}"


def _trend(previous: float | None, current: float) -> str:
    """两次目标值之间是降、是平、还是升。"""
    if previous is None:
        return UNKNOWN
    if abs(current - previous) <= PLATEAU_RELATIVE * max(abs(previous), 1.0):
        return PLATEAU
    return FALLING if current < previous else RISING


class LiveMetricsView(QWidget):
    """Project the live figures of a run: stage, objective, per-dataset J."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("liveMetricsView")
        self._values: dict[str, QLabel] = {}
        self._objectives: dict[str | None, float] = {}
        self._trends: dict[str | None, str] = {}
        self._order: list[str] = []
        self._members: tuple[str, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)
        for key in METRIC_KEYS:
            layout.addWidget(self._metric_row(key))
        caption = QLabel("各数据集目标值")
        caption.setObjectName("liveMetricsCaption")
        caption.setProperty("mutedText", True)
        layout.addWidget(caption)
        self.table = _ObjectiveTable()
        layout.addWidget(self.table)

    def _metric_row(self, key: str) -> QWidget:
        row = QWidget(self)
        row.setObjectName(f"liveMetric_{key}")
        cells = QHBoxLayout(row)
        cells.setContentsMargins(0, 0, 0, 0)
        cells.setSpacing(theme.SPACE_SM)
        label = QLabel(key)
        label.setObjectName(f"liveMetricKey_{key}")
        label.setProperty("mutedText", True)
        value = QLabel(UNKNOWN)
        value.setObjectName(f"liveMetricValue_{key}")
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        cells.addWidget(label)
        cells.addStretch(1)
        cells.addWidget(value)
        self._values[key] = value
        return row

    # ------------------------------------------------------------------ inputs

    def set_members(self, members: tuple[str, ...]) -> None:
        """Name the datasets a joint run shares its objective between.

        A joint progress event carries no dataset, so without the membership the
        table has nothing to list; an independent run leaves this empty and the
        table is built from the events themselves.
        """
        self._members = tuple(members)
        self._render_table()

    def reset(self) -> None:
        self._objectives.clear()
        self._trends.clear()
        self._order.clear()
        for value in self._values.values():
            value.setText(UNKNOWN)
        self._render_table()

    def set_progress(self, progress: api.FitProgress) -> None:
        trend = self._record(progress.dataset_id, progress.best_objective)
        for dataset_id, objective in progress.dataset_objectives or ():
            self._record(dataset_id, objective)
        self._show_metrics(progress, trend)
        self._render_table()

    def _record(self, key: str | None, objective: float) -> str:
        """记下一个读数，返回它相对同一个键上一次读数的趋势。"""
        value = float(objective)
        trend = _trend(self._objectives.get(key), value)
        self._objectives[key] = value
        self._trends[key] = trend
        if key is not None and key not in self._order:
            self._order.append(key)
        return trend

    def _show_metrics(self, progress: api.FitProgress, trend: str) -> None:
        arrow = "" if trend == UNKNOWN else f" {trend[0]}"
        self._values[METRIC_STAGE].setText(stage_text(progress.stage))
        self._values[METRIC_OBJECTIVE].setText(f"{_number(progress.best_objective)}{arrow}")
        self._values[METRIC_STEP].setText(f"{progress.completed} / {progress.total}")
        self._values[METRIC_ITERATION].setText(_count(progress.iteration))
        self._values[METRIC_NFEV].setText(_count(progress.nfev))
        self._values[METRIC_ACCEPTANCE].setText(_rate(progress.acceptance_rate))
        self._values[METRIC_STEP_SIZE].setText(_step(progress.step_size))

    # ----------------------------------------------------------------- reading

    def metric_text(self, key: str) -> str:
        return self._values[key].text()

    def objective_rows(self) -> tuple[tuple[str, str, str], ...]:
        return self.table.rows()

    # --------------------------------------------------------------- rendering

    def _reading(self, name: str) -> tuple[float, str] | None:
        """成员自己的 J 优先；后端没给分解时才退回那个共享的数。"""
        if name in self._objectives:
            return self._objectives[name], self._trends.get(name, UNKNOWN)
        shared = self._objectives.get(None)
        return None if shared is None else (shared, self._trends.get(None, UNKNOWN))

    def _rows(self) -> tuple[tuple[str, str, str], ...]:
        rows = []
        for name in self._members or tuple(self._order):
            reading = self._reading(name)
            if reading is not None:
                rows.append((name, _number(reading[0]), reading[1]))
        return tuple(rows)

    def _render_table(self) -> None:
        self.table.set_rows(self._rows())


class _ObjectiveTable(QTableWidget):
    """One row per dataset: its name, its objective and where it is heading."""

    def __init__(self) -> None:
        super().__init__(0, len(OBJECTIVE_HEADERS))
        self.setObjectName("liveObjectiveTable")
        self.setAccessibleName("各数据集目标值")
        self.setHorizontalHeaderLabels(OBJECTIVE_HEADERS)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.verticalHeader().hide()
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(OBJECTIVE_HEADERS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

    def set_rows(self, rows: tuple[tuple[str, str, str], ...]) -> None:
        self.setRowCount(len(rows))
        for index, cells in enumerate(rows):
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column == 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.setItem(index, column, item)
        self.setVisible(bool(rows))

    def rows(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            tuple(self.item(row, column).text() for column in range(self.columnCount()))
            for row in range(self.rowCount())
        )
