"""帧① 判定卡下的判读徽章与指标行。

证据面板已经把报告里的每一项都写成散文，但一句「Bootstrap 失败率：0.02」要读者
自己判断 0.02 是好还是坏。这里做的是判读：把同一份数据折成三句结论，各自带上
状态色，让「这次拟合能不能用」在一眼之内读完；细节仍留在下面的证据面板里。

指标行三项照设计稿：目标值 J、约化 χ²ᵥ、自助失败率。χ²ᵥ 没有存档字段，但它不是
需要新数据的量——候选解自带 ``weighted_residuals``（残差面板画的就是它），除以自由度
就是 χ²ᵥ。J 带正则项和标度自由度，只在这套代码内部可比；χ²ᵥ 是能跟文献里「拟合好不好」
对话的那个数，所以它紧跟 J 而不是排在最后。
"""

from __future__ import annotations

from math import fsum, isfinite

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from xrr_fitter.gui import theme
from xrr_fitter.gui.results.inference_text import correlation_unavailable_reason
from xrr_fitter.gui.wrapping import wrapping_row

OBJECTIVE_LABEL = "目标值 J"
REDUCED_CHI_SQUARED_LABEL = "约化 χ²ᵥ"
FAILURE_RATE_LABEL = "自助失败率"
UNAVAILABLE_TEXT = "不可用"
# 失败率超过这条线就不再打勾。与分类器的阈值同源：分类已经因为它把结论降级，
# 徽章跟着降级，两处才不会一个说「收敛」一个说「不可信」。
FAILURE_RATE_WARN = 0.1


def reduced_chi_squared(weighted_residuals: object | None, free_parameters: int) -> float | None:
    """χ²ᵥ = Σr²/ν，只数真的参与了拟合的那些点。

    候选解的 ``weighted_residuals`` 是整条曲线那么长的数组，拟合窗口外的位置留着
    ``nan``（``fit/candidates.py`` 先 ``np.full(..., np.nan)`` 再只往 ``fit_mask`` 里写）。
    所以求和前要先滤掉非有限值，否则一次正常的拟合也会算出 ``nan``。

    ν = N − p。除以 N 会让自由参数越多、χ²ᵥ 越好看，正好把过拟合报成拟合得更好；
    ν ≤ 0 时这个量没有定义，返回 ``None`` 让调用方写「不可用」。

    求和不走 numpy：架构契约只把 numpy 开给 ``gui.plots`` 与导入对话框，判定卡不在其中。
    几百到几千个点、一次标签刷新，纯 Python 的代价看不见，而 ``fsum`` 的舍入比逐项累加更准。
    """
    if weighted_residuals is None:
        return None
    finite = [value for value in (float(entry) for entry in weighted_residuals) if isfinite(value)]
    degrees_of_freedom = len(finite) - int(free_parameters)
    if not finite or degrees_of_freedom <= 0:
        return None
    return fsum(value * value for value in finite) / degrees_of_freedom


def free_parameter_count(result: object) -> int:
    """求解时真正自由的参数个数，也就是 χ²ᵥ 的 ν = N − p 里的 p。

    锁定与被约束的参数在参数表里各占一行，但它们没有参与求解，数进去会把「求解了几个」
    报成「表里有几行」，进而把 ν 报大、χ²ᵥ 报小。声明随结果一起存档，所以不必回读源文件。
    """
    return sum(1 for definition in result.parameter_definitions if not definition.locked and not definition.constrained)


def _percent(value: float) -> str:
    return f"{round(value * 100)}%"


def _convergence(report: object) -> tuple[str, str]:
    """自助收敛率与它的状态。

    报告存的是失败率，设计稿写的是收敛率，所以这里取补。没跑自助时失败率同样是
    0，直接取补会打出「收敛 100%」——一次没做的检查会读成一次通过的检查，所以
    未运行是独立的第三种说法，并且不带状态色——"没做"不是四种状态里的任何一种。
    """
    if not report.bootstrap_performed:
        return "— 未运行自助", ""
    rate = float(report.bootstrap_failure_rate)
    converged = _percent(1.0 - rate)
    if rate > FAILURE_RATE_WARN:
        return f"⚠ 自助收敛 {converged}", "warn"
    return f"✓ 自助收敛 {converged}", "ok"


def _boundary(report: object) -> tuple[str, str]:
    count = len(tuple(report.boundary_hits))
    if count == 0:
        return "✓ 无边界命中", "ok"
    return f"⚠ {count} 处边界命中", "warn"


def _correlation(report: object) -> tuple[str, str]:
    """强相关的对数，加上其中最强的系数。

    按绝对值取最强：负相关一样是纠缠，按带符号取最大会把 -0.93 让给 0.71，报出
    一个偏乐观的数。
    """
    if correlation_unavailable_reason(report) is not None:
        return "— 相关不可用", ""
    pairs = tuple(report.strong_correlations)
    if not pairs:
        return "✓ 无强相关", "ok"
    strongest = max(abs(float(value)) for _left, _right, value in pairs)
    return f"ℹ {len(pairs)} 处相关 ρ={strongest:.2f}", "info"


class VerdictEvidence(QWidget):
    """Read the uncertainty report as badges and metrics, not as prose."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("verdictEvidence")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        # 徽章换行，而不是把三枚之和当成地板。文案随判定变长（「⚠ 4 处边界命中」比
        # 「✓ 无边界命中」宽一截），求和的地板等于让判定内容决定右栏要多宽——检视器
        # 的水平滚动条是关掉的，超出的部分不是滚动而是无声裁掉。
        self._badge_host, self._badge_row = wrapping_row(self, spacing=theme.SPACE_XS + 2)
        self._badges = tuple(self._build_badge() for _ in range(3))
        for badge in self._badges:
            self._badge_row.addWidget(badge)
        layout.addWidget(self._badge_host)
        self._metric_grid = QGridLayout()
        self._metric_grid.setContentsMargins(0, 0, 0, 0)
        self._metric_grid.setHorizontalSpacing(theme.SPACE_SM)
        self._metric_grid.setVerticalSpacing(1)
        # 设计稿 ``.kv`` 是 space-between：键靠左，值靠右。让值列吃掉多余宽度，
        # 两行的数值才在同一条右边界上对齐。
        self._metric_grid.setColumnStretch(1, 1)
        self._metric_rows = (
            self._build_metric(0, OBJECTIVE_LABEL),
            self._build_metric(1, REDUCED_CHI_SQUARED_LABEL),
            self._build_metric(2, FAILURE_RATE_LABEL),
        )
        layout.addLayout(self._metric_grid)
        self._objective: float | None = None
        self._reduced_chi_squared: float | None = None
        self._report: object | None = None
        self._project()

    def _build_badge(self) -> QLabel:
        badge = QLabel(self)
        badge.setProperty("badge", True)
        return badge

    def _build_metric(self, row: int, caption: str) -> tuple[QLabel, QLabel]:
        key = QLabel(caption, self)
        key.setProperty("mutedText", True)
        value = QLabel(self)
        # 设计稿 ``.kv .v`` 用 tabular-nums。比例数字里 1 比 8 窄，上下两行的
        # 小数点会错开,读者得逐字对齐才知道哪个数更大。
        value.setProperty("mono", True)
        value.setProperty("tabularValue", True)
        self._metric_grid.addWidget(key, row, 0)
        self._metric_grid.addWidget(value, row, 1, Qt.AlignmentFlag.AlignRight)
        return key, value

    def badges(self) -> tuple[QLabel, ...]:
        return self._badges

    def metric_labels(self) -> tuple[QLabel, ...]:
        return tuple(value for _key, value in self._metric_rows)

    def metrics(self) -> tuple[tuple[str, str], ...]:
        """The rendered key/value pairs, or nothing while cleared."""
        if self._report is None:
            return ()
        return tuple((key.text(), value.text()) for key, value in self._metric_rows)

    def set_report(self, report: object | None) -> None:
        self._report = report
        self._project()

    def set_objective(self, objective: float | None) -> None:
        self._objective = objective
        self._project()

    def set_reduced_chi_squared(self, value: float | None) -> None:
        self._reduced_chi_squared = value
        self._project()

    def clear_report(self) -> None:
        self._report = None
        self._objective = None
        self._reduced_chi_squared = None
        self._project()

    def _project(self) -> None:
        report = self._report
        if report is None:
            self._hide_all()
            return
        readings = (_convergence(report), _boundary(report), _correlation(report))
        for badge, (text, kind) in zip(self._badges, readings, strict=True):
            badge.setText(text)
            theme.set_status_kind(badge, kind)
            badge.setVisible(True)
        correlation_reason = correlation_unavailable_reason(report) or ""
        self._badges[2].setToolTip(correlation_reason)
        self._badges[2].setAccessibleDescription(correlation_reason)
        objective = self._objective
        self._metric_rows[0][1].setText(UNAVAILABLE_TEXT if objective is None else f"{objective:.6g}")
        chi_squared = self._reduced_chi_squared
        # χ²ᵥ 的判读区间在 1 附近，三位有效数字就够；再多几位是把数值噪声当成信息。
        self._metric_rows[1][1].setText(UNAVAILABLE_TEXT if chi_squared is None else f"{chi_squared:.3g}")
        bootstrap = report.bootstrap_evidence
        self._metric_rows[2][1].setText(
            UNAVAILABLE_TEXT if bootstrap is None else _percent(float(bootstrap.failure_rate))
        )
        self._set_metrics_visible(True)
        self.setVisible(True)

    def _hide_all(self) -> None:
        for badge in self._badges:
            badge.setText("")
            badge.setToolTip("")
            badge.setAccessibleDescription("")
            badge.setVisible(False)
        self._set_metrics_visible(False)
        self.setVisible(False)

    def _set_metrics_visible(self, visible: bool) -> None:
        for key, value in self._metric_rows:
            key.setVisible(visible)
            value.setVisible(visible)
