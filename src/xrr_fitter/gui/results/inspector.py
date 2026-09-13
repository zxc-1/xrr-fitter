"""帧⑤ 右栏那三段：收敛诊断 / 后验分位 / 自助抽样。

同一份 ``UncertaintyReport`` 在这个程序里有两个读者。专家对话框里的 ``UncertaintyView``
把它念成一篇散文——每一句都带上下文，读的人是打算把这次采样从头看到尾的。这里三段读的是
同一份报告，但摆成读数：切到不确定度那一页的人正对着中栏的相关矩阵，他要问的是「这条链
能不能当证据用」，而那是六个数字加两个判定，不是一段话。

判定一律镜像既有出处，不在这里新造阈值：R̂ 与 ESS 的界取 ``results.uncertainty`` 的
``SPLIT_RHAT_LIMIT`` / ``EFFECTIVE_SAMPLE_FLOOR``（那两个又镜像 ``analysis.mcmc``
的 ``problem_mcmc_warnings``），接受率与 walkers 的界镜像同一个模块的
``problem_mcmc_warnings`` 与 ``_validate_walker_geometry``，失败率的界取
``results.verdict.FAILURE_RATE_WARN``。同一件事在两处各写一个数，迟早会分岔成两种判定。

归属也镜像：链的三段读 ``mcmc.candidate_id``，自助那一段读报告自己的
``candidate_id``（``plots.sld._owned_report``）。宁可整段空着，也不借一个读数来填——借
来的六行数字看不出是别人的。

numpy 与绘图归 ``gui.plots``（依赖门禁），所以这里不碰数组：分位段那张紧凑直方图是从
``plots.posterior`` 借来的控件，采样列由 ``results.uncertainty`` 的助手转置好再送进来。
"""

from __future__ import annotations

from math import fsum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from xrr_fitter.gui import theme
from xrr_fitter.gui.plots.correlation import _ranked_pairs
from xrr_fitter.gui.plots.parameter_labels import short_labels
from xrr_fitter.gui.plots.posterior import (
    POSTERIOR_QUANTILES,
    POSTERIOR_UNAVAILABLE_TEXT,
    PosteriorQuantilePlot,
)
from xrr_fitter.gui.plots.sld import _owned_report
from xrr_fitter.gui.results.uncertainty import (
    EFFECTIVE_SAMPLE_FLOOR,
    SPLIT_RHAT_LIMIT,
    _display_unit,
    _display_value,
    _finite_values,
    _linear_quantile,
    _sample_columns,
)
from xrr_fitter.gui.results.verdict import FAILURE_RATE_WARN, UNAVAILABLE_TEXT
from xrr_fitter.gui.wrapping import wrapping_row

# 接受率的健康区间，两端都是开的——镜像 ``analysis.mcmc.problem_mcmc_warnings`` 里
# ``(acceptance <= 0.10) | (acceptance >= 0.80)`` 那一句：贴着界的那次采样在那边已经被
# 报成问题，这里不能读成「刚好合格」。
ACCEPTANCE_FLOOR = 0.10
ACCEPTANCE_CEILING = 0.80

# walkers 的下界 ``2·n_free + 2``，镜像 ``analysis.mcmc._validate_walker_geometry``。拆成
# 两个名字是因为这一行要把式子念给读者听（「16（≥ 2·6+2）」），而念出来的那两个数必须
# 就是判定用的那两个。
WALKERS_FACTOR = 2
WALKERS_MARGIN = 2

# 合格与告警各一个字形。判定同时也染色，但颜色在灰度截图、色觉障碍和深色主题下各有各的
# 失灵方式，字形不会。
OK_MARK = "✓"
WARN_MARK = "⚠"

# 三段的行名，各按设计稿的次序。写成模块级常量而不是内联字面：这几个字面同时是测试的判据
# 与读屏念出的键名，改动时该只有一处。
CONVERGENCE_CAPTIONS = ("walkers", "步数 / 链", "接受率", "split-R̂ (max)", "ESS (min)", "燃烧期 burn-in")

# 「P50」后面缀「（中位）」：三行里只有它是个位置度量，而读者最常拿它当「那个值」用。
QUANTILE_CAPTIONS = ("P16", "P50（中位）", "P84")

BOOTSTRAP_CAPTIONS = ("重采样次数", "失败率", "边界命中")

# 没跑过自助抽样与跑过但没记下次数是两回事：前者的下一步是去跑，后者的下一步是去看这份
# 工程文件是哪个版本存的（``bootstrap_sample_count`` 是后加的字段）。
BOOTSTRAP_NOT_RUN_TEXT = "未运行"
BOOTSTRAP_COUNT_UNRECORDED_TEXT = "未记录"


def mcmc_evidence(result: object | None, candidate_id: str | None) -> object | None:
    """当前候选解自己的那条链，没有就是 ``None``。

    三条拒绝与 ``results.uncertainty.sampling_readings`` 逐条相同：没有报告、没有链、链
    挂在别的候选解上。``candidate_id`` 为 ``None`` 时也拒——那时屏上没有一个候选解可以
    认领这条链，而一条无主的链读起来像是当前这一个的。
    """
    report = None if result is None else result.uncertainty
    mcmc = None if report is None else report.mcmc
    if mcmc is None or candidate_id is None or mcmc.candidate_id != candidate_id:
        return None
    return mcmc


def _judged(text: str, healthy: bool) -> tuple[str, str]:
    """一行读数连它的判定。字形缀在数字后面，颜色跟着同一个判定走。"""
    return (f"{text} {OK_MARK if healthy else WARN_MARK}", "ok" if healthy else "warn")


def _acceptance_reading(values: object) -> tuple[str, str]:
    finite = _finite_values(values)
    if finite is None:
        return (UNAVAILABLE_TEXT, "")
    # 各 walker 的接受率取平均——这一行说的是这次采样整体的步长选得怎么样，而 walker 之间
    # 的差异是另一件事（``UncertaintyView`` 那边报的是区间）。``fsum`` 而不是内建 ``sum``：
    # 十几个同样的小数逐个累加会在末位上飘，而这一行只留两位，飘出去正好改掉显示值。
    mean = fsum(finite) / len(finite)
    return _judged(f"{mean:.2f}", ACCEPTANCE_FLOOR < mean < ACCEPTANCE_CEILING)


def _rhat_reading(values: object) -> tuple[str, str]:
    finite = _finite_values(values)
    if finite is None:
        return (UNAVAILABLE_TEXT, "")
    worst = max(finite)
    return _judged(f"{worst:.3f}", worst < SPLIT_RHAT_LIMIT)


def _sample_size_reading(values: object) -> tuple[str, str]:
    finite = _finite_values(values)
    if finite is None:
        return (UNAVAILABLE_TEXT, "")
    worst = min(finite)
    return _judged(f"{round(worst):,}", worst >= EFFECTIVE_SAMPLE_FLOOR)


def _burn_in_text(burn_in: int, total: int) -> str:
    return UNAVAILABLE_TEXT if total <= 0 else f"前 {round(burn_in / total * 100)}%"


def _walkers_are_enough(free_count: int, walkers: int) -> bool:
    """这么多 walkers 够不够这么多自由参数——``McmcControls.validated_config`` 拦不拦的那一条。"""
    return walkers >= WALKERS_FACTOR * free_count + WALKERS_MARGIN


def walkers_rule_summary(free_count: int, walkers: int) -> str:
    """左栏页脚那句 walkers 下界（设计稿帧⑤ 的 ``ds-summary``）。

    与 ``convergence_readings`` 第一行同一个判定，读的却是另外两个数：那一行说的是手上这条链
    跑了多少，这一句说的是下一条会不会被拦下来。所以这一句在链跑起来之前就该说得出话，而那
    一行要等报告。

    括号里念的是「几个自由参数 → 几条链」，与判定用的两个数是同一对——把式子念给读者听而不
    只给一个「已满足」，他才知道该往哪个方向调。
    """
    met = "已满足" if _walkers_are_enough(free_count, walkers) else "未满足"
    return f"walkers ≥ {WALKERS_FACTOR}·n_free + {WALKERS_MARGIN} · {met}（{free_count}→{walkers}）"


def convergence_readings(mcmc: object) -> tuple[tuple[str, str], ...]:
    """六行读数各连一个判定，次序照 ``CONVERGENCE_CAPTIONS``。

    R̂ 取最大、ESS 取最小——照 ``sampling_readings`` 同一个口径：收敛这件事没有平均可言，
    一个参数没收敛，整条链就不能当收敛用。

    walkers 与步数、燃烧期三行不挂字形。步数与燃烧期是这次采样的设置，没有合格不合格；
    walkers 有下界，但满足是常态，每次都挂一个 ✓ 只会把另外三个 ✓ 的分量摊薄，所以只在
    不足时才出声。
    """
    config = mcmc.config
    free_count = len(tuple(mcmc.parameter_names))
    walkers = int(config.walkers)
    geometry = f"{walkers}（≥ {WALKERS_FACTOR}·{free_count}+{WALKERS_MARGIN}）"
    enough = _walkers_are_enough(free_count, walkers)
    burn_in = int(config.burn_in)
    total = burn_in + int(config.production_steps)
    return (
        (geometry, "") if enough else _judged(geometry, False),
        (f"{total:,}", ""),
        _acceptance_reading(mcmc.acceptance_fraction),
        _rhat_reading(mcmc.split_rhat),
        _sample_size_reading(mcmc.effective_sample_size),
        (_burn_in_text(burn_in, total), ""),
    )


def quantile_target(report: object, mcmc: object) -> int | None:
    """哪一列采样点值得单独摆一段——按 |ρ| 降序的第一对里的第一位。

    与中栏矩阵旁那句「最强相关 d·aSi ↔ ρ·aSi」同一个序（``plots.correlation._ranked_pairs``）：
    同屏两处各挑一次参数的话，会出现「矩阵说最强是这一对、分位表却在报另一个参数」。

    相关矩阵与采样列各有自己的参数表，两份未必对齐（矩阵来自协方差，链只走自由参数），所以
    按名字回查而不是按位置。一对都回查不到时退到第一列——一个没有强相关的拟合照样有后验，
    而这一段的用处是看形状，不是看相关。
    """
    names = tuple(mcmc.parameter_names)
    if not names:
        return None
    positions = {name: index for index, name in enumerate(names)}
    for row, column, _value in _ranked_pairs(report.correlation_matrix):
        for index in (row, column):
            position = positions.get(report.correlation_names[index])
            if position is not None:
                return position
    return 0


def _crossing(
    low_value: float,
    low_objective: float,
    high_value: float,
    high_objective: float,
    threshold: float,
) -> float:
    """两个采样点之间目标值穿过阈值的位置，线性内插。"""
    span = high_objective - low_objective
    if span == 0.0:
        return low_value
    return low_value + (threshold - low_objective) / span * (high_value - low_value)


def _closed_profile_samples(profile: object) -> tuple[tuple[float, ...], tuple[float, ...], float] | None:
    """只让两端闭合、坐标与目标值都完整的扫描参与区间比较。"""
    threshold = profile.objective_threshold
    if threshold is None or not (profile.lower_closed and profile.upper_closed):
        return None
    values = _finite_values(profile.values)
    objectives = _finite_values(profile.objectives)
    if values is None or objectives is None or len(values) != len(objectives):
        return None
    return values, objectives, threshold


def _profile_interval(profile: object) -> tuple[float, float] | None:
    """这条 Profile 曲线圈出的区间，圈不出来时是 ``None``。

    阈值是绝对标度（``analysis.profiles`` 拿 ``objectives >= threshold`` 判两端闭合），所以
    「在里面」就是目标值低于阈值。两端都闭合才给区间：有一端跑到边界还没抬上去，那一侧根本
    没有界可言，把最后一个采样点当界会把一个敞开的方向报成一个数。
    """
    samples = _closed_profile_samples(profile)
    if samples is None:
        return None
    values, objectives, threshold = samples
    inside = [index for index, objective in enumerate(objectives) if objective < threshold]
    if not inside:
        return None
    first, last = inside[0], inside[-1]
    low = (
        values[first]
        if first == 0
        else _crossing(values[first - 1], objectives[first - 1], values[first], objectives[first], threshold)
    )
    high = (
        values[last]
        if last == len(values) - 1
        else _crossing(values[last], objectives[last], values[last + 1], objectives[last + 1], threshold)
    )
    return (min(low, high), max(low, high))


def _finite_pair(lower: object, upper: object) -> tuple[float, float] | None:
    """一对界，理顺次序。报告里这一栏没有 ``BootstrapResult`` 那道校验，所以自己查。"""
    pair = _finite_values((lower, upper))
    return None if pair is None else (min(pair), max(pair))


def _comparable_intervals(report: object) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """按参数名配对 Profile 与自助区间，排除未闭合或非有限的证据。"""
    bootstrap = {
        name: pair
        for name, lower, upper in report.bootstrap_intervals
        if (pair := _finite_pair(lower, upper)) is not None
    }
    return [
        (interval, bootstrap[profile.name])
        for profile in report.profiles
        if profile.name in bootstrap and (interval := _profile_interval(profile)) is not None
    ]


def profile_agreement(report: object) -> tuple[str, str]:
    """自助区间与 Profile 区间对不对得上——四种答案，各说一件不同的事。

    没有 Profile 曲线时说「未跑」：两种区间只有一种在手，谈不上一致或不一致。曲线在但一条也
    圈不出区间时说「未闭合」——那是 Profile 自己的话没说完，不是两者不合。

    「合」的判据是两个区间相交，不是相等，也不带容差：仓库里没有一个「自助与 Profile 差多少
    算合」的常量，这里不新造一个。相交这个谓词不需要阈值，而它答的正是读者要问的那件事——两
    种算法圈出的范围里有没有共同的一块。
    """
    if not report.profiles:
        return ("ℹ Profile 未跑", "info")
    comparable = _comparable_intervals(report)
    if not comparable:
        return ("ℹ Profile 未闭合", "info")
    disjoint = any(high < other_low or other_high < low for (low, high), (other_low, other_high) in comparable)
    if disjoint:
        return (f"{WARN_MARK} 与 Profile 不一致", "warn")
    return (f"{OK_MARK} 与 Profile 一致", "ok")


def correlation_note(report: object) -> tuple[str, str]:
    """重采样有没有把参数间的纠缠一并带上。

    自助抽样重采样的是数据，每一次都重新拟合，所以相关是自动带进去的——但只有拟合里真有强
    相关时，这句话才对读者有意义。一律写「已计入相关」的话，一份没有强相关的报告也在声称重
    采样替他处理了一件根本不存在的事。两句都是陈述而非判定，所以都用 info。
    """
    return ("ℹ 已计入相关", "info") if report.strong_correlations else ("ℹ 无强相关", "info")


def _boundary_reading(report: object) -> tuple[str, str]:
    """撞上参数边界的次数。一次都没有是好消息，撞上了则整段读数都要打折看。"""
    hits = tuple(report.boundary_hits)
    return (f"{len(hits):,}", "ok" if not hits else "warn")


def bootstrap_readings(report: object) -> tuple[tuple[str, str], ...]:
    """三行读数各连一个判定，次序照 ``BOOTSTRAP_CAPTIONS``。

    这三行不挂 ✓/⚠ 字形，只染色：收敛那一段的六行里有四行是拿数字跟阈值比，字形是那次比
    较的结论；这一段三行都是计数，「150 次」本身没有合格不合格，读者要的是「失败率高不高」
    这一层，而那正是颜色说的话。

    失败率补上基数：单给比例读不出量级——「丢了 2%」在 200 次和在 20 次上是两件事。
    """
    if not report.bootstrap_performed:
        return ((BOOTSTRAP_NOT_RUN_TEXT, ""), (UNAVAILABLE_TEXT, ""), _boundary_reading(report))
    count = int(report.bootstrap_sample_count)
    rate = float(report.bootstrap_failure_rate)
    kind = "ok" if rate <= FAILURE_RATE_WARN else "warn"
    # 次数为 0 说明这份工程文件存在 ``bootstrap_sample_count`` 之前——比例还在，基数没有，
    # 那就只念比例，不去拿 0 当基数算出一个「0/0」。
    share = f"{rate:.0%}" if count <= 0 else f"{rate:.0%}（{round(rate * count)}/{count:,}）"
    counted = BOOTSTRAP_COUNT_UNRECORDED_TEXT if count <= 0 else f"{count:,}"
    return ((counted, ""), (share, kind), _boundary_reading(report))


class _ReadingPanel(QWidget):
    """一段固定行名的读数：键在左、值在右，一行一个数。

    外观照 ``verdict.VerdictEvidence`` 的指标网格（键 ``mutedText``、值等宽右对齐、行距 1px）：
    这几段在同一栏里上下相邻，两套行距会读成两种东西。

    这个基类永不隐藏自己。三段的可见性归 ``window_layout.apply_step_scope``——它按步骤、当前
    分析页和 ``has_evidence()`` 一起判；面板自己再插一手，两处就会在某些切换次序上打架。而
    Qt 对显式 ``hide()`` 过的控件有记忆：那样的控件被加进布局后不会随窗口一起现身，整段于
    是永远不出现。
    """

    def __init__(self, captions: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._present = False
        self._body = QVBoxLayout(self)
        self._body.setContentsMargins(0, 0, 0, 0)
        self._body.setSpacing(theme.SPACE_SM)
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(theme.SPACE_SM)
        self._grid.setVerticalSpacing(1)
        # 只有值那一列可伸缩：键与单位都是短字，横向多出来的像素全给数字，右栏窄下去时
        # 先挤掉的也是数字与键之间的空白，而不是某一列的字。
        self._grid.setColumnStretch(1, 1)
        self._values = tuple(self._build_row(index, caption) for index, caption in enumerate(captions))
        self._body.addLayout(self._grid)

    def _build_row(self, row: int, caption: str) -> QLabel:
        key = QLabel(caption, self)
        key.setProperty("mutedText", True)
        value = QLabel(self)
        value.setProperty("mono", True)
        value.setProperty("tabularValue", True)
        self._grid.addWidget(key, row, 0)
        self._grid.addWidget(value, row, 1, Qt.AlignmentFlag.AlignRight)
        return value

    def has_evidence(self) -> bool:
        """这一段此刻手里有没有东西可读——右栏据此决定摆不摆它。"""
        return self._present

    def _write(self, readings: tuple[tuple[str, str], ...]) -> None:
        for label, (text, kind) in zip(self._values, readings, strict=True):
            label.setText(text)
            theme.set_status_kind(label, kind)
        self._present = True

    def _blank(self) -> None:
        for label in self._values:
            label.setText(UNAVAILABLE_TEXT)
            theme.set_status_kind(label, "")
        self._present = False


class ConvergencePanel(_ReadingPanel):
    """帧⑤ 右栏第一段：这条链能不能当证据用。

    六行读数全部来自当前候选解自己的那条链。别的候选解的链不在这里露面——右栏此刻正对着中栏
    那张相关矩阵，一行「split-R̂ 1.008」读起来说的就是屏上这一个拟合。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(CONVERGENCE_CAPTIONS, parent)
        self.setObjectName("mcmcConvergencePanel")
        self._blank()

    def set_result(self, result: object | None, candidate_id: str | None) -> None:
        mcmc = mcmc_evidence(result, candidate_id)
        if mcmc is None:
            self._blank()
            return
        self._write(convergence_readings(mcmc))

    def clear_result(self, message: str) -> None:
        """六行退回「不可用」。

        ``message`` 收下但不写进行里：那句话是给整栏的（「尚无拟合结果」），而这一列是右对齐
        的等宽数字，一句话摆进去会把六行读数的对齐全打散。空态本身也不会停在屏上——没有链时
        整段不摆（见 ``has_evidence``）。
        """
        self._blank()


class QuantilePanel(_ReadingPanel):
    """帧⑤ 右栏第二段：某一个参数的后验，形状在上、三个分位在下。

    「某一个」由中栏那张矩阵定（``quantile_target``），而卡抬头要把它的短名念出来。抬头归
    ``titled_card``，所以这里只管报名字：谁挂的抬头，谁去改那行字（``window_layout``）。
    """

    parameter_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(QUANTILE_CAPTIONS, parent)
        self.setObjectName("posteriorQuantilePanel")
        self._short_name = ""
        self.plot = PosteriorQuantilePlot()
        # 图摆在表上方，照设计稿把两者叠在同一段里的次序：先看形状，再读数。直接
        # ``insertWidget`` 而不在基类里留钩子——三段里只有这一段有图。
        self._body.insertWidget(0, self.plot)
        # 单位另起一列，不缀在数字后面：三行要按小数点对齐读，而「nm」跟在数字后面会把
        # 「32.32」与「330.00」的右边缘错开一个字符。
        self._units = tuple(self._build_unit(row) for row in range(len(QUANTILE_CAPTIONS)))
        self._blank()

    def _build_unit(self, row: int) -> QLabel:
        unit = QLabel(self)
        unit.setProperty("mutedText", True)
        self._grid.addWidget(unit, row, 2, Qt.AlignmentFlag.AlignLeft)
        return unit

    def _blank(self) -> None:
        super()._blank()
        for label in self._units:
            label.setText("")

    def current_short_name(self) -> str:
        """此刻在读哪个参数的短名，没在读时是空串。

        ``parameter_changed`` 之外另留这个读取器，是因为投影发生在 ``ResultsPanel`` 的构造期，
        早于 ``window_layout`` 把卡搭起来：只靠信号的话，第一次投影时抬头还没人接，那行字会永
        远停在「后验分位」。改名的助手因此先读一次打底再连接（照 ``_name_card_after_batch_mode``）。
        """
        return self._short_name

    def set_result(self, result: object | None, candidate_id: str | None) -> None:
        """画并念这个候选解自己那条链里最值得看的一列。"""
        report, message = _owned_report(result, candidate_id)
        mcmc = None if report is None else mcmc_evidence(result, candidate_id)
        columns = None if mcmc is None else _sample_columns(mcmc)
        index = None if mcmc is None else quantile_target(report, mcmc)
        if columns is None or index is None:
            self.clear_result(message if report is None else POSTERIOR_UNAVAILABLE_TEXT)
            return
        name = tuple(mcmc.parameter_names)[index]
        column = columns[index]
        # ``_linear_quantile`` 按位置插值，要的是排好序的一列；链里的采样点按步数排，不按值排。
        quantiles = tuple(_linear_quantile(tuple(sorted(column)), p) for p in POSTERIOR_QUANTILES)
        unit = _display_unit(name)
        self._write(tuple((f"{_display_value(name, value):.2f}", "") for value in quantiles))
        for label in self._units:
            label.setText(unit)
        # 图画在采样点原本的标度上（Å），表念的是显示标度（nm）——换算只在念的那一步做，所以
        # 三条线站的位置与三行数字说的是同一组数。
        self.plot.show_values(column, quantiles)
        self._restate(short_labels((name,), result.parameter_definitions)[0])

    def clear_result(self, message: str) -> None:
        """三行退回「不可用」，图上只留那句话，抬头退回不带参数名的写法。"""
        self._blank()
        self.plot.show_unavailable(message)
        self._restate("")

    def _restate(self, short_name: str) -> None:
        """换了参数才出声：同一个名字反复广播会让抬头每次投影都重排一次版。"""
        if short_name == self._short_name:
            return
        self._short_name = short_name
        self.parameter_changed.emit(short_name)


class BootstrapPanel(_ReadingPanel):
    """帧⑤ 右栏第三段：重采样怎么说。

    三行读数在上、两枚徽章在下。徽章说的是读数说不出的两件事：这组区间与 Profile 似然圈出的
    范围对不对得上，以及重采样有没有把参数之间的纠缠一并带进去。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(BOOTSTRAP_CAPTIONS, parent)
        self.setObjectName("bootstrapPanel")
        # 徽章走 ``WrappingRow``：两枚并排是常态，右栏窄下去时它们各占一行，而不是被无声裁掉
        # ——这一栏的横向滚动条是 ``AlwaysOff``，溢出没有任何声响。
        self._badge_host, self._badge_row = wrapping_row(self, spacing=theme.SPACE_XS + 2)
        self._badges = tuple(self._build_badge() for _ in range(2))
        for badge in self._badges:
            self._badge_row.addWidget(badge)
        self._body.addWidget(self._badge_host)
        self._blank()

    def _build_badge(self) -> QLabel:
        badge = QLabel(self)
        badge.setProperty("badge", True)
        return badge

    def _blank(self) -> None:
        super()._blank()
        for badge in self._badges:
            badge.setText("")
            badge.setVisible(False)

    def set_result(self, result: object | None, candidate_id: str | None) -> None:
        """三行读数与两枚徽章都出自这个候选解自己的那份报告。"""
        report, _message = _owned_report(result, candidate_id)
        if report is None:
            self._blank()
            return
        self._write(bootstrap_readings(report))
        readings = (profile_agreement(report), correlation_note(report))
        for badge, (text, kind) in zip(self._badges, readings, strict=True):
            badge.setText(text)
            theme.set_status_kind(badge, kind)
            badge.setVisible(True)

    def clear_result(self, message: str) -> None:
        """三行退回「不可用」，两枚徽章收起来。

        ``message`` 收下不用，同 ``ConvergencePanel.clear_result``：那句话是给整栏的，摆进右
        对齐的数字列里只会打散三行的对齐。
        """
        self._blank()
