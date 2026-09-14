"""「结果」那一步下面展开的不确定度方法子管线（设计稿帧⑤ 左栏）。

四种证据答的是同一个问题的四面：谁和谁纠缠（相关矩阵）、重采样后区间有多宽（自助抽样）、
单参数的谷有多宽（Profile 似然）、后验整体长什么样（MCMC）。它们摆成一列而不是四个并列的
按钮，因为代价是递增的——相关矩阵是拟合末尾那个 Hessian 顺带给的，MCMC 要采上万步——所以
从上往下就是建议的顺序，读者要知道自己走到了哪一种。

三态不另算一遍：读报告里那四样在不在即可（``correlation_names`` / ``bootstrap_performed``
/ ``profiles`` / ``mcmc``）。缺席的第一样是当前那一步，它后面的还没轮到。这样"哪一步没做"
只有一个出处——报告本身，而不是界面自己记的一份进度；界面记进度就会出现报告说没跑、左栏
说跑过的两份说法。

小字报的是这份报告此刻的数（Profile 覆盖几个参数、MCMC 几条链、自助抽样抽了多少次、失败
率多少），没跑过的报「未运行」。设计稿在自助抽样那行写的「200 次」读的是
``UncertaintyReport.bootstrap_sample_count``——报告落地时由 ``analysis/report.py`` 从证据反
解（成功样本数 ÷（1 − 失败率）），而不是回头去问预算 ``problem.config.budget``，GUI 也就不
必触达配置。这个字段之前存下的工程文件读出 0，此时退回报参数个数：宁可换一种读数，也不编一
个看着像次数的数字。

只有一件事读不出报告：链此刻在不在采。采样期间报告里还没有 ``mcmc``，按上面那套算的话第四行
读成「当前 · 未运行」——读者刚按下 ``▶ 运行 MCMC``、链已经在跑，左栏却说这一步没开始；跑过一
轮再跑第二轮更糟，四样都齐了，第四行读成「已完成」。所以「在跑」由外面推进来
（``sampling_walkers``），而且它一进来就先认走 ``current``：同一列里两处 ``current`` 会把
「读者现在在哪一步」问成两个答案。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.data.dataset_card import RUNNING_GLYPH
from xrr_fitter.gui.navigation.steps import build_step

METHOD_HEADING_TEXT = "不确定度方法"

# 标题、这一行是干什么的（tooltip 与没走到时的退路）。顺序即代价从低到高。
UNCERTAINTY_METHODS = (
    ("相关矩阵", "看参数之间的线性纠缠"),
    ("自助抽样", "重采样数据得出置信区间"),
    ("Profile 似然", "逐参数扫描目标函数的谷宽"),
    ("MCMC 后验", "采样后验，给非对称区间"),
)

# 相关矩阵那行报的是出处而不是尺寸：Hessian 只在最优点附近展开一次，谷子一歪那几个 ρ 就偏，
# 所以读者要先知道它不是采样得来的，才知道该多信它。
CORRELATION_SOURCE = "Hessian 派生"
NOT_RUN = "未运行"

# 链正在采时第四行那句小字。walkers 数取自 spin box 此刻的值——运行期间那几个 spin box 是禁用
# 的（``McmcControls.set_operation_state``），所以它就是这条活链的 walkers 数，而报告要等采完。
SAMPLING_SUFFIX = "采样中"


@dataclass(frozen=True, slots=True)
class MethodRow:
    """一行子步骤读出来的四样东西：标题、三态、小字、圆点里的字形。

    字形跟着一起算而不是留给控件按 ``state`` 反推：「序号 / ✓ / ◐」三选一是同一个判定的三个
    面，而 ``◐`` 那一档在 ``state`` 上与 ``current`` 重合——正在采的第四行与「轮到你了」的第
    四行都是 ``current``，只有这一层分得开它们。
    """

    title: str
    state: str
    caption: str
    glyph: str


def _correlation(report: api.UncertaintyReport) -> str | None:
    return CORRELATION_SOURCE if len(report.correlation_names) else None


def _bootstrap(report: api.UncertaintyReport) -> str | None:
    """跑过才有读数。区间为空而标记为跑过是另一回事——那是一个区间也没收住，属于失败率的事。"""
    if not report.bootstrap_performed:
        return None
    rate = f"失败 {report.bootstrap_failure_rate:.0%}"
    if report.bootstrap_sample_count:
        return f"{report.bootstrap_sample_count} 次 · {rate}"
    return f"{len(report.bootstrap_intervals)} 参数 · {rate}"


def _profiles(report: api.UncertaintyReport) -> str | None:
    return f"{len(report.profiles)} 参数" if report.profiles else None


def _mcmc(report: api.UncertaintyReport) -> str | None:
    return None if report.mcmc is None else f"{report.mcmc.config.walkers} walkers"


# 每种方法怎么从报告里读出自己那句小字，读不出来（``None``）就是没跑过。判据与文案是同一件
# 事：能报出数说明证据在手，报不出说明这一步还没走——分成两处写就会出现"标成走过却没有数"。
METHOD_READERS = (_correlation, _bootstrap, _profiles, _mcmc)


def _method_state(index: int, walked: int, sampling: int | None) -> tuple[str, str]:
    """先认正在采样的一行，再按证据的连续完成前缀标记其余行。"""
    if index == sampling:
        return "current", RUNNING_GLYPH
    if index < walked:
        return "done", "✓"
    # 前面那一步没走，后面的就还没轮到自己被标成走过——这一列是一条路，不是四个复选框。
    state = "current" if index == walked and sampling is None else "pending"
    return state, str(index + 1)


def uncertainty_method_rows(
    report: api.UncertaintyReport | None,
    *,
    sampling_walkers: int | None = None,
) -> tuple[MethodRow, ...]:
    """把一份报告（或没有报告）摊成四行。

    没有报告时站在第一行上：那是"刚跑完拟合、不确定度一样都没算"的样子。四行全画成未达会
    让读者以为这一段与自己无关，而其中三样确实还不能做——它们都要先有一份可信的最优解。

    ``sampling_walkers`` 不是 ``None`` 时链正在采：第四行改读那条活链的规模并认走 ``current``，
    别的行一律让出这一档（``walked`` 那一支因此要多问一句 ``sampling is None``）。判定摆在这个
    次序上是因为报告与「在跑」会各说一套：四样都齐时报告说第四行 ``done``，而链此刻正在跑第二
    轮；先按报告算再补采样的写法，得到的是「已完成 · 采样中」这样一行自相矛盾的读数。
    """
    captions = [None if report is None else reader(report) for reader in METHOD_READERS]
    walked = next((index for index, caption in enumerate(captions) if caption is None), len(captions))
    sampling = None if sampling_walkers is None else len(captions) - 1
    if sampling is not None:
        captions[sampling] = f"{sampling_walkers} walkers · {SAMPLING_SUFFIX}"
    rows = []
    for index, (title, _purpose) in enumerate(UNCERTAINTY_METHODS):
        state, glyph = _method_state(index, walked, sampling)
        rows.append(MethodRow(title=title, state=state, caption=captions[index] or NOT_RUN, glyph=glyph))
    return tuple(rows)


class UncertaintyMethodNav(QWidget):
    """The four evidence rows, headed by their own ``.nav-sec``.

    自成一段而不是接在六步后面当第七到第十步：这四行是「结果」这一步展开出来的内容，帧①③④
    的左栏里它们该整段消失（``PipelineNav`` 管可见性）。抬头归这个部件而不是归栏，理由与
    「分析管线」那句一样——它随自己命名的那四行一起出现、一起消失。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("uncertaintyMethodNav")
        self.setAccessibleName(METHOD_HEADING_TEXT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, theme.SPACE_SM, 0, 0)
        # 行与行之间不留空：连接线长在行内部，这里再留缝就把四枚圆点该读成的那一列断开了。
        layout.setSpacing(0)

        self.heading = QLabel(METHOD_HEADING_TEXT, self)
        self.heading.setObjectName("uncertaintyMethodHeader")
        theme.apply_section_heading(self.heading, tracking_px=theme.RAIL_SECTION_TRACKING_PX)
        self.heading.setContentsMargins(theme.SPACE_MD, 0, theme.SPACE_MD, theme.SPACE_SM)
        layout.addWidget(self.heading)

        self.labels: list[QLabel] = []
        self._dots: list[QLabel] = []
        self._captions: list[QLabel] = []
        last = len(UNCERTAINTY_METHODS) - 1
        for index, (title, purpose) in enumerate(UNCERTAINTY_METHODS):
            parts = build_step(self, title, purpose, connected=index < last)
            layout.addWidget(parts.row)
            self.labels.append(parts.label)
            self._dots.append(parts.marker)
            self._captions.append(parts.description)
        self.set_report(None)

    def set_report(
        self,
        report: api.UncertaintyReport | None,
        *,
        sampling_walkers: int | None = None,
    ) -> None:
        """Repaint the four rows for what this report carries."""
        for index, row in enumerate(uncertainty_method_rows(report, sampling_walkers=sampling_walkers)):
            marker = self._dots[index]
            # 字形照读数给的那个：✓ 与序号与「分析管线」那六行同一套记号，同一栏里两段进度不该
            # 用两套字母表；``◐`` 与帧④ 数据集行「在跑」那一枚同一个字形。
            marker.setText(row.glyph)
            theme.set_step_state(marker, row.state)
            theme.set_step_state(self.labels[index], row.state)
            self._captions[index].setText(row.caption)

    def captions(self) -> tuple[str, ...]:
        return tuple(label.text() for label in self._captions)
