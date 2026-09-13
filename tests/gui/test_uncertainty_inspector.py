"""帧⑤ 右栏那三段：MCMC 收敛诊断 / 后验分位 · <短名> / 自助抽样（Bootstrap）。

设计稿帧⑤ 的右栏和帧① 不是同一栏。帧① 是「拟合判定 / 参数 · 结果值 / 候选解」——一次拟合
交出了什么；帧⑤ 是「这条链可不可信 / 那个参数的后验长什么样 / 重采样怎么说」——同一份结果
换一套证据。实现此前只有帧① 那一套：切到不确定度那一页，右栏一动不动，读者对着中栏的相关
矩阵想确认「split-R̂ 到底多少」，得去专家菜单里另开一个对话框。

这个文件钉三件事：段在该在的时候在（且只在那时在）、抬头点名当前在读哪个参数、以及九行读数
逐个是算出来的。第三件最要紧——替身的数值全部与设计稿不同，任何一处照抄设计稿的字面都会红。
"""

from __future__ import annotations

from dataclasses import replace
from operator import itemgetter

import numpy as np
from PySide6.QtWidgets import QFrame, QLabel, QWidget
from tests.support.model_cases import dataset_project, final_fit_result

import xrr_fitter.api as api

# 六个自由参数，连它们在图上与卡抬头上的短名。短名的素材是 ``parameter_definitions``——与
# 相关矩阵的刻度、强相关那一行同一处出处（见 ``tests/gui/test_correlation_labels.py``）。
PARAMETERS = (
    ("component.0.thickness_a", "ox 厚度", "layer", "d·ox"),
    ("component.0.density_scale", "ox 相对密度", "layer", "ρ·ox"),
    ("component.1.thickness_a", "aSi 厚度", "layer", "d·aSi"),
    ("component.1.density_scale", "aSi 相对密度", "layer", "ρ·aSi"),
    ("component.1.roughness_a", "aSi 入射侧粗糙度", "layer", "σ·aSi"),
    ("instrument.scale", "尺度", "instrument", "scale"),
)

NAMES = tuple(map(itemgetter(0), PARAMETERS))

# 替身的采样规模。设计稿写 32 walkers / 8,000 步 / 前 25%；这里换成另一组，且仍然合法：
# ``McmcConfig`` 要求 walkers 是偶数，``McmcReport`` 要求 acceptance_fraction 与 walkers 等长。
WALKERS = 16
BURN_IN = 400
PRODUCTION = 1600

# 21 个采样点的厚度列（Å）。取奇数个是为了让 P50 正好落在一个样本上，P16/P84 落在两个样本
# 之间——线性插值那一步于是也在判据里：``_linear_quantile`` 算错端点或算错权重都会红。
THICKNESS_A = tuple(map(float, range(320, 341)))
DRAWS = len(THICKNESS_A)

# 上面那 21 个点摊出来的三个分位，换算成 nm（``thickness_a`` 是长度量，显示除 10）：
# P16 位置 20·0.16=3.2 → 323.2Å、P50 位置 10 → 330.0Å、P84 位置 20·0.84=16.8 → 336.8Å。
QUANTILE_NM = ("32.32", "33.00", "33.68")

# 设计稿的强相关是「d·aSi ↔ ρ·aSi」，抬头因此写「后验分位 · d·aSi」。替身把这一对做成
# 最强（|−0.72|），另配一对次强（0.55）——抬头要是取了索引最小的那一对，读到的是 d·ox。
STRONGEST = (2, 3, -0.72)
RUNNER_UP = (0, 1, 0.55)

BOOTSTRAP_COUNT = 150
BOOTSTRAP_FAILURE_RATE = 0.04


def _definition(name: str, display_name: str, category: str) -> api.ParameterDefinition:
    return api.ParameterDefinition(
        name=name,
        display_name=display_name,
        unit="",
        category=category,
        initial=1.0,
        lower=0.0,
        upper=2.0,
        transform="linear",
        locked=False,
    )


def _samples() -> np.ndarray:
    """(21, 6) 的采样矩阵；第 2 列是那条厚度列，别的列随手给个有限值。"""
    values = np.tile(np.linspace(0.8, 1.2, DRAWS)[:, None], (1, len(NAMES)))
    values[:, 2] = np.array(THICKNESS_A)
    return values


def _mcmc_report(*, candidate_id: str = "candidate-0") -> api.McmcReport:
    config = api.McmcConfig(
        walkers=WALKERS,
        burn_in=BURN_IN,
        production_steps=PRODUCTION,
        thin=1,
    )
    return api.McmcReport(
        config=config,
        child_seed=7,
        parameter_names=NAMES,
        samples_physical=_samples(),
        log_probability=np.linspace(-12.0, -10.0, DRAWS),
        # 每个 walker 的接受率都是 0.42：均值因此也是 0.42，且落在健康区间 (0.10, 0.80) 里。
        acceptance_fraction=np.full(WALKERS, 0.42),
        split_rhat=np.array([1.002, 1.021, 1.004, 1.011, 1.003, 1.007]),
        effective_sample_size=np.array([1400.0, 875.0, 1210.0, 990.0, 1330.0, 1180.0]),
        boundary_hits=(),
        candidate_id=candidate_id,
    )


def _uncertainty(*, mcmc: api.McmcReport | None = None) -> api.UncertaintyReport:
    matrix = np.eye(len(NAMES))
    for row, column, value in (STRONGEST, RUNNER_UP):
        matrix[row, column] = matrix[column, row] = value
    return api.UncertaintyReport(
        correlation_names=NAMES,
        correlation_matrix=matrix,
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=BOOTSTRAP_FAILURE_RATE,
        boundary_hits=(),
        strong_correlations=((NAMES[STRONGEST[0]], NAMES[STRONGEST[1]], STRONGEST[2]),),
        systematic_residual=False,
        diagnostics=(),
        mcmc=_mcmc_report() if mcmc is None else mcmc,
        candidate_id="candidate-0",
        bootstrap_sample_count=BOOTSTRAP_COUNT,
    )


def _result(*, mcmc: api.McmcReport | None = None) -> api.FitResult:
    """一份带 MCMC 证据的结果。

    ``final_fit_result()`` 的 ``uncertainty`` 是 ``None``，``parameter_definitions`` 是空
    元组——短名与三段读数的素材都得自己挂上去。
    """
    return replace(
        final_fit_result(),
        parameter_definitions=tuple(_definition(name, display, category) for name, display, category, _ in PARAMETERS),
        uncertainty=_uncertainty(mcmc=mcmc),
    )


def _project(*, mcmc: api.McmcReport | None = None) -> api.XrrProject:
    value = api.XrrProject.new((dataset_project(result=_result(mcmc=mcmc)),), master_seed=1201)
    value = replace(value, base_directory="/private/tmp")
    # ``XrrProject.new`` 不选数据集，而 ``ResultsPanel._refresh`` 是从「当前数据集」找结果的：
    # 少了这一句，整个面板停在「当前数据集尚无拟合结果」，右栏三段于是连一次投影都收不到。
    return api.select_active_dataset(value, "curve")


def _window(qtbot, project: api.XrrProject):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


# 三段的卡名自己写在这里，不从 ``STEP_INSPECTOR_SECTIONS`` 推导。
# ``test_step_scoped_views._all_section_names()`` 走的是推导那条路，于是它数不到只挂在
# 「视图」这一维上的卡——新段真的露在屏上时，那边的 ``== ["inspectorResults"]`` 照样为绿。
CARDS = ("inspectorMcmcConvergence", "inspectorPosteriorQuantile", "inspectorBootstrap")


def _visible_headings(window) -> list[str]:
    """右栏此刻露着的段落抬头，自上而下。

    读的是 ``sectionHeader`` 这枚属性而不是卡名：``titled_card`` 给卡抬头挂了它，帧① 的
    ``inspectorResults`` 里那三段各自的抬头也挂了它。同一把尺量两帧，「换了三段」与「三段
    没换」于是是同一个读数上的差别，而不是两套读法各说各话。
    """
    column = window.inspector_column
    headings = [
        label for label in column.findChildren(QLabel) if label.property("sectionHeader") and label.isVisibleTo(column)
    ]
    headings.sort(key=lambda label: label.mapTo(column, label.rect().topLeft()).y())
    return [label.text() for label in headings]


def _uncertainty_window(qtbot, *, mcmc: api.McmcReport | None = None):
    """停在「结果步 + 不确定度视图」上的主窗口——帧⑤ 的那一屏。"""
    window = _window(qtbot, _project(mcmc=mcmc))
    window.plot_panel.select_view("uncertainty")
    return window


def _reading_labels(card) -> list[QLabel]:
    """读数标签不包含段标题和独立判定徽章。"""
    return [
        label
        for label in card.findChildren(QLabel)
        if label.isVisibleTo(card)
        and label.text()
        and not label.property("sectionHeader")
        and not label.property("badge")
    ]


def _rows(window, name: str) -> dict[str, str]:
    """一张卡里那几行「键 → 值」。

    键值都是 ``QLabel``，按位置配对：同一行的两个 label 纵坐标相同。这样读不依赖控件的
    objectName，实现换用别的容器也还量得到同一件事。

    徽章排除在外：两枚徽章并排时纵坐标相同，按位置配对会把它们读成一行「键 → 值」。徽章说
    的是判定，不是读数，另有 ``test_the_bootstrap_badges_state_what_the_report_actually_says``
    盯着。
    """
    card = window.inspector_column.findChild(QFrame, name)
    assert card is not None, name
    labels = _reading_labels(card)
    lines: dict[int, list[QLabel]] = {}
    for label in labels:
        top = label.mapTo(card, label.rect().topLeft()).y()
        lines.setdefault(top, []).append(label)
    rows: dict[str, str] = {}
    for top in sorted(lines):
        row = sorted(lines[top], key=lambda label: label.mapTo(card, label.rect().topLeft()).x())
        if len(row) >= 2:
            rows[row[0].text()] = " ".join(label.text() for label in row[1:])
    return rows


def test_the_uncertainty_view_swaps_the_inspector_to_the_three_sections_frame_five_draws(qtbot) -> None:
    """切到不确定度那一页，右栏换成帧⑤ 画的三段。

    帧① 与帧⑤ 在实现里是同一个流程步（都是 ``RESULT_STEP_INDEX``），右栏此前只按步骤索引
    切段，于是两帧共用一栏。读者在中栏读相关矩阵、Profile 似然、后验直方图，右栏报的却是
    「参数 · 结果值」和「候选解」——三段证据里没有一段说得上这条链收敛没收敛。
    """
    window = _uncertainty_window(qtbot)

    assert _visible_headings(window) == ["MCMC 收敛诊断", "后验分位 · d·aSi", "自助抽样（Bootstrap）"]


def test_leaving_the_uncertainty_view_gives_the_inspector_back_to_frame_one(qtbot) -> None:
    """从不确定度切回别的分析页，右栏回到帧① 那三段。

    这一条与上一条是同一枚开关的两面。只钉「切进去会换」的话，实现可以把三段挂成常驻——
    帧① 的右栏于是变成六段，而设计稿的每一帧都恰好三段。
    """
    window = _uncertainty_window(qtbot)
    window.plot_panel.select_view("candidates")

    assert _visible_headings(window) == ["拟合判定", "参数 · 结果值", "候选解"]


def test_the_convergence_section_reads_the_chain_it_was_given(qtbot) -> None:
    """六行收敛读数逐个算自这条链，不是设计稿那份字面。

    替身给的是 16 walkers / 2,000 步 / 前 20% / 接受率 0.42 / split-R̂ 最大 1.021 / ESS 最小
    875；设计稿写的是 32 / 8,000 / 前 25% / 0.31 / 1.008 / 1,240。哪一处照抄，哪一处红。

    「最坏的那一个」是 R̂ 取最大、ESS 取最小——照 ``results.uncertainty.sampling_readings``
    同一个口径：收敛这件事没有平均可言，一个参数没收敛，整条链就不能当收敛用。
    """
    window = _uncertainty_window(qtbot)

    assert _rows(window, "inspectorMcmcConvergence") == {
        "walkers": f"{WALKERS}（≥ 2·{len(NAMES)}+2）",
        "步数 / 链": f"{BURN_IN + PRODUCTION:,}",
        "接受率": "0.42 ✓",
        "split-R̂ (max)": "1.021 ✓",
        "ESS (min)": "875 ✓",
        "燃烧期 burn-in": "前 20%",
    }


def test_the_convergence_section_flags_a_chain_that_has_not_converged(qtbot) -> None:
    """R̂ 越过 1.10、ESS 掉到 100 以下时，那两行不能还挂着 ✓。

    ✓ 若是写死的字面，上一条测试照样为绿——它给的正是一条收敛了的链。阈值出处是
    ``results.uncertainty`` 里镜像 ``analysis.mcmc.problem_mcmc_warnings`` 的那两个常数。
    """
    stalled = replace(
        _mcmc_report(),
        split_rhat=np.array([1.002, 1.310, 1.004, 1.011, 1.003, 1.007]),
        effective_sample_size=np.array([1400.0, 42.0, 1210.0, 990.0, 1330.0, 1180.0]),
    )
    window = _uncertainty_window(qtbot, mcmc=stalled)
    rows = _rows(window, "inspectorMcmcConvergence")

    assert rows["split-R̂ (max)"] == "1.310 ⚠"
    assert rows["ESS (min)"] == "42 ⚠"
    card = window.inspector_column.findChild(QFrame, "inspectorMcmcConvergence")
    values = [label for label in card.findChildren(QLabel) if label.text().endswith("⚠")]
    assert values, "没有一行被标成警告色"
    for label in values:
        assert label.property("statusKind") == "warn", (label.text(), label.property("statusKind"))


def test_the_quantile_section_names_the_parameter_the_matrix_calls_strongest(qtbot) -> None:
    """抬头右端那个短名，和相关矩阵读的是同一对里的同一位。

    帧⑤ 是一屏：中栏矩阵旁写「最强相关 d·aSi ↔ ρ·aSi」，右栏抬头写「后验分位 · d·aSi」。
    两处各自挑一次参数的话，同屏就会出现「矩阵说最强是这一对、分位表却在报另一个参数」。
    序取自 ``plots.correlation._ranked_pairs``（按 |ρ| 降序）——按索引顺序挑会得到 d·ox。
    """
    from xrr_fitter.gui.plots.correlation import _ranked_pairs
    from xrr_fitter.gui.plots.parameter_labels import label_map

    result = _result()
    row, column, _value = _ranked_pairs(result.uncertainty.correlation_matrix)[0]
    labels = label_map(result.parameter_definitions)
    expected = labels[result.uncertainty.correlation_names[row]]
    assert (expected, labels[result.uncertainty.correlation_names[column]]) == ("d·aSi", "ρ·aSi")

    window = _uncertainty_window(qtbot)
    card = window.inspector_column.findChild(QFrame, "inspectorPosteriorQuantile")
    heading = card.findChild(QLabel, "inspectorPosteriorQuantileTitle")

    assert heading.text() == f"后验分位 · {expected}"


def test_the_quantile_table_reads_the_three_quantiles_off_the_draws(qtbot) -> None:
    """P16 / P50（中位）/ P84 三行，值是从那 21 个采样点插出来的。

    设计稿写 48.52 / 48.70 / 48.89 nm；替身那条厚度列摊出来是 32.32 / 33.00 / 33.68。单位
    列写 nm 而不是 Å——``_display_value`` 把长度量除 10，两处必须同时换算，只换一处就是差
    一个数量级的读数。
    """
    window = _uncertainty_window(qtbot)

    assert _rows(window, "inspectorPosteriorQuantile") == {
        "P16": f"{QUANTILE_NM[0]} nm",
        "P50（中位）": f"{QUANTILE_NM[1]} nm",
        "P84": f"{QUANTILE_NM[2]} nm",
    }


def test_the_quantile_section_draws_the_histogram_the_design_puts_above_the_table(qtbot) -> None:
    """分位表上方那张紧凑直方图要真的在，且画的是那个参数的后验。

    三行数字说的是分布的三个点，形状说的是别的事：偏斜、双峰、贴边——设计稿把 26 根柱和
    P16/P50/P84 三条线画在表的上方正是为这个。图缺席时三行数字仍然对，读者却没法知道它们
    是从一个什么形状里取出来的。
    """
    window = _uncertainty_window(qtbot)
    card = window.inspector_column.findChild(QFrame, "inspectorPosteriorQuantile")
    plot = card.findChild(QWidget, "posteriorQuantilePlot")

    assert plot is not None, "分位段里没有直方图"
    assert plot.isVisibleTo(card)
    axes = plot.view.figure.axes[0]
    assert axes.patches, "直方图一根柱都没画"
    # 三条分位线：P16 / P50 / P84。柱子是 patches，线是 lines。
    assert len(axes.lines) == 3, [line.get_label() for line in axes.lines]


def test_the_bootstrap_section_reads_the_resampling_it_was_given(qtbot) -> None:
    """三行自助读数算自报告：次数、失败率（连基数）、边界命中数。

    设计稿写「200 / 2%（4/200）/ 0」；替身是 150 与 4%，摊出来是「4%（6/150）」。失败率
    单给比例读不出量级——「丢了 2%」在 200 次和在 20 次上是两件事，所以括号里补基数。
    """
    window = _uncertainty_window(qtbot)

    assert _rows(window, "inspectorBootstrap") == {
        "重采样次数": f"{BOOTSTRAP_COUNT}",
        "失败率": "4%（6/150）",
        "边界命中": "0",
    }


def test_the_bootstrap_badges_state_what_the_report_actually_says(qtbot) -> None:
    """两枚徽章读的是这份报告，不是设计稿上那两句话。

    「已计入相关」在设计稿里为真是因为那次拟合有强相关；一份没有强相关的报告挂着同一句，
    读者会以为重采样替他处理了一件根本不存在的事。写成常量时这条测试红。
    """
    window = _uncertainty_window(qtbot)
    card = window.inspector_column.findChild(QFrame, "inspectorBootstrap")
    badges = [label for label in card.findChildren(QLabel) if label.property("badge") and label.isVisibleTo(card)]
    badges.sort(key=lambda label: label.mapTo(card, label.rect().topLeft()).x())

    assert [label.text() for label in badges] == ["ℹ Profile 未跑", "ℹ 已计入相关"]
    assert [label.property("statusKind") for label in badges] == ["info", "info"]

    plain = replace(_result().uncertainty, strong_correlations=())
    window.result_panel.bootstrap_panel.set_result(replace(_result(), uncertainty=plain), "candidate-0")
    badges = [label for label in card.findChildren(QLabel) if label.property("badge") and label.isVisibleTo(card)]
    badges.sort(key=lambda label: label.mapTo(card, label.rect().topLeft()).x())
    assert [label.text() for label in badges] == ["ℹ Profile 未跑", "ℹ 无强相关"]


def test_the_three_sections_fit_the_inspector_without_a_sideways_scroll(qtbot) -> None:
    """三段的最小宽度都进得了右栏的视口——横向滚动条是 ``AlwaysOff``，溢出是无声裁掉。

    分位表因此不能用 ``QTableWidget``：表头与它自带的滚动条各有最小宽，两样加起来就把这
    一栏顶爆，而顶爆的表现不是滚动条出现，是右边那一列单位字符直接不见。
    """
    from PySide6.QtWidgets import QScrollArea

    window = _uncertainty_window(qtbot)
    scroll = window.inspector_column.findChild(QScrollArea, "inspectorScroll")
    viewport = scroll.viewport().width()

    for name in CARDS:
        card = window.inspector_column.findChild(QFrame, name)
        assert card.minimumSizeHint().width() <= viewport, (name, card.minimumSizeHint().width(), viewport)


def test_a_result_without_mcmc_evidence_keeps_the_inspector_on_frame_one(qtbot) -> None:
    """没有 MCMC 证据时不摆这三段——三张写着「不可用」的卡不是证据。

    帧⑤ 的右栏说的是「这条链怎么样」。链根本没跑过时，整段的答案是「去跑」，而那个入口在
    拟合菜单里，不在右栏。此前的空态是三行「不可用」，读者要读完三段才知道没东西可读。
    """
    without = replace(_result().uncertainty, mcmc=None)
    project = api.XrrProject.new(
        (dataset_project(result=replace(_result(), uncertainty=without)),),
        master_seed=1201,
    )
    window = _window(qtbot, replace(project, base_directory="/private/tmp"))
    window.plot_panel.select_view("uncertainty")

    assert _visible_headings(window) == ["拟合判定", "参数 · 结果值", "候选解"]


def test_the_evidence_belongs_to_the_candidate_on_screen(qtbot) -> None:
    """别的候选解的 MCMC 不是这一条的证据——归属不符时这三段不出现。

    照 ``results.uncertainty._mcmc_lines`` 与 ``sampling_readings`` 的规矩：宁可整段空着，
    也不借一个读数来填。借来的那六行数字看不出是别人的。

    ``candidate_id`` 只能取这份结果里真有的那几个（``project._validate_result_evidence``
    会拦下指向不存在候选解的证据），所以这里摆两个候选解，把链挂在第二个上，而检视器停在
    ``best_index`` 指的第一个上。
    """
    from tests.support.model_cases import fit_candidate

    borrowed = _mcmc_report(candidate_id="candidate-1")
    result = replace(
        final_fit_result(fit_candidate("candidate-0"), fit_candidate("candidate-1", objective=2.0)),
        parameter_definitions=tuple(_definition(name, display, category) for name, display, category, _ in PARAMETERS),
        uncertainty=replace(_uncertainty(mcmc=borrowed), candidate_id="candidate-1"),
    )
    project = api.XrrProject.new((dataset_project(result=result),), master_seed=1201)
    window = _window(qtbot, replace(project, base_directory="/private/tmp"))
    window.plot_panel.select_view("uncertainty")

    assert _visible_headings(window) == ["拟合判定", "参数 · 结果值", "候选解"]
