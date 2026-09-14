"""帧⑤ 左栏那两处运行态读数：第四行的 ``◐ · 采样中``，与页脚那句 walkers 下界。

设计稿帧⑤ 的左栏比帧① 多两样东西，两样都是读数而不是文案。

第四行（``MCMC 后验``）画成 ``.pstep.current``，圆点写 ``◐``，小字写 ``32 walkers · 采样中``。
实现此前只按报告里那四样在不在算三态——而链正在采的时候报告里还没有 ``mcmc``，第四行于是
读成 ``current`` + ``未运行``：读者按下 ``▶ 运行 MCMC``、链已经在跑，左栏却说这一步没开始。
四样都齐了（跑过一轮再跑第二轮）更糟：``walked`` 到顶，第四行读 ``done``。

页脚那句 ``walkers ≥ 2·n_free + 2 · 已满足（9→32）`` 只在这一屏出现，帧①③ 各写自己那句
（数据集清点 / 结构共享）。它预告的是「按下去会不会被 ``validated_config`` 拦」，所以两个数
取自 spin box 此刻的值与当前候选解的自由参数个数，不是报告里那条已经跑完的链。

判据的数全部出自替身：6 个自由参数、16 walkers，而报告里那条旧链是 4 walkers。设计稿写的是
9 与 32——三处两两不同，照抄哪一处哪一处红。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import QLabel

import xrr_fitter.api as api

METHOD_NAMES = ("相关矩阵", "自助抽样", "Profile 似然", "MCMC 后验")
SAMPLING_ROW = len(METHOD_NAMES) - 1

# 候选解的自由参数个数。报告里那条链只有 2 个参数名（见 ``_mcmc``）：页脚读错了出处就是「2→16」。
FREE_COUNT = 6
# spin box 此刻的值。下界 2·6+2 = 14，所以 16 满足、8 不满足。
WALKERS = 16
SHORT_WALKERS = 8
# 报告里那条跑完的链。与 ``WALKERS`` 不同，好让「采样中报的是哪一条」有个明确答案。
REPORT_WALKERS = 4

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)

WALKERS_RULE_PREFIX = "walkers ≥ 2·n_free + 2"
DATASET_TALLY = "共 1 个数据集 · 全部可拟合"


def _profile(name: str) -> api.ParameterProfile:
    return api.ParameterProfile(
        name=name,
        values=np.array([1.0, 2.0, 3.0]),
        objectives=np.array([2.0, 1.0, 2.0]),
        lower_closed=True,
        upper_closed=True,
    )


def _mcmc(walkers: int = REPORT_WALKERS) -> api.McmcReport:
    return api.McmcReport(
        config=api.McmcConfig(walkers=walkers, burn_in=2, production_steps=4),
        child_seed=7,
        parameter_names=("component.0.thickness_a", "instrument.scale"),
        samples_physical=np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]]),
        log_probability=np.zeros(4),
        acceptance_fraction=np.linspace(0.2, 0.8, walkers),
        split_rhat=np.array([1.05, 1.12]),
        effective_sample_size=np.array([120.0, 80.0]),
        boundary_hits=(),
        candidate_id="candidate-a",
    )


def _report(**changes) -> api.UncertaintyReport:
    values = {
        "correlation_names": ("scale", "thickness"),
        "correlation_matrix": np.eye(2),
        "profiles": (),
        "bootstrap_intervals": (),
        "bootstrap_failure_rate": 0.0,
        "boundary_hits": (),
        "strong_correlations": (),
        "systematic_residual": False,
        "diagnostics": (),
        "bootstrap_performed": False,
        "candidate_id": "candidate-a",
    }
    values.update(changes)
    return api.UncertaintyReport(**values)


def _full_report() -> api.UncertaintyReport:
    """四样证据都齐的报告——``walked`` 到顶，第四行本来会被读成 ``done``。"""
    return _report(bootstrap_performed=True, profiles=(_profile("scale"),), mcmc=_mcmc())


def _rows(report, **changes):
    from xrr_fitter.gui.navigation.methods import uncertainty_method_rows

    return uncertainty_method_rows(report, **changes)


def _project(tmp_path, *, uncertainty=None):
    """一份跑完拟合的工程，候选解有 6 个自由参数。

    数据是真写到盘上的（``add_dataset`` 会算源文件的 sha256，而页脚那句清点要读得出「可拟合」）。
    ``unit_vector`` 拉到 6 维是为了让页脚那两个数彼此不同：``FitCandidate`` 不校验它的长度，
    而 ``start_mcmc`` 正是用 ``len(candidate.unit_vector)`` 去喊 ``validated_config``。
    """
    from tests.support.model_cases import fit_candidate, fit_result

    path = tmp_path / "curve.xy"
    size = 64
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(size)) + "\n",
        encoding="utf-8",
    )
    project = api.add_dataset(api.new_project(), path, api.InstrumentSpec())
    project = api.set_structure(
        project,
        project.datasets[0].dataset_id,
        api.StructureSpec(AIR, (api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0),), SI),
    )
    candidate = replace(
        fit_candidate("candidate-a", 1.83),
        unit_vector=np.full(FREE_COUNT, 0.5),
        qz_a_inv=np.linspace(0.015, 0.25, size),
        model_normalized=np.geomspace(0.9, 2e-5, size),
        log_residuals_decades=np.full(size, 0.1),
        weighted_residuals=np.zeros(size),
    )
    result = api.FitResult.from_search(
        fit_result(candidate),
        confidence=api.ConfidenceClass.TRUSTED,
        uncertainty=uncertainty,
        classification_evidence=(),
    )
    project = replace(project, datasets=(replace(project.datasets[0], last_valid_result=result),))
    return api.select_active_dataset(project, project.datasets[0].dataset_id)


def _nav(qtbot, project):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.navigation.panel import PipelineNav

    nav = PipelineNav(ProjectDocument(project))
    qtbot.addWidget(nav)
    return nav


def _window(qtbot, project):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def _uncertainty_window(qtbot, tmp_path):
    """停在帧⑤ 那一屏上的主窗口，spin box 已经调到满足下界的那一档。"""
    window = _window(qtbot, _project(tmp_path, uncertainty=_full_report()))
    window.plot_panel.select_view("uncertainty")
    window.result_panel.walkers.setValue(WALKERS)
    return window


def _dot(nav) -> QLabel:
    marker = nav.findChild(QLabel, f"pipelineDot_{METHOD_NAMES[SAMPLING_ROW]}")
    assert marker is not None, METHOD_NAMES[SAMPLING_ROW]
    return marker


class _SamplingController:
    """一条正在采的链。

    ``FitController.is_running`` 是读 ``_job`` 的 property，所以伪造运行态得整体换掉这个对象
    （先例：``test_fit_progress`` / ``test_project_document``）。``refresh_operation_state``
    只问它这一件事。
    """

    is_running = True


# ----------------------------------------------------------------- 四行的读数


def test_a_live_chain_makes_the_fourth_row_the_step_you_are_on() -> None:
    """链在采：第四行是 ``current``，小字报这条链的 walkers 数与「采样中」。

    这条报告四样证据都齐（跑过一轮再跑第二轮），``walked`` 因此到顶——按「缺席的第一样是当前
    那一步」算的话第四行是 ``done``，读者于是在链正采的时候看见一行「已完成」。「在跑」不是
    报告里的字段，它只能从外面推进来。
    """
    rows = _rows(_full_report(), sampling_walkers=WALKERS)

    assert [row.state for row in rows] == ["done", "done", "done", "current"]
    assert rows[SAMPLING_ROW].caption == f"{WALKERS} walkers · 采样中"


def test_the_sampling_row_wears_the_half_filled_dot_frame_four_already_uses() -> None:
    """圆点是 ``◐``，与帧④ 数据集行「在跑」那一枚同一个字形。

    设计稿里 ``◐`` 只出现两处（帧④ 的三行数据集、帧⑤ 这一行），语义都是「在飞行中」。字形写在
    纯函数里而不是留给控件即兴决定：这一行的「序号 / ✓ / ◐」三选一是同一个判定的三个面。
    """
    from xrr_fitter.gui.data.dataset_card import RUNNING_GLYPH

    rows = _rows(_full_report(), sampling_walkers=WALKERS)

    assert rows[SAMPLING_ROW].glyph == RUNNING_GLYPH == "◐"
    assert [row.glyph for row in rows[:SAMPLING_ROW]] == ["✓", "✓", "✓"]


def test_sampling_takes_the_current_marker_off_the_row_that_held_it() -> None:
    """自助抽样还没跑而链已经在采：``current`` 归第四行，第二行退回 ``pending``。

    两处 ``current`` 会把「读者现在在哪一步」问成两个答案。判定顺序因此是「在采的那一行先认
    ``current``」，剩下的才按报告算——反过来写（先算 ``walked`` 再看采样）就是两枚亮点。
    """
    states = [row.state for row in _rows(_report(profiles=(_profile("scale"),)), sampling_walkers=WALKERS)]

    assert states.count("current") == 1
    assert states[SAMPLING_ROW] == "current"
    assert states[1] == "pending"


def test_an_idle_sampler_leaves_the_four_rows_reading_the_report() -> None:
    """没在采样时四行照报告读——``sampling_walkers=None`` 与不传是同一件事。

    新参数的默认值要是漏了，或者 ``None`` 走进了「在采」那一支，帧① 的左栏会跟着变。
    """
    report = _full_report()

    assert _rows(report, sampling_walkers=None) == _rows(report)
    assert _rows(report)[SAMPLING_ROW].caption == f"{REPORT_WALKERS} walkers"


# --------------------------------------------------------------------- 左栏接线


def test_the_rail_paints_the_sampling_row_the_way_the_reading_says(qtbot, tmp_path) -> None:
    """圆点、标题、小字三样一起换——设计稿那一行是 ``.pstep.current`` 整行变色。

    圆点与标题各挂一份 ``stepState``（``test_visual_contracts`` 盯着两者一致）：只改一处，那一行
    会读成「亮点配着灰字」。
    """
    nav = _nav(qtbot, _project(tmp_path, uncertainty=_full_report()))

    nav.set_sampling_walkers(WALKERS)

    marker = _dot(nav)
    label = nav.findChild(QLabel, f"pipelineLabel_{METHOD_NAMES[SAMPLING_ROW]}")
    assert marker.text() == "◐"
    assert (marker.property("stepState"), label.property("stepState")) == ("current", "current")
    assert nav.method_captions()[SAMPLING_ROW] == f"{WALKERS} walkers · 采样中"


def test_the_rail_goes_back_to_the_report_once_the_chain_stops(qtbot, tmp_path) -> None:
    """采完了那一行回到报告原样：✓ 与那条链自己的 walkers 数。

    与上一条是同一枚开关的两面。只钉「开始时会换」的话，实现可以换完不换回来——链停了半小时，
    左栏还在报「采样中」。
    """
    nav = _nav(qtbot, _project(tmp_path, uncertainty=_full_report()))

    nav.set_sampling_walkers(WALKERS)
    nav.set_sampling_walkers(None)

    assert _dot(nav).text() == "✓"
    assert _dot(nav).property("stepState") == "done"
    assert nav.method_captions()[SAMPLING_ROW] == f"{REPORT_WALKERS} walkers"


def test_a_running_sampler_pushes_its_walkers_to_the_rail(qtbot, tmp_path) -> None:
    """MCMC 一跑起来，左栏第四行就换成 ``◐ · 采样中``。

    推的是 spin box 里那个数：运行期间那几个 spin box 是禁用的（``set_operation_state``），所以
    它此刻的值正是这条活链的 walkers 数，而报告要等采完才有。
    """
    from xrr_fitter.gui.operation_state import refresh_operation_state

    window = _uncertainty_window(qtbot, tmp_path)
    live = window.result_panel.controller
    window.result_panel.controller = _SamplingController()

    refresh_operation_state(window)

    caption = window.pipeline_nav.method_captions()[SAMPLING_ROW]
    glyph = _dot(window.pipeline_nav).text()
    # 关窗时 ``closeEvent`` 会问每个控制器在不在跑、跑着就喊 ``cancel``：读数取到手就把真的换回去。
    window.result_panel.controller = live
    assert caption == f"{WALKERS} walkers · 采样中"
    assert glyph == "◐"


def test_the_rail_stops_reporting_sampling_when_the_run_ends(qtbot, tmp_path) -> None:
    """链停了再刷一次，左栏交还给报告。

    实现要是不判「在跑没跑」、一律把 spin box 的值推过去，上一条照样为绿：那个数一直在那里。
    """
    from xrr_fitter.gui.operation_state import refresh_operation_state

    window = _uncertainty_window(qtbot, tmp_path)
    live = window.result_panel.controller
    window.result_panel.controller = _SamplingController()
    refresh_operation_state(window)

    window.result_panel.controller = live
    refresh_operation_state(window)

    assert window.pipeline_nav.method_captions()[SAMPLING_ROW] == f"{REPORT_WALKERS} walkers"


# --------------------------------------------------------------------- 左栏页脚


def test_the_uncertainty_view_puts_the_walkers_rule_in_the_left_footer(qtbot, tmp_path) -> None:
    """帧⑤ 的页脚是那句 walkers 下界，两个数一个来自候选解、一个来自 spin box。

    ``（6→16）`` 里的 6 是候选解的自由参数个数，而这份报告里那条链只登记了 2 个参数名——读错
    出处会写成「2→16」，而下界判定跟着一起错（2·2+2=6，几乎永远「已满足」）。
    """
    window = _uncertainty_window(qtbot, tmp_path)

    assert window.dataset_summary.text() == f"{WALKERS_RULE_PREFIX} · 已满足（{FREE_COUNT}→{WALKERS}）"


def test_leaving_the_uncertainty_view_gives_the_dataset_tally_back(qtbot, tmp_path) -> None:
    """换回别的分析页，页脚回到这一栏本来那句清点。

    页脚在设计稿里逐帧换人（帧① 清点、帧③ 结构共享、帧⑤ walkers），帧②④⑥⑦ 干脆没有。把
    walkers 那句挂成常驻，帧① 的页脚就再也说不出「几个数据集可拟合」。
    """
    window = _uncertainty_window(qtbot, tmp_path)

    window.plot_panel.select_view("candidates")

    assert window.dataset_summary.text() == DATASET_TALLY


def test_the_footer_reads_the_spin_box_rather_than_the_chain_in_hand(qtbot, tmp_path) -> None:
    """把 walkers 调到下界以下，页脚当即改口说「未满足」。

    读报告里那条跑完的链的话，「已满足」是一句恒真的话——报告存在就说明它当初过了
    ``validated_config``。这一句的用处是预告下一次：按下 ``▶ 运行 MCMC`` 会不会被拦。
    """
    window = _uncertainty_window(qtbot, tmp_path)

    window.result_panel.walkers.setValue(SHORT_WALKERS)

    assert window.dataset_summary.text() == f"{WALKERS_RULE_PREFIX} · 未满足（{FREE_COUNT}→{SHORT_WALKERS}）"
