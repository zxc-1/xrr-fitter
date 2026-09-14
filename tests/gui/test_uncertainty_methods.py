"""左栏「结果」那一步下面展开的不确定度方法子管线。

设计稿帧⑤ 的左栏在数据集列表下面换了一段抬头：不是「分析管线」而是「不确定度方法」，下面
四行还是同一种 ``.pstep``——相关矩阵、自助抽样、Profile 似然、MCMC 后验，前三行 ✓，第四行
是当前那一步。四种证据各自答的是同一个问题的一面（谁和谁纠缠、边界撞没撞、单参数的谷有多
宽、后验长什么样），所以它们摆成一列而不是四个并列的按钮：读者要知道自己走到了哪一种。

三态不是另算一遍，而是读报告里那四样在不在：相关矩阵有名字、自助抽样跑过、profiles 非空、
``mcmc`` 不是 ``None``。缺席的第一样是当前那一步，它后面的还没轮到。这样"哪一步没做"这件
事只有一个出处——报告本身，而不是界面自己记的一份进度。

小字报的是这份报告此刻的数，不是方法的通用说明：Profile 报覆盖了几个参数，MCMC 报几条链，
自助抽样报失败率。没跑过的那几行报「未运行」——把它们写成方法学介绍会让四行在整个流程里
一个字都不变。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import xrr_fitter.api as api

METHOD_NAMES = ["相关矩阵", "自助抽样", "Profile 似然", "MCMC 后验"]


def _profile(name: str) -> api.ParameterProfile:
    return api.ParameterProfile(
        name=name,
        values=np.array([1.0, 2.0, 3.0]),
        objectives=np.array([2.0, 1.0, 2.0]),
        lower_closed=True,
        upper_closed=True,
    )


def _mcmc(walkers: int = 4) -> api.McmcReport:
    return api.McmcReport(
        config=api.McmcConfig(walkers=walkers, burn_in=2, production_steps=4),
        child_seed=7,
        parameter_names=("component.0.thickness_a", "instrument.scale"),
        samples_physical=np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]]),
        log_probability=np.zeros(4),
        # 每条链一个接受率——``McmcReport`` 拿这个长度校验 walkers 数，所以它得跟着参数走，
        # 不能钉死成 4。
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


def _rows(report):
    from xrr_fitter.gui.navigation.methods import uncertainty_method_rows

    return uncertainty_method_rows(report)


def _states(report) -> list[str]:
    return [row.state for row in _rows(report)]


def _captions(report) -> dict[str, str]:
    return {row.title: row.caption for row in _rows(report)}


def test_the_four_methods_keep_the_order_the_design_walks_them_in() -> None:
    """相关矩阵 → 自助抽样 → Profile 似然 → MCMC 后验，由便宜到贵。

    这个顺序不是随手排的：相关矩阵是拟合末尾那个 Hessian 顺带给的，自助抽样要重跑几百次，
    Profile 要逐参数扫，MCMC 要采上万步。读者从上往下走，每往下一步换来的确定性更多、代价
    也更大，所以列的顺序本身就是建议的顺序。
    """
    assert [row.title for row in _rows(_report())] == METHOD_NAMES


def test_the_first_missing_method_is_the_one_you_are_on() -> None:
    """相关矩阵在手、自助抽样还没跑：第二行是当前，它后面两行还没轮到。"""
    assert _states(_report()) == ["done", "current", "pending", "pending"]


def test_every_method_that_left_evidence_reads_done() -> None:
    """设计稿帧⑤ 的那一屏：前三样都在，MCMC 正等着——四行里只剩它不是 ✓。"""
    report = _report(
        bootstrap_performed=True,
        bootstrap_intervals=(("scale", 0.9, 1.1),),
        bootstrap_failure_rate=0.02,
        profiles=(_profile("scale"), _profile("thickness")),
    )

    assert _states(report) == ["done", "done", "done", "current"]


def test_a_finished_mcmc_leaves_no_step_to_walk() -> None:
    """四样齐了，四行全是 ✓——这一列到底了，没有「当前」那一步。"""
    report = _report(
        bootstrap_performed=True,
        bootstrap_intervals=(("scale", 0.9, 1.1),),
        profiles=(_profile("scale"),),
        mcmc=_mcmc(),
    )

    assert _states(report) == ["done", "done", "done", "done"]


def test_no_report_at_all_puts_you_at_the_first_method() -> None:
    """拟合出了结果、不确定度一样都没算：站在第一行上，后面三行还没轮到。

    这一屏是"刚跑完拟合"的样子。四行全画成未达会让读者以为这一段与自己无关，而其中三样
    确实还不能做——它们要先有一份可信的最优解。
    """
    assert _states(None) == ["current", "pending", "pending", "pending"]


def test_a_declined_bootstrap_still_counts_as_not_run() -> None:
    """``bootstrap_performed`` 为假时那行不算做过，哪怕报告里别的字段都在。

    自助抽样是可以被跳过的（点数太少、时间不够），报告照样成形。区间为空但标记为跑过的
    情形另说：那是跑了而一个区间也没收住，属于失败率的事，不是"没跑"。
    """
    states = _states(_report(bootstrap_performed=False, profiles=(_profile("scale"),)))

    assert states[1] == "current"
    # Profile 有证据在手，可这一列是一条路：前面那一步没走，它就还没轮到自己被标成走过。
    assert states[2] == "pending"


def test_each_method_reports_the_number_this_report_actually_carries() -> None:
    """小字是读数：Profile 覆盖几个参数、MCMC 几条链、自助抽样的失败率。"""
    report = _report(
        bootstrap_performed=True,
        bootstrap_intervals=(("scale", 0.9, 1.1), ("thickness", 39.0, 41.0)),
        bootstrap_failure_rate=0.02,
        profiles=(_profile("scale"), _profile("thickness")),
        mcmc=_mcmc(walkers=32),
    )

    captions = _captions(report)
    assert captions["Profile 似然"] == "2 参数"
    assert captions["MCMC 后验"] == "32 walkers"
    assert "2%" in captions["自助抽样"]


def test_the_bootstrap_line_reads_the_resample_count_the_design_asks_for() -> None:
    """设计稿这一行写的是「200 次 · 失败 2%」——次数是基数，缺了它失败率读不出量级。

    「丢了 2%」在 200 次里是 4 次、在 20 次里是 0.4 次；后者根本不成立，可见这两个数得一起读。
    """
    report = _report(
        bootstrap_performed=True,
        bootstrap_intervals=(("scale", 0.9, 1.1), ("thickness", 39.0, 41.0)),
        bootstrap_failure_rate=0.02,
        bootstrap_sample_count=200,
    )

    assert _captions(report)["自助抽样"] == "200 次 · 失败 2%"


def test_the_bootstrap_line_falls_back_to_the_parameter_count_without_a_recorded_total() -> None:
    """这个字段之前存下的工程文件不带次数，那时报参数数——总比把「0 次」摆出来像个读数好。"""
    report = _report(
        bootstrap_performed=True,
        bootstrap_intervals=(("scale", 0.9, 1.1), ("thickness", 39.0, 41.0)),
        bootstrap_failure_rate=0.02,
    )

    assert _captions(report)["自助抽样"] == "2 参数 · 失败 2%"


def test_the_correlation_line_says_where_the_matrix_came_from() -> None:
    """相关矩阵那行报的是出处——它是 Hessian 派生的，不是采样得来的。

    这个区别决定读者该多信它：Hessian 只在最优点附近展开一次，谷子一歪，那几个 ρ 就偏。
    设计稿在这一行写的正是这句话，而不是矩阵有多大。
    """
    assert _captions(_report())["相关矩阵"] == "Hessian 派生"


def test_a_method_that_has_not_run_says_so_rather_than_explaining_itself() -> None:
    """没跑过的行报「未运行」：这一段的小字位是状态位，不是方法学介绍位。"""
    captions = _captions(_report())

    assert captions["自助抽样"] == "未运行"
    assert captions["Profile 似然"] == "未运行"
    assert captions["MCMC 后验"] == "未运行"


# --------------------------------------------------------------------- 接线

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _project(tmp_path, *, uncertainty=None):
    """一份跑完拟合的工程——左栏因此算到「结果」那一步。"""
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


def test_the_methods_band_names_itself_under_the_result_step(qtbot, tmp_path) -> None:
    """设计稿帧⑤ 的第二段抬头是「不确定度方法」，四行挂在它下面。"""
    nav = _nav(qtbot, _project(tmp_path))

    assert nav.method_heading.text() == "不确定度方法"
    assert [label.text() for label in nav.method_labels] == METHOD_NAMES


def test_the_methods_band_stays_out_of_sight_until_you_reach_the_results(qtbot, tmp_path) -> None:
    """走到「结果」才展开：四种证据在还没有结果的项目上一样都做不了。

    左栏那六步是这一栏的骨架，264px 里塞得下它们加一张数据集列表就没多少余地了。四行子步骤
    因此不是常驻的第七到第十步，而是「结果」这一步展开出来的内容。
    """
    nav = _nav(qtbot, _project(tmp_path))
    assert nav.methods_visible() is True

    nav.select_step(1)

    assert nav.methods_visible() is False


def test_the_methods_band_follows_the_report_on_the_project(qtbot, tmp_path) -> None:
    """报告里有什么，四行就报什么——界面不自己记一份进度。"""
    report = _report(
        bootstrap_performed=True,
        bootstrap_intervals=(("scale", 0.9, 1.1),),
        bootstrap_failure_rate=0.02,
        bootstrap_sample_count=200,
        profiles=(_profile("scale"), _profile("thickness")),
    )
    nav = _nav(qtbot, _project(tmp_path, uncertainty=report))

    assert nav.method_captions() == ("Hessian 派生", "200 次 · 失败 2%", "2 参数", "未运行")
