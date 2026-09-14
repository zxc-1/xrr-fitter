"""运行中右栏的两段读数：实时指标，以及各数据集目标值。

设计稿帧④ 的右栏是三段——实时指标 / 各数据集目标值 / 控制。控制那段由 ``FitPanel``
自己画，前两段在这里。指标行的规矩是"只报求解器真数得出来的量"：``FitProgress`` 的
遥测尾巴每个字段都可以是 ``None``，缺的必须显示破折号，因为把缺量画成 0 就是一个看着
像读数的编造值。

目标值那张表有三种读法，测试三种都钉住：独立批量下每个数据集各自报自己的 J；联合批量
下后端给了每数据集分解就报各自的 J；分解缺席时才退回共享的那一个数。
"""

from __future__ import annotations

import xrr_fitter.api as api


def _view(qtbot):
    from xrr_fitter.gui.fitting.metrics import LiveMetricsView

    view = LiveMetricsView()
    qtbot.addWidget(view)
    return view


def _progress(
    stage: str,
    best: float,
    dataset_id: str | None = None,
    **telemetry: object,
) -> api.FitProgress:
    return api.FitProgress(
        dataset_id=dataset_id,
        stage=stage,
        completed=7,
        total=18,
        best_objective=best,
        message="",
        **telemetry,
    )


def test_the_metrics_name_the_stage_and_the_objective(qtbot) -> None:
    """当前阶段读的是阶段的中文名，不是求解器报的那个单字母键。"""
    view = _view(qtbot)
    view.set_progress(_progress("D", 2.61))

    assert view.metric_text("当前阶段") == "粗糙度与仪器精修 D"
    assert view.metric_text("全局目标 J") == "2.6100"
    assert view.metric_text("阶段进度") == "7 / 18"


def test_the_solver_telemetry_rows_read_a_dash_until_the_solver_publishes_them(qtbot) -> None:
    """求解器没报的指标显示破折号，不显示 0。

    一次性池扫描没有代数、``least_squares`` 没有接受率、MCMC 没有独立的函数评估计数。
    这些行的值是 ``None``，界面必须让"没有这个读数"和"读数是零"看起来不一样——把缺量画成
    0 就是四个看着像读数的编造值。
    """
    view = _view(qtbot)
    view.set_progress(_progress("A", 2.61, nfev=204))

    assert view.metric_text("迭代次数") == "—"
    assert view.metric_text("函数评估") == "204"
    assert view.metric_text("接受率") == "—"
    assert view.metric_text("步长") == "—"


def test_the_solver_telemetry_rows_report_what_the_sampler_published(qtbot) -> None:
    """采样器报出的步数、接受率与提议步长照原样显示。"""
    view = _view(qtbot)
    view.set_progress(
        _progress("MCMC", 2.61, iteration=18204, acceptance_rate=0.34, step_size=0.03125),
    )

    assert view.metric_text("迭代次数") == "18,204"
    assert view.metric_text("函数评估") == "—"
    assert view.metric_text("接受率") == "0.34"
    assert view.metric_text("步长") == "0.03125"


def test_a_joint_run_splits_the_objective_when_the_backend_reports_the_breakdown(qtbot) -> None:
    """后端给了每数据集分解，表就报各自的 J 和各自的趋势，而不是共享的那一个数。

    联合运行只有一个全局 J，但每条曲线各自的残差是求解器算过的——``JointEvaluation``
    每个成员一份。既然拿得到，就不该让三行显示同一个数：那会把"400 °C 拖后腿"这件事
    藏起来。
    """
    view = _view(qtbot)
    view.set_members(("aSi_25C", "aSi_200C", "aSi_400C"))
    view.set_progress(
        _progress("C", 2.61, dataset_objectives=(("aSi_25C", 2.2), ("aSi_200C", 2.7), ("aSi_400C", 3.1))),
    )
    view.set_progress(
        _progress("D", 2.55, dataset_objectives=(("aSi_25C", 2.14), ("aSi_200C", 2.7), ("aSi_400C", 3.4))),
    )

    assert view.objective_rows() == (
        ("aSi_25C", "2.1400", "↓ 下降"),
        ("aSi_200C", "2.7000", "≈ 平台"),
        ("aSi_400C", "3.4000", "↑ 上升"),
    )


def test_the_objective_reports_a_plateau_when_it_stops_moving(qtbot) -> None:
    """趋势是两次读数之间比出来的，第一次没有可比对象，所以只有第二次起才有箭头。"""
    view = _view(qtbot)
    view.set_progress(_progress("D", 2.61))
    view.set_progress(_progress("D", 2.61))

    assert view.metric_text("全局目标 J") == "2.6100 ≈"


def test_each_dataset_keeps_its_own_objective_under_independent_batches(qtbot) -> None:
    """独立批量下数据集轮流跑，每一行报的是它自己那次的 J。"""
    view = _view(qtbot)
    view.set_progress(_progress("B", 2.14, "aSi_25C"))
    view.set_progress(_progress("B", 3.05, "aSi_400C"))
    view.set_progress(_progress("C", 1.98, "aSi_25C"))

    assert view.objective_rows() == (
        ("aSi_25C", "1.9800", "↓ 下降"),
        ("aSi_400C", "3.0500", "—"),
    )


def test_a_joint_run_falls_back_to_the_shared_objective_without_a_breakdown(qtbot) -> None:
    """后端没报分解时，成员行报的是那个共享的全局 J——联合拟合本来只有一个目标函数。"""
    view = _view(qtbot)
    view.set_members(("aSi_25C", "aSi_200C", "aSi_400C"))
    view.set_progress(_progress("D", 2.61))

    assert view.objective_rows() == (
        ("aSi_25C", "2.6100", "—"),
        ("aSi_200C", "2.6100", "—"),
        ("aSi_400C", "2.6100", "—"),
    )
