"""每一步的那行小字报的是这个项目此刻的状态，不是这一步的通用说明。

设计稿的左栏在四帧里写的都是数字：帧① 是「4 导入 · 3 可拟合 / 5 层 · 含表面氧化 / 12 个
自由参数 / 全自动 · 已收敛 / 可信 · J=1.83」，帧③ 换成「3 可拟合 / 4 层 · 编辑中 / 待复核 /
未开始」，帧④ 换成「3 联合 / 4 层共享 / 9 共享 + 6 独立 / 进行中 62%」。同一个位置在不同
项目状态下说不同的话——它是六行状态读数，而不是六行操作指南。

此前这六行是常量：不管项目里有几个数据集、结构有几层、拟合有没有跑过，「导入并选择数据
集 / 定义样品结构 / 设定参数边界与先验」照原样念一遍。于是左栏在整个流程里一个字都不变，
读者要知道「现在几层」得去右栏或画布里找。

没状态可报的那几步，设计稿写的是一个破折号：帧③ 与帧④ 的「结果 / 导出」两行都是「—」。
所以规则分三档——有状态报状态；没状态但这一步还没轮到（在项目走到的那一步之后），报
「—」；没状态而这一步正是当前能动的那一步，才退回用途那句话。用途始终留在 tooltip 里。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path, points: int = 64) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(points)) + "\n",
        encoding="utf-8",
    )
    return path


def _structure(layers: int = 1) -> api.StructureSpec:
    return api.StructureSpec(
        AIR,
        tuple(api.LayerSpec(f"film{index}", SIO2, 40.0 + index, roughness_a=3.0) for index in range(layers)),
        SI,
    )


# 表面氧化层进结构时用的名字。``services.structures`` 接受氧化建议时写下的就是这一串
# （``f"{formula} native oxide"``），中文界面只在显示时把它译过来——左栏认的也是这一串。
SURFACE_OXIDE = "SiO2 native oxide"


def _oxidised_structure(layers: int = 2, *, location: str = "surface") -> api.StructureSpec:
    """一叠带氧化层的层。``surface`` 把它放在最上面，``backing`` 放在基底那一侧。

    两端插进来的层名一模一样：``services.structures`` 只按 ``location`` 决定插在哪一头。
    """
    films = _structure(layers).components
    oxide = api.LayerSpec(SURFACE_OXIDE, SIO2, 34.2, roughness_a=3.0)
    return api.StructureSpec(AIR, (oxide, *films) if location == "surface" else (*films, oxide), SI)


def _project(tmp_path, *, datasets: int = 1, layers: int = 0):
    project = api.new_project()
    for index in range(datasets):
        project = api.add_dataset(
            project,
            _write_curve(tmp_path / f"curve{index}.xy"),
            api.InstrumentSpec(),
        )
    if layers:
        for dataset in project.datasets:
            project = api.set_structure(project, dataset.dataset_id, _structure(layers))
    return api.select_active_dataset(project, project.datasets[0].dataset_id)


def _window(qtbot, project=None):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    document = ProjectDocument() if project is None else ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.show()
    qtbot.wait(1)
    return window


def _sub(window, title: str) -> str:
    label = window.pipeline_nav.findChild(QLabel, f"pipelineDescription_{title}")
    assert label is not None, title
    return label.text()


def test_the_data_step_counts_what_was_imported(qtbot, tmp_path) -> None:
    """全部能拟合时设计稿只写一个数：帧③ 是「3 可拟合」，不是「3 导入 · 3 可拟合」。

    第二个数只有在它和第一个不一样时才带信息量。三条进来三条能拟合，「3 导入 · 3 可拟合」
    是同一个数说两遍，还把这一行撑到 30px 底栏那种挤度；帧① 的 4 进 3 能拟合才需要两个数，
    因为差出来的那一条正是要读者去看的东西。

    可拟合的判定沿用数据面板那一条（掩码里够 30 点），两处不能各说一套：同一个项目在左栏
    报「3 可拟合」而在数据列表里标两行「数据点不足」的话，读者不知道该信哪个。
    """
    window = _window(qtbot, _project(tmp_path, datasets=3))

    assert _sub(window, "数据") == "3 可拟合"


def test_the_data_step_spells_out_both_counts_when_one_dataset_cannot_be_fitted(qtbot, tmp_path) -> None:
    """设计稿帧① 是「4 导入 · 3 可拟合」：有一条进不了拟合，两个数才都要说。"""
    from xrr_fitter.gui.navigation.panel import FITTABLE_POINT_FLOOR

    project = api.new_project()
    for index in range(3):
        project = api.add_dataset(project, _write_curve(tmp_path / f"curve{index}.xy"), api.InstrumentSpec())
    project = api.add_dataset(
        project,
        _write_curve(tmp_path / "short.xy", points=FITTABLE_POINT_FLOOR - 1),
        api.InstrumentSpec(),
    )
    window = _window(qtbot, api.select_active_dataset(project, project.datasets[0].dataset_id))

    assert _sub(window, "数据") == "4 导入 · 3 可拟合"


def test_the_data_step_falls_back_to_its_purpose_before_the_first_import(qtbot) -> None:
    """空项目没有数可报，这一行就该继续讲这一步是干什么的。"""
    window = _window(qtbot)

    assert _sub(window, "数据") == "导入并选择数据集"


def test_the_structure_step_counts_the_layers(qtbot, tmp_path) -> None:
    """「4 层」——层数是这一步唯一的读数，而它此前只在画布和右栏里有。"""
    window = _window(qtbot, _project(tmp_path, layers=4))

    assert _sub(window, "结构") == "6 层"


def test_the_structure_step_falls_back_to_its_purpose_before_a_structure_exists(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path))

    assert _sub(window, "结构") == "定义样品结构"


def test_the_structure_step_says_the_stack_carries_a_surface_oxide(qtbot, tmp_path) -> None:
    """设计稿帧① 这一行是「5 层 · 含表面氧化」，不是「5 层」。

    氧化层和别的层不是一回事：它由软件按氧化表建议、读者点过「接受」才进来，而它一进来就
    多吃三个自由参数。左栏这一行因此替读者记着这件事——回头问「12 个自由参数」里的 12 是
    怎么来的，答案有一部分在这一行里。
    """
    project = api.set_structure(_project(tmp_path), "curve0", _oxidised_structure())
    window = _window(qtbot, project)

    assert _sub(window, "结构") == "5 层 · 含表面氧化"


def test_a_backing_side_oxide_is_not_reported_as_a_surface_one(qtbot, tmp_path) -> None:
    """基底那一侧的氧化层同名，但它不在表面上——这一行不替它说话。

    ``services.structures`` 按 ``location`` 决定把氧化层插在哪一头，两头的层名一模一样。所以
    判的是「最上面那层是不是氧化层」，而不是「这叠层里有没有」：基底侧氧化的结构写「5 层」，
    读者不会照着这一行去表面上找一层不存在的氧化。
    """
    project = api.set_structure(_project(tmp_path), "curve0", _oxidised_structure(location="backing"))
    window = _window(qtbot, project)

    assert _sub(window, "结构") == "5 层"


def test_the_parameters_step_counts_the_free_parameters_once_it_is_behind_us(qtbot, tmp_path) -> None:
    """设计稿帧① 这一行是「12 个自由参数」——走过这一步之后，报的就是那个数。

    数不用自己再读一遍源文件：参数面板为参数表已经编译过一次声明，``set_parameter_definitions``
    把自由数交给左栏，左栏读现成的。所以这一份工程的源文件必须真的在盘上——``dataset_project``
    那份替身指向一个不存在的 ``curve.xy``，参数面板一个声明也编不出来。
    """
    from dataclasses import replace

    import numpy as np
    from tests.support.model_cases import fit_candidate, fit_result

    project = _project(tmp_path, layers=2)
    # 逐点数组要和上面写下的那条曲线一样长，画布在投影结果之前会核这一条。
    size = 64
    candidate = replace(
        fit_candidate("candidate-a", 1.83),
        qz_a_inv=np.linspace(0.015, 0.25, size),
        model_normalized=np.geomspace(0.9, 2e-5, size),
        log_residuals_decades=np.full(size, 0.1),
        residuals=np.full(size, 0.1),
        weighted_residuals=np.zeros(size),
    )
    result = api.FitResult.from_search(
        fit_result(candidate),
        confidence=api.ConfidenceClass.TRUSTED,
        uncertainty=None,
        classification_evidence=(),
    )
    project = replace(project, datasets=(replace(project.datasets[0], last_valid_result=result),))
    window = _window(qtbot, project)
    definitions = window.parameters_panel.definitions
    free = sum(1 for item in definitions if not item.locked)

    # 报的是自由数而不是声明总数：这份工程里仪器那几项本来就锁着，两个数不相等。
    assert 0 < free < len(definitions)
    assert _sub(window, "参数") == f"{free} 个自由参数"


def test_locking_a_bound_does_not_turn_this_row_into_a_tally_of_locks(qtbot, tmp_path) -> None:
    """轮到这一步、还没走过时，设计稿帧③ 写的是「待复核」——锁了几个不改变这句话。

    此前这一行报的是持久化字段里那三个数（共享 / 锁定 / 先验），于是随手锁一个边界就把
    「待复核」顶掉，写成「1 锁定」。设计稿六帧里没有任何一行是这个形状：走过的那一步报自由
    数（帧① 12 个自由参数，联合时帧④ 9 共享 + 6 独立），没走过的那一步报「待复核」。锁定数
    在参数表里逐行看得见，左栏这一行的活是说这一步到哪儿了。
    """
    project = _project(tmp_path, layers=2)
    definitions = api.describe_parameters(project, "curve0")
    target = definitions[0]
    project = api.set_parameter_settings(
        project,
        "curve0",
        (api.ParameterSetting(target.name, target.initial, target.lower, target.upper, api.ParameterFreedom.FIXED),),
    )
    window = _window(qtbot, project)

    assert _sub(window, "参数") == "待复核"


def test_the_parameters_step_reads_a_dash_before_a_structure_exists(qtbot, tmp_path) -> None:
    """还没有结构，参数就无从谈起——设计稿在这种位置写的是一个破折号。

    结构立起来之后同一行改说「待复核」（设计稿帧③，见 test_pipeline_nav_navigation）：一个
    边界都没动过也是有话可说的，说的是「轮到你了」。两句话要分得开，读者才知道现在是「还没
    轮到这一步」还是「轮到了，等你看」。
    """
    window = _window(qtbot, _project(tmp_path))

    assert _sub(window, "参数") == "—"


def test_the_fit_step_reads_not_started_until_a_run_lands(qtbot, tmp_path) -> None:
    """设计稿帧③ 这一行只有两个字：「未开始」。

    此前这里写「联合 · 未开始」/「独立 · 未开始」，把批量模式又念了一遍——而模式已经在
    「数据」那行（「3 联合」）和「结构」那行（「4 层共享」）各说过一次，状态栏里还有第三次。
    同一个事实在一栏里说三遍，挤掉的是这一行本来要报的那一件事：跑没跑过。
    """
    joint = _window(qtbot, api.set_batch_mode(_project(tmp_path, datasets=3, layers=2), "joint"))
    independent = _window(qtbot, _project(tmp_path, datasets=2, layers=2))

    assert (_sub(joint, "拟合"), _sub(independent, "拟合")) == ("未开始", "未开始")


def test_the_fit_step_reads_the_designs_wording_once_a_result_exists(qtbot, tmp_path) -> None:
    """设计稿帧① 是「全自动 · 已收敛」：跑的是全自动那条路，而它收住了。

    项目里没有 ``converged`` 这个字段，「已收敛」不能凭空造一个数——它读的就是「有一份
    ``last_valid_result``」：自动拟合只在收敛后才回写结果，所以这份结果的存在本身就是那句话。
    """
    from dataclasses import replace

    from tests.support.model_cases import dataset_project, final_fit_result

    value = replace(
        api.XrrProject.new((dataset_project(result=final_fit_result()),), master_seed=1201),
        base_directory=str(tmp_path),
    )
    window = _window(qtbot, value)

    assert _sub(window, "拟合") == "全自动 · 已收敛"


def test_the_export_step_names_the_three_formats_once_there_is_something_to_export(qtbot, tmp_path) -> None:
    """设计稿帧① 是「ORSO / Excel / 图」——三个格式名，而不是一句「导出 ORSO、Excel 与图」。

    这一行报的是「导出能给你什么」，读者一眼就要数得出有几种；写成一句话之后，那三个词埋在
    「导出……与……」里，还比 264px 的栏宽长出一截被截断。
    """
    from dataclasses import replace

    from tests.support.model_cases import dataset_project, final_fit_result

    value = replace(
        api.XrrProject.new((dataset_project(result=final_fit_result()),), master_seed=1201),
        base_directory=str(tmp_path),
    )
    window = _window(qtbot, value)

    assert _sub(window, "导出") == "ORSO / Excel / 图"


def test_the_result_step_reads_the_verdict_and_its_objective(qtbot, tmp_path) -> None:
    """「可信 · J=1.83」——判定和目标值，和右栏那张判定卡说的是同一件事。"""
    from dataclasses import replace

    from tests.support.model_cases import dataset_project, final_fit_result

    value = replace(
        api.XrrProject.new((dataset_project(result=final_fit_result()),), master_seed=1201),
        base_directory=str(tmp_path),
    )
    window = _window(qtbot, value)

    text = _sub(window, "结果")
    assert text.startswith("可信 · J="), text


def test_the_two_unreached_steps_read_a_dash_before_a_fit_lands(qtbot, tmp_path) -> None:
    """设计稿帧③ 与帧④ 的「结果 / 导出」两行都是「—」，不是两句操作说明。

    这两步都还没轮到，硬报一句用途会让左栏在没结果的时候看着像有内容；破折号说的正是
    「这一步现在没有可报的」，而用途那句话退到 tooltip 里，需要的人仍然找得到。
    """
    window = _window(qtbot, _project(tmp_path, layers=2))

    assert (_sub(window, "结果"), _sub(window, "导出")) == ("—", "—")


def test_every_step_keeps_its_purpose_in_the_tooltip(qtbot, tmp_path) -> None:
    """状态顶掉了正文里的用途，tooltip 就得接住它，否则那句话在界面上彻底消失。

    「结果 / 导出」这两行更要紧：设计稿在没结果时把它们写成「—」，正文里连一个字都不留，
    tooltip 是那句用途唯一的落脚处。
    """
    window = _window(qtbot, _project(tmp_path, datasets=3, layers=4))
    nav = window.pipeline_nav

    for title, purpose in (
        ("数据", "导入并选择数据集"),
        ("结构", "定义样品结构"),
        ("结果", "查看候选解与不确定度"),
        ("导出", "导出 ORSO、Excel 与图"),
    ):
        label = nav.findChild(QLabel, f"pipelineLabel_{title}")
        assert label is not None, title
        assert label.toolTip() == purpose
        assert _sub(window, title) != purpose


def test_the_sub_line_follows_the_project_without_a_rebuild(qtbot, tmp_path) -> None:
    """加一层，左栏那行数当场变——它读的是项目，不是构造那一刻拍下来的快照。"""
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = _project(tmp_path, layers=1)
    document = ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.show()
    qtbot.wait(1)
    assert _sub(window, "结构") == "3 层"

    document.replace_project(api.set_structure(project, "curve0", _structure(3)))
    qtbot.wait(1)

    assert _sub(window, "结构") == "5 层"


def test_the_joint_run_says_the_datasets_are_fitted_as_one(qtbot, tmp_path) -> None:
    """帧④ 的这一行写「3 联合」，不是「3 导入 · 3 可拟合」。

    联合批量下这三条曲线不再是三次独立拟合，而是一个共享层结构的目标函数；「3 可拟合」
    此刻说的是一件已经不成立的事——它数的是「有几条能各自拟合」。
    """
    project = api.set_batch_mode(_project(tmp_path, datasets=3, layers=4), "joint")
    window = _window(qtbot, project)

    assert _sub(window, "数据") == "3 联合"


def test_the_joint_run_says_the_stack_is_the_shared_one(qtbot, tmp_path) -> None:
    """帧④ 写「4 层共享」：联合拟合里这一叠层是三条曲线共用的那一叠。

    独立批量下同样的层数只讲活动数据集自己的结构，两者是不同的事实。
    """
    project = api.set_batch_mode(_project(tmp_path, datasets=3, layers=4), "joint")
    window = _window(qtbot, project)

    assert _sub(window, "结构") == "6 层共享"


def test_the_structure_step_counts_the_stack_rows_the_design_draws(qtbot, tmp_path) -> None:
    """设计稿帧③ 的样品是 SiO₂ + a-Si 两层，而这一行写的是「4 层」。

    数的是层堆叠里那几行：空气 / SiO₂ / a-Si / c-Si。入射介质与基底也各占一行，各自带着
    自己的粗糙度与密度进拟合；把它们排除在外，左栏报的数就和右栏那张列表、状态栏的
    「4 层 · 9 个自由参数」对不上——同一叠层在一屏里出现两个层数，读者不知道该信哪个。
    """
    window = _window(qtbot, _project(tmp_path, layers=2))

    assert _sub(window, "结构") == "4 层"


def test_the_joint_run_splits_the_parameters_into_shared_and_independent(qtbot, tmp_path) -> None:
    """帧④ 这一行写「9 共享 + 6 独立」，不是「9 共享」。

    联合拟合把一套层结构摊给三条曲线共用，而每条曲线仍有自己的标度与本底——「独立」那
    半边正是联合模式与「三次独立拟合」的分界。只报共享数会让读者以为剩下的都被绑在了一起。

    自由数由参数面板算（它已经为参数表编译过一次），左栏读那个现成的数，而不是自己再读一
    遍源文件。
    """
    project = api.set_batch_mode(_project(tmp_path, datasets=3, layers=2), "joint")
    definitions = api.describe_parameters(project, "curve0")
    free = tuple(item.name for item in definitions if not item.locked)
    shared = free[:2]
    project = api.set_sharing_rules(
        project,
        tuple(
            api.SharingRule(
                f"shared{index}",
                tuple(api.ParameterReference(dataset.dataset_id, name) for dataset in project.datasets),
            )
            for index, name in enumerate(shared)
        ),
    )
    window = _window(qtbot, project)

    independent = (len(free) - len(shared)) * 3
    assert _sub(window, "参数") == f"2 共享 + {independent} 独立"
