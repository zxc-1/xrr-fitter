"""The inspector and canvas show the step the project is on, not every step at once.

设计稿的六帧不是一张页面的六种滚动位置：每一帧的右栏都恰好三段，画布也跟着换内容。帧③
（结构）右栏是选中层 / 参数化 / 结构诊断，画布是层堆叠加 SLD 剖面；帧①（结果）右栏是
拟合判定 / 参数·结果值 / 候选解，画布只有反射率和残差两张图，层堆叠不在画面里；帧④
（拟合中）右栏只剩运行那一段。

在这之前三栏是「全都常驻」：五段检视器一栏堆到 1900px（视口 815px），层堆叠永远压在
绘图栈上方。于是不管项目走到哪一步，屏幕都是同一张合页——设计稿的四帧在实现里塌成了
一帧。这个文件把「哪一步露哪几段」钉住。
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QWidget
from tests.gui.plot_support import _project_with_curves
from tests.support.model_cases import dataset_project, final_fit_result

import xrr_fitter.api as api
from xrr_fitter.gui.window_layout import STEP_INSPECTOR_SECTIONS, apply_step_scope


def _window(qtbot, project: api.XrrProject | None = None):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument() if project is None else ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def _visible_cards(window) -> list[str]:
    """The inspector cards on screen, top to bottom."""
    column = window.inspector_column
    cards = [
        card
        for name in _all_section_names()
        if (card := column.findChild(QFrame, name)) is not None and card.isVisibleTo(column)
    ]
    cards.sort(key=lambda card: card.mapTo(column, card.rect().topLeft()).y())
    return [card.objectName() for card in cards]


def _all_section_names() -> tuple[str, ...]:
    seen: list[str] = []
    for names in STEP_INSPECTOR_SECTIONS.values():
        for name in names:
            if name not in seen:
                seen.append(name)
    return tuple(seen)


def _fitted_project() -> api.XrrProject:
    """A project whose active dataset carries a result, i.e. the 结果 step.

    ``base_directory`` 得给：数据集的 source_path 是相对路径，构造窗口时的来源校验会解析它。
    """
    value = api.XrrProject.new((dataset_project(result=final_fit_result()),), master_seed=1201)
    return replace(value, base_directory="/private/tmp")


def _structure_project() -> api.XrrProject:
    """有结构、还没有结果的项目：画布上半是层堆叠、下半是 SLD 剖面的那一屏（帧③）。

    活动数据集得显式选：没有它绘图面板停在空态页，四段一个都不画——而设计稿这几帧的左栏
    数据集树都有一条高亮行，状态栏也报着「活动数据集」。
    """
    value = api.XrrProject.new((dataset_project(),), master_seed=1201)
    return api.select_active_dataset(replace(value, base_directory="/private/tmp"), "curve")


def test_the_result_step_shows_the_three_sections_frame_one_draws(qtbot) -> None:
    """帧① 右栏就是 ``inspectorResults`` 一张卡，那三段全在它里面。

    「选中层」在结果态是纯噪声：这一步读的是判定和它背后的数字，而画布上此刻也没有层
    堆叠可点。留着它，读者要滚过两张跟当前问题无关的卡才够到判定。

    参数化与控制同样不在这一帧。设计稿帧① 的右栏数得出三段——拟合判定 / 参数 · 结果值 /
    候选解——而这三段是 ``ResultsPanel`` 自己画的，也就是这一张卡的内容。再挂上「参数化」
    等于把帧③ 那张编辑卡搬到结果态：它写的是交给求解器的初值，与上面刚读完的结果值同一
    个量级、不同的含义，两张表叠在一栏里读者要先判断哪张是结果。「控制」此前留着的理由是
    「重跑一次和强制停止都在那张卡上，不在这里就再也够不着」——命令栏有 ``⚡ 一键拟合``、
    拟合菜单有取消与强制停止，所以那个理由已经不成立。
    """
    window = _window(qtbot, _fitted_project())

    assert window.pipeline_nav.current_step_index() == 4
    assert _visible_cards(window) == ["inspectorResults"]


def _section_headings(window, name: str) -> list[str]:
    """一张检视器卡里画着的段落抬头，自上而下。"""
    card = window.inspector_column.findChild(QFrame, name)
    headings = [
        label for label in card.findChildren(QLabel) if label.property("sectionHeader") and label.isVisibleTo(card)
    ]
    headings.sort(key=lambda label: label.mapTo(card, label.rect().topLeft()).y())
    return [label.text() for label in headings]


def test_the_result_card_draws_the_three_headings_the_design_has_and_no_fourth(qtbot) -> None:
    """帧① 右栏是三个平级的 ``insp-sec``，没有把它们收进第四层抬头。

    实测把 ``ResultsPanel`` 那三张 ``titled_card`` 又套进一张叫「拟合结果」的卡里：屏幕上
    是两重边框，加一句设计稿没有的抬头。四句抬头里只有三句对应得上设计稿的段落，读者
    要先猜「拟合结果」和「拟合判定」是不是同一件事。
    """
    window = _window(qtbot, _fitted_project())

    assert _section_headings(window, "inspectorResults") == ["拟合判定", "参数 · 结果值", "候选解"]


def test_the_result_card_has_no_border_of_its_own_around_the_three_sections(qtbot) -> None:
    """外层不再是 ``sectionCard``：三段各自有边框，再套一层就是双线。"""
    window = _window(qtbot, _fitted_project())

    card = window.inspector_column.findChild(QFrame, "inspectorResults")
    assert card is not None
    assert card.property("sectionCard") is not True


def test_the_result_step_leaves_the_panels_extra_controls_out_of_the_inspector(qtbot) -> None:
    """设计稿帧① 右栏三段之外什么都没有：没有清除按钮、没有那段散文、没有状态行。

    这些控件本身要留着（清除结果与不确定度分析改从拟合菜单进），但它们不在设计稿画着的
    右栏里。留在栏里的话，三段读完下面还接着五件东西，「右栏恰好三段」就只是句注释。
    """
    window = _window(qtbot, _fitted_project())
    panel = window.result_panel

    for widget in (
        panel.automatic_points,
        panel.automatic_uniformity,
        panel.clear_button,
        panel.uncertainty,
        panel.uncertainty_button,
        panel.status_label,
    ):
        assert widget.isVisibleTo(window) is False, widget.objectName()


def test_the_result_card_keeps_the_designs_section_order(qtbot) -> None:
    """判定 → 参数 · 结果值 → 候选解。实测把两张自动模式的表插在后两段之间。"""
    window = _window(qtbot, _fitted_project())
    card = window.inspector_column.findChild(QFrame, "inspectorResults")

    order = [
        section.objectName()
        for section in sorted(
            (
                card.findChild(QFrame, "resultConfidenceCard"),
                card.findChild(QFrame, "resultValuesCard"),
                card.findChild(QFrame, "resultCandidatesCard"),
            ),
            key=lambda section: section.mapTo(card, section.rect().topLeft()).y(),
        )
    ]
    assert order == ["resultConfidenceCard", "resultValuesCard", "resultCandidatesCard"]


def test_the_structure_step_shows_the_three_sections_frame_three_draws(qtbot) -> None:
    """帧③ 右栏：选中层 / 参数化 / 结构诊断。候选解不在这一帧。

    还没拟合过就没有候选解可看，一张常驻空态卡占的是「改完这一层去哪儿复核」那句诊断
    的位置。
    """
    window = _window(qtbot)

    assert "inspectorResults" not in _visible_cards(window)
    assert _visible_cards(window)[:2] == ["inspectorSelectedLayer", "inspectorParameters"]
    assert "inspectorStructureDiagnostics" in _visible_cards(window)


def test_the_running_step_narrows_the_inspector_to_the_run_itself(qtbot) -> None:
    """帧④ 右栏只报这一次运行：实时指标与各数据集目标值，加控制。

    跑起来之后参数表和候选解都是上一轮的旧值，摆在进度旁边会被读成本轮的读数。
    """
    window = _window(qtbot, _fitted_project())

    window.fit_panel.running_changed.emit(True)

    assert _visible_cards(window) == ["inspectorLiveMetrics", "inspectorFit"]

    window.fit_panel.running_changed.emit(False)

    assert "inspectorResults" in _visible_cards(window)


def test_the_result_step_takes_the_layer_stack_out_of_the_canvas(qtbot) -> None:
    """帧① 的画布是反射率加残差两张图，画面里没有层堆叠。

    层堆叠是 ``central_stack`` 的兄弟，换页换不掉它；不显式收起，结果态的画布上半永远
    压着一个此刻没人要编辑的结构编辑器，两张图各自被挤掉一半高度。
    """
    window = _window(qtbot, _fitted_project())

    stack_pane = window.canvas_column.findChild(QWidget, "structurePanelScroll")
    assert stack_pane is not None
    assert stack_pane.isVisibleTo(window) is False
    assert window.plot_panel.isVisibleTo(window.canvas_column) is True


def test_the_structure_step_keeps_the_layer_stack_in_the_canvas(qtbot) -> None:
    """帧③ 的画布上半就是层堆叠，改一层看着下面的剖面动。"""
    window = _window(qtbot)

    stack_pane = window.canvas_column.findChild(QWidget, "structurePanelScroll")
    assert stack_pane is not None
    assert stack_pane.isVisibleTo(window) is True


def test_the_scoped_inspector_no_longer_overflows_its_viewport(qtbot) -> None:
    """三段装得下 815px 视口，这是步骤化真正解掉的那件事。

    五段常驻时内容 1900px、视口 815px，2.3 倍溢出，于是「哪一段落在折叠线下」只能靠
    滚动去救。按步骤只露三段之后，结果态那一栏根本不溢出——判定不需要被滚出来，它本来
    就在第一位且可见。
    """
    window = _window(qtbot, _fitted_project())

    scroll = window.inspector_column.findChild(QScrollArea, "inspectorScroll")
    assert scroll is not None
    verdict = window.result_panel.findChild(QFrame, "resultConfidenceCard")
    assert verdict is not None
    top = verdict.mapTo(scroll.viewport(), verdict.rect().topLeft()).y()
    assert 0 <= top < scroll.viewport().height(), f"判定卡顶边 y={top}，视口 {scroll.viewport().height()}px"


def test_the_result_step_canvas_holds_only_reflectivity_and_residual(qtbot) -> None:
    """帧① 的画布是两张图：对数反射率与加权残差。

    绘图栈的四段（反射率页 / 残差 / 分析页 / SLD 剖面）此前全都常驻，按 3:2:2:2 分同一
    个 ~800px 的画布，每段落到 180px 上下。180px 装不下一张带坐标轴和图例的图：帧⑤ 的
    相关矩阵和 Profile 似然叠在一起连标签都读不出来，SLD 剖面干脆被挤到折叠线以下。设计
    稿每一帧的画布都只有两张卡，所以这里也按步骤只留该露的那两段。
    """
    window = _window(qtbot, _fitted_project())
    panel = window.plot_panel

    assert panel.canvas_pane_keys() == ("reflectivity", "residual")
    assert panel.sld_pane.isVisibleTo(panel) is False


def test_the_structure_step_canvas_holds_the_sld_profile(qtbot) -> None:
    """帧③ 的画布下半是 SLD 深度剖面——上半的层堆叠改一层，它当场跟着动。

    反射率与残差在这一步是上一轮的旧曲线，摆在层堆叠下面会被读成「改完之后的样子」。
    """
    window = _window(qtbot, _structure_project())
    panel = window.plot_panel

    # 有结构、还没有结果，也就是画布上半是层堆叠的那几步之一。
    assert window.pipeline_nav.current_step_index() in (1, 2)
    assert panel.canvas_pane_keys() == ("sld",)
    assert panel.residual_pane.isVisibleTo(panel) is False
    # 而且是真画出来了，不是只留在 pane 列表里：这一步的画布只有剖面这一段，它不露就
    # 是一整块空白。默认项目的 ``显示高级选项`` 是关的，所以这条断言同时钉住「剖面不
    # 归高级选项管」。
    assert panel.sld_pane.isVisibleTo(panel) is True


def test_selecting_an_analysis_view_gives_it_the_residual_s_place(qtbot) -> None:
    """帧⑤ 那一屏：选到分析页，残差让位——分析页是另一种看法，不是第三张图。

    四段常驻时分析页只分到 ~180px，相关矩阵和 Profile 似然两张子图挤在一起，标签互相压
    掉。让残差退场之后中栏整段都归它，读得出坐标轴——那一段真实像素归
    ``test_visual_contracts`` 里那条最小窗口的量法钉，这里只钉「哪几段该露」。

    反射率也一并让位：设计稿帧⑤ 的 ``.canvas-top`` 只有四个标签（相关矩阵 / Profile 似然
    / SLD 可信带 / MCMC 后验），尾部空着——模式条是帧① 才有的角落控件。收起标签页不会把
    人困住：回反射率那一组走菜单「视图」，它不经过标签页；滚轮缩放直接连在画布上。
    """
    window = _window(qtbot, _fitted_project())
    panel = window.plot_panel

    panel.select_view("uncertainty")

    assert panel.canvas_pane_keys() == ("analysis",)
    assert panel.residual_pane.isVisibleTo(panel) is False
    assert panel.analysis_tabs.minimumHeight() >= 300


def test_leaving_the_analysis_view_brings_the_residual_back(qtbot) -> None:
    """回到反射率页，残差跟着回来——帧① 判读的就是这两张图。"""
    window = _window(qtbot, _fitted_project())
    panel = window.plot_panel
    panel.select_view("uncertainty")

    panel.select_view("log")

    assert panel.canvas_pane_keys() == ("reflectivity", "residual")


def test_the_running_step_canvas_drops_the_stale_companions(qtbot) -> None:
    """帧④ 的画布是总进度加实时反射率，残差和分析页都不在这一帧。

    跑到一半时残差与候选解比较画的都是上一轮的收敛结果，和正在动的进度摆在一起会被读
    成本轮读数——这和右栏在运行中收到只剩运行那一段是同一个理由。
    """
    window = _window(qtbot, _fitted_project())

    window.fit_panel.running_changed.emit(True)
    panel = window.plot_panel

    assert panel.canvas_pane_keys() == ("reflectivity",)
    assert panel.residual_pane.isVisibleTo(panel) is False
    assert panel.analysis_tabs.isVisibleTo(panel) is False


def test_the_running_run_reaches_the_dataset_cards_in_the_left_rail(qtbot) -> None:
    """帧④ 左栏那三行小字写着这一次运行的 J——所以进度事件得走到数据面板。

    卡片自己会说「拟合中 · J=…」，但只有窗口把运行状态和进度接过去，真跑起来的时候
    左栏才跟着动；不接线，那段文字在应用里永远不出现。
    """
    window = _window(qtbot, _fitted_project())

    window.fit_panel.running_changed.emit(True)
    window.fit_panel.controller.progress_changed.emit(
        api.FitProgress(dataset_id=None, stage="refine", completed=620, total=1000, best_objective=2.14, message="")
    )

    assert window.data_panel.run_subline("curve") == "拟合中 · J=2.14"

    window.fit_panel.running_changed.emit(False)

    assert window.data_panel.run_subline("curve") == ""


def _shared_stack_project(tmp_path) -> api.XrrProject:
    """两条真在磁盘上的曲线，拿着同一叠层：设计稿帧③ 左栏那种「结构共享」的局面。

    源文件得真存在——``ProjectDocument`` 构造时就跑来源校验，替身路径会被判成「源文件缺失」，
    于是一条可拟合的都没有，而页脚那句话的前提正是「有可拟合的数据集」。
    """
    from tests.support.model_cases import simple_structure

    value = _project_with_curves(tmp_path, 2)
    for dataset in value.datasets:
        value = api.set_structure(value, dataset.dataset_id, simple_structure())
    return api.select_active_dataset(value, value.datasets[0].dataset_id)


def test_the_structure_step_reaches_the_footer_under_the_dataset_cards(qtbot, tmp_path) -> None:
    """帧③ 左栏底下那句话得真跟着步骤换——面板自己会写，但要有人把步骤递进去。

    ``apply_step_scope`` 是步骤扇出的唯一一处：右栏分段、层堆叠、画布、状态栏都在这里换。
    页脚不接到这条边上，那句「结构对全部可拟合数据集共享」在应用里永远不出现，只在单测里成立。
    """
    window = _window(qtbot, _shared_stack_project(tmp_path))

    apply_step_scope(window, 1)

    assert window.data_panel.summary_label.text() == "结构对全部可拟合数据集共享 · 每集独立仪器/标度"

    apply_step_scope(window, 4)

    assert window.data_panel.summary_label.text() == "共 2 个数据集 · 全部可拟合"
