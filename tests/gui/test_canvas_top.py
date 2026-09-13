"""设计稿 ``.canvas-top``：画布顶上那一行「几个 tab，紧跟着这一页的动作」。

设计稿四帧画布都以同一行开头（帧① 反射率四视图 + 模式条、帧③ 样品结构三视图 + 三个
结构命令、帧④ 进度三视图 + 刷新徽标、帧⑤ 不确定度四视图），所以它是一个可复用的构件而
不是某一帧的装饰。这份文件钉的是这一行的两条骨架规则：

* 动作紧跟在 tab 后面，不贴右边框。设计稿里 ``.spring`` 只在 ``.cmdbar`` 和 ``.statusbar``
  里带 ``flex:1``，``.canvas-top`` 里它是一个零宽的 div——把动作推到最右是 Qt
  ``setCornerWidget`` 的行为，不是设计稿的。
* tab 与动作同处一行，所以这一行只花一段高度；帧③此前是八个按钮排成两行外加一条氧化层
  建议条，光行头就吃掉画布三段高度中的一段。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QPushButton, QSplitter, QTabBar, QToolButton, QWidget
from tests.support.model_cases import dataset_project, final_fit_result

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _panel(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.structure.panel import StructurePanel

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "sample.xy"),
        api.InstrumentSpec(instrument_id="canvas-top"),
    )
    panel = StructurePanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    panel.set_structure(
        api.StructureSpec(
            AIR,
            (
                api.LayerSpec("SiO₂ · 表面氧化层", SIO2, 34.2, roughness_a=5.1),
                api.LayerSpec("a-Si · 非晶硅薄膜", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),
            ),
            SI,
        )
    )
    return panel


def _row(panel) -> QWidget:
    row = panel.findChild(QWidget, "structureCanvasTop")
    assert row is not None, "结构画布没有设计稿那一行 .canvas-top"
    return row


def _window(qtbot, tmp_path):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "curve.xy"),
        api.InstrumentSpec(instrument_id="canvas-top"),
    )
    project = api.set_structure(
        project,
        "curve",
        api.StructureSpec(AIR, (api.LayerSpec("SiO₂", SIO2, 34.2, roughness_a=5.1),), SI),
    )
    window = MainWindow(ProjectDocument(api.select_active_dataset(project, "curve")))
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def _fitted_window(qtbot):
    """整只主窗口，活动数据集带着一份结果——分析那一组 tab 到这一步才都可选。

    与 ``_window`` 的差别只在「有没有结果」，而这正是分析组的门槛：没有结果时那五个 tab
    是收着的，``select_view`` 会直接拒掉。``base_directory`` 得给，数据集的 source_path 是
    相对路径，构造窗口时的来源校验会去解析它。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.XrrProject.new((dataset_project(result=final_fit_result()),), master_seed=1201)
    window = MainWindow(ProjectDocument(replace(project, base_directory="/private/tmp")))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_the_structure_canvas_opens_with_the_three_tabs_the_design_names(qtbot, tmp_path) -> None:
    """帧③ ``.canvas-top`` 的三个 tab，样品结构在前且是选中的那一个。

    这三个名字是画布这一列的目录：读者要先知道「这一列还能翻到 SLD 剖面和参数总览」，
    否则剖面卡就只是层堆叠下面一张没有出处的图。
    """
    panel = _panel(qtbot, tmp_path)
    tabs = _row(panel).findChild(QTabBar, "structureCanvasTabs")

    assert tabs is not None
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["样品结构", "SLD 深度剖面", "参数总览"]
    assert tabs.currentIndex() == 0


def test_the_structure_canvas_top_carries_exactly_the_designs_three_commands(qtbot, tmp_path) -> None:
    """设计稿这一行只画三个按钮：＋ 添加层 / 建议氧化层 / 周期结构…。

    此前是六个（添加普通层 · 添加周期块 · 编辑基底 · 删除 · 上移 · 下移）加上一条随建议
    出现的氧化层条，两行按钮压在层堆叠上面。设计稿把其中四个交还给行本身——排序是拖
    ⋮⋮，删除与编辑是右键行——留在行头的只有「往结构里加东西」这一类。
    """
    row = _row(_panel(qtbot, tmp_path))
    buttons = [button for button in row.findChildren(QPushButton) if button.isVisibleTo(row)]

    assert [button.text() for button in buttons] == ["＋ 添加层", "建议氧化层", "周期结构…"]


def test_the_commands_follow_the_tabs_instead_of_hugging_the_right_edge(qtbot, tmp_path) -> None:
    """设计稿 ``.canvas-top`` 里的 ``.spring`` 不带 ``flex:1``，所以它撑不开。

    Qt 的 ``setCornerWidget`` 会把这一组贴到最右——在 950px 宽的画布上那是离 tab 两百多
    像素的另一头，读者得横扫整行才找得到「＋ 添加层」和它作用的那张卡的关系。留白应该
    落在行尾，而不是插在 tab 和动作之间。
    """
    panel = _panel(qtbot, tmp_path)
    row = _row(panel)
    panel.resize(900, 400)
    panel.show()
    qtbot.waitExposed(panel)

    tabs = row.findChild(QTabBar, "structureCanvasTabs")
    buttons = [button for button in row.findChildren(QPushButton) if button.isVisibleTo(row)]
    gap = buttons[0].geometry().left() - tabs.geometry().right()
    trailing = row.width() - buttons[-1].geometry().right()

    assert 0 < gap <= 24, f"tab 与第一个动作之间空了 {gap}px"
    assert trailing > gap, "留白落在了 tab 和动作之间，而不是行尾"


def test_reordering_deleting_and_editing_moved_onto_the_row_they_act_on(qtbot, tmp_path) -> None:
    """设计稿删掉的那四个按钮变成行的右键菜单，命令本身一条不少。

    删掉按钮而不给替代路径就是砍功能；把它们挂到它们本来就作用的那一行上，选中哪一行、
    命令作用在哪一行这件事也不再需要读者自己对齐。
    """
    panel = _panel(qtbot, tmp_path)
    editor = panel.editor
    menu = editor.row_menu

    assert [action.text() for action in menu.actions()] == ["编辑基底", "删除", "上移", "下移", "忽略氧化层建议"]

    # 没选中任何一层时，三条针对层的命令都不该是可点的——菜单弹出来全灰比弹不出来更
    # 说明「先选一行」。
    editor.tree.setCurrentItem(None)
    assert [action.isEnabled() for action in menu.actions()[1:4]] == [False, False, False]

    editor.tree.setCurrentItem(editor.tree.topLevelItem(2))
    assert [action.isEnabled() for action in menu.actions()[1:4]] == [True, True, False]


def test_the_tabs_take_you_to_the_section_they_name(qtbot, tmp_path) -> None:
    """三个 tab 是目录不是分页：点一个，就把它指的那一段让到眼前。

    三段内容本来就都在画布列里，tab 藏不住任何东西——它解的是「这一列往下还有什么」在一
    屏放不下时读者看不见。所以点「SLD 深度剖面」得让剖面那一段真的有高度；否则这一行就
    只是三个不响应的名字，比不画还糟。
    """
    window = _window(qtbot, tmp_path)
    canvas = window.findChild(QSplitter, "canvasSplitter")
    # 剖面那一段是 ``central_stack``；它在 splitter 里排第几不写死——帧④ 的进度视图
    # 也住在这一列，段数会随设计变。
    sld = canvas.indexOf(window.central_stack)
    # 把它压到最小，正是读者看不见剖面的那一刻。
    sizes = [0] * canvas.count()
    sizes[0] = canvas.height()
    canvas.setSizes(sizes)
    QApplication.processEvents()
    before = canvas.sizes()[sld]

    window.structure_panel.editor.canvas_tabs.setCurrentIndex(1)
    QApplication.processEvents()

    assert canvas.sizes()[sld] > before, "点了「SLD 深度剖面」，剖面那一段没有被让出来"


def test_the_running_canvas_opens_with_the_three_tabs_frame_four_names(qtbot) -> None:
    """帧④ 画布顶栏：进度 / 实时反射率 / 目标值轨迹，加一枚刷新徽标。

    这一列在运行中有三段可看，而屏幕一次只放得下两段；没有这一行，读者看不见「往下
    还有实时反射率和目标值轨迹」。徽标报的是画面多久换一次——一条不动的曲线，究竟是
    收敛了还是界面卡住了，只有知道刷新周期才分得开。
    """
    from PySide6.QtWidgets import QLabel

    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    row = view.findChild(QWidget, "fitCanvasTop")
    assert row is not None
    tabs = row.findChild(QTabBar, "fitCanvasTabs")
    assert tabs is not None
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["进度", "实时反射率", "目标值轨迹"]
    assert tabs.currentIndex() == 0

    badge = row.findChild(QLabel, "fitRefreshBadge")
    assert badge is not None and badge.text() == "实时刷新 · 每 250 ms"


def test_the_refresh_badge_reports_the_interval_the_poller_is_actually_using(qtbot) -> None:
    """徽标读的是轮询器此刻的间隔，不是设计稿上那个写死的数。

    轮询在长时间没有事件时会退到空闲间隔；徽标仍写着活跃周期，就成了一句与画面对不上
    的保证。
    """
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    view.set_refresh_interval_ms(200)

    assert view.refresh_badge.text() == "实时刷新 · 每 200 ms"


def _running_progress(objective: float, completed: int = 500) -> object:
    import numpy as np

    import xrr_fitter.api as api

    return api.FitProgress(
        dataset_id=None,
        stage="refine",
        completed=completed,
        total=1000,
        best_objective=objective,
        message="",
        preview_qz_a_inv=np.array([0.02, 0.04, 0.06]),
        preview_model_normalized=np.array([1.0, 0.5, 0.2]),
    )


def test_the_three_tabs_each_open_their_own_page(qtbot) -> None:
    """三段是三页，不是三个只换标题的空壳。

    进度那页是总进度卡；实时反射率画本轮预览曲线；目标值轨迹把每一帧的 J 连成一条线。
    三样数据 ``FitProgress`` 每一帧都带着，所以这一列换页换的是同一次运行的三种读法。
    """
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    assert view.pages.count() == 3
    assert view.pages.currentIndex() == 0

    view.canvas_tabs.setCurrentIndex(2)

    assert view.pages.currentIndex() == 2


def test_the_live_reflectivity_page_draws_the_frame_preview(qtbot) -> None:
    """预览曲线走到实时反射率那页——这一帧模型长什么样，是运行中最要紧的一眼。"""
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    view.set_progress(_running_progress(2.14))

    preview = view.live_plot.preview_item
    assert preview is not None
    assert len(preview.getData()[0]) == 3


def test_the_objective_trace_accumulates_one_point_per_frame(qtbot) -> None:
    """目标值轨迹是这一次运行的 J 序列，逐帧攒出来的，不是单个读数。"""
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)

    view.set_progress(_running_progress(2.5, completed=300))
    view.set_progress(_running_progress(2.14, completed=600))

    xs, ys = view.trace_item.getData()
    assert list(ys) == [2.5, 2.14]
    assert len(xs) == 2

    view.reset()

    # 清空后 pyqtgraph 把两条数组置回 ``None``，所以这里问的是「没有点了」而不是长度。
    assert not len(view.trace_item.getData()[0] or ())


def _plot_panel(qtbot):
    """帧① 的画布：反射率四视图，模式条跟着 tab 排在同一行。

    要喂一份数据。``PlotPanel`` 没有数据集时整块画布让给空状态那一页，四个 tab 与模式条
    连同它们的父件一起被 ``QStackedLayout`` 藏起来——量到的是一行没有布过局的控件，几何
    断言会全过。
    """
    from tests.support.model_cases import prepared_data

    from xrr_fitter.gui.plots.panel import PlotPanel

    panel = PlotPanel()
    qtbot.addWidget(panel)
    panel.set_dataset("curve", prepared_data())
    panel.resize(950, 620)
    panel.show()
    qtbot.waitExposed(panel)
    return panel


def _modebar_glyphs(bar: QWidget) -> list[str]:
    """条上此刻摆着的按钮，按屏幕上的左右顺序。"""
    buttons = [button for button in bar.findChildren(QToolButton) if button.isVisibleTo(bar)]
    return [button.objectName() for button in sorted(buttons, key=lambda button: button.geometry().x())]


def test_the_reflectivity_modebar_wears_exactly_the_designs_four_glyphs(qtbot) -> None:
    """设计稿 ``.modebar`` 是四枚：``<b class="on">✥</b><b>⤢</b><b>⌂</b><b>▭</b>``。

    此前是九枚——查看 · 范围 · 掩膜 · 平移 · 框选放大 · 复位 · 缩放拟合区 · 全览 · 叠加
    对比——和四个 tab 挤在同一行里；四个 tab 加九枚按钮排不进 950px，条就把 tab 往回压。
    设计稿留下的是「怎么在图里走动」的三枚，加上「框出拟合范围」这一枚；容器的 title 恰好
    把这四枚念了一遍，所以这一行认的就是它自己写的那句话。
    """
    panel = _plot_panel(qtbot)

    assert _modebar_glyphs(panel.toolbar) == ["plotNavPan", "plotNavZoom", "plotNavHome", "plotModeRange"]
    assert panel.toolbar.toolTip() == "平移 / 缩放 / 复位 / 选择拟合范围"


# 一笔实线在 16px 上落的墨。这些字形共用 ``_pen`` 那支 1.8px 的圆头笔，画在 16 格上、开着
# 抗锯齿，一条描边最浓的那一格实测是 alpha 197——没有哪一格能到 255，除非那里是填充的箭头
# 头部。所以门槛取在描边之下、抗锯齿边缘（二十上下，见 ``GLYPH_INK_CEILING``）之上。
GLYPH_INK_FLOOR = 190

# 「这一格是空的」。设计稿那两枚字形都是空心的，而此前的「范围」在框内铺了一层 alpha 40 的
# 底色——门槛压在两者之间，空白与淡底色才分得开。
GLYPH_INK_CEILING = 24


def _glyph_image(name: str) -> QImage:
    """按设计稿那 16 格把一枚字形画出来。"""
    from xrr_fitter.gui.plots.plot_icons import plot_icon

    return plot_icon(name, size=16).pixmap(16, 16).toImage().convertToFormat(QImage.Format.Format_ARGB32)


def _ink(image: QImage, x: float, y: float) -> int:
    """字形在 (x, y) 那一格附近落的最重的墨：0 是空白，255 是实笔。

    坐标用设计稿那 16 格写，按图实际的像素比换算。取 3×3 邻域的最大值：笔是圆头又开了抗
    锯齿，单点采样会把一条正好擦过这一点的线读成空白。
    """
    scale = image.width() / 16.0
    values = []
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            px = round(x * scale) + offset_x
            py = round(y * scale) + offset_y
            if 0 <= px < image.width() and 0 <= py < image.height():
                values.append(QColor.fromRgba(image.pixel(px, py)).alpha())
    return max(values)


def test_the_zoom_glyph_is_the_designs_diagonal_arrow_not_a_magnifier(qtbot) -> None:
    """设计稿第二枚是 ``⤢``：一支从左下指到右上的双头箭头。

    此前画的是放大镜——镜筒一个圆压在左上、镜柄斜到右下、镜片里还有个 ``+``。放大镜说的是
    「放大多少」，而这一条做的是「拖出一个矩形，视野收到那块上」；``⤢`` 说的正是后者，也和
    条上另外三枚（``✥`` 十字、``⌂`` 房子）同属一套线条字形。两者在 16px 上的分界不是风格：
    放大镜的镜筒占着左上角、镜柄压着右下角，而 ``⤢`` 那条对角线整条是空的。
    """
    assert qtbot is not None
    image = _glyph_image("zoom")

    assert _ink(image, 12.5, 3.5) > GLYPH_INK_FLOOR, "右上没有箭头"
    assert _ink(image, 3.5, 12.5) > GLYPH_INK_FLOOR, "左下没有箭头"
    assert _ink(image, 3.5, 3.5) < GLYPH_INK_CEILING, "左上有墨——那是放大镜的镜筒"
    assert _ink(image, 12.5, 12.5) < GLYPH_INK_CEILING, "右下有墨——那是放大镜的镜柄"


def test_the_range_glyph_is_the_designs_hollow_rectangle(qtbot) -> None:
    """设计稿第四枚是 ``▭``：一个空心的横矩形，四条边都在。

    此前画的是两根竖线夹一层淡底色，中间不封口——在 16px 上那是个暂停符号，而暂停恰好是
    同一屏上「拟合进行中」那一列的意思。``▭`` 是框选：四条边围出的那一块就是要交给拟合的
    角度窗口，而框住的东西留在框里看得见，所以里面是空的。
    """
    assert qtbot is not None
    image = _glyph_image("range")

    assert _ink(image, 8.0, 5.0) > GLYPH_INK_FLOOR, "上边没有封口"
    assert _ink(image, 8.0, 11.0) > GLYPH_INK_FLOOR, "下边没有封口"
    assert _ink(image, 2.0, 8.0) > GLYPH_INK_FLOOR, "框没有铺到设计稿那么宽"
    assert _ink(image, 8.0, 8.0) < GLYPH_INK_CEILING, "框里有底色——设计稿那一枚是空心的"


def test_the_modebar_opens_with_pan_lit_because_the_view_really_is_panning(qtbot) -> None:
    """设计稿亮着的是 ``✥``，而画着的面板静止时本来就在平移态。

    pyqtgraph 的视图没有「什么都不做」这一档：左键拖就是平移。此前条上四枚全是灰的，
    读者据此以为得先点一下才能拖——亮起来不是装饰，是把面板此刻的真实状态说出来。
    """
    panel = _plot_panel(qtbot)

    assert panel.navigation_mode() == "pan"
    assert panel.navigation_buttons()["pan"].isChecked() is True


def test_the_five_controls_the_modebar_gave_up_are_still_reachable(qtbot) -> None:
    """收窄不是砍功能：查看 / 掩膜 / 缩放到拟合范围 / 恢复完整视图 / 叠加对比 一条不少。

    它们变成模式条自己的右键菜单——和结构面板把 编辑基底 / 删除 / 上移 / 下移 收到层行
    右键菜单上是同一种搬法：命令挂回它作用的那个东西身上。图body 的右键归 pyqtgraph
    的 ViewBox（菜单加右键拖动缩放），所以落点选在条上而不是图里。
    """
    panel = _plot_panel(qtbot)
    actions = [action for action in panel.toolbar.tool_actions() if not action.isSeparator()]

    assert [action.text() for action in actions] == [
        "查看",
        "范围",
        "掩膜",
        "缩放到拟合范围",
        "恢复完整视图",
        "叠加对比",
    ]
    assert panel.toolbar.contextMenuPolicy() == Qt.ContextMenuPolicy.ActionsContextMenu
    # 三档交互模式互斥、叠加对比是开关，所以这四条要能显示勾选状态；两个缩放命令是一次性
    # 动作，勾了反而是在说「还停在这个状态」。
    assert [action.text() for action in actions if action.isCheckable()] == ["查看", "范围", "掩膜", "叠加对比"]


def test_the_menu_entries_drive_the_same_modes_the_buttons_did(qtbot) -> None:
    """菜单里那几条得真的换模式，否则搬家只搬走了名字。"""
    panel = _plot_panel(qtbot)
    by_text = {action.text(): action for action in panel.toolbar.tool_actions() if not action.isSeparator()}

    by_text["掩膜"].trigger()
    assert panel.interaction_mode() == "mask"

    by_text["查看"].trigger()
    assert panel.interaction_mode() == "view"


def test_the_range_glyph_and_its_menu_entry_are_one_command(qtbot) -> None:
    """▭ 与菜单里的「范围」是同一条命令，勾选状态两边同步。

    留在条上的这一枚同时也在菜单里出现，读者从哪边进都该看到同一个状态；两套各记一份
    勾选，就会出现「条上亮着、菜单里没勾」这种自相矛盾的画面。
    """
    panel = _plot_panel(qtbot)
    button = panel.toolbar.findChild(QToolButton, "plotModeRange")
    range_action = next(action for action in panel.toolbar.tool_actions() if action.text() == "范围")

    button.click()

    assert panel.interaction_mode() == "range"
    assert range_action.isChecked() is True

    by_text = {action.text(): action for action in panel.toolbar.tool_actions() if not action.isSeparator()}
    by_text["查看"].trigger()

    assert button.isChecked() is False


def _reflectivity_tab_labels(panel) -> list[str]:
    tabs = panel.reflectivity_tabs
    return [tabs.tabText(index) for index in range(tabs.count())]


def test_the_reflectivity_strip_wears_the_designs_four_tabs(qtbot) -> None:
    """设计稿 帧① 的 ``.tabs`` 是四段，第三段带间隔号：

    ``对数反射率`` · ``原始数据与模型`` · ``qz⁴·R`` · ``加权残差``

    此前条上只有三段：加权残差被搬成了固定副图，于是这一行读不出「残差也是一个可以单独
    占满画布放大来看的视图」——它只能挤在下半段那一条里。``qz⁴·R`` 少了间隔号，「qz⁴R」
    念起来像一个变量名，而它其实是「qz 四次方乘以 R」这个乘积。
    """
    panel = _plot_panel(qtbot)

    assert _reflectivity_tab_labels(panel) == ["对数反射率", "原始数据与模型", "qz⁴·R", "加权残差"]


def test_the_reflectivity_modebar_follows_the_tabs_instead_of_hugging_the_right_edge(qtbot) -> None:
    """设计稿 帧① 的 ``.canvas-top`` 是 ``[tabs][modebar]`` 一个挨一个，不是两端对齐。

    Qt 的 ``setCornerWidget`` 只有「贴右边框」这一种摆法，于是模式条离开了它作用的那四个
    tab，跑到六百多像素之外——读者在 tab 上换视图，手要横穿整行才够得着换模式。设计稿的
    ``.spring`` 在 ``.canvas-top`` 里是零宽的（HTML 148 行），空白全留在行尾。
    """
    panel = _plot_panel(qtbot)
    row = panel.findChild(QWidget, "reflectivityCanvasTop")

    assert row is not None, "帧① 画布顶上没有 .canvas-top 那一行"
    assert row.isAncestorOf(panel.toolbar), "模式条不在那一行里"
    bar = row.findChild(QTabBar, "reflectivityCanvasTabs")
    assert bar is not None, "那一行里没有 tab 条"
    gap = panel.toolbar.geometry().left() - bar.geometry().right()
    trailing = row.width() - panel.toolbar.geometry().right()

    assert 0 < gap <= 24, f"tab 与模式条之间空了 {gap}px"
    assert trailing > gap, "留白落在了 tab 和模式条之间，而不是行尾"


def test_both_plot_strips_wear_the_flat_accent_tab_instead_of_a_boxed_one(qtbot) -> None:
    """两组 tab 都要走 ``canvasTab``，才拿得到设计稿那条 2px 下划线。

    ``QTabWidget`` 自带的 tab 是一只带边框的小盒子；设计稿的 ``.tab.on`` 是「文字换成
    accent 色 + 底下一条 2px accent 线」（HTML 155-157 行）。样式表里那条规则按
    ``QTabBar[canvasTab="true"]`` 选，所以认不认这一身皮全看这个属性。
    """
    panel = _plot_panel(qtbot)

    for name in ("reflectivityCanvasTabs", "analysisCanvasTabs"):
        bar = panel.findChild(QTabBar, name)
        assert bar is not None, f"没有 {name}"
        assert bar.property("canvasTab") is True, f"{name} 不认 canvasTab 皮肤"
        assert bar.drawBase() is False, f"{name} 还在自己画一条基线，和行底那条会叠成两条"


def test_the_pages_stop_drawing_a_second_tab_row_of_their_own(qtbot) -> None:
    """页面容器自带的那条 tab 条要收起来，否则一行 tab 变两行。

    页面还归 ``QTabWidget`` 管——``workspace`` 存盘、``interactions`` 换视图、可及性巡检都
    是按 ``reflectivityTabs`` / ``analysisTabs`` 这两个名字找它的。露在外面的只是设计稿那条
    扁 tab 条，容器自己那条重复画同样四个字。
    """
    panel = _plot_panel(qtbot)

    for pages in (panel.reflectivity_tabs, panel.analysis_tabs):
        # 断言的落点停在容器自己身上而不是 panel：``isVisibleTo`` 会一路往上走，分析页那
        # 一组没有结果时整组是收着的，问到 panel 就成了「祖先藏着所以看不见」——容器自己
        # 那条 tab 条露不露根本没被检查。
        assert pages.tabBar().isVisibleTo(pages) is False, f"{pages.objectName()} 还在画自己那条 tab 条"
    # 而露在外面的那条得是设计稿那条扁的。
    assert panel.findChild(QTabBar, "reflectivityCanvasTabs").isVisibleTo(panel) is True


def test_the_flat_bar_and_the_pages_stay_one_selection(qtbot) -> None:
    """扁 tab 条与页面容器必须是同一次选择：从哪边换，另一边跟着走。

    露出来的是 tab 条，而换视图的代码（菜单「视图」、Alt+N、``select_view``、恢复工作区）
    走的是容器。两边各记一份当前页，就会出现「条上亮着 qz⁴·R、画的却是对数反射率」。
    """
    panel = _plot_panel(qtbot)
    bar = panel.findChild(QTabBar, "reflectivityCanvasTabs")

    assert [bar.tabText(index) for index in range(bar.count())] == _reflectivity_tab_labels(panel)

    bar.setCurrentIndex(2)
    assert panel.reflectivity_tabs.currentIndex() == 2

    panel.reflectivity_tabs.setCurrentIndex(0)
    assert bar.currentIndex() == 0


def test_the_bar_lights_the_view_the_canvas_draws_inside_the_whole_window(qtbot) -> None:
    """装进主窗口之后，换视图仍要让条上亮的那只 tab 就是画布画的那一张。

    上一条测的是独立面板：那里 ``select_view`` 一路走到底没人打扰。装进主窗口后
    ``view_changed`` 会经 ``_plot_tab_changed`` → ``_capture_workspace`` →
    ``replace_project`` 绕回 ``project_project``，而那一次回投又把工作区落一遍到画布上。
    回投带的是分析组的旧序号，于是屏上出现「画的是不确定度、条上亮着候选解比较」——设计稿
    帧⑤ 的截图上就是这样，而按状态断言的测试一条都没红，因为容器那边的序号是对的。
    """
    window = _fitted_window(qtbot)
    panel = window.plot_panel
    bar = panel.findChild(QTabBar, "analysisCanvasTabs")

    panel.select_view("uncertainty")

    assert panel.current_view_key() == "uncertainty"
    assert bar.tabText(bar.currentIndex()) == "不确定度"


def test_projecting_the_project_again_keeps_the_analysis_choice(qtbot) -> None:
    """项目再落一遍到画布，分析组停在读者选的那一张，不退回第一个。

    ``project_project`` 每次都重放工作区，而拟合预览、撤销、改一个参数都会走到它。分析组
    的序号在 ``ui_state`` 里有自己的字段（``analysis_tab_index``，存盘也带它），漏着不传就
    等于每次投射都替读者按一次「候选解比较」。
    """
    window = _fitted_window(qtbot)
    panel = window.plot_panel
    bar = panel.findChild(QTabBar, "analysisCanvasTabs")
    panel.select_view("uncertainty")

    window.document.replace_project(window.document.project)

    assert panel.current_view_key() == "uncertainty"
    assert bar.tabText(bar.currentIndex()) == "不确定度"


def test_the_residual_tab_moves_the_one_card_instead_of_drawing_it_twice(qtbot) -> None:
    """设计稿 帧① 同时有「加权残差」这个 tab 和钉在下半段的那张残差卡。

    两处指的是同一张图。一个 ``LiveReflectivityPlot`` 只能有一个父件，真做成两份就是画两
    遍、各记一套缩放状态：读者在 tab 里放大完回到下半段，会看到另一个视野，而两条曲线本
    该是同一条。所以这张卡在两个位置之间搬家——选中「加权残差」时它就是这一页的正文，其余
    时候回到 splitter 里紧跟 tab 组的那一格。
    """
    panel = _plot_panel(qtbot)
    splitter = panel.plot_splitter
    card = panel.residual_pane
    pinned = splitter.indexOf(panel.reflectivity_group) + 1

    assert splitter.indexOf(card) == pinned
    assert card.isHidden() is False

    panel.select_view("residual")

    page = panel.reflectivity_tabs.currentWidget()
    assert page.isAncestorOf(card), "选中「加权残差」，这一页的正文却不是那张残差卡"
    assert splitter.indexOf(card) == -1, "卡还钉在 splitter 里，残差就画了两遍"
    assert card.isHidden() is False
    assert len(panel.findChildren(QWidget, "diagnosticCanvas:residual")) == 1, "残差图被建了两份"

    panel.select_view("log")

    assert splitter.indexOf(card) == pinned, "离开这一页，钉着的那一格没把卡收回去"
    assert card.isHidden() is False
