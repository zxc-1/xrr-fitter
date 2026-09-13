"""设计稿帧 ②/④ 的可读内容保真度。

引导模式面向不懂 XRR 的使用者，设计稿帧②把整段平白语言当成功能而非装饰：
层清单、氧化层解释、手动加删层入口。同样地，帧④的进度卡副标题报的是活动阶段
而不是一句静态说明。这些断言把那些文案钉在实现上，因为它们是这个模式唯一
的解释来源，删掉任何一条都会让画面重新变成一堆无解释的控件。
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QFont, QFontInfo
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QTableWidget, QWidget
from tests.gui.paint_support import painted
from tests.support.model_cases import final_fit_result, fit_candidate

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.model.parameters import ParameterValue


def _reads_as(text: str) -> str:
    """The words a提示 puts on screen, with the emphasis markup taken back out.

    这些断言钉的是"说了什么"，而字重是"怎么说"。把 ``<b>`` 留在断言里，会让任何一次
    强调位置的调整都变成一处文案回归——于是要么不敢动排版，要么把断言改成看不出漏了
    半句的宽松包含。去掉标记后仍然逐字比对整段。
    """
    return re.sub(r"</?b>", "", text)


# 引导页翻页的淡入时长，与 ``guidance.panel._animate_transition`` 里那条 ``setDuration`` 对齐。
# 量像素之前必须等它跑完：动画中途整页压着一层 ``QGraphicsOpacityEffect``，取样取到的是被淡化
# 过的颜色，而不是发货时读者看见的那一档。
GUIDANCE_FADE_MS = 200

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)
CURVE_POINTS = 64


def _write_curve(path: Path) -> Path:
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(CURVE_POINTS)) + "\n",
        encoding="utf-8",
    )
    return path


def _write_wide_curve(path: Path) -> Path:
    """A five-column scan whose third and fourth columns are a second usable curve.

    ``_write_curve`` only has two columns, so pointing the mapping at column 3 there
    is a genuine parse failure rather than a re-mapping. Five columns let the same
    file be read twice over — once as columns 1/2, once as 3/4 with 5 as σ — which is
    what the 列映射 cell has to keep up with.
    """
    rows = (
        f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g} "
        f"{0.06 + index * 0.02:.6f} {900.0 / (index + 1):.12g} {5.0 / (index + 1):.12g}"
        for index in range(64)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _project(tmp_path: Path, *, structured: bool = False, count: int = 1):
    project = api.new_project()
    for index in range(count):
        project = api.add_dataset(project, _write_curve(tmp_path / f"curve-{index}.xy"), api.InstrumentSpec())
        if structured:
            structure = api.StructureSpec(
                AIR,
                (
                    api.LayerSpec("SiO₂ 表面氧化层", SIO2, 34.0, roughness_a=3.0),
                    api.LayerSpec("a-Si 非晶硅薄膜", SI, 487.0, roughness_a=4.0),
                ),
                SI,
            )
            project = api.set_structure(project, f"curve-{index}", structure)
    return api.select_active_dataset(project, "curve-0")


def _window(qtbot, project=None):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    document = ProjectDocument() if project is None else ProjectDocument(project)
    window = MainWindow(document)
    qtbot.addWidget(window)
    window.resize(1280, 760)
    window.show()
    qtbot.wait(1)
    return window


def _span_text(window, name: str) -> str:
    """一段状态栏读出来是什么样：说明文字与取值按排版顺序连起来。

    设计稿帧② 那一段是「第 <b>2</b> 步 / 共 4 步」，只有序号加粗，所以序号与它两边的
    说明文字分属三个标签。断言写在连起来的这句话上，就不必替实现记住哪片归哪个标签。
    """
    from xrr_fitter.gui.status_bar import StatusSpan

    span = window.findChild(StatusSpan, name)
    assert span is not None, name
    layout = span.layout()
    pieces = []
    for index in range(layout.count()):
        label = layout.itemAt(index).widget()
        if isinstance(label, QLabel):
            pieces.append(label.text())
    return "".join(pieces)


def _stack_box(window) -> QWidget:
    """帧② 层堆叠那只盒子本身。"""
    stack = window.guidance.findChild(QWidget, "structureStepStack")
    assert stack is not None
    return stack


def _stack_rows(window) -> list[QFrame]:
    """帧② 层堆叠的四行，按从空气到基底的排版顺序。

    走 layout 而不是 ``findChildren``：后者返回的是整棵子树，行里那三个标签的父级也是
    ``QFrame`` 的话就会混进来，而这一列的顺序本身是断言的一部分。
    """
    layout = _stack_box(window).layout()
    return [layout.itemAt(index).widget() for index in range(layout.count())]


def _row_labels(row: QFrame) -> dict[str, QLabel]:
    """一行里那三段文字，按它们写着的字索引。

    色块也是 ``QLabel``（它搬的是一张 pixmap），但它的文本是空串，不会挤掉这三段。
    """
    return {label.text(): label for label in row.findChildren(QLabel)}


def test_the_live_pane_names_its_two_curves_the_way_the_key_does(qtbot) -> None:
    """帧①/④ 的图例写「观测数据」与「当前拟合模型」。

    「归一化」与「候选」都是流程内部词：前者说的是处理步骤，后者说的是搜索阶段，
    读图的人要的是"哪条是我测的、哪条是拟合出来的"。
    """
    from xrr_fitter.gui.plots.live import LiveReflectivityPlot

    pane = LiveReflectivityPlot()
    qtbot.addWidget(pane)

    assert pane.observed_item.name() == "观测数据"
    assert pane.model_item.name() == "当前拟合模型"


def test_the_progress_card_subtitle_names_the_running_stage(qtbot) -> None:
    """帧④ 的副标题是「阶段 4 / 9 · 单调递增 0–1000」，不是一句静态说明。

    卡片头是这张卡最先被读到的一行；把活动阶段写在那里，即使正文被滚出视野也
    还答得上"现在跑到哪了"。开跑前没有阶段可报，所以那时仍是静态的总述。
    """
    from xrr_fitter.gui.fitting.progress import ProgressView

    view = ProgressView()
    qtbot.addWidget(view)
    caption = view.findChild(QLabel, "fitProgressCardSubtitle")

    assert caption.text() == "共九阶段 · 单调递增 0–1000"

    view.set_progress(api.FitProgress("curve-0", "D", 3, 10, 1.25, "精修"))

    assert caption.text() == "阶段 4 / 9 · 单调递增 0–1000"
    assert caption.toolTip() == "阶段 4 / 9 · 单调递增 0–1000"

    view.reset()

    assert caption.text() == "共九阶段 · 单调递增 0–1000"


def test_each_rail_step_carries_the_subline_the_design_gives_it(qtbot, tmp_path) -> None:
    """帧② 的步骤轨每格都有第二行：一句"这一步会发生什么"。

    只写步骤名的轨道回答不了"点下去要付出什么"，而这四句正是让"开始拟合"读起来
    像"全自动"而不是"要我调参数"的原因。导入那句由真实数据集数量派生。
    """
    window = _window(qtbot, _project(tmp_path, count=2))

    assert window.guidance.step_header_sublines() == (
        "2 个文件 · 已就绪",
        "检查自动建议",
        "全自动",
        "置信度与参数",
    )


def test_the_import_subline_admits_an_empty_project(qtbot) -> None:
    """没有文件时不能谎报「已就绪」。"""
    window = _window(qtbot)

    assert window.guidance.step_header_sublines()[0] == "尚未导入"


def test_the_status_bar_states_which_guided_step_is_open(qtbot, tmp_path) -> None:
    """帧② 的状态栏分两段报「引导模式」与「第 2 步 / 共 4 步」。

    引导模式收走了检查器，状态栏因此是屏幕上唯一还能报位置的地方；专家模式下这一
    段无从谈起，所以它随模式隐藏而不是留一句过期的步骤号。

    设计稿把它拆成两段而不是一句「引导模式 · 第 2 步 / 共 4 步」：模式那一段带一颗
    ``--info`` 圆点，与其余各帧「就绪 / 拟合进行中」占同一个位置，而步骤号那一段只给
    序号加粗。写成一句话就既拿不到那颗圆点，也没法只加粗中间两个字。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    label = window.findChild(QLabel, "guidanceStepStatus")

    assert label is not None
    assert label.isVisibleTo(window) is False

    window.set_guidance_visible(True)

    assert label.isVisibleTo(window) is True
    assert _span_text(window, "statusStepSpan") == "第 1 步 / 共 4 步"
    # 第一段报模式而不是就绪，圆点跟着换成 --info。
    assert window.findChild(QLabel, "fitReadinessStatus").text() == "引导模式"
    assert window.findChild(QLabel, "fitReadinessDot").property("statusKind") == "info"

    window.guidance.show_step("fitStep")

    assert _span_text(window, "statusStepSpan") == "第 3 步 / 共 4 步"

    window.set_guidance_visible(False)

    assert label.isVisibleTo(window) is False


def test_the_structure_step_lists_the_stack_in_plain_language(qtbot, tmp_path) -> None:
    """帧② 用四行平白语言复述结构：名字、这层是什么、大概多厚。

    专家模式的结构树按列摊开厚度、粗糙度、密度，读它先要知道那些列是什么。引导
    模式给的是同一个结构的另一种投影：半无限的介质说清"无需参数"，有限层给一个
    量级而不是有效数字，用户要判断的只是"这个搭法是否合理"。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))

    assert window.guidance.structure_rows() == (
        ("空气", "入射介质 · 半无限，无需参数", "—"),
        ("SiO₂ 表面氧化层", "薄膜层 · 参与拟合", "≈ 3.4 nm"),
        ("a-Si 非晶硅薄膜", "薄膜层 · 参与拟合", "≈ 48.7 nm"),
        ("Si", "衬底 · 半无限", "—"),
    )


def test_each_guided_row_carries_the_colour_of_its_layer(qtbot, tmp_path) -> None:
    """帧② 的 ``.lyr`` 每行开头都有一块 ``.sw`` 色块，缺了它行与层之间就只剩名字对得上。

    引导用户看到的是这一张清单，而同一个结构在专家模式的树和色带图里都是按 hue 配对的；
    这一帧不上色，等于同一份结构在三处各说一套颜色，切模式后要重新找哪行是哪层。
    """
    from xrr_fitter.gui.structure.stack import component_fill

    window = _window(qtbot, _project(tmp_path, structured=True))
    stack = window.guidance.findChild(QWidget, "structureStepStack")

    chips = [chip for chip in stack.findChildren(QLabel) if chip.objectName() == "structureStepSwatch"]

    assert len(chips) == 4, f"四行只有 {len(chips)} 块色块"
    wanted = [theme.DATA_NEUTRAL, component_fill(0), component_fill(1), theme.DATA_NEUTRAL]
    shown = [QColor(chip.pixmap().toImage().pixelColor(5, 5)).name().upper() for chip in chips]
    assert shown == [QColor(colour).name().upper() for colour in wanted], shown


def test_the_guided_stack_is_one_bordered_box_and_not_four_separate_cards(qtbot, tmp_path) -> None:
    """设计稿 506-508 行的 ``.stack``：一只圆角盒子，里面每行用一道下边线分格。

    四张各带边框圆角的卡片说的是「四件并列的东西」，而这四行是**一叠**——从空气到基底，
    相邻两行贴着的那个面就是一道界面。盒子加分隔线把这层关系画出来了：外框说「这是一叠」，
    分隔线说「界面在这里」。行与行之间还留 4px 空隙时，那道界面变成了两行之间的空地。

    四行各自带哪几档属性由下一条测试钉；这里只问样式表把这只盒子画成了什么。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    sheet = theme.build_stylesheet(theme.light_palette())
    box = sheet.split("QWidget#structureStepStack {", 1)[1].split("}", 1)[0]
    divider = sheet.split('QFrame[stackRow="true"] {', 1)[1].split("}", 1)[0]

    assert f"border-radius: {theme.STACK_CONTAINER_RADIUS_PX}px" in box
    assert "border: 1px solid" in box
    assert "border-bottom: 1px solid" in divider
    assert 'QFrame[stackRow="true"][lastStackRow="true"] { border-bottom: 0px; }' in sheet
    assert _stack_box(window).layout().spacing() == 0


def test_each_stack_row_is_flagged_as_a_row_of_the_box_and_not_a_card(qtbot, tmp_path) -> None:
    """四行各自带着把它放进那只盒子里的几档属性。

    ``sectionCard`` 画的是一圈框，行上留着它就等于盒子里又套四只盒子。首末两行的外侧圆角替
    CSS 的 ``overflow:hidden``：方角的行压在圆角的框上会从四个角探出去。末行还要去掉那道下
    边线——盒子自己的框已经在那个位置了，再画一道就是两条并排的线。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    rows = _stack_rows(window)

    assert [bool(row.property("stackRow")) for row in rows] == [True] * 4
    assert [bool(row.property("sectionCard")) for row in rows] == [False] * 4
    assert [bool(row.property("firstStackRow")) for row in rows] == [True, False, False, False]
    assert [bool(row.property("lastStackRow")) for row in rows] == [False, False, False, True]


def test_the_two_semi_infinite_rows_read_as_the_bracket_they_are(qtbot, tmp_path) -> None:
    """``.lyr.semi``：首末两行压一层淡底，中间两行不压。

    空气与基底是半无限介质，它们不是这次拟合的对象，只是这一叠的上下界。淡底把「这两行
    与另两行不是一类」说在颜色上，读者不必先读完副行的「半无限」才明白第一行为什么没有厚度。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    rows = _stack_rows(window)
    sheet = theme.build_stylesheet(theme.light_palette())

    assert [bool(row.property("semiInfinite")) for row in rows] == [True, False, False, True]
    assert 'QFrame[stackRow="true"][semiInfinite="true"] {' in sheet


def test_each_stack_row_typesets_its_three_readings_at_the_designs_sizes(qtbot, tmp_path) -> None:
    """设计稿 511-513 行：层名 13px/600、副行 11px/400 最淡那档灰、读数 12px 静音档等宽数字。

    三段挤在一行里，谁是主角只能由字号和字重说。三段同号时这一行读起来是平的——层名、
    它的角色、它的厚度并列成三块同等重量的字，而读者扫这一列要找的是「这是哪一层、多厚」。
    副行取更淡的一档灰：它是主名的注脚，不是与厚度同级的第二个读数。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    labels = _row_labels(_stack_rows(window)[1])
    name = labels["SiO₂ 表面氧化层"]
    role = labels["薄膜层 · 参与拟合"]
    measure = labels["≈ 3.4 nm"]

    sizes = (
        QFontInfo(name.font()).pixelSize(),
        QFontInfo(measure.font()).pixelSize(),
        QFontInfo(role.font()).pixelSize(),
    )
    assert sizes == (13, 12, 11)
    # 顺序本身是断言：主名 > 读数 > 注脚，这一行的层次就是这么读出来的。
    assert list(sizes) == sorted(sizes, reverse=True)
    assert name.font().weight() == QFont.Weight.DemiBold
    assert role.font().weight() == QFont.Weight.Normal
    assert bool(role.property("faintText")) is True
    assert bool(role.property("mutedText")) is False
    assert bool(measure.property("mutedText")) is True
    # ``.mv`` 带 ``font-variant-numeric:tabular-nums``：这一列是上下扫着比厚度的。
    assert measure.font().isFeatureSet(QFont.Tag("tnum")) is True


def test_a_layer_named_without_chinese_does_not_come_out_a_shorter_row(qtbot, tmp_path) -> None:
    """四行等高，哪一行的名字里没有汉字都一样。

    设计稿的 ``.lyr`` 四行 ``padding`` 与字号都相同，所以四行等高。而 ``QLabel`` 的 sizeHint 是
    按它那句话实际用到的字体量的：``Si`` 全走主字体量到 16px，``空气`` 与 ``SiO₂ 表面氧化层`` 里
    的汉字走回退字体量到 19px。基底叫 ``Si``、``Au``、``Pt`` 的样品于是末行整行矮三像素——而这一叠
    画的正是一叠层，行高不齐读起来就是「这几层薄厚不同」，偏偏厚度是右边那一列已经写明的事。

    量的是未套样式表的行高，所以这里只有字撑出来的那部分：末行少的那道下边线是 QSS 的事，由
    ``lastStackRow`` 那条规则的断言管。断言落在整行的 sizeHint 上而不是层名标签的 ``height()``：
    没上过屏的标签高度还是默认值，四个一律相等，无论根因修没修。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    rows = _stack_rows(window)
    wanted = ("空气", "SiO₂ 表面氧化层", "a-Si 非晶硅薄膜", "Si")
    names = [_row_labels(row)[text] for row, text in zip(rows, wanted, strict=True)]
    assert len(names) == 4

    # 根因钉在这里：层名那一档字的高度只跟着字号走，不跟着这一行写的是汉字还是拉丁字母。
    assert [row.sizeHint().height() for row in rows] == [rows[0].sizeHint().height()] * 4


def test_the_stacks_outline_reaches_the_pixels_and_not_only_the_stylesheet(qtbot, tmp_path) -> None:
    """那圈框真的画出来了，不只是样式表里有一条规则。

    ``_StackList`` 是 ``QWidget`` 的子类，而样式表的 ``border``/``background`` 对自定义的
    ``QWidget`` 子类默认整条不走——规则写得再对也一个像素都不画，属性断言看不出这件事，因为
    规则确实在 sheet 里。只有量颜色才钉得住：盒子左缘那一列要比它里面深。

    发货时那份色板与样式表得自己装，那一步也得先翻到眼前：这里的窗口不经 ``apply_theme``，
    默认的离屏色板把 Window 与 Base 全给成白的，中央栈还停在专家模式的画布上。取样走整窗
    截图再映射坐标——单独 ``grab`` 一只嵌在栈里的子控件，离屏平台交回的是一整片黑。

    ``show_step`` 会给新翻上来那一页挂 200ms 的淡入（``_animate_transition``），不等它跑完量到
    的是动画中途那一帧：实测框色从该有的 34/255 被压到 7/255，跟白底差六级灰，任何「这一列该
    比里面深」的断言都会失手。
    """
    palette = theme.light_palette()
    window = _window(qtbot, _project(tmp_path, structured=True))
    window.set_guidance_visible(True)
    window.guidance.show_step("structureStep")
    window.setPalette(palette)
    window.setStyleSheet(theme.build_stylesheet(palette))
    qtbot.waitExposed(window)
    qtbot.wait(GUIDANCE_FADE_MS * 2)
    stack = _stack_box(window)
    assert stack.isVisible() is True
    shot = painted(window)
    # 取某一行（非半无限）的垂直中心：半无限行（首末）有淡底，中间薄膜行背景是 transparent。
    # 不取整个盒子的正中——偶数行时恰好落在行分隔线（border-bottom）上，
    # edge 和 inside 都读到分隔线颜色，断言就废了。
    row_count = stack.layout().count()
    row_height = stack.height() // max(row_count, 1)
    middle = row_height + row_height // 2  # 第 1 行（0-indexed）的垂直中心

    edge = shot.at_point(stack.mapTo(window, QPoint(0, middle)))
    inside = shot.at_point(stack.mapTo(window, QPoint(3, middle)))
    assert edge.getRgb()[:3] != inside.getRgb()[:3], (edge.name(), inside.name())
    assert edge.lightnessF() < inside.lightnessF(), (edge.name(), inside.name())


def test_two_phases_of_one_element_do_not_collapse_into_one_name(qtbot, tmp_path) -> None:
    """帧② 写「a-Si 非晶硅薄膜」「c-Si 晶体硅基底」，两行不能都读成「Si」。

    非晶硅膜和晶硅基底的 formula 都是 ``Si``，相只写在材料名里。清单原先取 formula，于
    是这份样品最要紧的区分——测的那层和它底下的衬底是两种东西——在读者眼里消失了。
    """
    a_si = api.MaterialSpec("a-Si", "Si", 2.28)
    c_si = api.MaterialSpec("c-Si", "Si", 2.33)
    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve-0.xy"), api.InstrumentSpec())
    project = api.set_structure(
        project,
        "curve-0",
        api.StructureSpec(AIR, (api.LayerSpec("a-Si 非晶硅薄膜", a_si, 487.0),), c_si),
    )
    window = _window(qtbot, api.select_active_dataset(project, "curve-0"))

    rows = window.guidance.structure_rows()

    assert rows[1][0] == "a-Si 非晶硅薄膜"
    assert rows[2][0] == "c-Si"


def test_a_lone_layer_is_named_as_the_thing_being_measured(qtbot, tmp_path) -> None:
    """只有一层时它就是被测对象，说得出来就不必让用户自己猜。"""
    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve-0.xy"), api.InstrumentSpec())
    project = api.set_structure(
        project,
        "curve-0",
        api.StructureSpec(AIR, (api.LayerSpec("SiO2", SIO2, 487.0),), SI),
    )
    window = _window(qtbot, api.select_active_dataset(project, "curve-0"))

    assert window.guidance.structure_rows()[1] == ("SiO2", "主体层 · 你要测量的对象", "≈ 48.7 nm")


def test_the_structure_step_says_nothing_before_a_structure_exists(qtbot, tmp_path) -> None:
    """没有结构就没有层可列，空清单胜过一张写着占位符的假清单。"""
    window = _window(qtbot, _project(tmp_path))

    assert window.guidance.structure_rows() == ()


def test_the_structure_step_offers_a_manual_way_out(qtbot, tmp_path) -> None:
    """帧② 的「＋ 我要手动加/删层」是自动建议不对时唯一的出口。

    引导模式本身没有层编辑器，所以这个入口只能把人交给专家模式；不给它，一个
    看出建议不对的用户就只剩"照着错的结构拟合"这一条路。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    window.set_guidance_visible(True)
    window.guidance.show_step("structureStep")
    button = window.guidance.findChild(QPushButton, "structureStepManual")

    assert button is not None
    assert button.text() == "＋ 我要手动加/删层"

    button.click()

    assert window.guidance_is_visible() is False


def test_the_guided_ctas_wear_the_designs_own_large_size_class(qtbot, tmp_path) -> None:
    """帧② 那一行 CTA 是设计稿的 ``.btn.lg``：两枚都 44px 高，并排站着严丝合缝。

    设计稿给 ``.btn.lg`` 定了 ``height:44px;padding:0 22px;font-size:14px;border-radius:8px``，
    并把引导页那一行的两枚按钮都写成 ``lg``——``看起来没问题，开始拟合 →`` 是 ``primary lg``，
    ``＋ 我要手动加/删层`` 是普通描边的 ``btn lg``。这一行是整屏唯一的出路，一行里只有它们，
    所以它们自己定这一行的高度，比应用里其他按钮高一档。

    高度得是钉死的同一个数，不能各自按文字与边框撑开：``.ctarow`` 是 ``align-items:center``，
    两枚差一像素就会把矮的那枚往下推，读起来像整行没对齐。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    # fixture 不走 ``apply_theme``，而这两枚的高度全由 QSS 的 min/max-height 说话；不自己套
    # 一遍发货的 stylesheet，量到的就不是 app 实际发出去的那个控件。
    window.setStyleSheet(theme.build_stylesheet(window.palette()))
    window.set_guidance_visible(True)
    window.guidance.show_step("structureStep")
    action = window.guidance.findChild(QPushButton, "structureStepAction")
    manual = window.guidance.findChild(QPushButton, "structureStepManual")

    assert action.height() == theme.LARGE_BUTTON_H
    assert manual.height() == theme.LARGE_BUTTON_H
    # 两枚同父（都挂在卡片上），``y()`` 就是同一把尺子量出来的行内位置。
    assert action.y() == manual.y()
    # 设计稿那一枚是描边的 ``btn lg``：``ghost`` 是无框档，用在这里会让一行里一枚有框一枚
    # 没框，而设计稿画的是两枚同框。
    assert manual.property("ghost") in (None, False)


def test_an_auto_added_oxide_layer_explains_itself(qtbot, tmp_path) -> None:
    """帧② 的 💡 说明「已自动加入表面氧化层」并给出移除它的条件。

    一层用户没画过的东西凭空出现在结构里，如果不解释就是软件在替他做决定；说出
    它为什么在那、什么情况下该拿掉，那一层才从"意外"变成"可复核的判断"。
    """
    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve-0.xy"), api.InstrumentSpec())
    project = api.set_structure(
        project,
        "curve-0",
        api.StructureSpec(AIR, (api.LayerSpec("film", SI, 487.0),), SI),
    )
    suggestion = next(s for s in api.suggest_oxide_layers(project.datasets[0].structure) if s.location == "surface")
    project = api.accept_oxide_suggestion(project, "curve-0", suggestion)
    window = _window(qtbot, api.select_active_dataset(project, "curve-0"))
    window.set_guidance_visible(True)
    window.guidance.show_step("structureStep")
    tip = window.guidance.findChild(QLabel, "structureStepOxideTip")

    assert tip is not None
    assert tip.isVisibleTo(window) is True
    assert _reads_as(tip.text()) == (
        "💡 已自动加入表面氧化层。金属与半导体表面在空气中通常生成几纳米氧化层。"
        "若你的样品在惰性气氛中制备并即时测量，可以移除它。"
    )
    assert window.guidance.structure_rows()[1] == ("SiO₂ 表面氧化层", "自动建议 · 可移除", "≈ 1.0 nm")


def test_the_oxide_tip_stays_hidden_when_no_oxide_was_added(qtbot, tmp_path) -> None:
    """没有自动加层就没有要解释的事。"""
    window = _window(qtbot, _project(tmp_path, structured=True))
    window.set_guidance_visible(True)
    window.guidance.show_step("structureStep")

    assert window.guidance.findChild(QLabel, "structureStepOxideTip").isVisibleTo(window) is False


def test_the_structure_action_confirms_a_stack_that_already_exists(qtbot, tmp_path) -> None:
    """帧② 的主按钮是「看起来没问题，开始拟合 →」——确认，而不是重建。

    结构已经搭好时，"初始化样品结构"会把用户刚看过的那份结构推倒重来，正好是这一
    步要他确认的东西。所以按钮的文案与动作都跟着结构在不在走。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    window.set_guidance_visible(True)
    window.guidance.show_step("structureStep")
    action = window.guidance.findChild(QPushButton, "structureStepAction")

    assert action.text() == "看起来没问题，开始拟合 →"

    action.click()

    assert window.guidance.current_step() == "fitStep"


def test_the_structure_action_builds_a_stack_that_is_missing(qtbot, tmp_path) -> None:
    window = _window(qtbot, _project(tmp_path))
    window.set_guidance_visible(True)
    window.guidance.show_step("structureStep")

    assert window.guidance.findChild(QPushButton, "structureStepAction").text() == "初始化样品结构"


def test_the_structure_step_body_confirms_instead_of_instructing(qtbot, tmp_path) -> None:
    """结构已在时，正文说的是"我们搭好了，你看一眼"，不是"去初始化"。

    帧② 的这段话是整个引导模式里唯一交代"结构从哪来、不准会怎样"的地方；结构还
    没有时它无从谈起，所以那时仍是那句建结构的说明。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    body = window.guidance.findChild(QLabel, "structureStepBody")

    assert body.text() == (
        "我们根据你的数据和常见薄膜体系，自动搭好了一个初始结构。看一眼是否合理——"
        "不确定也没关系，拟合会在允许范围内自动修正每一层。"
    )

    empty = _window(qtbot, _project(tmp_path))

    assert empty.guidance.findChild(QLabel, "structureStepBody").text() == (
        "初始化样品结构，必要时添加膜层。默认基底为 Si，可在结构面板调整。"
    )


def test_the_structure_step_points_at_expert_mode_for_fine_control(qtbot, tmp_path) -> None:
    """帧② 收尾一句把"我要更细的控制"引向专家模式，而不是留在引导里找不到。

    设计稿 585 行把「专家模式」加粗：这一句是一段无关紧要的脚注，除了它指向的那个去处——
    加粗的正是读者要去找的那颗开关的名字，其余半句是理由。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    window.set_guidance_visible(True)
    hint = window.guidance.findChild(QLabel, "structureStepExpertHint")

    assert hint is not None
    assert _reads_as(hint.text()) == "需要精细控制每个参数的边界、先验或跨数据集共享？切换到专家模式即可。"
    assert "<b>专家模式</b>" in hint.text()


def test_the_inspector_warns_that_thickness_and_density_can_trade_off(qtbot, tmp_path) -> None:
    """帧③ 右栏第三段「结构诊断」：改结构时就说 d 与 ρ 可能相关。

    薄层上厚度与密度换着走能给出几乎同样的曲线，所以一个收敛得很好的拟合仍可能
    两个值都不可信。这句话必须出现在编辑结构的地方——等到拟合完再说，人已经把那
    组数字记成结论了。措辞照设计稿留「可能」，并且指向不确定度页，因为这里只能提
    示相关的可能性，真的相关要用相关矩阵和 Profile 似然去核。
    """
    from xrr_fitter.gui.window_layout import INSPECTOR_SECTIONS

    window = _window(qtbot, _project(tmp_path, structured=True))

    section = [entry for entry in INSPECTOR_SECTIONS if entry[3] == "structure_diagnostics_panel"]
    assert section, "右栏没有结构诊断段"
    assert section[0][1] == "结构诊断"

    hint = window.findChild(QLabel, "structureCorrelationHint")
    assert hint is not None
    assert "厚度 d 与密度 ρ 可能相关" in hint.text()
    assert "不确定度" in hint.text()
    assert hint.wordWrap()


# 设计稿的 ``.insp-sec .h`` 是一行 ``space-between``（:175）：标题靠左，右端那句说明只在
# 写了 ``<span class="faint">`` 的地方存在。数一遍六帧的右栏，带右端说明的只有三处计数
# （帧① 的 aSi_ML_25C / 12 自由 / 3 个）和帧④ 实时指标的「联合」；帧③ 那三段和「控制」
# 段的抬头都只有标题一个词。
#
# 少画的这四句不是省略——它们是发明。「材料与几何」「初值 · 先验 · 约束」这类副标题在
# 设计稿里根本没有对应文本，写上去等于在每段抬头右边加一句谁也没要求的解释，而抬头右
# 端在设计稿里是留给计数的位置：读者扫到那儿要读的是「几个」，不是又一句形容。
#
# 「哪几段该有右端说明」这条判据照旧。改掉的是实时指标那一格的取值：设计稿帧④ 写的
# 「联合」不是这张卡的标题装帧，而是那一帧此刻的批量模式。把它当字面文案钉死，等于让
# 独立批量的项目也在右栏报「联合」，而同一屏顶栏那枚高亮写的是「独立」——两处说的是
# 同一件事，读者只能二选一地信。默认项目是 ``batch_mode="independent"``（``model/
# project.py``），所以这里的期望值是「独立」，联合那一档由下一条用例接管。
INSPECTOR_CAPTIONS = {
    "inspectorSelectedLayer": "",
    "inspectorParameters": "",
    "inspectorStructureDiagnostics": "",
    "inspectorLiveMetrics": "独立",
    "inspectorFit": "",
}


def test_only_the_sections_the_design_captions_carry_a_right_hand_caption(qtbot, tmp_path) -> None:
    """右栏每段抬头的右端：设计稿写了说明的才有，其余留空。"""
    window = _window(qtbot, _project(tmp_path, structured=True))

    for name, caption in INSPECTOR_CAPTIONS.items():
        label = window.findChild(QLabel, f"{name}Subtitle")
        assert label is not None, name
        assert label.text() == caption, name


def test_the_live_metrics_caption_reports_the_batch_mode_in_force(qtbot, tmp_path) -> None:
    """实时指标抬头右端报的是此刻的批量模式，跟着项目走。

    设计稿帧④ 那一格写「联合」是因为那一帧正在联合拟合：右栏这句与顶栏那枚模式高亮
    读的是同一个字段，所以切了模式它必须跟着换，否则两处同屏互相打脸。tooltip 一起
    改——``theme.titled_card`` 用副标题当 tooltip，只换可见文字会留下一句过期的悬停。
    """
    project = _project(tmp_path, structured=True, count=2)
    window = _window(qtbot, project)
    caption = window.findChild(QLabel, "inspectorLiveMetricsSubtitle")
    assert caption is not None
    assert caption.text() == "独立"

    window.document.replace_project(api.set_batch_mode(project, "joint"))

    assert caption.text() == "联合"
    assert caption.toolTip() == "联合"


def test_the_selected_layer_header_writes_the_layer_name_with_one_separator(qtbot, tmp_path) -> None:
    """设计稿帧③ 的抬头是「选中层 · a-Si 非晶硅」：一整行里只有一个间隔点。

    层名自己可能带着间隔点（专家列表里写「a-Si · 非晶硅薄膜」）。原样接在「选中层 · 」
    后面就成了「选中层 · a-Si · 非晶硅薄膜」——三段之间两个同样的符号，读者分不出哪一
    个是抬头与层名的界，哪一个是层名内部的界。引导列表早就有这条收法：把层名内部那个
    间隔点收成空格。
    """
    window = _window(qtbot, _project(tmp_path, structured=True))
    structure = api.StructureSpec(AIR, (api.LayerSpec("a-Si · 非晶硅薄膜", SI, 487.0, roughness_a=4.0),), SI)
    window.structure_panel.editor.load(structure)

    window.structure_panel.editor._select_component(0)

    heading = window.findChild(QLabel, "inspectorSelectedLayerTitle")
    assert heading is not None
    assert heading.text() == "选中层 · a-Si 非晶硅薄膜"


def _correlated_report(candidate_id: str, correlations) -> api.UncertaintyReport:
    import numpy as np

    return api.UncertaintyReport(
        correlation_names=("component.0.thickness_a", "component.0.density"),
        correlation_matrix=np.array([[1.0, -0.72], [-0.72, 1.0]]),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=correlations,
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate_id,
    )


def _uncertainty_view(qtbot, correlations):
    from dataclasses import replace

    from tests.support.model_cases import final_fit_result, fit_candidate

    from xrr_fitter.gui.results.uncertainty import UncertaintyView

    candidate = fit_candidate("candidate-a", 0.2)
    result = replace(
        final_fit_result(candidate),
        uncertainty=_correlated_report("candidate-a", correlations),
    )
    view = UncertaintyView()
    qtbot.addWidget(view)
    view.set_result(result, "candidate-a")
    return view


def test_a_strong_correlation_is_read_out_not_just_listed_as_a_number(qtbot) -> None:
    """帧⑤ 相关矩阵旁的 ⚠ 提示框：说清 −0.72 意味着什么。

    「强相关：d/ρ=-0.72」只是把数字搬到了屏幕上。看得懂这个数的人不需要它，看不懂
    的人也不会因此改变读数方式——而这里要改变的恰恰是读数方式：两个参数纠缠时 ±1σ
    会低估真实不确定度，必须换用 Profile 似然判读。设计稿把这句判读放进提示框，就
    是要它在数字旁边被读到，而不是躺在证据清单的第四行里。
    """
    view = _uncertainty_view(qtbot, (("component.0.thickness_a", "component.0.density", -0.72),))

    callout = view.findChild(QLabel, "uncertaintyCorrelationCallout")
    assert callout is not None, "相关矩阵旁没有判读提示"
    assert callout.isVisibleTo(view)
    assert "低估" in callout.text()
    assert "Profile 似然" in callout.text()
    assert callout.property("hintBox") is True, "判读提示没做成设计稿的提示框"


def test_the_callout_asks_for_its_own_height_instead_of_taking_the_evidence(qtbot) -> None:
    """The view sizes itself by evidence lines, and the callout is not one of them.

    ``sizeHint`` counts text lines plus the card header. A third widget appearing
    inside the same card without being counted comes out of the only flexible
    thing in there — the evidence box — so the caution would arrive by clipping
    the report it is commenting on.
    """
    pair = (("component.0.thickness_a", "component.0.density", -0.72),)
    with_callout = _uncertainty_view(qtbot, pair)
    without = _uncertainty_view(qtbot, ())

    callout_height = with_callout.correlation_callout.sizeHint().height()
    assert callout_height > 0
    grew = with_callout.sizeHint().height() - without.sizeHint().height()
    assert grew >= callout_height, "提示框没有为自己要高度，会挤掉证据行"


def test_every_callout_bolds_the_clause_that_says_what_happened(qtbot) -> None:
    """设计稿的 ``.hint b`` 把首句加粗，让提示能被扫读而不必逐字读完。

    三条提示的正文都是「为什么」和「怎么办」，两三行；真正要先看到的是第一句——发生了
    什么。全篇同一个字重时，一条两行的旁注要读完才知道值不值得读，于是多半不读。
    """
    from xrr_fitter.gui.guidance.panel import OXIDE_TIP_TEXT
    from xrr_fitter.gui.results.uncertainty import CORRELATION_CALLOUT_TEXT
    from xrr_fitter.gui.window_layout import STRUCTURE_CORRELATION_HINT_TEXT

    leads = {
        OXIDE_TIP_TEXT: "已自动加入表面氧化层。",
        STRUCTURE_CORRELATION_HINT_TEXT: "厚度 d 与密度 ρ 可能相关。",
        CORRELATION_CALLOUT_TEXT: "存在强相关参数：",
    }
    for text, lead in leads.items():
        assert f"<b>{lead}</b>" in text, f"首句没有加粗：{lead}"


def test_no_strong_correlation_leaves_the_callout_away(qtbot) -> None:
    """A caution that is always on screen stops being read as a caution.

    The callout speaks to a specific finding — two parameters that traded off in
    this fit — so with no such pair it would be asserting something the evidence
    does not say.
    """
    view = _uncertainty_view(qtbot, ())

    callout = view.findChild(QLabel, "uncertaintyCorrelationCallout")
    assert callout is None or not callout.isVisibleTo(view)


def _thin_curve(path: Path, count: int = 24) -> Path:
    """一条读得出来、但有效点不足 30 的曲线——设计稿那一行「参考」曲线的真实来源。"""
    path.write_text(
        "\n".join(f"{0.10 + index * 0.10:.6f} {1000.0 / (index + 1):.12g}" for index in range(count)) + "\n",
        encoding="utf-8",
    )
    return path


def _blank_source(path: Path) -> Path:
    """一个连续两行数字都找不到的文件：读取器在这里抛 ``ValueError``。"""
    path.write_text("# 无数值列\n", encoding="utf-8")
    return path


def test_the_import_dialog_heads_itself_with_the_batch_preview_title(qtbot, tmp_path) -> None:
    """帧⑥a 的 ``.dh``：「📥 导入数据 · 批量预览」。

    窗管标题栏在设计稿里画的是应用外壳，而这张对话框是画在画面正中的一张卡；卡上没有
    标题栏，说明「这是哪张对话框」的只有这一行。之前只有窗口标题「导入 XRR 数据」，
    截图里那行字根本不在像素里。
    """
    from xrr_fitter.gui.data.import_dialog import DIALOG_HEADING, ImportDialog

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)

    heading = dialog.findChild(QLabel, "importDialogHeading")

    assert heading is not None and heading.isVisibleTo(dialog)
    assert heading.text() == DIALOG_HEADING == "📥 导入数据 · 批量预览"


def test_the_import_preview_names_the_angle_then_the_columns_it_read(qtbot, tmp_path) -> None:
    """帧⑥a 的表头是五列：文件 · 角度 · 列映射 · 行数 · 状态。

    行数与状态只说解析成功了，不说解析成了什么；一个把强度读成 2θ 的文件同样报
    「✓ 就绪」，而错在哪只有把这次实际用的角度约定与列写出来才看得见。列映射照设计稿
    写成 ``2θ列 → 强度列``，列号从 1 数起——文件里数得出来的是第一列、第二列，界面上
    不该出现「第 0 列」这种打开文件也数不出来的东西；``DataColumnMapping`` 那边照旧存 0 基。
    """
    from xrr_fitter.gui.data.import_dialog import PREVIEW_ANGLE_TWO_THETA, PREVIEW_HEADERS, ImportDialog

    assert PREVIEW_HEADERS == ("文件", "角度", "列映射", "行数", "状态")

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)
    table = dialog.preview_table

    assert table.columnCount() == len(PREVIEW_HEADERS)
    assert table.item(0, 1).text() == PREVIEW_ANGLE_TWO_THETA == "2θ"
    assert table.item(0, 2).text() == "1 → 2"

    # 覆盖映射后这一格要跟着走：偏移量得落在每一个字段上，不只落在默认值上。
    wide = ImportDialog((_write_wide_curve(tmp_path / "five_columns.xy"),))
    qtbot.addWidget(wide)
    wide.set_column_mapping(two_theta=2, intensity=3, intensity_sigma=4)

    assert wide.preview_table.item(0, 2).text() == "3 → 4 · σ:5"


def test_a_file_that_cannot_be_parsed_leaves_the_angle_and_mapping_cells_empty(qtbot, tmp_path) -> None:
    """设计稿第三行：解析不了的文件，角度与列映射都是 ``—``。

    读不出数据列时「按 2θ 读了第 1 列」根本没发生过，把默认映射照写上去会读成
    「列没问题，只是行数是 0」。破折号说的是「这一格无从谈起」。
    """
    from xrr_fitter.gui.data.import_dialog import PREVIEW_FAILED, PREVIEW_UNKNOWN, ImportDialog

    dialog = ImportDialog((_blank_source(tmp_path / "blank_run.dat"),))
    qtbot.addWidget(dialog)
    table = dialog.preview_table

    assert table.item(0, 1).text() == PREVIEW_UNKNOWN == "—"
    assert table.item(0, 2).text() == PREVIEW_UNKNOWN
    assert table.item(0, 3).text() == "0"
    assert table.item(0, 4).text() == PREVIEW_FAILED == "✕ 无数据列"


def test_a_parseable_but_thin_file_takes_the_third_badge_tier(qtbot, tmp_path) -> None:
    """设计稿状态列有三档：``ok`` / ``info`` / ``error``——中间那档不是失败也不是就绪。

    24 行的参考曲线读得出来（角度、列映射、行数都成立），但有效唯一点不足 30，
    ``PreparedData.fit_ready`` 因此是 ``False``：导入照做，拟合用不了。两档配色会把它
    并进「✓ 就绪」，读者要到导入之后才发现这条曲线拟不了。
    """
    from xrr_fitter.gui.data.import_dialog import PREVIEW_THIN, ImportDialog

    dialog = ImportDialog((_thin_curve(tmp_path / "Si_substrate_ref.xy"),))
    qtbot.addWidget(dialog)
    table = dialog.preview_table
    tokens = theme.palette_tokens(dialog.palette())

    assert table.item(0, 1).text() == "2θ"
    assert table.item(0, 3).text() == "24"
    status = table.item(0, 4)
    assert status.text() == PREVIEW_THIN == "ℹ 点数不足"
    assert status.foreground().color().name().lower() == QColor(tokens.info).name().lower()
    assert status.toolTip()


def test_the_dialog_states_the_angle_convention_and_the_delimiter_it_detected(qtbot, tmp_path) -> None:
    """帧⑥a 的第一组 ``.wrap2``：角度约定（全局默认）＋ 列分隔符。

    两档角度约定都是真的：``2θ（衍射角）`` 照原样读，``θ（掠射角）`` 在导入时把角度列
    ×2 归一到散射角——模型内部的轴始终是 ``two_theta_deg``，两条路径之后走同一套计算。
    分隔符那一栏才是真的只有一档：每行先把逗号换成空白再切分，空白与逗号一律认得，所以
    它灰着并带一句说明，比摆一个按不下去也没别的可选的下拉更接近实情。
    """
    from xrr_fitter.gui.data.import_dialog import (
        ANGLE_THETA_TEXT,
        ANGLE_TWO_THETA_TEXT,
        DELIMITER_TEXT,
        ImportDialog,
    )

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)

    two_theta = dialog.two_theta_convention
    theta = dialog.theta_convention
    delimiter = dialog.delimiter_select

    assert two_theta.text() == ANGLE_TWO_THETA_TEXT == "2θ（衍射角）"
    assert two_theta.isChecked() and two_theta.isEnabled()
    assert theta.text() == ANGLE_THETA_TEXT == "θ（掠射角）"
    assert theta.isEnabled() and theta.toolTip()
    assert [delimiter.itemText(index) for index in range(delimiter.count())] == [DELIMITER_TEXT]
    assert delimiter.currentText() == "自动检测（空白/逗号）"
    assert delimiter.toolTip()


def test_the_advanced_box_states_the_header_skip_and_the_log_option_it_does_not_have(qtbot, tmp_path) -> None:
    """帧⑥a 的 ``高级`` 组：跳过表头行「自动」＋ 强度列取对数「是（若为线性计数）」。

    表头是真自动的：读取器从第一处「连续两行都是数字」开始取数，所以这一格只读不写。
    取对数则整个不存在——``io/xy.py`` 里唯一带 log 的是拟合窗口的 ``log_domain_mask``，
    导入这一步按线性计数读强度列。设计稿那枚方框本来也是未勾选的，所以灰着＋未勾选正好
    是一句真话，而不是一个按了没反应的开关。
    """
    from xrr_fitter.gui.data.import_dialog import (
        ADVANCED_TITLE,
        HEADER_SKIP_TEXT,
        LOG_INTENSITY_TEXT,
        ImportDialog,
    )

    dialog = ImportDialog((_write_curve(tmp_path / "sample.xy"),))
    qtbot.addWidget(dialog)

    box = dialog.advanced_box
    assert box.title() == ADVANCED_TITLE == "高级"
    assert dialog.header_skip_field.text() == HEADER_SKIP_TEXT == "自动"
    assert dialog.header_skip_field.isReadOnly()
    log_option = dialog.log_intensity_check
    assert log_option.text() == LOG_INTENSITY_TEXT == "是（若为线性计数）"
    assert not log_option.isChecked() and not log_option.isEnabled()
    assert log_option.toolTip()


def test_the_import_dialog_drops_the_single_file_preview_plot(qtbot, tmp_path) -> None:
    """设计稿这张对话框里没有曲线图，摘掉它是为了给那两组 ``.wrap2`` 腾出高度。

    单文件曲线图占 150–200px，而它只画 ``paths[0]``：批量导入里其余文件在图上没有出处，
    而每个文件的行数与状态在表里已经逐行写着。解析报错仍然进状态格的 tooltip，
    所以那句错误没有跟着图一起消失。
    """
    from xrr_fitter.gui.data.import_dialog import ImportDialog

    dialog = ImportDialog((_blank_source(tmp_path / "blank_run.dat"),))
    qtbot.addWidget(dialog)

    assert dialog.findChild(QWidget, "importPreviewPlot") is None
    assert dialog.findChild(QLabel, "importPreviewError") is None
    assert dialog.preview_table.item(0, 4).toolTip().startswith("ValueError")


def test_the_export_dialog_heads_itself_and_offers_the_vector_profile(qtbot) -> None:
    """帧⑥b 的 ``.dh``「📤 导出结果」，以及默认勾上的「SLD 深度剖面 SVG（矢量）」。

    两张对话框在设计稿里是并排的两张卡，各自靠头一行说明自己是哪张；没有这一行时
    截图里只剩三组选项，读者要靠页脚按钮反推这是导出。矢量剖面是 ``include_svg``
    这个真开关，设计稿把它画成勾上的——它是投稿要的那一份，默认给出来。
    """
    from xrr_fitter.gui.export.dialog import EXPORT_HEADING, OrtOptionDialog

    dialog = OrtOptionDialog()
    qtbot.addWidget(dialog)

    heading = dialog.findChild(QLabel, "exportDialogHeading")

    assert heading is not None and heading.isVisibleTo(dialog)
    assert heading.text() == EXPORT_HEADING == "📤 导出结果"
    assert dialog.include_svg is True


# 设计稿帧① 431-450：检视器第二段是「参数 · 结果值〈12 自由〉」，四列 参数 / 结果值 / ±1σ /
# 单位。层名在分组行上写成「说明 · 名字」，层内三行按 厚度 → 粗糙度 → 密度 排，密度写的是
# 绝对密度（g·cm⁻³）而不是相对密度；基底那组两行，密度是锁定的；仪器那组只有强度标度与本底。
C_SI = api.MaterialSpec("c-Si · 晶体硅基底", "Si", 2.329)
DESIGN_OXIDE_DENSITY = 2.19
DESIGN_FILM_DENSITY = 2.28
DESIGN_VALUES = {
    "component.0.thickness_a": 34.2,
    "component.0.roughness_a": 5.1,
    "component.0.density_scale": DESIGN_OXIDE_DENSITY / 2.20,
    "component.1.thickness_a": 487.0,
    "component.1.roughness_a": 4.4,
    "component.1.density_scale": DESIGN_FILM_DENSITY / 2.329,
    "backing.roughness_a": 3.0,
    "instrument.scale": 0.982,
    "instrument.background": 3.1e-7,
}
DESIGN_VALUE_ROWS = (
    ("表面氧化层 · SiO₂",),
    ("厚度 d", "3.420", "nm"),
    ("粗糙度 σ", "0.510", "nm"),
    ("密度 ρ", "2.190", "g·cm⁻³"),
    ("非晶硅薄膜 · a-Si",),
    ("厚度 d", "48.700", "nm"),
    ("粗糙度 σ", "0.440", "nm"),
    ("密度 ρ", "2.280", "g·cm⁻³"),
    ("晶体硅基底 · c-Si",),
    ("粗糙度 σ", "0.300", "nm"),
    ("密度 ρ", "2.329", "g·cm⁻³"),
    ("仪器",),
    ("强度标度 scale", "0.982", "—"),
    ("本底 background", "3.1e-7", "—"),
)


def _design_structure() -> api.StructureSpec:
    """设计稿那两层：厚度/粗糙度按图上的 nm 读数折回 Å，密度按绝对密度折回相对值。"""
    return api.StructureSpec(
        AIR,
        (
            api.LayerSpec(
                "SiO₂ · 表面氧化层",
                SIO2,
                34.2,
                roughness_a=5.1,
                density_scale=DESIGN_VALUES["component.0.density_scale"],
            ),
            api.LayerSpec(
                "a-Si · 非晶硅薄膜",
                SI,
                487.0,
                roughness_a=4.4,
                density_scale=DESIGN_VALUES["component.1.density_scale"],
            ),
        ),
        C_SI,
        backing_roughness_a=3.0,
    )


def _fitted_project(tmp_path: Path):
    """一份带结果的真实项目：声明来自编译器，值按设计稿的读数填。

    值不是手写一张声明表凑出来的——那样会把「表里怎么排」和「求解器交回来什么」两件事
    混成一件。这里让 ``describe_parameters`` 给出真正的 17 条声明，逐条取设计稿的读数、
    其余取声明自己的初值，于是候选解覆盖每一条声明，表格拿到的是它在应用里真会拿到的东西。
    """
    project = api.new_project()
    project = api.add_dataset(project, _write_curve(tmp_path / "curve-0.xy"), api.InstrumentSpec())
    project = api.set_structure(project, "curve-0", _design_structure())
    definitions = tuple(api.describe_parameters(project, "curve-0"))
    values = tuple(
        ParameterValue(
            definition.name,
            DESIGN_VALUES.get(definition.name, definition.initial),
            definition.lower,
            definition.upper,
        )
        for definition in definitions
    )
    points = CURVE_POINTS
    candidate = replace(
        fit_candidate("candidate-a", 0.2),
        parameters=values,
        unit_vector=np.zeros(len(values)),
        qz_a_inv=np.linspace(0.01, 0.2, points),
        model_normalized=np.linspace(1.0, 0.1, points),
        log_residuals_decades=np.zeros(points),
        weighted_residuals=np.zeros(points),
    )
    result = replace(final_fit_result(candidate), parameter_definitions=definitions)
    dataset = replace(project.datasets[0], last_valid_result=result)
    project = replace(project, datasets=(dataset,))
    return api.select_active_dataset(project, "curve-0")


def _value_rows(table: QTableWidget) -> tuple[tuple[str, ...], ...]:
    """把结果值表读成设计稿那张表：分组行只有名字，参数行读 名字 / 结果值 / 单位。"""

    def text(row: int, column: int) -> str:
        item = table.item(row, column)
        return "" if item is None else item.text()

    rows = []
    for row in range(table.rowCount()):
        if table.columnSpan(row, 0) == table.columnCount():
            rows.append((text(row, 0),))
        else:
            rows.append((text(row, 0), text(row, 1), text(row, 3)))
    return tuple(rows)


def test_the_result_value_table_reads_as_the_design_frame_one(qtbot, tmp_path) -> None:
    """帧① 的「参数 · 结果值」逐行对齐：命名、组内顺序、绝对密度、仪器只留两行。

    这张表此前读的是声明的原文——「相对密度 ρ」配一个没有单位的 0.995、「入射侧粗糙度 σ」、
    「尺度」、「常数背景」，加上十行仪器机械量。设计稿读的是物理量本身：密度是能跟文献对
    照的 g·cm⁻³，粗糙度就叫粗糙度，仪器那组只留下每条曲线都要拟合的强度标度与本底。
    抬头的「12 自由」数的仍是全部自由声明，所以表里少画的那几行不会被读成少拟合的参数。
    """
    window = _window(qtbot, _fitted_project(tmp_path))

    table = window.findChild(QTableWidget, "resultValueTable")
    subtitle = window.findChild(QLabel, "resultValuesCardSubtitle")

    assert table is not None and subtitle is not None
    assert _value_rows(table) == DESIGN_VALUE_ROWS
    assert subtitle.text() == "12 自由"


def test_the_locked_substrate_density_row_reads_as_locked(qtbot, tmp_path) -> None:
    """设计稿基底那组第二行是「密度 ρ ▣」配 ±1σ 栏的「锁定」。

    基底密度不是求解量：它是那块衬底的体密度，写在 ``MaterialSpec`` 上，没有对应的可拟合
    声明。设计稿把它画成锁定行而不是省掉，因为读者要拿它跟上面两层的密度比——所以这一行
    是显示层合成的，值取自材料本身，并且必须带上锁定标记，否则它会被读成一个拟合出来的值。
    """
    from xrr_fitter.gui.results.values import LOCKED_ROLE

    window = _window(qtbot, _fitted_project(tmp_path))
    table = window.findChild(QTableWidget, "resultValueTable")

    row = DESIGN_VALUE_ROWS.index(("密度 ρ", "2.329", "g·cm⁻³"))

    assert table.item(row, 2).text() == "锁定"
    assert table.item(row, 0).data(LOCKED_ROLE) is True
    assert "bulk_density_g_cm3" in table.item(row, 0).toolTip()


def test_the_correlation_hint_names_a_page_the_canvas_actually_has(qtbot, tmp_path) -> None:
    """结构诊断那句指路语点的页名，得就是画布 tab 条上那一格的名字。

    设计稿把这一页称作「不确定度」页（HTML 724），实现把那句话逐字还原进了
    ``STRUCTURE_CORRELATION_HINT_TEXT``；而承载它的 tab 一度叫「相关性与区间」。读者照提示
    去 tab 条上找「不确定度」是找不到的——一句指向界面里不存在的名字的指路语，比不指路更
    糟：它让读者以为自己漏看了什么。

    页名从提示语里取、tab 名从发货的画布上读，两处任何一方改名都会在这里红，而不是各自
    对着一个写死的中文串。
    """
    from xrr_fitter.gui.window_layout import STRUCTURE_CORRELATION_HINT_TEXT

    window = _window(qtbot, _fitted_project(tmp_path))
    page_name = re.search(r"「(.+?)」", STRUCTURE_CORRELATION_HINT_TEXT)

    assert page_name is not None
    assert page_name.group(1) in window.plot_panel.tab_titles()
