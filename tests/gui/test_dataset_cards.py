"""帧① 左栏的数据集卡片：字形 + 名字 + 一行小字，以及它上面那条分区抬头。

设计稿把每个数据集画成一张两行卡（``.ds``）：左边一枚置信度字形，右边上行文件名、
下行一句小字；列表上方是一条 ``数据集 ＋`` 的分区抬头（``.nav-sec``）。应用此前把同样
的数据摊成三列表格行——名字、状态、拟合各占一列，264px 的窄栏里三列互相挤宽度。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter
from PySide6.QtWidgets import QLabel, QStyleOptionViewItem, QToolButton, QTreeWidget, QVBoxLayout, QWidget
from tests.gui.paint_support import painted_at
from tests.gui.plot_support import _project_with_curves, _write_curve
from tests.support.model_cases import final_fit_result, fit_candidate

import xrr_fitter.api as api
from xrr_fitter.model.analysis import ConfidenceClass


def _panel(qtbot, document=None):
    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    panel = DataPanel(ProjectDocument(api.new_project()) if document is None else document)
    qtbot.addWidget(panel)
    return panel


def _one_dataset_panel(qtbot, tmp_path: Path, name: str = "sample"):
    document_project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / f"{name}.xy"),
        api.InstrumentSpec(instrument_id=name),
    )
    from xrr_fitter.gui.document import ProjectDocument

    return _panel(qtbot, ProjectDocument(document_project))


def _fitted_document(tmp_path: Path, verdict: ConfidenceClass, name: str = "fitted"):
    """一个源文件真实存在、且已有拟合判定的库。

    源文件必须真在磁盘上：``ProjectDocument`` 构造时就跑 ``inspect_sources``，替身路径
    会被判成「源文件缺失」，而那句话在小字里优先级高于拟合判定——夹具会盖掉被测的那件事。
    所以数据集走真实导入（顺带算出对得上的 SHA），只有拟合结果是替身。
    """
    from xrr_fitter.gui.document import ProjectDocument

    value = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / f"{name}.xy"),
        api.InstrumentSpec(instrument_id=name),
    )
    result = replace(final_fit_result(fit_candidate("candidate-a", 0.2)), confidence=verdict)
    dataset = replace(value.datasets[0], last_valid_result=result)
    return ProjectDocument(replace(value, datasets=(dataset,)))


def test_the_card_carries_the_glyph_the_name_and_one_line_of_small_print(qtbot, tmp_path: Path) -> None:
    """卡片有三段：字形、名字、小字。设计稿 ``.ds`` 就是这个形状。"""
    from xrr_fitter.gui.data.dataset_card import dataset_card

    panel = _one_dataset_panel(qtbot, tmp_path)

    card = dataset_card(panel, "sample")

    assert card.title == "sample"
    assert card.subline == "θ/2θ · 32 点"
    assert card.glyph and card.kind


def test_an_unfitted_dataset_takes_the_hollow_ring_so_the_titles_stay_in_one_column(
    qtbot,
    tmp_path: Path,
) -> None:
    """未拟合也要有字形。

    ``fit_status_marker`` 对未拟合返回空串，那是给表格单元用的——旁边一格已经写着
    「未拟合」，再补一枚实心字形会读成「有结果」。卡片里空串的代价不同：字形格空了，
    这一行的名字就比上下行左移一格，四行卡片的名字对不齐一列。空心环读作「还没测」而
    不是第五档可信度，所以它填这个格。
    """
    from xrr_fitter.gui import theme
    from xrr_fitter.gui.data.dataset_card import dataset_card

    panel = _one_dataset_panel(qtbot, tmp_path)

    card = dataset_card(panel, "sample")

    assert card.glyph == theme.CONFIDENCE_FALLBACK_GLYPH
    assert card.kind == theme.CONFIDENCE_FALLBACK_KIND


@pytest.mark.parametrize(
    ("verdict", "glyph", "kind"),
    (
        (ConfidenceClass.TRUSTED, "●", "ok"),
        (ConfidenceClass.CORRELATED, "◆", "info"),
        (ConfidenceClass.MULTIPLE, "▲", "warn"),
        (ConfidenceClass.UNTRUSTED, "■", "error"),
    ),
)
def test_each_verdict_takes_its_own_shape_and_its_own_colour(qtbot, tmp_path: Path, verdict, glyph, kind) -> None:
    """四档判定四种形状四种颜色——这是并排比较的列表，不折叠任何一档。

    命令栏那条状态栏把「可用但相关」折到 warn 上（它没有并排对象，少一种颜色更清楚），
    这里不折：四行卡片摆在一起时，两档共用一色就没法区分谁是哪一档。
    """
    from xrr_fitter.gui.data.dataset_card import dataset_card

    panel = _panel(qtbot, _fitted_document(tmp_path, verdict))

    card = dataset_card(panel, "fitted")

    assert (card.glyph, card.kind) == (glyph, kind)


def test_the_small_print_reads_the_point_count_then_the_one_fact_that_matters_most(
    qtbot,
    tmp_path: Path,
) -> None:
    """小字只有两段，第二段是此刻最要紧的那句，不是把三句判定串起来。

    设计稿写「512 点 · 已拟合」——点数加一句状态。应用手上有两句：源校验加可拟合性
    （``status_text``），以及拟合判定（``fit_status_text``）。两句都写就成了
    「32 点 · 可拟合 · 未拟合」，读者要先绕过一句「能拟合」才看到「还没拟合」。所以
    第二段按紧急度取一句：源文件出问题说源文件，点数不够说点数不够，其余说这一集该怎么办。
    """
    from xrr_fitter.gui.data.dataset_card import dataset_card

    fitted = _panel(qtbot, _fitted_document(tmp_path, ConfidenceClass.TRUSTED))
    assert dataset_card(fitted, "fitted").subline == "θ/2θ · 32 点 · 已拟合"
    assert "可拟合" not in dataset_card(fitted, "fitted").subline

    unfitted = _one_dataset_panel(qtbot, tmp_path)
    assert dataset_card(unfitted, "sample").subline == "θ/2θ · 32 点"


def test_a_broken_source_outranks_the_fit_verdict_in_the_small_print(qtbot, tmp_path: Path) -> None:
    """源文件没了就先说这件事，并带上那枚警示字形——旧判定此刻不再回答「能不能用」。"""
    from xrr_fitter.gui.data.dataset_card import dataset_card

    source = _write_curve(tmp_path / "sample.xy")
    panel = _panel(qtbot)
    panel.add_paths((source,), beam=api.BeamSpec("monochromatic"), instrument=api.InstrumentSpec("sample"))
    source.unlink()
    panel.document.refresh_sources()

    card = dataset_card(panel, "sample")

    assert card.subline == "⛔ θ/2θ · 32 点 · 源文件缺失"


def test_the_list_shows_one_column_and_no_header_row(qtbot, tmp_path: Path) -> None:
    """卡片是一列自绘的行，所以表头和其余六列都收起来。

    七列仍在模型里——详情标签和各列的 tooltip 都从它们读，删列会连带删掉那些出处。
    收起的是显示：设计稿的 ``.ds`` 列表没有表头，也没有第二列。
    """
    panel = _one_dataset_panel(qtbot, tmp_path)

    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None
    assert tree.isHeaderHidden()
    assert not tree.isColumnHidden(0)
    assert all(tree.isColumnHidden(column) for column in range(1, tree.columnCount()))


def test_the_row_hands_the_delegate_its_card_and_keeps_prose_for_the_screen_reader(
    qtbot,
    tmp_path: Path,
) -> None:
    """自绘的四段挂在自定义角色上，DisplayRole 仍是一句能读出来的散文。"""
    from xrr_fitter.gui.data.dataset_card import DATASET_CARD_ROLE, DatasetCard, DatasetCardDelegate

    panel = _one_dataset_panel(qtbot, tmp_path)
    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None
    item = tree.topLevelItem(0)

    assert isinstance(item.data(0, DATASET_CARD_ROLE), DatasetCard)
    assert item.text(0) == "sample"
    assert isinstance(tree.itemDelegateForColumn(0), DatasetCardDelegate)


def test_the_card_is_tall_enough_for_two_lines_of_text(qtbot, tmp_path: Path) -> None:
    """两行卡就得有两行高——照单行行高摆，小字会被裁掉。"""
    panel = _one_dataset_panel(qtbot, tmp_path)
    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None

    height = tree.sizeHintForRow(0)

    assert height >= 2 * tree.fontMetrics().height()


def test_the_rail_heads_the_list_with_a_dataset_section_and_a_plus(qtbot, tmp_path: Path) -> None:
    """设计稿的分区抬头：左边「数据集」，右边一枚 ``＋``。"""
    panel = _one_dataset_panel(qtbot, tmp_path)

    heading = panel.findChild(QLabel, "dataPanelHeader")
    add = panel.findChild(QToolButton, "datasetAddButton")

    assert heading is not None and heading.isVisibleTo(panel)
    assert heading.text() == "数据集"
    assert add is not None and add.isVisibleTo(panel)
    assert add.text() == "＋"
    assert add.accessibleName() == "导入数据集"
    assert add.toolTip()


# 一个像素与底色的差距，三通道相加。抗锯齿的边缘落在二三十上下，一笔实墨在两百以上，
# 门槛压在两者之间：空白与字形才分得开。
INK_DELTA = 60


def _ink_span(host, widget) -> tuple[int, int]:
    """控件在屏幕上落墨的最左、最右两列，折回控件自己的逻辑坐标。

    量的是宿主的截图而不是控件自己的：``QToolButton`` 的底由它坐的那一栏铺，按钮单独
    ``grab()`` 出来的缓冲区没有那层底，底色取不准，空白与字形就分不开。Retina 上截图是
    逻辑尺寸的两倍，所以按实际比例折回去。
    """
    image = host.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    scale = image.width() / max(host.width(), 1)
    origin = widget.mapTo(host, QPoint(0, 0))
    background = QColor.fromRgba(image.pixel(round(origin.x() * scale), round(origin.y() * scale)))
    inked = []
    for x in range(widget.width()):
        for y in range(widget.height()):
            colour = QColor.fromRgba(image.pixel(round((origin.x() + x) * scale), round((origin.y() + y) * scale)))
            delta = (
                abs(colour.red() - background.red())
                + abs(colour.green() - background.green())
                + abs(colour.blue() - background.blue())
            )
            if delta > INK_DELTA:
                inked.append(x)
                break
    assert inked, "控件上一个像素的墨都没有"
    return inked[0], inked[-1]


def test_the_plus_stands_alone_without_qts_drop_down_arrow(qtbot, tmp_path: Path) -> None:
    """设计稿 ``.nav-sec .add`` 是一枚 ``＋``，它右边什么都没有。

    ＋ 背后挂着一只 ``QMenu``（三条导入命令住在里面），而 Qt 只要看见菜单就替按钮画一枚
    ``::menu-indicator`` 下拉箭头——那枚 ``⌄`` 吃掉按钮右边一半，抬头于是读成「数据集 ＋⌄」，
    而设计稿那一行只有一枚字形。菜单要留着（展开仍然选文件夹、换预设），画出来的箭头不能留。
    """
    from xrr_fitter.gui import theme

    panel = _one_dataset_panel(qtbot, tmp_path)
    host = _railed(qtbot, panel)
    button = panel.findChild(QToolButton, "datasetAddButton")
    assert button is not None

    left, right = _ink_span(host, button)
    glyph = QFontMetrics(button.font()).horizontalAdvance(button.text())

    # 落墨的那一段只够放一枚 ``＋``；箭头会把它撑到两枚字形宽。
    assert right - left + 1 <= glyph + 2, f"墨迹横跨 {right - left + 1}px，一枚 ＋ 只有 {glyph}px 宽"
    # 按钮本身也该收到字形上：设计稿那枚 ``＋`` 是行内 span，没有边框也没有内边距。
    assert button.width() <= glyph + 2 * theme.SPACE_XS, f"按钮 {button.width()}px 宽，字形只有 {glyph}px"


def test_the_joint_run_heads_the_list_with_the_joint_section(qtbot, tmp_path: Path) -> None:
    """帧④ 的分区抬头写「联合拟合 · 数据集」，不只是「数据集」。

    联合批量下这一列不再是几条各自拟合的曲线，而是一次共享层结构的运行的成员表；抬头
    是读者判断「下面这三行是三次拟合还是一次」的第一处线索。
    """
    from xrr_fitter.gui.document import ProjectDocument

    project = api.set_batch_mode(_project_with_curves(tmp_path, 3), "joint")
    panel = _panel(qtbot, ProjectDocument(project))

    heading = panel.findChild(QLabel, "dataPanelHeader")

    assert heading is not None and heading.text() == "联合拟合 · 数据集"


def _joint_panel(qtbot, tmp_path: Path):
    from xrr_fitter.gui.document import ProjectDocument

    project = api.set_batch_mode(_project_with_curves(tmp_path, 3), "joint")
    return _panel(qtbot, ProjectDocument(project))


def _progress(objective: float) -> api.FitProgress:
    """一次联合运行的进度事件：``dataset_id`` 是 ``None``，J 是三条曲线共享的那一个。"""
    return api.FitProgress(
        dataset_id=None,
        stage="refine",
        completed=620,
        total=1000,
        best_objective=objective,
        message="",
    )


def test_a_running_dataset_says_it_is_fitting_and_where_the_objective_is(qtbot, tmp_path: Path) -> None:
    """帧④ 的每一行小字写「拟合中 · J=2.14↓」——此刻这条曲线在被拟合，收敛到哪儿。

    不跑的时候这一行写的是点数与拟合判定，两句都是上一轮的旧事实；运行中把它们摆在
    进度条旁边，会被读成本轮的读数。这和右栏在运行中收到只剩运行那一段是同一个理由。
    """
    from xrr_fitter.gui.data.dataset_card import dataset_card

    panel = _joint_panel(qtbot, tmp_path)

    panel.set_running(True)
    panel.set_run_progress(_progress(2.5))
    panel.set_run_progress(_progress(2.14))

    card = dataset_card(panel, "curve-0")
    assert card.subline == "拟合中 · J=2.14↓"
    assert card.glyph == "◐"
    assert card.kind == "accent"


def test_the_finished_run_gives_the_cards_their_verdict_back(qtbot, tmp_path: Path) -> None:
    """跑完之后小字回到点数与判定：运行中的 J 已经是上一轮的数了。"""
    from xrr_fitter.gui.data.dataset_card import dataset_card

    panel = _joint_panel(qtbot, tmp_path)
    panel.set_running(True)
    panel.set_run_progress(_progress(2.14))

    panel.set_running(False)

    assert "拟合中" not in dataset_card(panel, "curve-0").subline


def test_the_plus_holds_the_three_import_commands_so_the_rail_needs_no_button_row(
    qtbot,
    tmp_path: Path,
) -> None:
    """三个导入命令挂在 ``＋`` 的菜单里，抬头那行因此只有一枚字形宽。

    这三颗按钮此前并排摊在抬头行里，光文字就要 152px，专家模式再加「更换测量预设」
    共 242px——264px 的栏放不下，第三颗量出来是 0 宽。设计稿那行只有 ``数据集 ＋``。
    按钮本身留着：可及名、tooltip 和点击处理都在它们身上，菜单托的就是同一颗按钮，
    所以既有的 ``findChild(QPushButton, "importFilesButton").click()`` 照旧成立。
    """
    from PySide6.QtWidgets import QPushButton, QWidgetAction

    panel = _one_dataset_panel(qtbot, tmp_path)
    add = panel.findChild(QToolButton, "datasetAddButton")
    assert add is not None
    menu = add.menu()
    assert menu is not None

    hosted = {
        action.defaultWidget().objectName()
        for action in menu.actions()
        if isinstance(action, QWidgetAction) and action.defaultWidget() is not None
    }
    assert hosted == {"importFilesButton", "importFolderButton", "changeMeasurementPresetButton"}
    for name in hosted:
        assert isinstance(panel.findChild(QPushButton, name), QPushButton)


def test_the_plus_opens_the_file_chooser_directly(qtbot, tmp_path: Path, monkeypatch) -> None:
    """点字形本身走最常用那条路（导入文件），不必先展开菜单。"""
    calls: list[int] = []
    panel = _one_dataset_panel(qtbot, tmp_path)
    monkeypatch.setattr(panel, "_import_files", lambda: calls.append(1))
    add = panel.findChild(QToolButton, "datasetAddButton")
    assert add is not None

    add.defaultAction().trigger()

    assert calls == [1]


def test_the_active_card_is_the_selected_row(qtbot, tmp_path: Path) -> None:
    """``.ds.on`` 是选中态：活动数据集就是列表的当前行，卡片不另存一份「谁在亮」。"""
    from xrr_fitter.gui.document import ProjectDocument

    panel = _panel(qtbot, ProjectDocument(_project_with_curves(tmp_path, count=2)))
    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None

    active = tree.currentItem()

    assert active is not None
    assert str(active.data(0, Qt.ItemDataRole.UserRole)) == panel.active_dataset_id


def _railed(qtbot, panel):
    """把数据集列表摆回它在左栏里的真实位置：一只名叫 ``navigationColumn`` 的机架里，套上主题。

    列表自己不该挑底色——屏幕上那块底是它下面那一栏铺的。宿主不叫这个名字、样式表不套上去，
    量到的就只是 Qt 默认的内容色，那不是应用发货时的样子。调色板取 ``light_palette()``——
    发货时装的就是它——断言于是不依赖跑测试那台机器的配色。
    """
    from xrr_fitter.gui import theme

    host = QWidget()
    host.setObjectName("navigationColumn")
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(panel)
    palette = theme.light_palette()
    host.setPalette(palette)
    host.setStyleSheet(theme.build_stylesheet(palette))
    host.resize(264, 360)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    return host


def _row_edge_colour(host, tree: QTreeWidget, row: int) -> QColor:
    """量一行最右边那块空处的颜色——委托的字形与两行文字都不画到那里。"""
    rect = tree.visualItemRect(tree.topLevelItem(row))
    spot = tree.viewport().mapTo(host, QPoint(rect.right() - 4, rect.center().y()))
    return painted_at(host, spot)


def test_the_rows_sit_straight_on_the_rail_and_only_the_active_one_takes_the_panel_white(
    qtbot,
    tmp_path: Path,
) -> None:
    """设计稿 ``.ds`` 直接坐在机架上，``.ds.on`` 才是 ``#fff``：白色是「此刻在看这一集」的意思。

    通用列表规则给每张列表铺一层内容色底，左栏于是整段被涂成白的——机架色一个像素也看不见，
    而设计稿里 ``--panel-2`` 正是数据集行与管线步骤共处的那块底。底色一平，「哪一集是活动的」
    就只剩委托画的那条 accent 竖条在说，设计稿是用整行的白说的。
    """
    from xrr_fitter.gui.document import ProjectDocument

    panel = _panel(qtbot, ProjectDocument(_project_with_curves(tmp_path, count=2)))
    host = _railed(qtbot, panel)
    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None
    assert tree.topLevelItemCount() == 2
    active_row = tree.currentIndex().row()
    assert active_row in (0, 1)

    active = _row_edge_colour(host, tree, active_row)
    idle = _row_edge_colour(host, tree, 1 - active_row)

    assert idle.name().lower() == "#f7f8fa", f"未选中的行底是 {idle.name()}，机架色没透上来"
    assert active.name().lower() == "#ffffff", f"活动行底是 {active.name()}，不是设计稿的 --panel"


def test_the_card_names_the_scan_convention_before_the_point_count(qtbot, tmp_path) -> None:
    """设计稿的小字是「θ/2θ · 512 点 · 状态」——点数前面先说这是哪种扫描。

    只报点数说不清这 512 个数是沿什么轴排的：同一条曲线按 θ 还是 2θ 读，出来的层厚
    差一倍。约定不是猜的，项目把它存在 ``input_angle_kind`` 里，且模型只接受
    ``two_theta_deg``——写 θ/2θ 是把已有字段读出来，不是替数据做主张。
    """
    from xrr_fitter.gui.data.dataset_card import dataset_card

    panel = _one_dataset_panel(qtbot, tmp_path)

    assert dataset_card(panel, "sample").subline == "θ/2θ · 32 点"


# 设计稿帧① 左栏那四行小字的尾巴（HTML 368-371）：● 与 ◆ 都写「已拟合」，▲ 写「需复核」。
# 判定名（可信 / 可用但相关 / 多解 / 不可信）左边那枚字形已经在说了，尾巴说的是「所以
# 该怎么办」——两处写同一个词，卡片就有一半的字在重复。
VERDICT_TAILS = (
    (ConfidenceClass.TRUSTED, "已拟合"),
    (ConfidenceClass.CORRELATED, "已拟合"),
    (ConfidenceClass.MULTIPLE, "需复核"),
    (ConfidenceClass.UNTRUSTED, "需复核"),
)


@pytest.mark.parametrize(("verdict", "tail"), VERDICT_TAILS)
def test_the_small_print_says_what_to_do_next_not_which_verdict_it_repeats(
    qtbot,
    tmp_path: Path,
    verdict: ConfidenceClass,
    tail: str,
) -> None:
    """小字的第三段是动作，不是判定名——判定名由左边那枚字形负责。

    设计稿四行卡的尾巴只有两种写法：可信与可用但相关都写「已拟合」（这一集不用再动），
    多解与不可信都写「需复核」（这一集还要人来看）。此前这里直接把 ``fit_status_text``
    抄进小字，于是「◆ 可用但相关」这一行读作「字形说相关、小字也说相关」，而设计稿在
    这一格里给的是下一步。
    """
    from xrr_fitter.gui.data.dataset_card import dataset_card

    panel = _panel(qtbot, _fitted_document(tmp_path, verdict))

    card = dataset_card(panel, "fitted")

    assert card.subline == f"θ/2θ · 32 点 · {tail}"
    assert str(verdict.value) not in card.subline


def test_the_structure_step_stops_the_small_print_at_the_point_count(qtbot, tmp_path: Path) -> None:
    """设计稿帧③ 三行小字都停在点数（HTML 628-630），只有帧① 带那截尾巴（368-370）。

    两帧的字形一样是 ●◆▲——判定仍然报，只是不再用词重复一遍。改结构这一步问的是「我编的
    这叠层落在哪几条曲线上」：点数回答得了，而「已拟合」说的是上一轮的事，正要被这次编辑
    作废。
    """
    from xrr_fitter.gui import theme
    from xrr_fitter.gui.data.dataset_card import dataset_card

    panel = _panel(qtbot, _fitted_document(tmp_path, ConfidenceClass.TRUSTED))
    assert dataset_card(panel, "fitted").subline == "θ/2θ · 32 点 · 已拟合"

    panel.set_step(1)

    card = dataset_card(panel, "fitted")
    assert card.subline == "θ/2θ · 32 点"
    assert card.glyph == theme.CONFIDENCE_GLYPHS["可信"]


def test_the_structure_step_still_says_the_source_is_gone(qtbot, tmp_path: Path) -> None:
    """收起的只有判定那截尾巴——「源文件缺失」在哪一步都要说。

    一叠层落在一条读不到的曲线上，这一步就是白编的；而这句话与判定不同，它不是上一轮的
    结论，是此刻的事实。
    """
    from xrr_fitter.gui.data.dataset_card import dataset_card

    source = _write_curve(tmp_path / "sample.xy")
    panel = _panel(qtbot)
    panel.add_paths((source,), beam=api.BeamSpec("monochromatic"), instrument=api.InstrumentSpec("sample"))
    source.unlink()
    panel.document.refresh_sources()

    panel.set_step(1)

    assert dataset_card(panel, "sample").subline == "⛔ θ/2θ · 32 点 · 源文件缺失"


def test_the_two_lines_of_a_card_take_the_designs_two_type_sizes(qtbot) -> None:
    """``.ds .nm .t{font-size:13px;font-weight:600}`` 对着 ``.s{font-size:11.5px}``——两行不同号。

    此前两行同号：离屏实测基准是 9pt，名字与小字的 ``QFontMetrics.height()`` 都是 15，
    ``_small_font`` 把 ``FONT_PT_SM`` 设回同一个点数，等于没设。于是两行只靠颜色分层次，
    卡片读起来是两行一样重的字，设计稿的「名字大、小字小」这层结构没了。名字那行还是
    ``setBold(True)`` 即 700，比设计稿的 600 粗一档——600 在 Qt 里是 ``DemiBold``。

    11.5px 表达不出来：``QFont.setPixelSize`` 只收整数。取 12px，与管线步骤说明字
    （``navigation/steps`` 的 ``STEP_DESC_FONT_PX``）同一号，左栏两处小字因此同高。
    """
    _ = qtbot
    from xrr_fitter.gui.data.dataset_card import _small_font, _title_font

    base = QFont()
    base.setPointSize(9)

    title = _title_font(base)
    small = _small_font(base)

    assert title.pixelSize() == 13
    assert title.weight() == QFont.Weight.DemiBold
    assert small.pixelSize() == 12
    assert QFontMetrics(title).height() > QFontMetrics(small).height()


def _painted_lines(monkeypatch, tree, row: int) -> list[tuple[str, int, object, QColor]]:
    """委托画这一行时，每一行文字实际用的 (文字, 像素字号, 字重, 颜色)。

    取的是委托自己那一步而不是渲染出来的像素：字号与颜色都带反锯齿，从图上取色只能取到
    混过底的中间值，分不清 ``muted_text``（α .549）和 ``faint_text``（α .427）。
    """
    from xrr_fitter.gui.data.dataset_card import DatasetCardDelegate

    records: list[tuple[str, int, object, QColor]] = []
    original = DatasetCardDelegate._paint_line

    def spy(self, painter, rect, text, font, colour) -> None:
        records.append((text, font.pixelSize(), font.weight(), QColor(colour)))
        original(self, painter, rect, text, font, colour)

    monkeypatch.setattr(DatasetCardDelegate, "_paint_line", spy)
    index = tree.model().index(row, 0)
    option = QStyleOptionViewItem()
    tree.initViewItemOption(option)
    option.rect = tree.visualRect(index)
    image = QImage(max(option.rect.width(), 1), max(option.rect.height(), 1), QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    tree.itemDelegateForColumn(0).paint(painter, option, index)
    painter.end()
    return records


def test_the_small_print_is_painted_a_size_smaller_and_a_shade_fainter(qtbot, tmp_path: Path, monkeypatch) -> None:
    """两行字上屏时用的就是那两号，小字的墨色是设计稿最淡的那一档。

    ``.ds .nm .s`` 写的是 ``color:var(--ink-faint)``；此前画的是 ``muted_text``，也就是
    ``--ink-muted``。两者只差一档透明度（α .427 对 .549），但小字整段都在这一档上，淡
    一档才让名字那行独占视线。字号在这里再验一次是因为 ``sizeHint`` 与 ``paint`` 各自
    取字体：只在助手函数上断言，改了其中一处仍能过。
    """
    from xrr_fitter.gui import theme

    panel = _one_dataset_panel(qtbot, tmp_path)
    host = _railed(qtbot, panel)
    qtbot.waitExposed(host)
    tree = panel.findChild(QTreeWidget, "datasetTree")
    assert tree is not None
    tokens = theme.palette_tokens(tree.palette())

    lines = _painted_lines(monkeypatch, tree, 0)

    assert [text for text, *_ in lines] == ["sample", "θ/2θ · 32 点"]
    assert (lines[0][1], lines[0][2]) == (13, QFont.Weight.DemiBold)
    assert lines[1][1] == 12
    assert lines[1][3] == theme.token_colour(tokens.faint_text)
    assert lines[1][3] != theme.token_colour(tokens.muted_text)
