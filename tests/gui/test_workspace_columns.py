"""Fixed three-column workspace.

The mockup's shell is a grid, not a dock area: ``grid-template-columns:264px 1fr
340px`` with a navigation rail, an adaptive canvas, and a context inspector.  The
previous build spent that layout on six QDockWidgets, which put a title bar and a
close button on every column, showed one right-hand panel at a time behind tabs,
and let the whole arrangement be dragged apart.  None of that is in the design.

The columns are therefore a QSplitter with pinned seams: no title bars, no
floating, no tabs, no grip to pull the columns apart, and the inspector's sections
scroll together so 判定 / 参数 / 候选解 are all readable at once.  Column widths
still round-trip through the project, and a width record written by another build
must never make a project unopenable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFontInfo, QPalette
from PySide6.QtWidgets import QDockWidget, QFrame, QLabel, QScrollArea, QSplitter, QWidget
from tests.gui.paint_support import painted, painted_at

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.window_layout import (
    CANVAS_COLUMN_FLOOR,
    INSPECTOR_SECTIONS,
    LEFT_COLUMN_WIDTH,
    RIGHT_COLUMN_WIDTH,
    STEP_INSPECTOR_SECTIONS,
    STRUCTURE_PANE_FLOOR_PX,
)

COLUMN_NAMES = ("navigationColumn", "canvasColumn", "inspectorColumn")


def _window(qtbot):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def _painted_window(qtbot):
    """一只已经上好设计色板与样式表、并且已经落定在屏幕上的窗口。

    这个 fixture 不走 ``create_application``，所以发货时那份亮色板得自己装：默认的离屏板把
    Window / Base / AlternateBase 三个角色全给成白的，机架色与画布色于是相等，任何「这块该
    是机架色」的断言都会退化成同义反复。样式表同理——``apply_theme`` 也没人替这里调。
    """
    from xrr_fitter.gui.theme import build_stylesheet, light_palette

    window = _window(qtbot)
    palette = light_palette()
    window.setPalette(palette)
    window.setStyleSheet(build_stylesheet(palette))
    window.resize(1280, 760)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_the_workspace_is_a_splitter_not_a_dock_area(qtbot) -> None:
    """The shell is the design's grid, so no column is a dockable panel."""
    window = _window(qtbot)

    splitter = window.findChild(QSplitter, "workspaceSplitter")
    assert splitter is not None
    assert splitter.orientation() == Qt.Orientation.Horizontal
    assert window.findChildren(QDockWidget) == []


def test_the_three_columns_are_the_splitter_in_reading_order(qtbot) -> None:
    window = _window(qtbot)

    splitter = window.workspace_splitter
    order = [splitter.widget(index).objectName() for index in range(splitter.count())]
    assert order == list(COLUMN_NAMES)


def test_only_the_canvas_absorbs_the_spare_width(qtbot) -> None:
    """``264px 1fr 340px``: the side columns are budgets, the canvas is the rest.

    Asserted by widening the shell rather than by reading the stretch factors:
    ``QSplitter`` has ``setStretchFactor`` and no matching getter, so the
    configured value is unobservable and only its effect can be measured.
    """
    window = _window(qtbot)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    before = window.workspace_splitter.sizes()

    window.resize(1700, 900)
    qtbot.waitUntil(lambda: sum(window.workspace_splitter.sizes()) > sum(before))

    left, canvas, right = window.workspace_splitter.sizes()
    assert left == before[0]
    assert right == before[2]
    assert canvas > before[1]
    assert window.workspace_splitter.childrenCollapsible() is False


def test_the_side_columns_open_at_their_documented_widths(qtbot) -> None:
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)

    left, _canvas, right = window.workspace_splitter.sizes()
    assert left == LEFT_COLUMN_WIDTH
    assert right == RIGHT_COLUMN_WIDTH


def test_the_side_columns_cannot_be_dragged_to_another_width(qtbot) -> None:
    """设计稿的壳是 ``grid-template-columns:264px 1fr 340px``：两侧是定值，只有画布是 ``1fr``。

    定值的意思不只是「打开时是这个数」，还包括用户改不动它。分栏器默认把每道手柄都做成可拖
    的，于是三栏能被拖散——量出来 264/340 只说明初值对，不说明它们是固定列。这条断言拖一
    把：手柄不接受拖动，栏宽因此和拖之前一样。

    钉的是用户那条路，不是所有路：``setSizes`` 仍然有效，恢复布局、读回工程宽度、引导态收
    起侧栏都还走它。
    """
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    splitter = window.workspace_splitter
    handles = [splitter.handle(index) for index in range(1, splitter.count())]
    before = splitter.sizes()

    assert handles
    assert all(not handle.isEnabled() for handle in handles), "手柄还能拖，三栏就不是固定列"

    grip = handles[0]
    qtbot.mousePress(grip, Qt.MouseButton.LeftButton, pos=QPoint(0, grip.height() // 2))
    qtbot.mouseMove(grip, QPoint(120, grip.height() // 2))
    qtbot.mouseRelease(grip, Qt.MouseButton.LeftButton, pos=QPoint(120, grip.height() // 2))

    assert splitter.sizes() == before


def test_the_seam_between_columns_is_one_pixel_of_border(qtbot) -> None:
    """设计稿 ``.nav{border-right:1px solid var(--border)}``：栏之间是一道线。

    分栏器手柄的默认宽度由样式给（这里是 4px），底色是窗口色：屏幕上那不是设计稿的细线，
    是一条比两侧都暗的空档。此前这一列的文档说它「就是那 1px 分隔线」——量出来不是。

    线交给手柄画而不是画在栏上：手柄那几像素挪不走，边框画在栏上时，线和画布之间会留下
    手柄那条空档成为第二道缝。
    """
    window = _painted_window(qtbot)

    assert window.workspace_splitter.handleWidth() == 1

    shot = painted(window)
    y = window.height() // 2
    edge = window.canvas_column.geometry().left()
    seam = shot.at(edge - 1, y)
    wash = shot.at(edge + 2, y)
    assert wash == window.palette().color(QPalette.ColorRole.Base), wash.name()
    assert seam.lightness() < wash.lightness(), f"缝 {seam.name()} 没有比画布底 {wash.name()} 暗"


def test_the_rail_wash_is_not_covered_by_its_own_scroll_surfaces(qtbot) -> None:
    """机架色得真的画到屏幕上，而不是被自己的滚动区糊掉。

    ``QScrollArea`` 给视口和被滚动的那只控件都开了 ``autoFillBackground``，两层默认铺的是内容
    色：机架整条被涂成白的，栏上那条 ``--panel-2`` 一个像素也看不见。设计稿 ``.nav`` 里管线步骤
    之间和末步之下露出来的都是机架色，量的就是末步之下那块空处。
    """
    window = _painted_window(qtbot)

    pipeline = window.nav_column.findChild(QWidget, "pipelineNav")
    assert pipeline is not None
    steps = [child for child in pipeline.findChildren(QWidget) if child.objectName().startswith("pipelineStep_")]
    assert steps, "管线里一个步骤也没有，量不到空处"
    # 末步之下那块空处现在就是 ``.pipe{padding:var(--sm) 0}`` 自己的 8px：管线按内容定高
    # 之后，栏里多出来的高度一次顶到页脚之上，不再垫在带子末端。取半格落在这 8px 中间。
    free_y = max(step.geometry().bottom() for step in steps) + theme.SPACE_SM // 2
    assert free_y < pipeline.height(), f"末步之下没有空处：{free_y} >= {pipeline.height()}"

    spot = pipeline.mapTo(window, QPoint(pipeline.width() // 2, free_y))
    shown = painted_at(window, spot)
    palette = window.palette()
    assert shown != palette.color(QPalette.ColorRole.Base), "机架仍被内容色盖着"
    assert shown == palette.color(QPalette.ColorRole.AlternateBase), shown.name()


def test_the_rail_has_no_divider_of_its_own(qtbot) -> None:
    """设计稿左栏是一整块连续的机架：数据集与「分析管线」之间没有任何分隔。

    栏内那只竖分栏器的手柄默认 4px 且铺窗口色，在机架上割出一条比两侧都亮的横带——设计稿里
    没有这条带。手柄收到 0，两块就像设计稿那样直接相接；分栏尺寸仍照旧存进项目文件。
    """
    window = _window(qtbot)
    rail = window.nav_column.findChild(QSplitter, "leftSplitter")
    assert rail is not None
    assert rail.handleWidth() == 0


def test_the_navigation_column_holds_datasets_pipeline_and_a_summary(qtbot) -> None:
    """One rail, in the design's order: what to work on, where you are, and how much.

    Splitting these across three stacked docks gave each its own title bar and
    close button, so the single 264px rail the design draws read as three
    dismissable panels.

    The summary is asserted by ownership rather than visibility: it reports "共 N
    个数据集 · 可拟合 …", which has nothing to say before the first import, so the
    panel hides it on an empty project.  ``isVisibleTo`` would call that a
    missing widget when what it means is an empty one.
    """
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)

    column = window.nav_column
    assert window.data_panel.isVisibleTo(column) is True
    assert window.pipeline_nav.isVisibleTo(column) is True
    assert column.isAncestorOf(window.dataset_summary) is True
    order = [window.data_panel, window.pipeline_nav]
    tops = [widget.mapTo(column, widget.rect().topLeft()).y() for widget in order]
    assert tops == sorted(tops)


def test_the_inspector_shows_this_step_all_at_once(qtbot) -> None:
    """一步之内的那几段一起答同一个问题，所以不能再拿标签页把三分之二藏起来。

    「全都常驻」和「用标签页分开」之间还有第三种：按项目走到的那一步露相关的那几段，段与
    段之间不加门。设计稿的每一帧右栏都恰好三段（帧③ 是选中层/参数化/结构诊断，帧① 是判定/
    参数·结果值/候选解），常驻五段在 1400×900 会溢出到 2.3 倍视口，那时哪一段落在折叠线下
    就成了配置问题。这个测试守的是「同一步之内没有门」——这一步该露的都在场且都可见，具体
    哪一步露哪几段由 `test_step_scoped_views.py` 钉住。
    """
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)

    column = window.inspector_column
    names = STEP_INSPECTOR_SECTIONS[window.pipeline_nav.current_step_index()]
    assert len(names) >= 3
    for name in names:
        card = column.findChild(QFrame, name)
        assert card is not None, name
        assert card.isVisibleTo(column) is True, name


def test_one_build_order_reads_correctly_on_every_step(qtbot) -> None:
    """每一步声明的段序都是构造顺序的子序列，所以换步只是换可见性、不必重排布局。

    顺序是这一栏的全部论证：帧① 要判定在前、产生它的数字在后、它被从谁当中选出来的最后；
    帧③ 要选中层在前、参数化在后、诊断收尾。这两条约束能被同一个固定构造顺序同时满足
    （候选解 / 选中层 / 参数化 / 结构诊断 / 控制），于是卡片一次建好就不动——按步骤把卡从
    布局里摘下来重插，会让挂在构造期的那些接线随步骤反复断连。这个测试守的正是这个前提：
    哪一步的段序一旦不再是构造顺序的子序列，就不能再靠切可见性了。
    """
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)

    build_order = [name for name, _title, _subtitle, _attribute in INSPECTOR_SECTIONS]
    for step, names in STEP_INSPECTOR_SECTIONS.items():
        positions = [build_order.index(name) for name in names]
        assert positions == sorted(positions), f"第 {step} 步的段序和构造顺序相反"

    column = window.inspector_column
    visible = [
        card
        for name in build_order
        if (card := column.findChild(QFrame, name)) is not None and card.isVisibleTo(column)
    ]
    tops = [card.mapTo(column, card.rect().topLeft()).y() for card in visible]
    assert tops == sorted(tops)


def test_the_inspector_scrolls_rather_than_clipping_its_sections(qtbot) -> None:
    window = _window(qtbot)

    from PySide6.QtWidgets import QScrollArea

    scroll = window.inspector_column.findChild(QScrollArea, "inspectorScroll")
    assert scroll is not None
    assert scroll.widgetResizable() is True
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_the_canvas_holds_the_plot_and_never_becomes_a_dock(qtbot) -> None:
    window = _window(qtbot)

    assert window.central_stack.currentWidget() is window.plot_panel
    assert window.central_stack.isVisibleTo(window.canvas_column) is True


def test_guidance_takes_the_whole_shell_and_drops_both_side_columns(qtbot) -> None:
    """Frame ② draws no ``appbody`` at all: the guided step owns the full width.

    The rail used to stay because this contract cited ``.appbody.two``
    (``grid-template-columns:264px 1fr``).  That rule is *defined* in the mockup
    and never *applied* -- all four ``appbody`` elements are the plain
    three-column form -- and frame ② has no ``appbody`` element, only
    ``cmdbar`` → ``wizhead`` → ``wizbody`` → ``statusbar``.  Its lead states the
    requirement outright: 「隐藏停靠面板与高级批量选项」.  Leaving the rail up put
    two competing numberings on one screen -- the六段管线's 「5 结果 / 6 导出」
    beside the header's 「第 N 步 / 共 4 步」 -- which is the one thing the guided
    surface exists to avoid.
    """
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)

    window.set_guidance_visible(True)

    assert window.central_stack.currentWidget() is window.guidance
    assert window.inspector_column.isVisibleTo(window) is False
    assert window.nav_column.isVisibleTo(window) is False
    # 层堆叠栏是画布列里 ``central_stack`` 的兄弟，换页换不掉它。留着它，专家结构编辑器
    # 就压在引导抬头上方；而引导第 2 步自己有一份平实语言的结构清单，同屏会出现两个结构
    # 编辑器，其中一个正是这一步想替读者省掉的。
    stack_pane = window.canvas_column.findChild(QWidget, "structurePanelScroll")
    assert stack_pane is not None
    assert stack_pane.isVisibleTo(window) is False

    window.set_guidance_visible(False)

    assert window.inspector_column.isVisibleTo(window) is True
    assert window.nav_column.isVisibleTo(window) is True
    assert stack_pane.isVisibleTo(window) is True


def test_column_widths_round_trip_through_a_saved_project(
    qtbot,
    tmp_path: Path,
) -> None:
    """A recorded column geometry has to survive save and reopen."""
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    left, canvas, right = window.workspace_splitter.sizes()
    window.workspace_splitter.setSizes([left + 60, canvas - 60, right])
    window._capture_workspace()
    widened = window.document.project.ui_state.workspace_splitter_sizes
    assert widened is not None

    target = tmp_path / "columns.xrrproj.json"
    api.save_project(window.document.project, target)
    reopened = _window(qtbot)
    reopened.document.open(target)

    assert reopened.document.project.ui_state.workspace_splitter_sizes == widened


def test_reset_layout_returns_the_columns_to_their_budgets(qtbot) -> None:
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    left, canvas, right = window.workspace_splitter.sizes()
    window.workspace_splitter.setSizes([left + 90, canvas - 90, right])

    window.reset_layout()

    restored_left, _canvas, restored_right = window.workspace_splitter.sizes()
    assert restored_left == LEFT_COLUMN_WIDTH
    assert restored_right == RIGHT_COLUMN_WIDTH


def test_a_legacy_dock_state_does_not_make_a_project_unopenable(qtbot) -> None:
    """Projects saved by the dock build carry a ``dock_state`` no column consumes.

    The field stays in the schema so those files still load; nothing reads it, so
    a value from any Qt build -- including one that never decodes -- degrades to
    the default column widths instead of refusing to open.
    """
    window = _window(qtbot)
    window.show()

    legacy = api.set_dock_state(window.document.project, "!!!not-base64!!!")
    window.document.replace_project(legacy)

    assert window.workspace_splitter.count() == len(COLUMN_NAMES)
    for name in COLUMN_NAMES:
        assert window.findChild(type(window.nav_column), name) is not None


def test_a_legacy_dock_state_survives_a_capture_and_a_reopen(qtbot, tmp_path: Path) -> None:
    """A field this build stops consuming still must not be a field it destroys.

    The dock arrangement is opaque base64 from another build's ``saveState()``.
    Columns ignore it, but capture rewrites ``ui_state`` on every layout change --
    so the one way this value dies is a capture that rebuilds the state instead of
    replacing the fields it owns.  A user who opens a dock-build project here,
    saves, and opens it in that build again gets their arrangement back.
    """
    window = _window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    arrangement = "AAAA/wAAAAD9AAAAAgAAAAA="
    window.document.replace_project(api.set_dock_state(window.document.project, arrangement))

    window._capture_workspace()
    target = tmp_path / "docked.xrrproj.json"
    api.save_project(window.document.project, target)

    assert api.load_project(target).ui_state.dock_state == arrangement


AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)


def _write_curve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(32)) + "\n",
        encoding="utf-8",
    )
    return path


def _populated(qtbot, tmp_path: Path, *, datasets: int, layers: int, expert: bool = False):
    """A window holding the requested dataset and layer counts.

    The project is handed to the document the window is built on, rather than
    swapped in afterwards, so the source-validation records the panels consult
    are populated the same way opening a project populates them.
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.new_project()
    for index in range(datasets):
        project = api.add_dataset(
            project,
            _write_curve(tmp_path / f"curve{index}.xy"),
            api.InstrumentSpec(instrument_id="columns"),
        )
    if datasets and layers:
        structure = api.StructureSpec(
            AIR,
            tuple(api.LayerSpec(f"layer{index}", SIO2, 40.0, roughness_a=3.0) for index in range(layers)),
            SI,
        )
        project = api.set_structure(project, "curve0", structure)
    if datasets:
        project = api.select_active_dataset(project, "curve0")
    if expert:
        project = api.set_expert_mode(project, True)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    return window


def test_the_dataset_list_height_follows_the_datasets_it_holds(
    qtbot,
    tmp_path: Path,
) -> None:
    """One dataset must not ask for the height that ten datasets need.

    The dataset list shares the 264px rail with the pipeline and the summary, so
    a view reporting a fixed 192px whatever it holds hands Qt the same request
    either way and starves whichever neighbour it is stacked against.
    """
    one = _populated(qtbot, tmp_path / "one", datasets=1, layers=0)
    many = _populated(qtbot, tmp_path / "many", datasets=10, layers=0)

    assert one.data_panel.tree.sizeHint().height() < many.data_panel.tree.sizeHint().height()


def test_the_summary_sits_on_the_rail_s_bottom_edge_once_it_has_a_count(
    qtbot,
    tmp_path: Path,
) -> None:
    """``.ds-summary{margin-top:auto;border-top:1px solid var(--border)}``.

    The count is the rail's footer, below the pipeline rather than tucked under
    the dataset list it counts: read from inside the list it looks like one more
    row, and it would scroll away with them.
    """
    window = _populated(qtbot, tmp_path, datasets=2, layers=0)
    window.show()
    qtbot.waitExposed(window)

    summary = window.dataset_summary
    assert summary.text() != ""
    assert summary.isVisibleTo(window.nav_column) is True
    column = window.nav_column
    summary_top = summary.mapTo(column, summary.rect().topLeft()).y()
    pipeline = window.pipeline_nav
    assert summary_top >= pipeline.mapTo(column, pipeline.rect().bottomLeft()).y()


def test_the_navigation_column_fits_its_budget(qtbot, tmp_path: Path) -> None:
    """The whole rail -- datasets, pipeline, summary -- must fit 264px.

    Horizontal scrolling is off, so a column whose minimum exceeds its budget is
    clipped outright: there is no scrollbar to reach the rest of it, and widening
    the column would take the width the design gives the canvas.
    """
    window = _populated(qtbot, tmp_path, datasets=1, layers=0)
    window.show()
    qtbot.waitExposed(window)

    floor = window.nav_column.minimumSizeHint().width()
    assert floor <= LEFT_COLUMN_WIDTH, f"rail asks {floor}px of a {LEFT_COLUMN_WIDTH}px column"


def test_the_structure_list_height_follows_the_layers_it_holds(
    qtbot,
    tmp_path: Path,
) -> None:
    """The same defect as the dataset list, in the panel stacked against it.

    The structure list shares the rail's spare height with the dataset list
    through ``leftSplitter``; two views that both report a fixed 192px hand Qt
    identical requests and are given identical shares, whatever they hold.
    """
    two = _populated(qtbot, tmp_path / "two", datasets=1, layers=2)
    twelve = _populated(qtbot, tmp_path / "twelve", datasets=1, layers=12)

    assert two.structure_panel.editor.tree.sizeHint().height() < twelve.structure_panel.editor.tree.sizeHint().height()


def test_one_dataset_does_not_claim_as_much_height_as_a_twelve_layer_stack(
    qtbot,
    tmp_path: Path,
) -> None:
    """The reported complaint: a two-row dataset table crowding out the layers."""
    window = _populated(qtbot, tmp_path, datasets=1, layers=12)

    datasets = window.data_panel.tree.sizeHint().height()
    layers = window.structure_panel.editor.tree.sizeHint().height()
    assert datasets < layers, f"dataset list asks {datasets}px, layer list {layers}px"


def test_the_data_panel_fits_the_rail(qtbot, tmp_path: Path) -> None:
    """A populated data panel must not ask for more width than the rail has.

    栏里每一样东西都得是宽度的消费者，不能是出价者：列表的行由委托按分到的宽度画，抬头是
    一句短语，合计那行会省略。此前顶破预算的是那块四行详情——它按 ``details_label.width()``
    截断，报出的最小宽度因此来自它上一次写进去的文本，而不是它要住进去的那一列。
    """
    window = _populated(qtbot, tmp_path, datasets=1, layers=0)
    window.show()
    qtbot.waitExposed(window)

    floor = window.data_panel.minimumSizeHint().width()
    assert floor <= LEFT_COLUMN_WIDTH, f"data panel asks {floor}px of a {LEFT_COLUMN_WIDTH}px column"


def test_the_structure_panel_fits_the_canvas_at_the_shells_narrowest(qtbot, tmp_path: Path) -> None:
    """结构面板得能在窗口拖到最窄时排完，因为画布列也关掉了横向滚动。

    此前这条量的是导航栏那 264px——理由是「最窄的那一列，装得下它就哪儿都装得下」。设计
    稿把结构编辑放进画布，而画布的行头是一行三个 tab 加���个命令（``.canvas-top``），264px
    连三个 tab 都排不完；拿一条这个面板永远不会住进去的列当尺子，量的就不是它会不会被裁。
    真正的地板是窗口最窄时画布分到的宽度。

    留出的余量仍然是有限的：再往行头加两三个命令就会撞线，那时该退回菜单而不是让面板在
    列边缘被裁掉。
    """
    window = _populated(qtbot, tmp_path, datasets=1, layers=2)
    window.show()
    qtbot.waitExposed(window)

    floor = window.structure_panel.minimumSizeHint().width()
    assert floor <= CANVAS_COLUMN_FLOOR, f"structure panel asks {floor}px of a {CANVAS_COLUMN_FLOOR}px column"


def test_the_parameter_name_column_is_not_the_narrowest_one(qtbot, tmp_path: Path) -> None:
    """The field a reader uses to tell rows apart cannot be the table's thinnest.

    The sized columns take what their contents need and the name absorbs only what
    is left: at the inspector's budget that was 16px against 上限's 64px, so every
    name elided to nothing while columns holding two characters kept their full
    share.  Three of those columns were floored by their own headers rather than
    their contents -- 初值, 单位 and 锁定 each took 44px to show a 25px value, and
    the last two have since moved inside the name cell where the design draws them.

    The header padding only exists once the theme is on, so the stylesheet has to
    be applied here the way ``apply_theme`` applies it to the running app.
    """
    from xrr_fitter.gui.theme import build_stylesheet

    window = _populated(qtbot, tmp_path, datasets=1, layers=12)
    window.setStyleSheet(build_stylesheet(window.palette()))
    window.show()
    qtbot.waitExposed(window)
    # The inspector has to be held at its documented budget: left to itself the
    # test window is wide enough to hide the starvation this is about.
    left, canvas, right = window.workspace_splitter.sizes()
    window.workspace_splitter.setSizes([left, canvas + right - RIGHT_COLUMN_WIDTH, RIGHT_COLUMN_WIDTH])
    qtbot.wait(1)
    table = window.parameters_panel.parameter_table

    name = table.columnWidth(0)
    others = {
        table.horizontalHeaderItem(column).text(): table.columnWidth(column) for column in range(1, table.columnCount())
    }
    assert name >= max(others.values()), (
        f"name column {name}px against {others} (viewport {table.viewport().width()}px)"
    )


def test_the_structure_editor_lives_in_the_canvas_not_on_the_rail(qtbot) -> None:
    """帧③把层堆叠画在中央画布的上半，左栏只有数据集、分析管线和汇总。

    结构编辑器原来挤在 264px 的导航栏里，六个命令按钮和一棵层树共用一列，层名被
    裁、命令要换行；设计稿把它放在画布里，和它下面实时联动的 SLD 深度剖面同屏，
    正是「点一层、看剖面怎么动」的那条工作路径。这条断言只问它在哪一列。
    """
    window = _window(qtbot)

    assert window.structure_panel.parent() is not None
    assert _column_of(window, window.structure_panel) == "canvasColumn"
    assert window.nav_column.findChild(QWidget, "structurePanelScroll") is None


def test_the_structure_pane_sits_above_the_sld_profile_in_the_canvas(qtbot) -> None:
    """两张卡的顺序：层堆叠在上，SLD 深度剖面在下，中间只许进度视图。

    设计稿的 canvas-body 是这两张卡，上下相邻；结构改动的即时反馈就在正下方。层堆叠
    在页面栈之外，因为绘图面板在导入数据前会整体换成空态——而那正是结构刚建起来、
    最需要看得见的时候。夹在中间的进度视图只在拟合运行时露面（帧④ 的画布列就是它压
    着反射率），不运行时它不占高度，两张卡仍是相邻的。
    """
    window = _window(qtbot)
    splitter = window.canvas_column.findChild(QSplitter, "canvasSplitter")

    order = [splitter.widget(index) for index in range(splitter.count())]
    assert order[0].isAncestorOf(window.structure_panel), order[0].objectName()
    assert order[1] is window.fit_panel.progress_view
    assert order[1].isVisibleTo(window) is False
    assert order[2] is window.central_stack
    assert order[2].isAncestorOf(window.plot_panel.sld_pane)


def _column_of(window, widget) -> str | None:
    """Which of the three columns holds ``widget``."""
    for name in COLUMN_NAMES:
        column = window.findChild(QWidget, name)
        if column is not None and column.isAncestorOf(widget):
            return name
    return None


def test_the_structure_pane_opens_tall_enough_to_read_the_stack(qtbot) -> None:
    """层堆叠开局要露出命令条加数层，而不是被绘图栈挤成一条缝。

    ``QSplitter`` 按 stretch 分的是“多余”高度，地板由子控件的最小高度决定；滚动区
    的天然最小高度只有几十像素，所以不给地板时它会被下面 624px 的绘图栈压到只剩标题。
    """
    window = _window(qtbot)
    window.resize(1600, 980)
    window.show()
    qtbot.waitExposed(window)

    scroll = window.canvas_column.findChild(QScrollArea, "structurePanelScroll")
    assert scroll is not None
    assert scroll.height() >= STRUCTURE_PANE_FLOOR_PX, scroll.height()


def test_a_taller_window_gives_the_whole_stack_card_instead_of_capping_it(qtbot) -> None:
    """窗口高到装得下时，层堆叠要拿到它需要的全部高度，包括最下面的层列表。

    ``QSplitter`` 只在建栈时问过一次“你要多高”，那时结构还是空的；加层之后卡片长高了，
    却没人再问，于是层列表的最后一行永远差一截。高度需求变了就得重新分。
    """
    window = _window(qtbot)
    window.resize(1600, 1400)
    window.show()
    qtbot.waitExposed(window)
    window.structure_panel.editor.load(_structure())
    qtbot.wait(1)

    scroll = window.canvas_column.findChild(QScrollArea, "structurePanelScroll")
    assert scroll.height() >= window.structure_panel.sizeHint().height(), (
        scroll.height(),
        window.structure_panel.sizeHint().height(),
    )


def _inspector_scroll(window) -> QScrollArea:
    return window.inspector_column.findChild(QScrollArea, "inspectorScroll")


def _is_inside_viewport(scroll: QScrollArea, card: QWidget) -> bool:
    viewport = scroll.viewport()
    top = card.mapTo(viewport, card.rect().topLeft()).y()
    return 0 <= top < viewport.height()


def test_a_finished_fit_brings_its_verdict_into_view(qtbot) -> None:
    """拟合出结果时，判定卡要当场可见并且在视口里，而不是留在折叠线下等人去找。

    等了一百秒的那个结论默认不可见，是这一栏最贵的一个缺陷。它此前的形态是溢出：五段常驻
    检视器在 1400×900 下内容高约 1900px、视口只有 815px，判定卡排在第五段、顶边比视口底还
    低 266px。

    现在按步骤只露相关的那三段，溢出本身没了（结构态 0px、结果态 129px），判定卡在结果态
    是第一段、顶边 y=12。所以这个测试量的不再是「滚了没有」而是那个更强的保证：结果一到，
    判定既在场也在视口里。换步和滚动哪一条起了作用都算数——``result_published`` 比项目状态
    先到一拍，这条边就是为那一拍接的。
    """
    window = _window(qtbot)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)

    scroll = _inspector_scroll(window)
    card = window.result_panel.findChild(QFrame, "resultConfidenceCard")
    assert card is not None
    assert card.isVisibleTo(window.inspector_column) is False, "还没拟合，判定卡就已经在栏里了"

    window.fit_panel.result_published.emit(_fit_result())

    assert card.isVisibleTo(window.inspector_column) is True, "拟合出了结果，判定卡还没入场"
    assert _is_inside_viewport(scroll, card), "拟合出了结果，判定卡还在折叠线下"


def test_selecting_a_layer_brings_the_layer_editor_into_view(qtbot) -> None:
    """点画布里的一层，右栏的「选中层」要滚进视野，否则改的是看不见的那张卡。

    量这条规则得先找到真会溢出的一屏。结构那两步只露三段，参数表搬去画布列之后它们连窗口
    压到自己的下限（``minimumHeight()`` = 693px）都不溢出：视口 606px、内容也是 606px。第 2 步
    多露一段「控制」，四段合起来 681px（选中层 322 + 参数化 70 + 结构诊断 99 + 控制 190），
    1400×700 下视口 613px——滚到底部就把选中层顶到视口上方 68px，正是这个测试要的起点。

    按步骤分段解掉的是常驻五段那 2.3 倍，解不掉这一屏，所以「谁现在相关就把谁滚出来」这条
    规则得留着：从滚到底部的位置选中一层，编辑它的那张卡要回到视口里。
    """
    from xrr_fitter.gui.theme import build_stylesheet
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window(qtbot)
    # 这一屏的算术是按发货时那份样式表来的：``.card`` 的内边距与 ``.h`` 的行高都由 QSS 给，
    # 少了它四段合起来短过视口，滚到底部等于没滚，起点就不成立了。
    window.setStyleSheet(build_stylesheet(window.palette()))
    window.resize(1400, 700)
    window.show()
    qtbot.waitExposed(window)
    apply_step_scope(window, 2)
    # 切段是改可见性，滚动范围要等布局把这一段的高度算进内容里才更新；不放一拍过去，
    # ``maximum()`` 还是 0，滚到底部等于没滚，测出来的是「卡本来就在视口顶上」。
    qtbot.wait(1)

    scroll = _inspector_scroll(window)
    card = window.inspector_column.findChild(QFrame, "inspectorSelectedLayer")
    assert card is not None
    assert card.isVisibleTo(window.inspector_column) is True, "结构那几步里选中层卡本该在场"
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    assert not _is_inside_viewport(scroll, card), "选中层卡已在视野内，这个测试量不到滚动"

    window.structure_panel.editor.load(_structure())
    window.structure_panel.editor.component_selected.emit(0, _structure().components[0])

    assert _is_inside_viewport(scroll, card), "选中了一层，编辑它的那张卡还在视野外"


def test_the_inspector_never_needs_more_width_than_its_viewport_gives(qtbot, tmp_path: Path) -> None:
    """检视器横向不许溢出：它的横向滚动条是关掉的，溢出就等于无声裁掉。

    纵向溢出至少还有滚动条兜着，横向没有——``inspectorScroll`` 关了水平滚动条，内容
    比视口宽多少就直接看不见多少，屏幕上也没有任何东西说明少了一截。1400×900 下这是
    视口 322px、内容地板 437px，「拟合结果」被从字中间切断，结果表的 ±1σ 与 单位 两列
    整列消失。

    地板一路传到 ``verdictEvidence``：它的徽章行是普通 ``QHBoxLayout``，最小宽是三枚
    徽章之和。这里量的是那条链的出口——检视器要的宽不许超过它拿到的宽。
    """
    from xrr_fitter.gui.theme import build_stylesheet

    # 专家模式：「开始拟合」只在专家模式下露面，非专家模式量到的按钮行少一枚按钮，
    # 也就少一截宽度——量到的不是发货状态下最宽的那一档。
    window = _populated(qtbot, tmp_path, datasets=2, layers=3, expert=True)
    window.setStyleSheet(build_stylesheet(window.palette()))
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    window.fit_panel.result_published.emit(_fit_result())
    # 判定徽章的宽度随文案变，空报告下三枚都是空标签——量不到真实地板。这里推的是
    # 带边界命中与强相关的报告，也就是徽章最宽的那一档。
    window.result_panel.verdict_evidence.set_report(_wide_uncertainty())
    qtbot.wait(1)

    scroll = _inspector_scroll(window)
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    budget = scroll.viewport().width()
    # 每一段都量，不只量此刻露着的那几段：换一步就换一批卡，只量当前可见的那几张会
    # 让另一步的溢出溜过去（拟合卡就是这么溜过去的）。
    floors = {
        name: window.inspector_column.findChild(QWidget, name).minimumSizeHint().width()
        for name, _title, _subtitle, _attribute in INSPECTOR_SECTIONS
    }
    over = {name: floor for name, floor in floors.items() if floor > budget}
    assert not over, f"视口只有 {budget}px，这几段要得更多且横向滚动条是关的：{over}"
    body_floor = scroll.widget().minimumSizeHint().width()
    assert body_floor <= budget, f"检视器内容要 {body_floor}px，视口只有 {budget}px，差的部分被无声裁掉"


def test_the_verdict_badges_wrap_instead_of_widening_the_column(qtbot) -> None:
    """三枚判定徽章的地板是最宽的那一枚，不是三枚之和。

    徽章文案随判定变长（「⚠ 4 处边界命中」比「✓ 无边界命中」宽），用求和的地板等于
    让判定内容去决定整栏宽度。改成换行行之后，宽度不够时它往下折一行，而不是把右栏
    连同结果表一起顶出窗口。
    """
    from xrr_fitter.gui.results.verdict import VerdictEvidence

    evidence = VerdictEvidence()
    qtbot.addWidget(evidence)
    evidence.set_report(_wide_uncertainty())

    badges = evidence.badges()
    assert len(badges) == 3
    widest = max(badge.sizeHint().width() for badge in badges)
    total = sum(badge.sizeHint().width() for badge in badges)
    assert total > widest, "三枚徽章一样宽的话这个测试量不到求和"
    assert evidence.minimumSizeHint().width() < total, (
        f"判定证据的地板 {evidence.minimumSizeHint().width()}px 还是三枚徽章之和 {total}px"
    )


def _wide_uncertainty() -> api.UncertaintyReport:
    """判定徽章文案最长的那一档：命中边界、有强相关、自助跑过但收敛率不满。"""
    import numpy as np

    return api.UncertaintyReport(
        correlation_names=("thickness", "roughness"),
        correlation_matrix=np.eye(2),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.12,
        boundary_hits=("thickness", "density", "roughness", "scale"),
        strong_correlations=(("thickness", "roughness", -0.93), ("density", "scale", 0.71)),
        systematic_residual=False,
        diagnostics=(),
        bootstrap_performed=True,
    )


def _fit_result() -> api.ProjectFitResult:
    """A published independent fit carrying no dataset results.

    The scroll reacts to a fit having finished, not to what it found, so the
    thinnest legal payload is the right one here.
    """
    return api.ProjectFitResult(
        mode="independent",
        datasets=(),
        warnings=(),
        updated_project=api.XrrProject.new((), master_seed=7),
    )


def _structure() -> api.StructureSpec:
    """A two-layer stack, tall enough that the card outgrows its build-time hint."""
    oxide = api.LayerSpec("SiO2", api.MaterialSpec("SiO2", "SiO2", 2.2), 21.0, roughness_a=4.0)
    film = api.LayerSpec("a-Si", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=30.0)
    return api.StructureSpec(
        api.MaterialSpec("Air", "N", 0.0012),
        (oxide, film),
        api.MaterialSpec("c-Si", "Si", 2.33),
        backing_roughness_a=3.0,
    )


def test_the_layer_list_lives_inside_the_card_whose_caption_describes_it(qtbot) -> None:
    """设计稿「层堆叠」这张卡装的就是层列表，卡头说的「点击选中 · 拖动排序」也是行的事。

    把抬头扣在色带图上、列表甩在卡外面，等于让抬头描述了一个它没盖住的控件：图既不能
    点选也不能拖排序，而真能的那个还得自己找。
    """
    window = _window(qtbot)
    editor = window.structure_panel.editor

    card = window.structure_panel.findChild(QFrame, "structureStackCard")

    assert card is not None
    assert editor.tree.parentWidget() is card, editor.tree.parentWidget().objectName()


def test_the_default_window_shows_the_layer_rows_and_not_only_their_header(qtbot) -> None:
    """1400×900 是默认尺寸，层列表在这个尺寸下必须看得见行，不能只剩一行列头。

    色带图按「装得下九条带」固定预留 160px，而常见结构只有一两层；那份预留把面板顶出
    视口，超支的一段正好是列表。核视口里放得下几行，因为策略对而行仍在折叠线下的话，
    屏幕上依旧是一行列头。
    """
    window = _window(qtbot)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    window.structure_panel.editor.load(_structure())
    qtbot.wait(1)

    tree = window.structure_panel.editor.tree
    scroll = window.canvas_column.findChild(QScrollArea, "structurePanelScroll")
    viewport = scroll.viewport()
    top = tree.mapTo(viewport, tree.rect().topLeft()).y()
    header = tree.header().height()
    rows = (min(tree.height(), viewport.height() - top) - header) // max(tree.sizeHintForRow(0), 1)

    assert rows >= 4, f"只有 {rows} 行落在视口内（空气 + 两层 + 基底 = 4）"


def test_the_diagram_reserves_height_for_the_layers_it_has(qtbot) -> None:
    """色带图的高度下限该随实际成分数走，而不是按九条带一律预留。

    下限是为了「每条带都还点得中、放得下标签」；一层的结构不需要九条带的地方，多留的
    那一截是从列表身上扣的。
    """
    from xrr_fitter.gui.structure.stack import MEDIUM_BAND_H, MIN_BAND_H

    window = _window(qtbot)
    editor = window.structure_panel.editor
    editor.load(_structure())
    qtbot.wait(1)

    floor = 2 * MEDIUM_BAND_H + len(_structure().components) * MIN_BAND_H

    assert editor.stack.minimumHeight() == floor, editor.stack.minimumHeight()


def test_both_rail_headings_carry_the_navigation_section_typography(qtbot, tmp_path: Path) -> None:
    """左栏两处抬头都是 ``.nav-sec``，所以都得挂那一档字号字色字距。

    ``sectionHeader`` 只给 ``font-weight: 700``，字号沿用默认 9pt、字色沿用正文——比设计稿的
    11px/``--ink-faint`` 大一号也深一档。抬头跟它下面的列表因此没有层级差。两处一起量：这两
    句在同一栏里上下相邻，只对上一句就会让另一句自己漂。
    """
    window = _populated(qtbot, tmp_path, datasets=1, layers=0)
    headings = (
        window.data_panel.findChild(QLabel, "dataPanelHeader"),
        window.pipeline_nav.findChild(QLabel, "pipelineNavHeader"),
    )

    for heading in headings:
        assert heading is not None
        assert heading.property("faintText") is True
        assert heading.property("sectionHeader") is True
        # 像素读数：字号是按像素给的，``pointSize()`` 会返回 -1。
        assert QFontInfo(heading.font()).pixelSize() == theme.SECTION_HEADING_FONT_PX
        assert heading.font().letterSpacing() == pytest.approx(theme.RAIL_SECTION_TRACKING_PX, abs=1 / 64)


def test_the_rail_carries_no_dataset_detail_block(qtbot, tmp_path: Path) -> None:
    """设计稿的左栏只有抬头、列表和底部合计——没有那四行 源文件/光路/仪器/状态。

    那一块是把列表里已收起的几列又抄了一遍：264px 装不下它们，于是每一行还要按标签宽度截断，
    截出来的字既读不全也和卡片小字重复。真正需要它们的时候是悬停在那一行上，而第 0 列的
    tooltip 本来就在讲状态和拟合，把这四项并进去就不必在栏里常驻一块设计稿没有的方块。
    """
    window = _populated(qtbot, tmp_path, datasets=1, layers=0)
    panel = window.data_panel

    assert panel.findChild(QLabel, "datasetDetails") is None
    tooltip = panel.tree.topLevelItem(0).toolTip(0)
    assert "源文件：curve0.xy" in tooltip
    assert "光路：" in tooltip
    assert "仪器：columns" in tooltip
    assert "SHA-256" in tooltip
