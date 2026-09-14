"""Shell chrome contract: menus, toolbar, status bar, theme, and empty states."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMenuBar,
    QPushButton,
    QStatusBar,
    QToolBar,
    QToolButton,
    QWidget,
)

import xrr_fitter.api as api


def _write_curve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _window(qtbot):
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    return window


def _span_text(window, name: str) -> str:
    """一段状态栏读出来是什么样：说明文字与取值按排版顺序连起来。

    设计稿的一段是 ``活动数据集：<b>aSi_ML_25C</b>``——说明文字与取值分属两个标签，
    但读者看见的是一句话。断言写在这句话上，就不必替实现记住「冒号归哪个标签」，
    换了拆法也不会误报。
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


def _menu_action(window, object_name):
    """The menu's action for a command, wherever chrome parked it."""
    for action in window.actions():
        if action.objectName() == object_name:
            return action
    action = window.chrome_actions.get(object_name)
    assert action is not None, f"missing menu action: {object_name}"
    return action


def test_main_window_has_menu_bar_with_workflow_menus(qtbot) -> None:
    window = _window(qtbot)

    bar = window.findChild(QMenuBar, "mainMenuBar")
    assert bar is not None
    titles = [action.text() for action in bar.actions()]
    assert titles == ["文件", "编辑", "视图", "拟合", "帮助"]

    file_menu = bar.actions()[0].menu()
    names = [action.objectName() for action in file_menu.actions() if not action.isSeparator()]
    for required in (
        "newProjectAction",
        "openProjectAction",
        "saveProjectAction",
        "saveAsProjectAction",
        "importFilesMenuAction",
        "importFolderMenuAction",
        "reloadSourceAction",
        "relinkSourceAction",
        "exportResultsAction",
    ):
        assert required in names


def test_toolbar_hosts_project_commands_and_export(qtbot) -> None:
    window = _window(qtbot)

    toolbar = window.findChild(QToolBar, "mainToolbar")
    assert toolbar is not None
    assert not toolbar.isMovable()
    new_button = window.findChild(QPushButton, "newProjectButton")
    export_button = window.findChild(QPushButton, "exportResultsButton")
    assert toolbar.isAncestorOf(new_button)
    assert toolbar.isAncestorOf(export_button)


def test_batch_mode_rides_with_the_reflectivity_canvas(qtbot, tmp_path) -> None:
    """批量 独立|联合 只在画布画着反射率曲线时出现。

    帧① 和帧④ 有它，帧③（结构编辑）和帧⑤（不确定度）没有。这不是「专家/引导」也不是
    「跑没跑」能分开的：帧① 和帧⑤ 同在 结果 这一步，帧③ 和帧① 同样不在跑。分界线是画布
    此刻画的是什么——批量说的是「屏幕上这几条曲线归一个结构还是各归一个」，画布换成共享
    结构的剖面（帧③）或某一个候选的相关矩阵（帧⑤）时，这个问题根本没在屏幕上问。

    帧④ 正在跑却仍然显示 批量 联合：先前的实现一律在运行态收起它，于是设计稿里唯一
    需要看清「这轮是联合拟合」的那一帧反而看不到。
    """
    from xrr_fitter.gui.chrome import set_command_bar_running
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    window.show()
    qtbot.waitExposed(window)
    group = window.findChild(QWidget, "batchModeGroup")

    apply_step_scope(window, 4)
    assert "reflectivity" in window.plot_panel.canvas_pane_keys()
    assert group.isVisibleTo(window) is True

    apply_step_scope(window, 1)
    assert "reflectivity" not in window.plot_panel.canvas_pane_keys()
    assert group.isVisibleTo(window) is False

    apply_step_scope(window, 3)
    set_command_bar_running(window, True)

    assert group.isVisibleTo(window) is True
    assert window.findChild(QWidget, "runningCommandGroup").isVisibleTo(window) is True


def test_the_reduced_command_bar_keeps_every_command_reachable_from_a_menu(qtbot) -> None:
    """Hiding a button must not hide the command: the menu twin stays enabled.

    This is what makes the reduction safe.  A guided user who does need 另存为 or
    导出结果 finds it in 文件 / 拟合 exactly as before, so the guided bar is a
    smaller offer rather than a smaller program.
    """
    window = _window(qtbot)

    window.set_guidance_visible(True)

    for object_name in (
        "saveAsProjectAction",
        "reloadSourceAction",
        "relinkSourceAction",
        "exportResultsAction",
        "startFitAction",
        "cancelFitAction",
    ):
        assert _menu_action(window, object_name).isVisible() is True, object_name


def test_status_bar_reports_readiness_and_active_dataset(qtbot) -> None:
    """空项目里只有就绪那一句话；「活动数据集」那一段没有数据集可报就整段退场。

    从前它打的是「活动数据集：无活动数据集」——说明文字问一遍、取值再答一遍同一件事，
    而设计稿五帧底栏没有任何一段是这样自问自答的。段里没有取值就把段藏起来，右半边
    留给真有话说的那几段。
    """
    from xrr_fitter.gui.status_bar import StatusSpan

    window = _window(qtbot)

    bar = window.findChild(QStatusBar, "mainStatusBar")
    assert bar is not None
    readiness = window.findChild(QLabel, "fitReadinessStatus")
    dataset = window.findChild(QLabel, "activeDatasetStatus")
    assert readiness is not None and dataset is not None
    assert "导入" in readiness.text()
    assert dataset.text() == ""
    assert window.findChild(StatusSpan, "statusDatasetSpan").isVisibleTo(window) is False


def test_view_menu_switches_plot_views_and_syncs_expert_mode(qtbot) -> None:
    window = _window(qtbot)

    bar = window.findChild(QMenuBar, "mainMenuBar")
    view_menu = bar.actions()[2].menu()
    by_name = {action.objectName(): action for action in view_menu.actions()}

    log_action = by_name["plotViewAction:log"]
    log_action.trigger()
    assert window.plot_panel.current_view_key() == "log"

    expert_action = by_name["expertModeAction"]
    assert expert_action.isCheckable()
    expert_action.setChecked(True)

    assert window.parameters_panel.expert_toggle.isChecked()
    window.parameters_panel.expert_toggle.setChecked(False)
    assert not expert_action.isChecked()


def test_the_view_menu_offers_one_layout_command_and_no_panel_toggles(qtbot) -> None:
    """A fixed-column shell has nothing to hide, so the menu offers the way back.

    The dock build listed a visibility toggle per panel.  The design's grid has
    no dismissable columns, and the one column that does come and go -- the
    inspector, which the guided surface drops -- is already driven by 引导模式
    above; a second control for it would let the menu and the surface disagree
    about whether it is showing.  Asserted as a closed set so a re-added toggle
    fails here rather than quietly reopening that disagreement.
    """
    window = _window(qtbot)

    bar = window.findChild(QMenuBar, "mainMenuBar")
    view_menu = bar.actions()[2].menu()
    names = {action.objectName() for action in view_menu.actions() if not action.isSeparator()}
    borrowed = {name for name in names if name.startswith(("plotViewAction:", "plotToolAction:"))}

    assert names - borrowed == {
        "resetLayoutAction",
        "guidanceModeAction",
        "expertModeAction",
    }
    # 设计稿的 ``.modebar`` 只摆四枚字形，其余五条退到绘图条自己的右键菜单——右键是隐藏
    # 入口，菜单栏得留一条明路。挂的是条上那几条 action 本身，所以这里认的就是同一批对象。
    assert borrowed >= {f"plotToolAction:{key}" for key in ("view", "mask", "zoom_to_range", "reset_zoom", "overlay")}
    by_name = {action.objectName(): action for action in view_menu.actions()}
    assert by_name["plotToolAction:mask"] is window.plot_panel.mode_actions()["mask"]


def test_view_menu_groups_cover_every_diagnostic_tab() -> None:
    """The menu's grouping is hand-written, so a new tab must be added to it.

    ``_sync_view_actions`` looks up an action for every ``TAB_SPECS`` entry, so a
    tab missing from the grouping raises ``KeyError`` on the next view change
    rather than merely going unlisted in the menu.
    """
    from xrr_fitter.gui.chrome import VIEW_GROUPS
    from xrr_fitter.gui.plots.diagnostics import TAB_SPECS

    grouped = tuple(key for _title, keys in VIEW_GROUPS for key in keys)

    assert sorted(grouped) == sorted(key for key, _title, _description in TAB_SPECS)
    assert len(grouped) == len(set(grouped))


def test_theme_applies_idempotent_application_stylesheet(qtbot) -> None:
    from xrr_fitter.gui.theme import apply_theme

    application = QApplication.instance()
    previous = application.styleSheet()
    try:
        first = apply_theme(application)
        assert first
        assert application.styleSheet() == first
        assert apply_theme(application) == first
    finally:
        application.setStyleSheet(previous)


def test_plot_panel_shows_import_guidance_until_data_arrives(qtbot) -> None:
    from tests.support.model_cases import prepared_data

    window = _window(qtbot)
    panel = window.plot_panel

    empty_state = panel.findChild(QWidget, "plotEmptyState")
    assert empty_state is not None
    assert empty_state.isVisibleTo(panel)
    assert not panel.tabs.isVisibleTo(panel)

    panel.set_dataset("sample", prepared_data())

    assert not empty_state.isVisibleTo(panel)
    assert panel.tabs.isVisibleTo(panel)


def test_empty_state_import_button_opens_import_dialog(qtbot, monkeypatch) -> None:
    window = _window(qtbot)
    calls: list[str] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: (calls.append("opened"), ([], ""))[1],
    )

    button = window.findChild(QPushButton, "emptyStateImportButton")
    assert button is not None
    button.click()

    assert calls == ["opened"]


def test_open_project_activates_first_dataset_when_none_selected(
    qtbot,
    tmp_path,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument

    project = api.new_project()
    project = api.add_dataset(project, _write_curve(tmp_path / "a.xy"), api.InstrumentSpec())
    project = api.add_dataset(project, _write_curve(tmp_path / "b.xy"), api.InstrumentSpec())
    project = api.select_active_dataset(project, None)
    target = tmp_path / "project.xrrproj.json"
    api.save_project(project, target)

    document = ProjectDocument()
    document.open(target)

    assert document.active_dataset_id == "a"
    assert not document.is_dirty


def test_fit_progress_view_is_hidden_while_idle(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    project = api.new_project()
    project = api.add_dataset(project, _write_curve(tmp_path / "c.xy"), api.InstrumentSpec())
    panel = FitPanel(ProjectDocument(project))
    qtbot.addWidget(panel)

    assert not panel.progress_view.isVisibleTo(panel)

    panel._project_running_state(True)
    assert panel.progress_view.isVisibleTo(panel)

    panel._project_running_state(False)
    assert not panel.progress_view.isVisibleTo(panel)


def _one_dataset_panel(qtbot, tmp_path):
    """一个只装了 ``d.xy`` 的数据面板。"""
    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    project = api.new_project()
    project = api.add_dataset(project, _write_curve(tmp_path / "d.xy"), api.InstrumentSpec())
    panel = DataPanel(ProjectDocument(project))
    qtbot.addWidget(panel)
    return panel


def test_dataset_tree_shows_only_the_card_column(qtbot, tmp_path) -> None:
    """一列自绘卡片，没有表头，其余六列都不显示。

    此前这里显示两列（名字、状态）并把另外四列收起。设计稿的 ``.ds`` 是一张两行卡：
    字形、名字、一句小字，没有第二列也没有表头。
    """
    # panel 要留个名字：``qtbot.addWidget`` 记的是弱引用，取完 ``.tree`` 就让面板被回收，
    # 树跟着析构，断言撞上 "Internal C++ object already deleted"。
    panel = _one_dataset_panel(qtbot, tmp_path)
    tree = panel.tree

    assert tree.isHeaderHidden()
    hidden = [column for column in range(tree.columnCount()) if tree.isColumnHidden(column)]
    assert hidden == list(range(1, tree.columnCount()))


def test_dataset_tree_keeps_the_hidden_columns_in_the_model(qtbot, tmp_path) -> None:
    """收起的列仍带着值。

    列没有删——卡片的小字和各列 tooltip 都从模型读，删列会连带删掉那些出处；改的只是显示。
    """
    from xrr_fitter.gui.data.dataset_card import DATASET_CARD_ROLE

    panel = _one_dataset_panel(qtbot, tmp_path)
    item = panel.tree.topLevelItem(0)

    assert item.text(4) == f"{panel.point_count_text('d')} · {panel.status_text('d')}"
    assert item.data(0, DATASET_CARD_ROLE).title == "d"

    tooltip = item.toolTip(0)
    assert "源文件：d.xy" in tooltip
    assert panel.sha256_text("d") in tooltip


def test_dataset_details_keep_one_field_per_line_in_the_row_tooltip(qtbot, tmp_path) -> None:
    """四项详情一项一行，长仪器名也不许把某一项挤掉或折断。

    这几项此前住在栏里一个 ``datasetDetails`` 标签上，按标签宽度逐行省略：一个带几何足迹的
    具名仪器要 434px 才排得完，而栏只有 320px 上下，中文又允许在任意两字之间断行，于是
    「分辨率 q」被折成「分辨」「率 q」——量和它的单位断在了两行。设计稿的左栏没有这块方块，
    这几项改挂到行的 tooltip 上；tooltip 不受那一列的宽度约束，但一项一行这条还得成立。
    """
    from math import asin, degrees

    from xrr_fitter.gui.data.panel import DataPanel
    from xrr_fitter.gui.document import ProjectDocument

    instrument = api.InstrumentSpec(
        instrument_id="Rigaku SmartLab",
        footprint_mode="geometry",
        sample_length_mm=10.0,
        beam_width_mm=0.1,
        footprint_spill_angle_deg=degrees(asin(0.1 / 10.0)),
    )
    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "d.xy"), instrument)
    panel = DataPanel(ProjectDocument(project))
    qtbot.addWidget(panel)

    lines = panel.tree.topLevelItem(0).toolTip(0).splitlines()

    assert "源文件：d.xy" in lines
    assert f"仪器：{panel.instrument_text('d')}" in lines
    assert f"光路：{panel.beam_text('d')}" in lines
    assert f"SHA-256：{panel.sha256_text('d')}" in lines


def test_the_appearance_toggle_sits_at_the_far_right_and_flips_the_palette(qtbot) -> None:
    """设计稿四张帧的命令栏最右边都是 ☾：一枚把整屏换成深色的开关。

    它排在弹簧之后、所有命令之外——外观不是项目里的一步，把它塞进命令序列里读者会
    以为按下去会改数据。
    """
    from xrr_fitter.gui import theme

    window = _window(qtbot)
    button = window.findChild(QToolButton, "appearanceToggleButton")
    assert button is not None
    assert button.text() == theme.APPEARANCE_DARK_GLYPH
    assert button.isCheckable() is True

    application = QApplication.instance()
    previous = application.palette()
    # 回程要落回按之前那一档，而不是写死「亮色」：跑测试的机器可能本来就是深色系统外观。
    started = theme.palette_tokens(previous)
    try:
        button.click()
        assert theme.palette_tokens(application.palette()) is theme.DARK_TOKENS
        # 按下之后字形改口，否则读者只能靠屏幕颜色猜这枚键现在处在哪一边。
        assert button.text() == theme.APPEARANCE_LIGHT_GLYPH
        button.click()
        assert theme.palette_tokens(application.palette()) is started
        assert button.text() == theme.APPEARANCE_DARK_GLYPH
    finally:
        application.setPalette(previous)
        theme.apply_theme(application)


def test_the_command_bar_swaps_the_fit_group_for_pause_and_stop_while_running(qtbot) -> None:
    """设计稿帧④ 的命令栏在跑起来之后换了一套：⏸ 暂停 与 ⏹ 停止 顶替 一键拟合/导出。

    跑到一半时「一键拟合」和「导出结果」都按不动，留着只是两块灰；真正要用的两个命令
    反而只在检视器底下。所以运行态把这一组换掉，而不是把它灰着摆在原位。
    """
    from xrr_fitter.gui.chrome import set_command_bar_running

    window = _window(qtbot)
    running_group = window.findChild(QWidget, "runningCommandGroup")
    assert running_group is not None
    assert running_group.isVisibleTo(window) is False
    assert window.expert_command_group.isVisibleTo(window) is True

    set_command_bar_running(window, True)
    assert running_group.isVisibleTo(window) is True
    assert window.expert_command_group.isVisibleTo(window) is False

    pause = window.findChild(QToolButton, "toolbarPauseButton")
    stop = window.findChild(QToolButton, "toolbarStopButton")
    assert pause is not None and stop is not None
    # 命令栏这两枚是检视器里那两枚的同一条命令，不是各走各的第二条路：点击转发过去，
    # 所以暂停/继续的措辞、可用状态与快捷键都只有一份。
    window.fit_panel.pause_button.setEnabled(True)
    with qtbot.waitSignal(window.fit_panel.pause_button.clicked, timeout=500):
        pause.click()
    window.fit_panel.cancel_button.setEnabled(True)
    with qtbot.waitSignal(window.fit_panel.cancel_button.clicked, timeout=500):
        stop.click()

    set_command_bar_running(window, False)
    assert running_group.isVisibleTo(window) is False
    assert window.expert_command_group.isVisibleTo(window) is True


def _window_with_structure(qtbot, tmp_path):
    """一个已经立起两层结构的窗口：SiO₂ 表面氧化层压在 a-Si 薄膜上。

    帧③ 那一屏就是这个形状——层堆叠四行（空气 / SiO₂ / a-Si / c-Si），状态栏因此写「4 层」。
    """
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (
            api.LayerSpec("SiO2 native oxide", api.MaterialSpec("SiO2", "SiO2", 2.19), 34.2, roughness_a=5.1),
            api.LayerSpec("a-Si · 非晶硅薄膜", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),
        ),
        api.MaterialSpec("c-Si · 晶体硅基底", "Si", 2.329),
    )
    project = api.set_structure(project, "curve", structure)
    window = MainWindow(ProjectDocument(api.select_active_dataset(project, "curve")))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.show()
    qtbot.wait(1)
    return window


def _structure_status_labels(window):
    """结构步骤状态栏的那几个标签，按规格统一拿出来。"""
    scale = window.findChild(QLabel, "structureScaleStatus")
    selected = window.findChild(QLabel, "selectedLayerStatus")
    return {"scale": scale, "selected": selected}


def test_the_structure_step_reports_the_stack_scale_and_the_selected_layer(qtbot, tmp_path) -> None:
    """设计稿帧③ 的状态栏读的是这一步的事：「4 层 · 9 个自由参数」与「选中：a-Si」。

    状态栏在设计稿里逐帧换内容——帧① 报判定与 J，帧③ 报这一叠层有多大、手上选的是哪一
    层。此前它是固定的那几段，于是站在结构这一步上，屏幕最下面那一行讲的是一件与手上动作
    无关的事（拟合判定），而「选了哪一层」除了右栏抬头之外无处可读。

    选中那一段只报名字的前半截：设计稿写「选中：a-Si」，不是把「· 非晶硅薄膜」也拖进来——
    状态栏一行里同时还站着层数与下一步，说明性的后半截在这里只会挤掉别的。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)
    labels = _structure_status_labels(window)
    free = sum(1 for item in window.parameters_panel.definitions if not item.locked)

    assert labels["scale"].isVisibleTo(window) is True
    assert labels["scale"].text() == f"4 层 · {free} 个自由参数"

    # 点层堆叠里那条带子——和读者用鼠标走的是同一条路。
    window.structure_panel.editor.stack.component_selected.emit(0)
    qtbot.wait(1)
    assert _span_text(window, "statusSelectedSpan") == "选中：SiO₂"

    apply_step_scope(window, 4)
    assert labels["scale"].isVisibleTo(window) is False
    assert labels["selected"].isVisibleTo(window) is False


def test_the_structure_step_puts_the_next_action_where_the_verdict_would_be(qtbot, tmp_path) -> None:
    """帧③ 的状态栏右半边是「选中：a-Si」与「下一步：开始拟合」，没有判定、批量与数据集。

    这一步还没有判定可报——那三段留在屏幕上，读者会拿上一次运行的结论去读手上这一叠还没
    拟过的层。腾出来的位置交给「下一步」：站在结构这一步，最该知道的是改完往哪走。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)

    following = window.findChild(QLabel, "nextActionStatus")
    assert following is not None
    assert following.isVisibleTo(window) is True
    assert _span_text(window, "statusNextSpan") == "下一步：开始拟合"
    for name in ("fitQualityStatus", "fitQualityDot", "batchModeStatus", "activeDatasetStatus"):
        label = window.findChild(QLabel, name)
        assert label is not None, name
        assert label.isVisibleTo(window) is False, name

    apply_step_scope(window, 4)
    assert following.isVisibleTo(window) is False
    assert window.findChild(QLabel, "activeDatasetStatus").isVisibleTo(window) is True


def test_the_next_action_is_the_one_coloured_word_in_the_bar(qtbot, tmp_path) -> None:
    """设计稿帧③ 把「开始拟合」写成 ``<b style="color:var(--accent)">``（HTML 733 行）。

    底栏这一行整条是次要色的白文，唯一带色的就是这个词——它是这一屏唯一的出口。字重已经在
    ``add_text(bold=True)`` 里，缺的是色：``theme`` 里那条 ``QLabel[statusKind="accent"]``
    早就为这一处写着（连注释都点着「下一步：开始拟合」），只是没人给标签挂上这个属性。
    """
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)

    following = window.findChild(QLabel, "nextActionStatus")
    assert following.property("statusKind") == "accent"
    # 前面那句说明文字（「下一步：」）不跟着变色：设计稿只给取值上色。
    caption = window.findChild(QLabel, "nextActionCaption")
    assert caption.property("statusKind") in (None, "")


def _edit_the_stack(window) -> None:
    """把手上这叠层改一下——走的是读者拖 SLD 或改数字之后同一条路。"""

    structure = window.document.project.datasets[0].structure
    edited = replace(structure, backing_roughness_a=structure.backing_roughness_a + 0.7)
    assert window.structure_panel.set_structure(edited) is True


def test_an_unsaved_stack_edit_is_what_the_bar_reports_on_the_structure_step(qtbot, tmp_path) -> None:
    """设计稿帧③ 底栏第一段是 warn 圆点 +「结构已修改（未保存）」（HTML 729 行）。

    这一屏读者刚动过层——底栏第一段该说的就是这件事，而它此前照样念「就绪」。「就绪」在这
    里不算错（拟合前提确实齐了），但它答的是另一个问题，于是「改过、还没落盘」这件事在整
    个窗口里除了标题栏那个圆点之外无处可读，而标题栏的圆点对任何改动都亮，不单指结构。

    判据是两件事的与：结构确实被提交过一次编辑，且这次编辑还没存。只看 ``is_dirty`` 太松
    ——切换活动数据集也会置脏；只看「站在结构这一步」是撒谎，读者可以什么都没改就走过来。
    """
    from xrr_fitter.gui import messages
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)
    label = window.findChild(QLabel, "fitReadinessStatus")
    dot = window.findChild(QLabel, "fitReadinessDot")
    # 结构是在造窗口之前立起来的，所以这一屏起手并没有「改过还没存」可报。
    assert label.text() == messages.READY_TEXT

    _edit_the_stack(window)

    assert label.text() == "结构已修改（未保存）"
    assert dot.property("statusKind") == "warn"


def test_accepting_the_suggested_oxide_counts_as_an_unsaved_stack_edit(qtbot, tmp_path) -> None:
    """设计稿帧③ 那一屏正是「点了建议氧化层、接受了、还没存」——底栏得认这一次编辑。

    接受建议往层堆叠里插进一整层，四行里的 SiO₂ 就是它。它却走的是自己那条提交路（``api.
    accept_oxide_suggestion`` 直接落盘到文档），不经过 ``set_structure``；于是「改过结构」这
    件事在这条路上没人宣布，底栏照念「就绪」。用户视角里两条路没有分别：都是层堆叠多了/
    少了东西。所以宣布口必须一致——谁把新结构落了盘，谁就得说一声。
    """
    from xrr_fitter.gui import messages
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow
    from xrr_fitter.gui.window_layout import apply_step_scope

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "curve.xy"), api.InstrumentSpec())
    # 只有一层薄膜压在 c-Si 上：表面还没有氧化层，``suggest_oxide_layers`` 因此有话可说。
    project = api.set_structure(
        project,
        "curve",
        api.StructureSpec(
            api.MaterialSpec("Air", None, None, 0.0j),
            (api.LayerSpec("a-Si · 非晶硅薄膜", api.MaterialSpec("a-Si", "Si", 2.28), 487.0, roughness_a=4.4),),
            api.MaterialSpec("c-Si · 晶体硅基底", "Si", 2.329),
        ),
    )
    window = MainWindow(ProjectDocument(api.select_active_dataset(project, "curve")))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.show()
    qtbot.wait(1)
    apply_step_scope(window, 1)
    label = window.findChild(QLabel, "fitReadinessStatus")
    dot = window.findChild(QLabel, "fitReadinessDot")
    assert label.text() == messages.READY_TEXT
    assert window.structure_panel.current_oxide_suggestion() is not None

    window.structure_panel.accept_current_oxide()

    assert len(window.document.project.datasets[0].structure.components) == 2
    assert label.text() == "结构已修改（未保存）"
    assert dot.property("statusKind") == "warn"


def test_saving_the_project_takes_the_unsaved_stack_note_back_down(qtbot, tmp_path) -> None:
    """存过之后这句话必须消失，否则它报的是「改过」而不是「没存」。

    锁存器不能只加不减：存盘把项目落到磁盘上，「未保存」当场不成立；此后再置脏（比如换活
    动数据集）也不该把这句话叫回来——它说的是结构，不是任何改动。
    """
    from xrr_fitter.gui import messages
    from xrr_fitter.gui.window_layout import apply_step_scope

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)
    label = window.findChild(QLabel, "fitReadinessStatus")
    _edit_the_stack(window)
    assert label.text() == "结构已修改（未保存）"

    window.document.save(tmp_path / "saved.xrrproj.json")

    assert label.text() == messages.READY_TEXT
    window.document.mark_dirty()
    assert label.text() == messages.READY_TEXT


def test_a_live_run_outranks_the_unsaved_stack_note(qtbot, tmp_path) -> None:
    """在跑的时候底栏第一段是「拟合进行中」——设计稿帧④ 如此，而且它更紧要。

    两句话都想占第一段：正在跑是此刻唯一会自己变化的事，改没改存没存等它跑完再说也不迟。
    跑完之后那句话得自己回来，锁存器不因为跑过一次就作废。
    """
    from xrr_fitter.gui.operation_state import refresh_operation_state
    from xrr_fitter.gui.window_layout import apply_step_scope

    class _LiveJob:
        def cancel(self) -> None:
            return None

        def force_stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    window = _window_with_structure(qtbot, tmp_path)
    apply_step_scope(window, 1)
    label = window.findChild(QLabel, "fitReadinessStatus")
    _edit_the_stack(window)

    window.fit_panel.controller._job = _LiveJob()
    refresh_operation_state(window)
    assert label.text() == "拟合进行中"

    window.fit_panel.controller._job = None
    refresh_operation_state(window)
    assert label.text() == "结构已修改（未保存）"
