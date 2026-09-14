"""Menu bar, toolbar, and status bar assembly for the main window.

Chrome reuses the window's existing QAction and QPushButton objects so every
command keeps one identity across menus, the toolbar, keyboard shortcuts, and
accessibility metadata.  Only presentation lives here; workflows stay on the
window and its panels.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMenuBar,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QToolButton,
    QWidget,
)

from xrr_fitter.gui import messages, status_bar, theme
from xrr_fitter.gui.command_icons import command_icon
from xrr_fitter.gui.dialog_template import StyledDialog
from xrr_fitter.gui.plots.diagnostics import TAB_SPECS
from xrr_fitter.gui.results.uncertainty import sampling_readings
from xrr_fitter.gui.results.verdict import free_parameter_count, reduced_chi_squared
from xrr_fitter.gui.segmented import SegmentedControl
from xrr_fitter.gui.structure import naming

# The design's 引导↔专家 segment (frame ①), described there as "两种模式是同一文档的
# 两个投影，功能等价".  The halves name surfaces, not depths: the deeper parameter
# columns are the parameters panel's own 显示高级选项, which is persisted on the
# project, so a surface switch must not touch it.  Only the segment wears 专家.
WORKSPACE_MODE_SPECS = (
    ("guided", "workspaceModeGuided", "引导", "引导：按导入、结构、拟合、导出四步逐一完成"),
    ("expert", "workspaceModeExpert", "专家", "专家：完整三栏工作区，导航、图表与检查器同屏"),
)

# 帧①④ 的 ``批量 独立|联合``.  它决定的是整屏参数表读作什么——联合时厚度、粗糙度、
# 密度由三条曲线共享，独立时各自一套——所以它是项目级状态，不是某张卡的局部设置。
BATCH_MODE_SPECS = (
    ("independent", "batchModeIndependent", "独立", "独立：每个数据集各自拟合，参数互不相干"),
    ("joint", "batchModeJoint", "联合", "联合：多个数据集共享同一套层结构，各自保留标度与本底"),
)
BATCH_MODE_LABEL = "批量"

# 设计稿帧⑤ 命令栏右端那枚执行命令的字面。字形在文字里，与 ``⚡ 一键拟合`` 同一个写法。
SAMPLING_COMMAND_TEXT = "▶ 运行 MCMC"

# 拒绝一次操作的提示停留多久。够读完一句话，又不至于压住之后的常驻读数。
REFUSAL_MESSAGE_MS = 4000

# The SLD profile is not a selectable view: it is a permanent companion pane,
# so the menu lists only the switchable diagnostic tabs.  The weighted residual
# is both -- the design's fourth reflectivity tab *and* the card pinned under
# the strip -- so it does belong here.
# The view menu sections, which group the diagnostic tabs by the question they
# answer rather than by tab order. Every tab needs an entry here: the menu is
# the only way to reach a view once the tab bar scrolls, and the checked-state
# sync walks TAB_SPECS expecting to find an action for each key.
VIEW_GROUPS = (
    ("反射率", ("log", "raw", "qz4", "residual")),
    ("诊断", ("candidates", "residual_map", "parameter_map", "uncertainty", "trend")),
)

# 帧④ 状态栏右边那一段的取值。说明文字「模式：」住在段里（见 ``status_bar``），
# 所以这里只报取值。
BATCH_MODE_TEXTS = {"independent": "独立批量", "joint": "联合批量"}

# 帧② 状态栏第一段的取值。引导模式下那一屏没有检查器，说清「现在是被领着走的」比重复
# 一句项目就绪更有用——就绪与否那一屏的每一步自己会说。
GUIDED_READINESS_TEXT = "引导模式"

# 帧③ 状态栏第一段的取值。改过层堆叠而还没存盘时，「就绪」说的是「现在可以开拟合」——
# 那句话没错，但它盖住了此刻更要紧的一件事：盘上那一份和屏幕上这一份已经不是同一个样品。
UNSAVED_STRUCTURE_TEXT = "结构已修改（未保存）"

ABOUT_TEXT = (
    "XRR Fitter\n\n"
    "X 射线反射率全自动拟合桌面应用。\n"
    "导入 .xy / .dat / .txt 反射率数据，初始化样品结构，"
    "一键拟合并导出结果。\n\n"
    "支持的 Python 边界：xrr_fitter.api"
)


def _window_action(window: QWidget, object_name: str) -> QAction:
    for action in window.actions():
        if action.objectName() == object_name:
            return action
    raise LookupError(f"missing window action: {object_name}")


def _action(
    window: QWidget,
    object_name: str,
    text: str,
    callback,
    *,
    checkable: bool = False,
    command: str | None = None,
) -> QAction:
    action = QAction(text, window)
    action.setObjectName(object_name)
    action.setToolTip(text)
    action.setStatusTip(text)
    action.setCheckable(checkable)
    # The icon follows the command's identity, so this menu action renders the
    # same glyph as any toolbar twin that names the same callback.
    if command is not None:
        action.setIcon(command_icon(command))
    if checkable:
        action.toggled.connect(callback)
    else:
        action.triggered.connect(lambda _checked=False: callback())
    return action


def install_chrome(window: QWidget) -> None:
    """Assemble toolbar, menus, and the status bar around existing commands."""
    window.chrome_actions = {}
    _install_toolbar(window)
    _install_menu_bar(window)
    _install_status_bar(window)
    _connect_view_sync(window)


def _install_toolbar(window: QWidget) -> None:
    toolbar = QToolBar("主工具栏", window)
    toolbar.setObjectName("mainToolbar")
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    # Buttons and actions already carry their command icon from creation, so the
    # toolbar only lays them out; it no longer re-applies glyphs of its own.
    toolbar.addWidget(window.project_actions)
    toolbar.addSeparator()
    toolbar.addWidget(_install_mode_segment(window))
    window.batch_mode_action = toolbar.addWidget(_install_batch_segment(window))
    # 设计稿四张专家帧的命令栏都在 批量 段之后放一段弹簧，把「执行 / 导出 / 外观」顶到右
    # 端：左边那串是「这一屏是什么」，右端那几枚是「从这一屏出去」。弹簧排在执行命令之后
    # 时，它们会贴着 批量 段坐在栏的中段，读成模式选择的一部分。
    spring = QWidget(toolbar)
    spring.setObjectName("commandBarSpring")
    spring.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    toolbar.addWidget(spring)
    window.expert_command_action = toolbar.addWidget(_install_expert_commands(window))
    window.running_command_action = toolbar.addWidget(_install_running_commands(window))
    toolbar.addWidget(_install_appearance_toggle(window))
    window.addToolBar(toolbar)
    # 三组可切换的命令都是先建后藏。``QToolBar.addWidget`` 把控件包进一个 QWidgetAction：
    # 建的时候就藏着，那枚 action 会连着禁用且再也解不回来，组里的按钮永远点不动；只藏
    # 控件不藏 action，工具栏下一次重排又会把它显示回来。所以两边一起切。
    _sync_command_bar(window)


def _install_appearance_toggle(window: QWidget) -> QWidget:
    """设计稿命令栏最右边的 ☾：一枚把整屏换成深色的开关。"""
    button = QToolButton(window)
    button.setObjectName("appearanceToggleButton")
    button.setCheckable(True)
    button.setText(theme.APPEARANCE_DARK_GLYPH)
    button.setAccessibleName("切换深色外观")
    button.setToolTip("切换深色外观")
    button.toggled.connect(lambda dark: _apply_appearance(button, dark))
    window.appearance_toggle = button
    return button


def _apply_appearance(button: QToolButton, dark: bool) -> None:
    application = QApplication.instance()
    if application is not None:
        theme.set_dark_appearance(application, dark)
    # 字形报的是按下去会去哪儿：深色时给太阳，亮色时给月亮。
    button.setText(theme.APPEARANCE_LIGHT_GLYPH if dark else theme.APPEARANCE_DARK_GLYPH)


def _toolbar_separator(parent: QWidget) -> QWidget:
    """The design's ``.vsep``, as a widget rather than a toolbar separator.

    A separator added with ``QToolBar.addSeparator`` is an action on the toolbar
    and cannot travel with the group it introduces, so hiding the group in guided
    mode would leave its divider behind.
    """
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setProperty("pipelineRail", True)
    return line


def _install_expert_commands(window: QWidget) -> QWidget:
    """Group ⚡一键拟合 / 导出… so guided mode can set them aside.

    Frame ②'s cmdbar carries 新建/打开/保存 and the 引导·专家 segment, nothing
    else: a newcomer on step ① is offered 导出结果 two steps before there is a
    result.  The commands keep their single ``QAction`` -- the toolbar buttons
    adopt it through ``setDefaultAction`` -- so shortcuts, the enabled-state sync
    in ``operation_state`` and the menu twins all keep working while the group is
    hidden.  Hiding the actions themselves would take the menu entries with them.

    取消拟合 不上这条栏。设计稿帧①③ 的右端是「一枚实心的一键拟合 + 导出…」，帧⑤ 把执行
    那一枚换成 ``▶ 运行 MCMC``（见 ``_sync_execution_command``），真跑起来则换成帧④ 的
    ⏸暂停 / ⏹停止；空闲态摆一枚 取消 是在为一件还没发生的事留位子。它连着 Esc 留在 拟合
    菜单里。
    """
    group = QWidget(window)
    group.setObjectName("expertCommandGroup")
    layout = QHBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_XS)
    action = _window_action(window, "startFitAction")
    # 设计稿写作 ``⚡ 一键拟合``：字形在文字里。它挂在 ``iconText`` 上而不是 ``text`` 上，
    # 因为 QToolButton 读 ``iconText``、QMenu 读 ``text``——命令栏拿到设计稿的写法，拟合
    # 菜单仍是干净的「一键拟合」，而两处仍然是同一条 QAction。按钮自己 ``setText`` 顶不住：
    # ``operation_state`` 每次改可用状态都会发 ``QAction::changed``，QToolButton 收到就
    # 重新从 action 抄一遍文字。
    action.setIconText("⚡ 一键拟合")
    button = QToolButton(group)
    button.setObjectName("startFitToolButton")
    button.setDefaultAction(action)
    # 字形已经在文字里，再挂一枚图标就是同一件事说两遍。
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    # 设计稿的 ``.btn primary``：整屏每次只推荐一个下一步动作，实心那枚就是它。
    button.setProperty("primary", True)
    layout.addWidget(button)
    window.start_fit_button = button
    layout.addWidget(_install_sampling_command(window, group))
    # 导出 紧跟在执行命令后面，中间不画竖线：设计稿右端那几枚之间只有间距，与左边隔开
    # 靠的是弹簧那整片空白。
    layout.addWidget(window.export_button)
    window.expert_command_group = group
    return group


def _install_sampling_command(window: QWidget, group: QWidget) -> QWidget:
    """帧⑤ 顶替 ⚡一键拟合 的那一枚：``▶ 运行 MCMC``。

    点击转发给检视器那枚采样按钮，照的是 ``⏸暂停`` / ``⏹停止`` 的同一条规矩——walkers
    校验、候选解归属与「链跑着时不许再起一条」都留在 ``ResultsPanel`` 那一份实现里。

    设计稿这一枚没有省略号，也没有 ``primary``：不带省略号说的是按下去就开跑（同帧的
    ``导出…`` 带，它确实开对话框）；不实心说的是这一页手上已经有一份读得下去的结果，采样
    是往上补一层证据，而实心那枚的意思是「整屏此刻只推荐这一个下一步」。
    """
    button = QToolButton(group)
    button.setObjectName("runMcmcToolButton")
    # 字形在文字里，与 ⚡一键拟合 同一个写法。这一枚不挂 QAction：采样的可用状态由
    # ``ResultsPanel`` 那枚按钮算（见 ``_sync_execution_command``），此处再立一条 QAction
    # 就等于把同一个判断摆到两个地方。
    button.setText(SAMPLING_COMMAND_TEXT)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    button.setAccessibleName("运行 MCMC 采样")
    button.setToolTip("对当前候选解运行 MCMC 采样")
    button.clicked.connect(window.result_panel.mcmc_button.click)
    window.sampling_command_button = button
    return button


def _install_batch_segment(window: QWidget) -> QWidget:
    """Build frame ①'s ``批量 独立|联合`` group; ``_connect_view_sync`` wires it up.

    The choice used to be a combo box on the fit card, which put a project-wide
    fact -- whether three curves share one stack or hold three -- behind a scroll
    of the inspector, while the parameter table two cards above changes meaning
    with it.  The design puts it on the command bar next to the projection
    segment, where a glance answers it.
    """
    group = QWidget(window)
    group.setObjectName("batchModeGroup")
    layout = QHBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_XS)
    layout.addWidget(_toolbar_separator(group))
    label = QLabel(BATCH_MODE_LABEL, group)
    label.setObjectName("batchModeLabel")
    # 设计稿的 ``.faint``：它是段的名字，不是一条与段并重的命令。
    label.setProperty("mutedText", True)
    layout.addWidget(label)
    segment = SegmentedControl(BATCH_MODE_SPECS, name="batchModeSegment", parent=group)
    segment.setAccessibleName(BATCH_MODE_LABEL)
    label.setBuddy(segment)
    layout.addWidget(segment)
    window.batch_mode_group = group
    window.batch_mode_segment = segment
    return group


def _install_running_commands(window: QWidget) -> QWidget:
    """帧④ 的运行态命令组：⏸ 暂停 与 ⏹ 停止。

    这两枚是检视器控制段那两枚按钮的同一条命令（转发点击而不是各自接一次控制器），
    所以暂停/继续的措辞、可用状态与快捷键都只有一份。
    """
    group = QWidget(window)
    group.setObjectName("runningCommandGroup")
    layout = QHBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_XS)
    panel = window.fit_panel
    window.toolbar_pause_button = None
    for object_name, source, danger in (
        ("toolbarPauseButton", panel.pause_button, False),
        ("toolbarStopButton", panel.cancel_button, True),
    ):
        button = QToolButton(group)
        button.setObjectName(object_name)
        button.setText(source.text())
        button.setAccessibleName(source.accessibleName())
        button.setToolTip(source.toolTip())
        if danger:
            # 设计稿把停止画成红字红边：它是这一栏里唯一会丢掉正在跑的东西的命令。
            button.setProperty("danger", True)
        button.clicked.connect(lambda _checked=False, target=source: target.click())
        layout.addWidget(button)
    # 暂停键按下之后改口说「▶ 继续」；命令栏这一枚跟着源按钮改，否则两处措辞会打架。
    window.toolbar_pause_button = group.findChild(QToolButton, "toolbarPauseButton")
    panel.pause_button.clicked.connect(lambda: window.toolbar_pause_button.setText(panel.pause_button.text()))
    window.running_command_group = group
    return group


def _sync_command_bar(window: QWidget) -> None:
    guided = bool(getattr(window, "_command_bar_guided", False))
    running = bool(getattr(window, "_command_bar_running", False))
    _set_group_visible(window, "expert_command", not guided and not running)
    _set_group_visible(window, "running_command", running)
    # 批量 独立|联合 说的是「这一批数据集各自拟合，还是共享结构一起拟合」，只有画布上画着
    # 反射率曲线时它才对得上眼前的东西。设计稿就是这么切的：帧①（反射率+残差）与帧④
    # （反射率）栏上有它，帧③（SLD 剖面）与帧⑤（不确定度分析）没有。运行中不收起来——
    # 帧④ 正在跑，那一段仍在栏上，因为「联合」就是这一次运行的口径。
    _set_group_visible(window, "batch_mode", not guided and _canvas_shows_reflectivity(window))
    _sync_execution_command(window)


def _sync_execution_command(window: QWidget) -> None:
    """右端那枚执行命令换人：不确定度那一页给 ``▶ 运行 MCMC``，别的页给 ``⚡ 一键拟合``。

    这一枚只看画布停在哪一页，不看采样跑过没有——右栏那三段读数没证据时该整段收起（见
    ``window_layout._uncertainty_evidence_is_ready``），而这一枚恰恰在一次都没跑过时最该
    在栏上：它就是去把那三段填上的入口。

    可用状态抄 ``ResultsPanel`` 那枚采样按钮，不自己判一次：候选解够不够跑、链是不是正跑
    着，都归 ``candidate_is_mcmc_ready`` 与 ``_refresh_mcmc_buttons``。
    """
    start = getattr(window, "start_fit_button", None)
    sampling = getattr(window, "sampling_command_button", None)
    if start is None or sampling is None:  # pragma: no cover - 命令栏总是先建齐再同步
        return
    on_uncertainty = _canvas_shows_uncertainty(window)
    start.setVisible(not on_uncertainty)
    sampling.setVisible(on_uncertainty)
    sampling.setEnabled(window.result_panel.mcmc_button.isEnabled())


def _canvas_shows_uncertainty(window: QWidget) -> bool:
    panel = getattr(window, "plot_panel", None)
    if panel is None:  # pragma: no cover - the window always builds its canvas
        return False
    return panel.active_analysis_view() == "uncertainty"


def _canvas_shows_reflectivity(window: QWidget) -> bool:
    panel = getattr(window, "plot_panel", None)
    if panel is None:  # pragma: no cover - the window always builds its canvas
        return True
    return "reflectivity" in panel.canvas_pane_keys()


def refresh_command_bar(window: QWidget) -> None:
    """画布换了内容之后重算命令栏该露哪几组。

    批量 那一组跟着画布走（见 ``_sync_command_bar``），而画布内容由步骤决定，所以
    ``apply_step_scope`` 每换一步都要回头喊一次这里。
    """
    _sync_command_bar(window)


def _set_group_visible(window: QWidget, name: str, visible: bool) -> None:
    action = getattr(window, f"{name}_action", None)
    group = getattr(window, f"{name}_group", None)
    if action is not None:
        action.setVisible(visible)
    if group is not None:
        group.setVisible(visible)


def set_command_bar_running(window: QWidget, running: bool) -> None:
    """运行态把 一键拟合/导出 那一组换成 ⏸ 暂停 / ⏹ 停止。"""
    window._command_bar_running = running
    _sync_command_bar(window)
    pause = getattr(window, "toolbar_pause_button", None)
    if pause is not None:
        pause.setText(window.fit_panel.pause_button.text())


def set_command_bar_guided(window: QWidget, guided: bool) -> None:
    """Reduce the command bar to frame ②'s offer, or restore the expert bar.

    ``project_actions`` 不再参与增减：新建/打开/保存 三颗在引导和专家两态都在栏上，
    正是设计稿六张帧的一致画法。这里要收放的只有 ``expert_command`` / ``batch_mode``
    两组。
    """
    window._command_bar_guided = guided
    _sync_command_bar(window)


def _install_mode_segment(window: QWidget) -> QWidget:
    """Build the design's 引导↔专家 segment; ``_connect_view_sync`` wires it up.

    The state it shows belongs to ``guidanceModeAction``, which the menu bar
    creates after this toolbar exists, so the widget is assembled here and
    connected once both halves of the chrome are in place.
    """
    segment = SegmentedControl(WORKSPACE_MODE_SPECS, name="workspaceModeSegment", parent=window)
    segment.setAccessibleName("工作界面")
    segment.setToolTip("在引导流程与完整工作区之间切换，两者读写同一个项目")
    window.workspace_mode_segment = segment
    return segment


def _install_menu_bar(window: QWidget) -> None:
    bar = QMenuBar(window)
    bar.setObjectName("mainMenuBar")
    _install_file_menu(window, bar)
    _install_edit_menu(window, bar)
    _install_view_menu(window, bar)
    _install_fit_menu(window, bar)
    _install_help_menu(window, bar)
    window.setMenuBar(bar)


def _install_file_menu(window: QWidget, bar: QMenuBar) -> None:
    menu = QMenu("文件", bar)
    menu.setObjectName("fileMenu")
    for name in ("newProjectAction", "openProjectAction"):
        menu.addAction(_window_action(window, name))
    menu.addSeparator()
    for name in ("saveProjectAction", "saveAsProjectAction"):
        menu.addAction(_window_action(window, name))
    menu.addSeparator()
    specs = (
        (
            "importFilesMenuAction",
            "导入数据文件…",
            window.data_panel.import_files_button.click,
            "import_files",
        ),
        (
            "importFolderMenuAction",
            "导入数据文件夹…",
            window.data_panel.import_folder_button.click,
            "import_folder",
        ),
        ("reloadSourceAction", "重新加载数据源", window.reload_source_dialog, "reload_source_dialog"),
        ("relinkSourceAction", "重新链接数据源…", window.relink_source_dialog, "relink_source_dialog"),
    )
    for object_name, text, callback, command in specs:
        action = _action(window, object_name, text, callback, command=command)
        window.chrome_actions[object_name] = action
        menu.addAction(action)
        if object_name in ("importFolderMenuAction", "relinkSourceAction"):
            menu.addSeparator()
    menu.addAction(_window_action(window, "exportResultsAction"))
    bar.addMenu(menu)


def _install_edit_menu(window: QWidget, bar: QMenuBar) -> None:
    menu = QMenu("编辑", bar)
    menu.setObjectName("editMenu")
    undo = _action(
        window,
        "undoAction",
        "撤销",
        window.document.undo,
        command="undo",
    )
    undo.setShortcut("Ctrl+Z")
    undo.setEnabled(window.document.can_undo)
    window.chrome_actions["undoAction"] = undo
    menu.addAction(undo)
    redo = _action(
        window,
        "redoAction",
        "重做",
        window.document.redo,
        command="redo",
    )
    redo.setShortcut("Ctrl+Shift+Z")
    redo.setEnabled(window.document.can_redo)
    window.chrome_actions["redoAction"] = redo
    menu.addAction(redo)
    window.document.undo_state_changed.connect(
        lambda can_undo, can_redo: (
            undo.setEnabled(can_undo),
            redo.setEnabled(can_redo),
        )
    )
    bar.addMenu(menu)


def _install_view_menu(window: QWidget, bar: QMenuBar) -> None:
    menu = QMenu("视图", bar)
    menu.setObjectName("viewMenu")
    group = QActionGroup(menu)
    group.setExclusive(True)
    for title, keys in VIEW_GROUPS:
        menu.addSection(title)
        for key in keys:
            menu.addAction(_view_action(window, group, key))
    _install_plot_tool_actions(window, menu)
    _install_layout_actions(window, menu)
    menu.addSeparator()
    guidance = _action(
        window,
        "guidanceModeAction",
        "引导模式",
        window.set_guidance_visible,
        checkable=True,
    )
    guidance.setChecked(window.guidance_is_visible())
    window.chrome_actions["guidanceModeAction"] = guidance
    menu.addAction(guidance)
    menu.addSeparator()
    expert = _action(
        window,
        "expertModeAction",
        # The entry is the checkbox by another route, so it borrows the checkbox's
        # own label rather than keeping a second copy of it here to drift.
        window.parameters_panel.expert_toggle.text(),
        window.parameters_panel.expert_toggle.setChecked,
        checkable=True,
    )
    window.chrome_actions["expertModeAction"] = expert
    window.parameters_panel.expert_toggle.toggled.connect(expert.setChecked)
    expert.setChecked(window.parameters_panel.expert_toggle.isChecked())
    menu.addAction(expert)
    bar.addMenu(menu)


def _install_plot_tool_actions(window: QWidget, menu: QMenu) -> None:
    """把绘图条上那一组命令借到 视图 菜单里，同一批 ``QAction``。

    设计稿的 ``.modebar`` 只摆四枚字形，查看 · 掩膜 · 缩放到拟合范围 · 恢复完整视图 ·
    叠加对比 因此退到条自己的右键菜单。右键菜单是隐藏入口——菜单栏得有一条明路，否则读者
    只能靠猜。挂的是条上那几条 action 本身而不是副本，所以勾选状态永远只有一份。
    """
    menu.addSection("绘图工具")
    for action in window.plot_panel.toolbar.tool_actions():
        menu.addAction(action)


def _install_layout_actions(window: QWidget, menu: QMenu) -> None:
    """Offer the one layout command a fixed column shell still needs.

    The dock build listed a visibility toggle per panel; the design's grid has
    no hideable panels, so what remains is the way back from a drag that went
    too far.  The guided surface is what hides the inspector now, and it has its
    own entry above -- a second control for the same column would let the menu
    and the surface disagree about whether it is showing.
    """
    menu.addSection("面板")
    reset = _action(
        window,
        "resetLayoutAction",
        "重置面板布局",
        window.reset_layout,
    )
    window.chrome_actions["resetLayoutAction"] = reset
    menu.addAction(reset)


def _view_action(window: QWidget, group: QActionGroup, key: str) -> QAction:
    title = next(title for name, title, _description in TAB_SPECS if name == key)
    action = _action(
        window,
        f"plotViewAction:{key}",
        title,
        lambda view_key=key: _select_view(window, view_key),
    )
    action.setCheckable(True)
    group.addAction(action)
    window.chrome_actions[action.objectName()] = action
    return action


def _select_view(window: QWidget, key: str) -> None:
    try:
        window.plot_panel.select_view(key)
    except ValueError:
        # The refusal has to name the switch that lifts it, and name it the way the
        # inspector labels it -- a reader sent to look for 专家模式 finds the command
        # bar's surface segment, which is not what hides this tab.
        label = window.parameters_panel.expert_toggle.text()
        window.statusBar().showMessage(f"该诊断视图需在检查器勾选「{label}」后可用", REFUSAL_MESSAGE_MS)
        _sync_view_actions(window)


def _install_fit_menu(window: QWidget, bar: QMenuBar) -> None:
    menu = QMenu("拟合", bar)
    menu.setObjectName("fitMenu")
    cancel = _window_action(window, "cancelFitAction")
    menu.addAction(_window_action(window, "startFitAction"))
    menu.addAction(cancel)
    force = _action(
        window,
        "forceStopFitAction",
        "强制停止拟合",
        window.fit_panel.controller.force_stop,
        command="force_stop",
    )
    force.setEnabled(cancel.isEnabled())
    cancel.changed.connect(lambda: force.setEnabled(cancel.isEnabled()))
    window.chrome_actions["forceStopFitAction"] = force
    menu.addAction(force)
    _install_result_actions(window, menu)
    bar.addMenu(menu)


def _install_result_actions(window: QWidget, menu: QMenu) -> None:
    """Give result operations and read-only evidence a menu home.

    设计稿帧① 的右栏到候选解为止：清除结果与不确定度分析都不在那一栏里画。撤掉控件不等于
    撤掉命令——这两条做的是「对已经拿到的这份结果」的操作，所以排在开始 / 取消 / 强制停止
    后面。按的还是面板自己那两个按钮，所以启用条件、确认框、专家门都只有一份实现。
    """
    menu.addSeparator()
    clear = _action(
        window,
        "clearResultsAction",
        window.result_panel.clear_button.text(),
        window.result_panel.clear_button.click,
    )
    window.chrome_actions["clearResultsAction"] = clear
    menu.addAction(clear)
    evidence = _action(
        window,
        "viewInferenceEvidenceAction",
        "查看推断证据…",
        lambda: window.result_panel.open_uncertainty_dialog(sampling=False).show(),
    )
    window.chrome_actions["viewInferenceEvidenceAction"] = evidence
    menu.addAction(evidence)
    uncertainty = _action(
        window,
        "openUncertaintyAction",
        window.result_panel.uncertainty_button.text(),
        window.result_panel.uncertainty_button.click,
    )
    window.chrome_actions["openUncertaintyAction"] = uncertainty
    menu.addAction(uncertainty)
    # 按钮此前只在专家模式露面，换成菜单项后这道门不能悄悄消失。跟着按钮的状态走而不是自己
    # 再读一遍 ``ui_state``：门槛只有一处判断，两个入口就不会一个露一个藏。
    _sync_result_actions(window)
    window.document.project_changed.connect(lambda _project: _sync_result_actions(window))
    window.result_panel.results_cleared.connect(lambda _requested: _sync_result_actions(window))


def _sync_result_actions(window: QWidget) -> None:
    """Mirror the result buttons' state onto their menu twins.

    读的是 ``isHidden``、不是 ``isVisibleTo``：检视器那一栏把整个 ``secondary`` 容器关掉了，
    问「在面板里可见吗」永远得到否。按钮自己的隐藏标志仍然只由专家模式改写，那才是这道门。
    """
    actions = window.chrome_actions
    panel = window.result_panel
    actions["clearResultsAction"].setEnabled(panel.clear_button.isEnabled())
    actions["viewInferenceEvidenceAction"].setEnabled(panel.candidates.result is not None)
    actions["openUncertaintyAction"].setVisible(not panel.uncertainty_button.isHidden())


def _show_about_dialog(window: QWidget) -> None:
    dialog = StyledDialog(
        "关于 XRR Fitter",
        description="X 射线反射率全自动拟合桌面应用",
        parent=window,
        buttons=QDialogButtonBox.StandardButton.Close,
    )
    from xrr_fitter.gui.application import _build_app_icon

    icon_label = QLabel()
    icon_label.setPixmap(_build_app_icon().pixmap(64, 64))
    icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    dialog.body_layout.addWidget(icon_label)
    info = QLabel(
        "导入 .xy / .dat / .txt 反射率数据，初始化样品结构，\n"
        "一键拟合并导出结果。\n\n"
        "支持的 Python 边界：xrr_fitter.api"
    )
    info.setWordWrap(True)
    dialog.body_layout.addWidget(info)
    dialog.exec()


def _install_help_menu(window: QWidget, bar: QMenuBar) -> None:
    menu = QMenu("帮助", bar)
    menu.setObjectName("helpMenu")
    about = _action(
        window,
        "aboutAction",
        "关于 XRR Fitter",
        lambda: _show_about_dialog(window),
    )
    window.chrome_actions["aboutAction"] = about
    menu.addAction(about)
    bar.addMenu(menu)


def _install_status_bar(window: QWidget) -> None:
    """把设计稿那一行状态栏装上，形状归 ``status_bar``，读数归这里。

    段的排布、间距、圆点直径、等宽数字都在 ``status_bar`` 里，因为那些是同一张设计稿的
    同一条规则；这个模块只负责把项目状态投影成各段的文字与语义色。
    """
    bar = QStatusBar(window)
    bar.setObjectName("mainStatusBar")
    bar.setSizeGripEnabled(False)
    status_bar.build(window, bar)
    window.setStatusBar(bar)


def _connect_view_sync(window: QWidget) -> None:
    window.plot_panel.view_changed.connect(lambda _index: _sync_view_actions(window))
    window.parameters_panel.expert_toggle.toggled.connect(lambda _checked: _sync_view_actions(window))
    _connect_mode_segment(window)
    _connect_batch_segment(window)
    _sync_view_actions(window)


def _connect_mode_segment(window: QWidget) -> None:
    """Bind the command bar's segment to the guidance command, both ways.

    The action stays the one source of truth: pressing a half sets it, and every
    change to it -- from the menu, from a shortcut, from the guided flow
    finishing -- lights the matching half.  Neither control can therefore show a
    surface the other has left.
    """
    segment = window.workspace_mode_segment
    action = window.chrome_actions["guidanceModeAction"]
    segment.chosen.connect(lambda value: action.setChecked(value == "guided"))
    action.toggled.connect(lambda checked: segment.set_value("guided" if checked else "expert"))
    segment.set_value("guided" if action.isChecked() else "expert")


def _connect_batch_segment(window: QWidget) -> None:
    """Bind the batch segment to ``project.batch_mode``, both ways.

    The project stays the one source of truth.  A refused switch -- joint needs at
    least two datasets -- has to be read back, because the pressed half checked
    itself before the service was asked and would otherwise sit lit over a project
    that never changed.
    """
    segment = window.batch_mode_segment
    segment.chosen.connect(lambda value: _choose_batch_mode(window, value))
    window.document.project_changed.connect(lambda _project: _sync_batch_segment(window))
    window.fit_panel.running_changed.connect(lambda running: window.batch_mode_group.setEnabled(not running))
    _sync_batch_segment(window)


def _choose_batch_mode(window: QWidget, mode: str) -> None:
    try:
        window.fit_panel.set_batch_mode(mode)
    except ValueError as error:
        window.statusBar().showMessage(str(error), REFUSAL_MESSAGE_MS)
        _sync_batch_segment(window)


def _sync_batch_segment(window: QWidget) -> None:
    window.batch_mode_segment.set_value(window.document.project.batch_mode)


def _sync_view_actions(window: QWidget) -> None:
    current = window.plot_panel.current_view_key()
    for key, _title, _description in TAB_SPECS:
        action = window.chrome_actions[f"plotViewAction:{key}"]
        action.setChecked(key == current)


def _active_dataset_text(window: QWidget) -> str:
    """帧① 右边那一段的取值：数据集自己的名字。

    说明文字「活动数据集：」住在段里（见 ``status_bar``），所以这里只报名字。没有活动
    数据集时报空串，整段随之藏起来，而不是打出「活动数据集：无活动数据集」这种自问自答。
    """
    dataset = _active_dataset(window)
    if dataset is None:
        return ""
    return dataset.display_name or dataset.dataset_id


def _active_dataset(window: QWidget) -> object | None:
    dataset_id = window.document.active_dataset_id
    if dataset_id is None:
        return None
    return next(
        (value for value in window.document.project.datasets if value.dataset_id == dataset_id),
        None,
    )


def _active_result(window: QWidget) -> object | None:
    dataset = _active_dataset(window)
    return None if dataset is None else dataset.last_valid_result


def _reported_candidate(window: QWidget, result: object) -> object | None:
    """状态栏该报哪一条候选解：右栏选中的那条，没有选中就是最优解。

    读选中项而不是一律读最优解，是为了让底栏那两个数与右栏候选解列表指向同一次求解；
    否则点开第二候选解看细节时，屏幕最下面报的还是最优解的 J。
    """
    panel = getattr(window, "result_panel", None)
    selected = None if panel is None else panel.selected_candidate_id()
    if selected is not None:
        for candidate in result.candidates:
            if candidate.candidate_id == selected:
                return candidate
    return result.best_candidate


def _fit_quality(window: QWidget) -> tuple[str, str, str]:
    """Report the active dataset's finished verdict as word, semantic kind, and glyph.

    Readiness stays true about starting another fit even when the run that just
    finished is 不可信, so the bar would otherwise read as reassuring about a
    result it says nothing about.

    判定词与它的读数分成两段：设计稿帧① 底栏是「拟合结果：● 可信」与
    「J = 1.83 · χ²ᵥ = 1.14」两段，中间只有段间距，所以 J 不再折进这一段。

    颜色读 theme 那张比较用的表，不再自己重述一份。设计稿的 ``.cf-correlated`` 是
    ``--info``（帧⑤ 底栏「判定：◆ 可用但相关」），把它折到 ``warn`` 上会画错颜色。
    """
    result = _active_result(window)
    if result is None:
        return "", "", theme.CONFIDENCE_FALLBACK_GLYPH
    verdict = str(result.confidence.value)
    return (
        verdict,
        theme.CONFIDENCE_COMPARISON_KINDS.get(verdict, ""),
        theme.CONFIDENCE_GLYPHS.get(verdict, theme.CONFIDENCE_FALLBACK_GLYPH),
    )


def _fit_metrics_text(window: QWidget) -> str:
    """帧① 底栏中间那一段：``J = 1.83 · χ²ᵥ = 1.14``。

    J 与噪声模式、残差单位一起读。只有 Gaussian 标准化残差才能解释成 χ²ᵥ；ν ≤ 0 或
    残差全是非有限值时也不写它。没有候选解时整段留空——这一行是读数，不是占位。
    """
    result = _active_result(window)
    if result is None:
        return ""
    candidate = _reported_candidate(window, result)
    if candidate is None:
        return ""
    parts = [f"J = {candidate.objective:g}", f"{candidate.noise_model} [{candidate.residual_unit}]"]
    if candidate.noise_model == "gaussian":
        chi = reduced_chi_squared(candidate.weighted_residuals, free_parameter_count(result))
        if chi is not None:
            parts.append(f"χ²ᵥ = {chi:.3g}")
    return " · ".join(parts)


def _sampling_projection(window: QWidget) -> tuple[str, str, str]:
    """帧⑤ 底栏那一段的 ``split-R̂`` 与 ``ESS``，读的是与 J、χ²ᵥ 同一条候选解。

    两个读数与它们的颜色一并由 ``results.uncertainty`` 给出——阈值与「报最坏的那一个」
    这条口径归它，底栏只负责把结果贴到两个标签上。
    """
    result = _active_result(window)
    if result is None:
        return "", "", ""
    candidate = _reported_candidate(window, result)
    return sampling_readings(result, None if candidate is None else candidate.candidate_id)


def _readiness_projection(
    readiness: object,
    running: str | None,
    guided: bool,
    sampling: str = "",
    unsaved_structure: bool = False,
) -> tuple[str, str]:
    """Say what the first status segment should read, and in which semantic kind.

    Readiness is derived from the project alone, so it keeps asserting 已就绪 while
    an operation runs: true about starting again, misleading about the window the
    user is looking at. A live run therefore owns the line, in the accent kind the
    design's frame ④ paints that dot, rather than in the ``ok`` that would read as
    an invitation.

    引导模式下这一段报的是模式（设计稿帧② 是「引导模式」加一颗 ``--info`` 圆点）：
    那一屏把检查器收走了，一步一步走到哪里由它右边那一段报，这里说清「现在是被领着走的」
    才对得上那个序号；而运行中仍然优先，运行是比模式更要紧的当下状态。

    采样那一趟的圆点是设计稿里唯一不用强调色的运行态（帧⑤ 是 ``--ok``）：采样期间那一行
    右边就跟着 split-R̂ 与 ESS，圆点报的是这两个读数的结论而不是「有事在跑」。所以拿得到
    收敛结论时用它，拿不到才退回强调色。

    改过结构而没存盘时报那件事，圆点是设计稿帧③ 的 ``warn``。它排在运行与模式之下：那两
    个说的是「此刻正在发生什么」，而未存盘说的是「盘上那一份已经旧了」——后者等得起。
    """
    if running == "mcmc" and sampling:
        return messages.running_text(running), sampling
    if running is not None:
        return messages.running_text(running), "accent"
    if guided:
        return GUIDED_READINESS_TEXT, "info"
    if unsaved_structure:
        return UNSAVED_STRUCTURE_TEXT, "warn"
    return messages.readiness_text(readiness.message), "ok" if readiness.ready else "warn"


def _write_run_summary(window: QWidget, stage: str, position: str, remaining: str) -> None:
    """把一次运行的三段读数写进底栏，段的显隐交给 ``status_bar.sync``。"""
    for name, text in (
        ("fitStageStatus", stage),
        ("fitPositionStatus", position),
        ("fitRemainingStatus", remaining),
    ):
        label = window.findChild(QLabel, name)
        if label is not None:
            label.setText(text)
    status_bar.sync(window)


def refresh_run_summary(window: QWidget, summary: object) -> None:
    """Show what the progress card published about the run it is tracking.

    The card owns the stage ordinal, the monotonic position and the smoothed
    remaining estimate; the bar only repeats them, so the two surfaces cannot
    disagree about where the same run stands.

    三个取值分开报：设计稿帧④ 把阶段与进度留在左边、把「预计剩余」挪到弹簧右边，
    一句拼好的话没法这么排。
    """
    _write_run_summary(window, summary.stage, summary.position, summary.remaining)


def _project_run_summary(window: QWidget, running: str | None) -> None:
    """Empty the run segments once nothing is running, and only then.

    A refresh can be triggered by any project change, including several during a
    run, and none of those callers know where the fit has reached -- so clearing
    unconditionally would blank a live summary until the next frame arrived. An
    idle window, on the other hand, has no position at all, and leaving the last
    run's numbers beside a 已就绪 line would read as a run still going.
    """
    if running is None:
        _write_run_summary(window, "", "", "")


# 层堆叠列表两端那两行：入射介质与基底。它们不在 ``components`` 里，却各占一行也各自带
# 参数进拟合——设计稿帧③ 的「4 层」把它们算在内，左栏那行层数用的也是同一个口径。
STACK_ENDS = 2

# 结构这一步屏幕最下面报的是这一步的事：这一叠层有多大、手上选中的是哪一层、下一步做什么。
STRUCTURE_STEP_SPANS = (
    "statusScaleSpan",
    "statusSelectedSpan",
    "statusNextSpan",
)

# 同一帧里让位的那几段：判定与它的读数还没有，批量与活动数据集、源校验在这一步无关手上的
# 动作。设计稿帧③ 底栏只有上面那三段加就绪，所以这五段在这一步退场。
STRUCTURE_STEP_YIELDS = (
    "statusVerdictSpan",
    "statusMetricsSpan",
    "statusBatchSpan",
    "statusDatasetSpan",
    "statusSourceSpan",
)

# 右半边报什么，跟着「有没有操作在跑」换。把五张底栏排在一起看就是这条规矩：闲着时（帧①②）
# 报手上这份活儿的来路——活动数据集与源校验；拟合跑着时（帧④）来路不再是要紧事，要紧的是
# 这一趟怎么跑的——模式与预计剩余；采样跑着时（帧⑤）连模式也退场，只留判定挪到右边。
# 拟合跑着时它自己会把读数换掉，所以判定与 J、χ²ᵥ 这两段在那一趟里也不作声。
RUN_SCOPES = (
    ("statusBatchSpan",),
    ("statusDatasetSpan", "statusSourceSpan", "statusMetricsSpan"),
    ("statusVerdictSpan",),
)


def _scope_by_run(window: QWidget, running: str | None) -> None:
    """按「有没有操作在跑」记下右半边各段的进出，并把判定挪到该在的那一边。

    这一维与管线步骤那一维分开记（见 ``status_bar.allow`` 的 ``gate``）：模式那一段既要在
    结构那一步让位，也只在拟合跑着时才有话可说，两个判断落在两个属性上才不会互相擦掉——
    否则进了结构步再来一次项目变更，活动数据集就会在帧③ 里冒出来。
    """
    for names, permitted in zip(
        RUN_SCOPES,
        (running == "fit", running is None, running != "fit"),
        strict=True,
    ):
        status_bar.allow(window, names, allowed=permitted, gate=status_bar.SPAN_RUN_SCOPE)
    # 判定在帧① 读作「拟合结果：」站在左边，在帧⑤ 读作「判定：」挪到右边——采样那一趟里
    # 它就是右半边唯一的一段。
    status_bar.place_verdict(window, on_right=running == "mcmc")


def _structure_scale_text(window: QWidget) -> str:
    """帧③ 中间那一段：「4 层 · 9 个自由参数」。

    自由数读的是参数面板已经编译好的那份声明，而不是自己再调一次 ``describe_parameters``
    ——后者会读源文件，源文件不在时还会抛，而状态栏这一行不该为此空掉。
    """
    dataset = _active_dataset(window)
    if dataset is None or dataset.structure is None:
        return ""
    rows = len(dataset.structure.components) + STACK_ENDS
    free = sum(1 for item in window.parameters_panel.definitions if not item.locked)
    return f"{rows} 层 · {free} 个自由参数"


def _selected_layer_text(component: object) -> str:
    """帧③ 右边那一段的取值：「a-Si」。

    只取显示名的前半截。设计稿这一行写的是「a-Si」而不是「a-Si · 非晶硅薄膜」——同一行里
    还站着层数与下一步，说明性的后半截在这里只会把别的段挤出去；要看全名右栏抬头就是。
    """
    if component is None:
        return ""
    label = naming.expert_name(getattr(component, "name", ""))
    return label.split(naming.SEPARATOR)[0]


def refresh_structure_status(window: QWidget, *, active: bool) -> None:
    """让这几段跟着管线步骤进出，并在露面时把读数刷新一遍。

    进出记在段上、由 ``status_bar.sync`` 统一落实，而不是在这里逐个 ``setVisible``：
    一段还得有取值可报才该露面，否则退出结构那一步时会露出一个只剩「拟合结果：」的空段。
    """
    scale = window.findChild(QLabel, "structureScaleStatus")
    if scale is not None and active:
        scale.setText(_structure_scale_text(window))
    status_bar.allow(window, STRUCTURE_STEP_SPANS, allowed=active)
    status_bar.allow(window, STRUCTURE_STEP_YIELDS, allowed=not active)
    status_bar.sync(window)


def set_selected_layer_status(window: QWidget, component: object) -> None:
    """层堆叠的选中变了，状态栏那一段跟着改。"""
    selected = window.findChild(QLabel, "selectedLayerStatus")
    if selected is not None:
        selected.setText(_selected_layer_text(component))
        status_bar.sync(window)


def _write_sampling(window: QWidget, rhat_text: str, size_text: str, kind: str) -> None:
    """把采样那一段的两个读数贴上去；颜色只落在 R̂ 上。

    设计稿帧⑤ 里 ESS 是加粗的白文而 R̂ 带语义色——收敛与否的结论是 R̂ 在说，ESS 是它旁边
    的一个规模，两个都染色反而看不出哪个在报警。
    """
    rhat = window.findChild(QLabel, "mcmcRhatStatus")
    if rhat is not None:
        rhat.setText(rhat_text)
        theme.set_status_kind(rhat, kind)
    size = window.findChild(QLabel, "mcmcEssStatus")
    if size is not None:
        size.setText(size_text)


def _write_verdict(window: QWidget) -> None:
    """判定那一段：字形与判定词同色，读数（J、χ²ᵥ）在它右边另起一段。"""
    quality_label = window.findChild(QLabel, "fitQualityStatus")
    if quality_label is None:
        return
    quality_text, quality_kind, quality_glyph = _fit_quality(window)
    quality_label.setText(quality_text)
    theme.set_status_kind(quality_label, quality_kind)
    quality_dot = window.findChild(QLabel, "fitQualityDot")
    if quality_dot is not None:
        # 字形与判定词是一个整体（``<b class="cf-trusted">● 可信</b>``），整段的进出归
        # ``status_bar.sync``，所以这里不单独藏这颗字形。
        quality_dot.setText(quality_glyph)
        theme.set_status_kind(quality_dot, quality_kind)
    metrics_label = window.findChild(QLabel, "fitMetricsStatus")
    if metrics_label is not None:
        metrics_label.setText(_fit_metrics_text(window))


def _write_context(window: QWidget) -> None:
    """右半边报来路的那两段：活动数据集与批量模式。源校验由项目动作自己刷。"""
    dataset_label = window.findChild(QLabel, "activeDatasetStatus")
    if dataset_label is not None:
        dataset_label.setText(_active_dataset_text(window))
    batch_label = window.findChild(QLabel, "batchModeStatus")
    if batch_label is not None:
        batch_label.setText(BATCH_MODE_TEXTS.get(window.document.project.batch_mode, ""))


def refresh_status(window: QWidget, readiness: object, running: str | None = None) -> None:
    """Project readiness, active dataset, and source-action enablement."""
    label = window.findChild(QLabel, "fitReadinessStatus")
    if label is None:
        return
    guided = getattr(window, "guidance_is_visible", None)
    rhat_text, size_text, sampling_kind = _sampling_projection(window)
    readiness_text, readiness_kind = _readiness_projection(
        readiness,
        running,
        bool(guided and guided()),
        sampling_kind,
        bool(getattr(window, "structure_edit_unsaved", False)),
    )
    # 就绪那一段的文字在设计稿五帧里一律是次要色的白文，语义色只落在它前面那颗圆点上
    # （``.statusbar .dot`` 的 ``background``）。所以这里只给圆点着色。
    label.setText(readiness_text)
    _project_run_summary(window, running)
    readiness_dot = window.findChild(QLabel, "fitReadinessDot")
    if readiness_dot is not None:
        theme.set_status_kind(readiness_dot, readiness_kind)
    _write_verdict(window)
    _write_sampling(window, rhat_text, size_text, sampling_kind)
    _write_context(window)
    _scope_by_run(window, running)
    status_bar.sync(window)
    has_active = window.document.active_dataset_id is not None
    for name in ("reloadSourceAction", "relinkSourceAction"):
        window.chrome_actions[name].setEnabled(has_active)
