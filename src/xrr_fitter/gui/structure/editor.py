"""Concrete structure value editor and deterministic tree renderer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.canvas_top import canvas_top
from xrr_fitter.gui.structure import naming, rows
from xrr_fitter.gui.structure.dialogs import BackingDialog, LayerDialog, PeriodicDialog
from xrr_fitter.gui.structure.reorder import ReorderableTree
from xrr_fitter.gui.structure.stack import StackView, component_fill

# 一行画成什么样，在建项时就随项存下来（颜色 / 名字 / 副行 / 读数）。项建完整棵树才挂
# 得上行控件——``setItemWidget`` 要求项已经在树里——所以两步之间得有个地方放这份配方。
ROW_SPEC = Qt.ItemDataRole.UserRole + 1

CommitStructure = Callable[[api.StructureSpec], object]
OxideAction = Callable[[], object]

# 设计稿这张卡是一列 ``.lyr``，不是表格：厚度、粗糙度、密度收进每行的副行，周期数收进
# 周期块自己那一行。六列摊在 264px 的画布列里每列只剩四十来像素，数字全被截断，而列头
# 还要先吃掉一整行高度。
STRUCTURE_TREE_COLUMNS = 1

# 设计稿 ``.lyr .nm small`` 的数字都是两位小数（``48.70``、``0.51``、``2.19``）。定点比
# ``:g`` 更适合一列上下相邻的读数：小数点对得齐，长短不一的有效数字不会让一列看起来像
# 在跳。
NM_DECIMALS = 2

# 一层喂给拟合的三个量：厚度、粗糙度、密度。周期块按它的子层数乘。
PARAMETERS_PER_LAYER = 3

# The tree annotates a drifting block in its tooltip rather than adding a
# column, so these mirror the dialog's Chinese labels for kind and target.
DRIFT_KIND_LABELS = {"linear": "线性", "sine": "正弦", "random": "随机"}
DRIFT_TARGET_LABELS = {"thickness": "厚度", "roughness": "粗糙度"}

# The diagram and the tree state the same structure in two registers, so the
# diagram needs a title to say which one it is and a subtitle to say that it is
# interactive; unlabelled, a proportional section reads as an ornament.
# 列表按行数要高度，到这个行数为止；再深的堆叠由列表自己滚，因为到那时面板已经是列表
# 的天下，让整块面板跟着长高只会把下面的绘图区挤没。
TREE_ROWS_BEFORE_SCROLL = 8

STACK_CARD_TITLE = "层堆叠"
STACK_CARD_SUBTITLE = "从空气到基底 · 点击选中 · 拖动排序"

# 设计稿帧③ ``.canvas-top`` 的三个 tab（HTML 644 行）与它们各自指向的那张卡。tab 不藏
# 内容——三张卡在画布列里都在——它是这一列的目录，告诉读者往下还有什么。
CANVAS_VIEWS = (
    ("structure", "样品结构"),
    ("sld", "SLD 深度剖面"),
    ("parameters", "参数总览"),
)

# 设计稿这一行右边那一组只有三个（HTML 648 行），且都属于「往结构里加东西」这一类。
ADD_LAYER_TEXT = "＋ 添加层"
OXIDE_TEXT = "建议氧化层"
PERIODIC_TEXT = "周期结构…"


def _material_text(material: api.MaterialSpec) -> str:
    return material.formula.strip() if material.formula is not None else "显式 SLD"


def _nm(value_a: float) -> str:
    return f"{value_a / 10.0:g}"


def _nm2(value_a: float) -> str:
    """One length in nm at the design's fixed two decimals."""
    return f"{value_a / 10.0:.{NM_DECIMALS}f}"


def _substance_text(material: api.MaterialSpec) -> str:
    """副行末段：有密度就报密度，没有就报材料本身是怎么给的。

    六列时代 ``材料`` 和 ``密度`` 各占一列，其中总有一列是空的——显式 SLD 的层没有密度，
    有密度的层的化学式又已经写在名字里。合成一段之后两种给法都还在，且不再为对方留白。
    """
    density = material.bulk_density_g_cm3
    return f"密度 {density:.{NM_DECIMALS}f}" if density is not None else _material_text(material)


def _layer_detail(layer: api.LayerSpec) -> str:
    """设计稿 ``.lyr .nm small``：厚度 · 粗糙 · 密度，一行说完这层是什么。"""
    return " · ".join(
        (
            f"厚度 {_nm2(layer.thickness_a)}",
            f"粗糙 {_nm2(layer.roughness_a)}",
            _substance_text(layer.material),
        )
    )


def _medium_detail(material: api.MaterialSpec, roughness_a: float | None) -> str:
    """半无限介质的副行。入射侧没有粗糙度可报，基底有。

    基底的密度后面跟一个「（锁定）」，设计稿帧③ 就是这么写的，而它在我们这儿字面为真：
    ``default_parameter_definitions`` 给基底只发一个声明（``backing.roughness_a``）。层的
    密度有 ``component.N.density_scale`` 可以放开，基底的密度连声明都没有，任何拟合都动不
    了它。这一行的两个数因此一个能拟合一个不能——不点出来，读者只会去参数表里找那个永远
    找不到的密度。

    显式给 SLD 的基底（``_substance_text`` 那时报的是材料本身而不是密度）不加这个后缀：
    那种给法下 ``_material_definitions`` 反而会发出 ``backing.sld_real_a2``，是能拟合的。
    """
    if roughness_a is None:
        return "入射介质 · 半无限"
    substance = _substance_text(material)
    if material.bulk_density_g_cm3 is not None:
        substance = f"{substance}（锁定）"
    return f"粗糙 {_nm2(roughness_a)} · {substance}"


def _parameter_count(component: object) -> int:
    """这一行喂给拟合几个参数：一层三个（厚度 / 粗糙度 / 密度）。

    设计稿右端读数报的是「3 参数」。真正的自由度还要看锁定与共享，那是参数表那一步的
    事；这里报的是这一行的结构规模，和右栏状态栏的「9 个自由参数」不是同一个数。
    """
    if isinstance(component, api.PeriodicBlock):
        return PARAMETERS_PER_LAYER * len(component.layers)
    return PARAMETERS_PER_LAYER


def _transition_text(transition: api.InterfaceTransition) -> str:
    """Spell out the transition in the roughness column without adding one."""
    branches = "、".join(
        f"{branch.kind} 权重 {branch.weight:g} 宽度 {_nm(branch.thickness_a)} nm" for branch in transition.branches
    )
    return f"界面过渡取代粗糙度：{branches}；切片上限 {_nm(transition.microslab_max_a)} nm"


def _drift_tooltip(drift: api.DriftSpec) -> str:
    """Name the drift law and target in the block row without a new column."""
    kind = DRIFT_KIND_LABELS.get(drift.kind, drift.kind)
    target = DRIFT_TARGET_LABELS.get(drift.target, drift.target)
    return f"漂移：{kind} · 作用于{target}"


def _oxide_identity(value: object) -> tuple[object, ...]:
    material = value.oxide_material
    formula = material.formula if isinstance(material, api.MaterialSpec) else material
    return (
        value.base_material,
        formula,
        value.location,
        value.oxide_table_version,
    )


class StructureEditor(QWidget):
    """Edit a detached structure while a panel owns API publication."""

    # 帧③ 把「选哪一层」和「这一层是什么」分给画布和右栏两处，这个信号是它们之间
    # 唯一的接线：载荷是 ``(index, component)``，没有选中普通层时是 ``(None, None)``。
    component_selected = Signal(object, object)

    # 层数变了，卡片要的高度也就变了。摆在画布里的容器只在建栈时问过一次高度，靠这个
    # 信号才知道该重新问；与 ``StructurePanel.structure_changed`` 不同，它不管有没有提
    # 交到 API，``load`` 也算——重绘过就发。
    contents_resized = Signal()

    # 画布列的目录换了一页。载荷是 ``CANVAS_VIEWS`` 的键；哪一页对应挪动哪块面板由窗口
    # 决定，编辑器只报「读者点到了哪一页」。
    canvas_view_changed = Signal(str)

    def __init__(
        self,
        commit_structure: CommitStructure,
        accept_oxide: OxideAction,
        refuse_oxide: OxideAction,
        parent: QWidget | None = None,
        *,
        master_seed_source: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("structureEditor")
        self.setAccessibleName("结构编辑")
        self._commit_structure = commit_structure
        self._accept_oxide = accept_oxide
        self._refuse_oxide = refuse_oxide
        self._master_seed_source = master_seed_source
        self._structure: api.StructureSpec | None = None
        self._decisions: tuple[api.OxideDecision, ...] = ()
        self._suggestion: api.OxideSuggestion | None = None
        self._build_widgets()
        self.clear()

    def _build_widgets(self) -> None:
        self.tree = ReorderableTree()
        self.tree.setObjectName("structureTree")
        self.tree.setItemDelegate(rows.LayerRowDelegate(self.tree))
        # 设计稿 ``.stack`` 是一列 ``.lyr``：一行一层，没有列头。每行的内容由
        # ``LayerRow`` 画，树只剩「按顺序排、点一行选中、拖一行换位」这三件事。
        self.tree.setColumnCount(STRUCTURE_TREE_COLUMNS)
        self.tree.setHeaderHidden(True)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        # 隔行底色是给多列表格分行用的；这里分行靠 ``.lyr`` 自己的下边框，隔行色反而会
        # 和半无限那两行的 ``.semi`` 底色打架，读成四档深浅。
        self.tree.setAlternatingRowColors(False)
        # 行高不再一致：带副行的层比只有一行字的介质行高一截，统一行高会按最矮的那行裁。
        self.tree.setUniformRowHeights(False)
        # 顶层行左边不留展开箭头的位置，四行才和设计稿一样贴着盒子左缘；周期块的子层
        # 仍然缩进，因为它们是子项而不是顶层项。
        self.tree.setRootIsDecorated(False)
        self.stack = StackView()
        self.tree.currentItemChanged.connect(self._refresh_action_state)
        self.tree.currentItemChanged.connect(self._sync_stack_selection)
        self.tree.currentItemChanged.connect(self._sync_row_selection)
        # 拖放和上移/下移走同一条提交路径，所以拖出来的层序受同一套校验。
        self.tree.component_moved.connect(self.move_component)
        self.stack.component_selected.connect(self._select_component)
        self.add_layer_button = self._button(ADD_LAYER_TEXT, "addLayerButton")
        self.oxide_button = self._button(OXIDE_TEXT, "oxideSuggestionButton")
        self.add_periodic_button = self._button(PERIODIC_TEXT, "addPeriodicBlockButton")
        self._build_row_menu()
        self.error_label = QLabel()
        self.error_label.setObjectName("structureValidationError")
        self.error_label.setProperty("statusKind", "error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self._connect_actions()
        # 设计稿 ``.canvas-top``：三个 tab，紧跟着这一页的三个动作，留白落在行尾。此前是
        # 六个按钮排成两行、外加一条随建议进出的氧化层条——三行行头压在层列表上面，而画布
        # 列一共只有三段高度。氧化层建议改成常驻但按有无建议启用：一个按钮时隐时现，会让
        # 它旁边那两个每次都换位置。
        self.canvas_row, self.canvas_tabs, actions = canvas_top(
            self,
            name="structureCanvasTop",
            tabs_name="structureCanvasTabs",
            titles=[title for _key, title in CANVAS_VIEWS],
        )
        actions.addWidget(self.add_layer_button)
        actions.addWidget(self.oxide_button)
        actions.addWidget(self.add_periodic_button)
        self.canvas_tabs.currentChanged.connect(self._announce_view)
        self.stack_card, stack_body = theme.titled_card(
            self,
            "structureStackCard",
            STACK_CARD_TITLE,
            STACK_CARD_SUBTITLE,
        )
        # 设计稿这张卡装的就是层列表，卡头「点击选中 · 拖动排序」说的也是行的事。色带
        # 示意图不在这一帧的画面里：它按厚度分段，1400×900 下光是给它留的那 160px 就把
        # 层列表整段顶出视口，而它能说的（谁在谁上面、哪一层多厚）列表已经逐行写清了。
        # 图本身留着——它仍是行配色（``component_fill``）与选中互指的那一份模型。
        stack_body.addWidget(self.tree, 1)
        self.stack.setParent(self.stack_card)
        self.stack.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(self.canvas_row)
        layout.addWidget(self.stack_card, 1)
        layout.addWidget(self.error_label)

    def _build_row_menu(self) -> None:
        """设计稿行头只画三个按钮，其余命令回到它们作用的那一行上。

        编辑基底 / 删除 / 上移 / 下移 / 忽略氧化层建议此前都是行头的按钮，而它们全都是
        「对着某一行做」——按钮在行头，作用点在列表里，读者得自己把两处对齐。挂到行的右键
        菜单上之后，点哪一行、命令作用在哪一行是同一个动作；快捷键让它们不必先找到菜单。
        上移 / 下移在设计稿里另有一条路：行首的 ⋮⋮ 拖拽柄。
        """
        self.row_menu = QMenu(self)
        self.row_menu.setObjectName("structureRowMenu")
        self.edit_backing_action = self._action(
            "编辑基底",
            "editBackingAction",
            "编辑半无限基底的材料与粗糙度",
            self._choose_backing,
        )
        self.remove_action = self._action(
            "删除",
            "removeComponentAction",
            "删除当前选中的结构组件",
            self._remove_selected,
            QKeySequence.StandardKey.Delete,
        )
        self.up_action = self._action(
            "上移",
            "moveComponentUpAction",
            "将当前结构组件上移（也可拖动行首的 ⋮⋮）",
            lambda: self._move_selected(-1),
            QKeySequence("Ctrl+Up"),
        )
        self.down_action = self._action(
            "下移",
            "moveComponentDownAction",
            "将当前结构组件下移（也可拖动行首的 ⋮⋮）",
            lambda: self._move_selected(1),
            QKeySequence("Ctrl+Down"),
        )
        self.refuse_action = self._action(
            "忽略氧化层建议",
            "oxideSuggestionRefuseAction",
            "记录并隐藏当前氧化层建议",
            self._refuse_oxide,
        )
        for action in (
            self.edit_backing_action,
            self.remove_action,
            self.up_action,
            self.down_action,
            self.refuse_action,
        ):
            self.row_menu.addAction(action)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_row_menu)

    def _action(
        self,
        text: str,
        name: str,
        hint: str,
        slot: Callable[[], object],
        shortcut: QKeySequence | QKeySequence.StandardKey | None = None,
    ) -> QAction:
        action = QAction(text, self)
        action.setObjectName(name)
        action.setToolTip(hint)
        action.triggered.connect(slot)
        if shortcut is not None:
            action.setShortcut(shortcut)
            # 快捷键的作用域是列表本身：这些命令都吃「当前行」，焦点不在列表上时按下
            # Delete，读者心里的目标行未必还是列表里高亮的那一行。
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        # 加到树上不是为了它自带的菜单（策略是 CustomContextMenu），而是为了让快捷键有
        # 一个宿主控件——没有宿主的 QAction 的 shortcut 永远不会触发。
        self.tree.addAction(action)
        return action

    def _show_row_menu(self, position: QPoint) -> None:
        item = self.tree.itemAt(position)
        if item is not None:
            self.tree.setCurrentItem(item)
        self.row_menu.exec(self.tree.viewport().mapToGlobal(position))

    def _announce_view(self, index: int) -> None:
        if 0 <= index < len(CANVAS_VIEWS):
            self.canvas_view_changed.emit(CANVAS_VIEWS[index][0])

    def _button(self, text: str, name: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(name)
        button.setAccessibleName(text)
        button.setProperty("commandBar", True)
        return button

    def _connect_actions(self) -> None:
        self.add_layer_button.clicked.connect(self._choose_layer)
        self.add_periodic_button.clicked.connect(self._choose_periodic)
        self.oxide_button.clicked.connect(self._accept_oxide)

    @property
    def structure(self) -> api.StructureSpec | None:
        return self._structure

    def load(
        self,
        structure: api.StructureSpec,
        decisions: tuple[api.OxideDecision, ...] = (),
        *,
        keep_selection: bool = False,
    ) -> None:
        """Show ``structure``, optionally landing back on the row that was selected.

        重建整棵树会把选中一起清掉。改一层的厚度走的正是这条路——``replace_component``
        保住了那一层在堆叠里的位置，选中却没保住，右栏那张卡因为自己那次编辑而变成
        「未选择」，接着改第二个数还得回堆叠里再点一次。

        只有调用方确认这是「同一份结构被自己的编辑刷新了」时才恢复：换数据集时第 1 行是
        另一层，落回去等于选错了层还不作声。
        """
        previous = self._selected_index() if keep_selection else None
        self._structure = structure
        self._decisions = tuple(decisions)
        self.error_label.hide()
        self._render()
        self.stack.load(structure)
        if previous is not None and previous < len(structure.components):
            self._select_component(previous)
        self._refresh_oxide()
        # 树和示意图都摆好了才报高度：在 ``_render`` 里发会赶在示意图长高之前，报的是
        # 上一版的数。
        self.contents_resized.emit()

    def clear(self) -> None:
        self._structure = None
        self._decisions = ()
        self._suggestion = None
        self.tree.clear()
        self.stack.clear()
        for button in (
            self.add_layer_button,
            self.add_periodic_button,
        ):
            button.setEnabled(False)
        self.edit_backing_action.setEnabled(False)
        self.oxide_button.setEnabled(False)
        self.refuse_action.setEnabled(False)
        self._refresh_action_state()
        self.error_label.hide()
        self.contents_resized.emit()

    def current_oxide_suggestion(self) -> api.OxideSuggestion | None:
        return self._suggestion

    def add_layer(self, layer: api.LayerSpec) -> None:
        structure = self._require_structure()
        self._commit_structure(replace(structure, components=(*structure.components, layer)))

    def add_periodic_block(self, block: api.PeriodicBlock) -> None:
        structure = self._require_structure()
        self._commit_structure(replace(structure, components=(*structure.components, block)))

    def set_backing(self, backing: api.MaterialSpec, roughness_a: float) -> None:
        structure = self._require_structure()
        self._commit_structure(
            replace(
                structure,
                backing=backing,
                backing_roughness_a=roughness_a,
            )
        )

    def replace_component(self, index: int, component: object) -> None:
        """Swap one component for an edited copy, leaving the stack's order alone.

        Editing an existing layer used to mean removing it and adding a new one,
        which appended it to the bottom of the stack and lost the selection with
        it.  A replacement keeps the position, so a typed thickness change reads
        as the same layer with a different number.
        """
        structure = self._require_structure()
        self._require_index(index, len(structure.components))
        components = list(structure.components)
        components[index] = component
        self._commit_structure(replace(structure, components=tuple(components)))

    def remove_component(self, index: int) -> None:
        structure = self._require_structure()
        self._require_index(index, len(structure.components))
        components = (*structure.components[:index], *structure.components[index + 1 :])
        self._commit_structure(replace(structure, components=components))

    def move_component(self, source: int, destination: int) -> bool:
        structure = self._require_structure()
        count = len(structure.components)
        self._require_index(source, count)
        self._require_index(destination, count)
        if source == destination:
            return False
        components = list(structure.components)
        component = components.pop(source)
        components.insert(destination, component)
        self._commit_structure(replace(structure, components=tuple(components)))
        return True

    def _require_structure(self) -> api.StructureSpec:
        if self._structure is None:
            raise ValueError("active dataset has no structure")
        return self._structure

    def _require_index(self, index: int, count: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < count:
            raise IndexError("component index out of range")

    def _render(self) -> None:
        structure = self._require_structure()
        self.tree.clear()
        self.tree.addTopLevelItem(self._medium_item("空气", structure.fronting, None))
        for index, component in enumerate(structure.components):
            self.tree.addTopLevelItem(self._component_item(component, index))
        backing_name = structure.backing.name.strip() or "基底"
        self.tree.addTopLevelItem(self._medium_item(backing_name, structure.backing, structure.backing_roughness_a))
        self._dress_rows()
        self._fit_tree_to_rows()
        self.add_layer_button.setEnabled(True)
        self.add_periodic_button.setEnabled(True)
        self.edit_backing_action.setEnabled(True)
        self._refresh_action_state()

    def _dress_rows(self) -> None:
        """Hang a `LayerRow` on every item, then let the tree size to them.

        行控件要先挂上去才量得出高度，而 ``setItemWidget`` 又要项已经在树里；所以整棵树
        先建完，这里再走一遍。
        """
        last = self.tree.topLevelItemCount() - 1
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            row = item.data(0, ROW_SPEC)
            widget = rows.LayerRow(*row)
            widget.set_semi_infinite(index in (0, last))
            widget.set_last(index == last)
            self.tree.setItemWidget(item, 0, widget)
            item.setSizeHint(0, widget.sizeHint())
            for child in range(item.childCount()):
                self._dress_child(item.child(child))

    def _dress_child(self, item: QTreeWidgetItem) -> None:
        widget = rows.LayerRow(*item.data(0, ROW_SPEC))
        self.tree.setItemWidget(item, 0, widget)
        item.setSizeHint(0, widget.sizeHint())

    def _sync_row_selection(self) -> None:
        """设计稿 ``.lyr.sel`` 的左缘强调条：树的高亮被行底色盖住，行得自己再戴一次。"""
        current = self.tree.currentItem()
        for item in self._all_items():
            widget = self.tree.itemWidget(item, 0)
            if widget is not None:
                widget.set_selected(item is current)

    def _all_items(self) -> list[QTreeWidgetItem]:
        found: list[QTreeWidgetItem] = []
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            found.append(item)
            found.extend(item.child(child) for child in range(item.childCount()))
        return found

    def _fit_tree_to_rows(self) -> None:
        """Ask for the height this many rows need, up to `TREE_ROWS_BEFORE_SCROLL`.

        Left at the default the list reports a minimum of about three rows however
        many it holds, so a four-row stack scrolls inside a list that is itself
        inside a scrolling panel -- two bars for one overflow, and the row the
        second bar hides is the one the first bar suggested was all there was.
        """
        rows_shown = min(self.tree.topLevelItemCount(), TREE_ROWS_BEFORE_SCROLL)
        row_height = max(self.tree.sizeHintForRow(0), theme.CONTROL_MIN_H)
        header = self.tree.header()
        chrome = (0 if header.isHidden() else header.sizeHint().height()) + 2 * self.tree.frameWidth()
        self.tree.setMinimumHeight(rows_shown * row_height + chrome)

    def _medium_item(
        self,
        name: str,
        material: api.MaterialSpec,
        roughness_a: float | None,
    ) -> QTreeWidgetItem:
        reading = "半无限" if roughness_a is not None else "SLD 0 · 半无限"
        item = QTreeWidgetItem((name,))
        item.setData(0, Qt.ItemDataRole.UserRole, None)
        item.setData(0, ROW_SPEC, (theme.DATA_NEUTRAL, name, _medium_detail(material, roughness_a), reading))
        return item

    def _component_item(self, component: object, index: int) -> QTreeWidgetItem:
        label = naming.expert_name(component.name)
        if isinstance(component, api.PeriodicBlock):
            item = QTreeWidgetItem((label,))
            detail = f"周期 ×{component.repeats} · {len(component.layers)} 层"
            if component.drift is not None:
                item.setToolTip(0, _drift_tooltip(component.drift))
            for layer in component.layers:
                item.addChild(self._layer_item(layer))
        elif isinstance(component, api.GradientLayerSpec):
            item = QTreeWidgetItem((label,))
            detail = f"厚度 {_nm2(component.thickness_a)} · 粗糙 {_nm2(component.roughness_a)} · SLD 梯度"
        else:
            item = self._layer_item(component)
            detail = _layer_detail(component)
        item.setData(0, Qt.ItemDataRole.UserRole, index)
        item.setData(
            0,
            ROW_SPEC,
            (component_fill(index), label, detail, f"{_parameter_count(component)} 参数"),
        )
        return item

    def _layer_item(self, layer: api.LayerSpec) -> QTreeWidgetItem:
        label = naming.expert_name(layer.name)
        item = QTreeWidgetItem((label,))
        if layer.transition is not None:
            item.setToolTip(0, _transition_text(layer.transition))
        # A block's children share their parent's single band, so they are left
        # bare: a chip on each would read as a band per child layer.
        item.setData(0, ROW_SPEC, (None, label, _layer_detail(layer), f"{PARAMETERS_PER_LAYER} 参数"))
        return item

    def _refresh_oxide(self) -> None:
        structure = self._require_structure()
        decided = {_oxide_identity(value) for value in self._decisions}
        self._suggestion = next(
            (value for value in api.suggest_oxide_layers(structure) if _oxide_identity(value) not in decided),
            None,
        )
        visible = self._suggestion is not None
        # 设计稿这一行的三个按钮是固定的三个，「建议氧化层」不随建议进出：一个按钮时隐时
        # 现，它右边那个每次都要换位置，而读者刚记住的正是「第三个是周期结构」。没有建议
        # 时置灰即可——灰着的按钮至少还说明「这里有过这件事」。
        self.oxide_button.setEnabled(visible)
        self.refuse_action.setEnabled(visible)

    def _selected_index(self) -> int | None:
        item = self.tree.currentItem()
        if item is None or item.parent() is not None:
            return None
        value = item.data(0, Qt.ItemDataRole.UserRole)
        return value if isinstance(value, int) else None

    def _sync_stack_selection(self, *_args) -> None:
        """Mirror the tree's current row into the diagram and the inspector."""
        index = self._selected_index()
        self.stack.set_selected_index(index)
        component = None
        if index is not None and self._structure is not None:
            components = self._structure.components
            component = components[index] if index < len(components) else None
        self.component_selected.emit(index if component is not None else None, component)

    def _select_component(self, index: int) -> None:
        """Make the clicked band the tree's current row.

        The buttons act on the tree's current row, so a click that highlighted only
        the diagram would leave 删除 pointing at whatever was selected before.
        """
        for position in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(position)
            if item.data(0, Qt.ItemDataRole.UserRole) == index:
                self.tree.setCurrentItem(item)
                return

    def _refresh_action_state(self, *_args) -> None:
        index = self._selected_index()
        count = 0 if self._structure is None else len(self._structure.components)
        self.remove_action.setEnabled(index is not None)
        self.up_action.setEnabled(index is not None and index > 0)
        self.down_action.setEnabled(index is not None and index + 1 < count)

    def _remove_selected(self) -> None:
        index = self._selected_index()
        if index is not None:
            self.remove_component(index)

    def _move_selected(self, offset: int) -> None:
        index = self._selected_index()
        if index is not None:
            self.move_component(index, index + offset)

    def _choose_layer(self) -> None:
        LayerDialog(self, commit_layer=self.add_layer).exec()

    def _choose_periodic(self) -> None:
        structure = self._require_structure()
        PeriodicDialog(
            self,
            commit_block=self.add_periodic_block,
            master_seed=self._master_seed(),
            block_offset=len(structure.components),
        ).exec()

    def _master_seed(self) -> int:
        if self._master_seed_source is None:
            return 0
        return self._master_seed_source()

    def _choose_backing(self) -> None:
        structure = self._require_structure()
        BackingDialog(
            structure.backing,
            structure.backing_roughness_a,
            self,
            commit_backing=self.set_backing,
        ).exec()
