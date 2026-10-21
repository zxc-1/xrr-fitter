"""Active-dataset parameter, sharing, and expert-mode coordinator."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QLabel,
    QMenu,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui import theme
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.parameters.constraints import ConstraintEditor
from xrr_fitter.gui.parameters.disposition import ParameterDisposition
from xrr_fitter.gui.parameters.grouping import bulk_densities, component_captions, group_key
from xrr_fitter.gui.parameters.sharing import SharingEditor
from xrr_fitter.gui.parameters.table import (
    FREEDOM_BY_CHECK_STATE,
    FREEDOM_CHECK_STATES,
    GROUP_KEY_ROLE,
    VALUE_COLUMNS,
    ParameterTable,
)

# Wash of the error color behind a cell whose entered bound is self-inconsistent.
INVALID_CELL_BRUSH = QBrush(QColor(179, 38, 30, 48))
CONSTRAINT_CONTEXT_TOOLTIP = "该参数由表达式约束驱动，请先删除约束再修改"


FIX_GROUP_TEXT = "固定本组全部参数"
FREE_GROUP_TEXT = "放开本组全部参数"


def _group_setting(
    definition: api.ParameterDefinition,
    setting: api.ParameterSetting | None,
    freedom: api.ParameterFreedom,
) -> api.ParameterSetting | None:
    """整组改档时这一个量的新 setting；不该动或已经在这一档时是 ``None``。"""
    if setting is None:
        # 没有 setting 的量按声明走：声明锁住的不归「整组放开」管，声明自由的才固定它。
        if definition.locked or freedom is not api.ParameterFreedom.FIXED:
            return None
        return api.ParameterSetting(definition.name, definition.initial, definition.lower, definition.upper, freedom)
    if setting.freedom is freedom:
        return None
    return replace(setting, freedom=freedom)


def bounds_problem(initial: float, lower: float, upper: float) -> str | None:
    """Name the first self-inconsistency in an entered bound triple.

    This is a display-space sanity check that runs the instant a cell is
    edited, so the user sees why a value is rejected without waiting for the
    fit preflight. The public API stays the sole authority on whether a
    consistent triple is scientifically admissible; this only catches the
    ordering mistakes that are wrong on their face.
    """
    if lower > upper:
        return "下限不能大于上限"
    if initial < lower:
        return "初值不能小于下限"
    if initial > upper:
        return "初值不能大于上限"
    return None


class ParametersPanel(QWidget):
    """Project parameters whose only mutation boundary is the public API."""

    settings_changed = Signal(str, tuple)
    sharing_changed = Signal(tuple)
    constraints_changed = Signal(tuple)
    expert_mode_changed = Signal(bool)
    # 这一屏刚编译出来的参数声明。左栏那行「N 共享 + M 独立」要数自由参数，而
    # ``describe_parameters`` 会读源文件——挂在 project_changed 上等于把这里做过的读取再做
    # 一遍。所以由算过的这一方报出来，别处只订阅。
    definitions_changed = Signal(tuple)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("parametersPanel")
        self._definitions: tuple[api.ParameterDefinition, ...] = ()
        # Named for what it uncovers, not for who is presumed to be looking: the
        # command bar's 引导↔专家 segment already owns 专家, and it switches which
        # surface is on screen rather than how deep this one goes.  Both carrying the
        # one word meant the guided surface's 「切换到专家模式」 hint landed on a
        # workspace where a control of the same name was still unchecked.
        # 它还决定一键拟合走哪条路线（``MainWindow._uses_automatic_fit``），提示里必须说出来。
        # 「关闭」只对按测量预设导入、逐个拟合的项目意味着自动路线，联合拟合与手动导入不受它影响。
        self.expert_toggle = QCheckBox("高级模式")
        self.expert_toggle.setObjectName("expertModeToggle")
        self.expert_toggle.setAccessibleName("切换高级模式")
        self.expert_toggle.setToolTip(
            "开启后显示高级参数、诊断视图与 MCMC 采样，并且「一键拟合」改用手动拟合路线；"
            "关闭时，按测量预设导入、逐个拟合的项目使用自动拟合路线。两条路线的分析步骤与耗时不同。"
            "与顶栏的引导↔专家不是同一个开关。"
        )
        self.parameter_table = ParameterTable()
        self.disposition = ParameterDisposition()
        self.disposition.freedom_requested.connect(self._set_freedom)
        self.sharing_editor = SharingEditor(document)
        self.sharing_editor.rules_changed.connect(self.sharing_changed.emit)
        self.constraint_editor = ConstraintEditor(document)
        self.constraint_editor.constraints_changed.connect(self.constraints_changed.emit)
        self.status_label = QLabel()
        self.status_label.setObjectName("parameterStatus")
        self.status_label.setProperty("mutedText", True)
        self.status_label.setWordWrap(True)
        tabs = QTabWidget()
        tabs.setObjectName("parameterTabs")
        tabs.addTab(self.parameter_table, "参数")
        tabs.addTab(self.sharing_editor, "共享")
        tabs.addTab(self.constraint_editor, "约束")
        self.tabs = tabs
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(self.expert_toggle)
        # ``disposition`` 不在这个面板的布局里：设计稿帧③ 把三档与两枚徽标画在右栏「参数化」
        # 那一段（HTML 711-720），而这张表在画布列的「参数总览」卡里。它仍归这个面板所有——
        # 接线、锁定回写都在这里——只是由 ``window_layout`` 挂到右栏那一段去。
        layout.addWidget(tabs)
        layout.addWidget(self.status_label)
        self.expert_toggle.toggled.connect(self._toggle_expert_mode)
        self.parameter_table.itemChanged.connect(self._table_setting_changed)
        # 抬头和这一段跟同一行走。两个发信方都必须是表自己：挂到 ``table.model()`` 上的
        # 连接会在表的 Python 包装失效之后再响一次，抛 shiboken 的「对象已删除」，而那个
        # 异常落在 Qt 事件循环里会被 pytest-qt 记到*下一条*用例头上。
        self.parameter_table.currentCellChanged.connect(self._follow_current_row)
        # 换数据集、换结构都会重填整张表，选中行随之作废——那次作废发生在
        # ``QSignalBlocker`` 里，``currentCellChanged`` 一声不响，所以由表自己补一声。
        self.parameter_table.rows_reloaded.connect(self._follow_current_row)
        self.parameter_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.parameter_table.customContextMenuRequested.connect(self._show_row_context_menu)
        document.project_changed.connect(self._refresh)
        self._refresh()

    @property
    def definitions(self) -> tuple[api.ParameterDefinition, ...]:
        return self._definitions

    @property
    def visible_definitions(self) -> tuple[api.ParameterDefinition, ...]:
        return self.parameter_table.definitions

    @property
    def row_names(self) -> tuple[str, ...]:
        return self.parameter_table.row_names

    @property
    def expert_mode(self) -> bool:
        return self.document.project.ui_state.expert_mode

    @property
    def sharing_rules(self) -> tuple[api.SharingRule, ...]:
        return self.sharing_editor.rules

    @property
    def constraint_rules(self) -> tuple[api.ConstraintRule, ...]:
        return self.constraint_editor.rules

    def display_values(self, name: str) -> tuple[float, float, float]:
        return self.parameter_table.display_values(name)

    def display_unit(self, name: str) -> str:
        return self.parameter_table.display_unit(name)

    def set_parameter(
        self,
        name: str,
        *,
        initial: float,
        lower: float,
        upper: float,
        freedom: api.ParameterFreedom,
    ) -> bool:
        self._definition(name)
        setting = api.ParameterSetting(name, initial, lower, upper, freedom)
        dataset_id = self._require_active_dataset_id()
        dataset = self._dataset(dataset_id)
        settings = self._with_setting(dataset.parameter_settings, setting)
        current = self.document.project
        updated = api.set_parameter_settings(current, dataset_id, settings)
        if updated is current:
            return False
        self.document.replace_project(updated)
        persisted = self._dataset(dataset_id).parameter_settings
        self.settings_changed.emit(dataset_id, persisted)
        return True

    def set_display_parameter(
        self,
        name: str,
        *,
        initial: float,
        lower: float,
        upper: float,
        freedom: api.ParameterFreedom,
    ) -> bool:
        values = self.parameter_table.to_persisted_values(
            name,
            initial,
            lower,
            upper,
        )
        return self.set_parameter(
            name,
            initial=values[0],
            lower=values[1],
            upper=values[2],
            freedom=freedom,
        )

    def set_expert_mode(self, enabled: bool) -> bool:
        current = self.document.project
        updated = api.set_expert_mode(current, enabled)
        if updated is current:
            return False
        self.document.replace_project(updated)
        self.expert_mode_changed.emit(enabled)
        return True

    def reset_parameter(self, name: str) -> bool:
        """Drop a parameter's persisted setting so it falls back to its default.

        The declared default lives in ``describe_parameters`` and is derived
        from the structure; removing the user's override lets that default
        reassert itself without the user having to remember the original number.
        """
        self._definition(name)
        dataset_id = self._require_active_dataset_id()
        dataset = self._dataset(dataset_id)
        retained = tuple(value for value in dataset.parameter_settings if value.name != name)
        if len(retained) == len(dataset.parameter_settings):
            return False
        current = self.document.project
        updated = api.set_parameter_settings(current, dataset_id, retained)
        if updated is current:
            return False
        self.document.replace_project(updated)
        self.settings_changed.emit(
            dataset_id,
            self._dataset(dataset_id).parameter_settings,
        )
        return True

    def apply_sharing_rules(self, rules) -> bool:
        return self.sharing_editor.apply_rules(rules)

    def remove_sharing_rule(self, sharing_key: str) -> bool:
        return self.sharing_editor.remove_rule(sharing_key)

    def eligible_sharing_names(self, dataset_ids) -> tuple[str, ...]:
        return self.sharing_editor.eligible_names(dataset_ids)

    def sharing_error_text(self) -> str:
        return self.sharing_editor.error_text()

    def apply_constraint_rules(self, rules) -> bool:
        return self.constraint_editor.apply_rules(rules)

    def remove_constraint_rule(self, target: api.ParameterReference) -> bool:
        return self.constraint_editor.remove_rule(target)

    def eligible_constraint_targets(self, dataset_id: str) -> tuple[api.ParameterReference, ...]:
        return self.constraint_editor.eligible_targets(dataset_id)

    def eligible_constraint_sources(self, dataset_id: str) -> tuple[api.ParameterReference, ...]:
        return self.constraint_editor.eligible_sources(dataset_id)

    def constraint_error_text(self) -> str:
        return self.constraint_editor.error_text()

    def _toggle_expert_mode(self, enabled: bool) -> None:
        self.set_expert_mode(enabled)

    def _follow_current_row(self, *_args) -> None:
        """把「参数化」那一段切到表里当前那一行上。"""
        name = self.parameter_table.current_name()
        definition = None if name is None else self._visible_definition(name)
        self.disposition.show_definition(
            definition,
            quantity=self.parameter_table.current_quantity(),
            # 档位读表的那份映射，不重新去 project 里翻 setting：表刚才就是按它画的，另找一
            # 条来源等于给同一件事开第二个真相。
            freedom=None if name is None else self.parameter_table.freedom_of(name),
            sharing_rules=self.sharing_rules,
            dataset_id=self.document.active_dataset_id,
        )

    def _visible_definition(self, name: str) -> api.ParameterDefinition | None:
        matches = tuple(item for item in self.visible_definitions if item.name == name)
        return matches[0] if matches else None

    def _set_freedom(self, name: str, freedom: api.ParameterFreedom) -> None:
        """走名字格那个勾原本那条提交路径，而不是另起一条。

        另起一条的话，同一件事有两个入口各提交各的：改完档位之后表里的勾会和这一段的档位
        对不上，而两处都没标注自己读的是哪一边。
        """
        row = self._row_of(name)
        if row is None:
            return
        item = self.parameter_table.item(row, 0)
        if item is None:
            return
        item.setCheckState(FREEDOM_CHECK_STATES[freedom])

    def _row_of(self, name: str) -> int | None:
        for row in range(self.parameter_table.rowCount()):
            item = self.parameter_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == name:
                return row
        return None

    def _show_row_context_menu(self, position: object) -> None:
        item = self.parameter_table.itemAt(position)
        if item is None:
            return
        name_item = self.parameter_table.item(item.row(), 0)
        if name_item is None:
            return
        name = name_item.data(Qt.ItemDataRole.UserRole)
        # 分组标题行不带参数名，只有整组那两条；以前它也走参数菜单，名字读成 "None" 当场抛错。
        key = name_item.data(GROUP_KEY_ROLE) if name is None else group_key(self._definition(str(name)))
        menu = self._group_context_menu(key) if name is None else self._row_context_menu(str(name))
        chosen = menu.exec(self.parameter_table.viewport().mapToGlobal(position))
        handler = (
            None
            if chosen is None
            else self._menu_handlers(key, None if name is None else str(name)).get(chosen.objectName())
        )
        if handler is not None:
            handler()

    def _menu_handlers(self, key: str, name: str | None) -> dict[str, object]:
        handlers = {
            "fixGroupAction": lambda: self.set_group_freedom(key, api.ParameterFreedom.FIXED),
            "freeGroupAction": lambda: self.set_group_freedom(key, api.ParameterFreedom.FREE),
        }
        if name is not None:
            handlers.update(
                {
                    "resetParameterAction": lambda: self._reset_parameter_row(name),
                    "editPriorAction": lambda: self._edit_prior_row(name),
                    "clearPriorAction": lambda: self._clear_prior_row(name),
                }
            )
        return handlers

    def _row_context_menu(self, name: str) -> QMenu:
        definition = self._definition(name)
        menu = QMenu(self.parameter_table)
        reset = menu.addAction("恢复默认值")
        reset.setObjectName("resetParameterAction")
        edit_prior = menu.addAction("编辑先验")
        edit_prior.setObjectName("editPriorAction")
        clear_prior = menu.addAction("清除先验")
        clear_prior.setObjectName("clearPriorAction")
        if definition.constrained:
            for action in (reset, edit_prior, clear_prior):
                action.setEnabled(False)
                action.setToolTip(CONSTRAINT_CONTEXT_TOOLTIP)
        menu.addSeparator()
        self._add_group_actions(menu)
        return menu

    def _group_context_menu(self, key: str) -> QMenu:
        menu = QMenu(self.parameter_table)
        self._add_group_actions(menu)
        return menu

    def _add_group_actions(self, menu: QMenu) -> None:
        fix = menu.addAction(FIX_GROUP_TEXT)
        fix.setObjectName("fixGroupAction")
        fix.setToolTip("把这一组里此刻在拟合的量都钉在当前初值上；被约束驱动的量不受影响")
        free = menu.addAction(FREE_GROUP_TEXT)
        free.setObjectName("freeGroupAction")
        free.setToolTip("撤掉这一组里你设过的「固定」；声明本身锁定的量（例如未启用的背景项）仍保持锁定")

    def set_group_freedom(self, key: str, freedom: api.ParameterFreedom) -> bool:
        """一次把 ``key`` 那一组（一层、基底或仪器）的量都设成 ``freedom``。

        只动用户能动的那些：被约束驱动的不碰；「放开」只撤掉 setting 里的固定，声明自己锁住
        的量（未启用的背景、拓扑整数）不会因为整组放开就被解锁——那会让整批设定被拒。
        """
        dataset_id = self._require_active_dataset_id()
        settings = {setting.name: setting for setting in self._dataset(dataset_id).parameter_settings}
        changed = self._group_changes(key, settings, freedom)
        if not changed:
            self.status_label.setText("这一组没有可以改档的参数")
            return False
        merged = dict(settings)
        merged.update((setting.name, setting) for setting in changed)
        current = self.document.project
        updated = api.set_parameter_settings(current, dataset_id, tuple(merged.values()))
        if updated is current:
            return False
        self.document.replace_project(updated)
        verb = "固定" if freedom is api.ParameterFreedom.FIXED else "放开"
        self.status_label.setText(f"已{verb}这一组的 {len(changed)} 个参数")
        self.settings_changed.emit(dataset_id, self._dataset(dataset_id).parameter_settings)
        return True

    def _group_changes(
        self,
        key: str,
        settings: dict[str, api.ParameterSetting],
        freedom: api.ParameterFreedom,
    ) -> list[api.ParameterSetting]:
        proposed = (
            _group_setting(definition, settings.get(definition.name), freedom)
            for definition in self.visible_definitions
            if group_key(definition) == key and not definition.constrained
        )
        return [setting for setting in proposed if setting is not None]

    def focus_component(self, index: int | None) -> None:
        """把参数表滚到第 ``index`` 个组件那一组，并停在组里第一个参数上。

        给结构编辑器选层时调：读者在层堆叠里点了 a-Si，要调的是 a-Si 的参数，不该再去十几
        行里找。表在别的标签页时先切回来，否则滚动发生在看不见的地方。
        """
        if index is None:
            return
        table = self.parameter_table
        rows = table.group_rows(f"component.{index}")
        if not rows:
            return
        self.tabs.setCurrentWidget(table)
        parameter = next((row for row in rows if table.row_names[row]), rows[0])
        table.setCurrentCell(parameter, 0)
        table.scrollToItem(table.item(rows[0], 0), QAbstractItemView.ScrollHint.PositionAtTop)

    def _reset_parameter_row(self, name: str) -> None:
        try:
            if self.reset_parameter(name):
                self.status_label.setText(f"{name} 已恢复默认值")
        except (KeyError, ValueError) as error:
            self._refresh()
            self.status_label.setText(str(error))

    def _edit_prior_row(self, name: str) -> None:
        from xrr_fitter.gui.parameters.dialogs import PriorDialog

        definition = self._definition(name)
        dialog = PriorDialog(
            definition,
            self,
            existing_prior=definition.prior,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        spec = dialog.spec()
        if spec is None:
            return
        self._commit_prior(name, self._with_prior(name, spec))

    def _clear_prior_row(self, name: str) -> None:
        self._commit_prior(name, self._without_prior(name))

    def _commit_prior(self, name: str, priors: tuple[api.ParameterPrior, ...]) -> None:
        try:
            dataset_id = self._require_active_dataset_id()
            current = self.document.project
            updated = api.set_parameter_priors(current, dataset_id, priors)
            if updated is current:
                return
            self.document.replace_project(updated)
            self.status_label.setText(f"{name} 先验已更新")
        except (KeyError, ValueError) as error:
            self._refresh()
            self.status_label.setText(str(error))

    def _with_prior(self, name: str, spec: api.PriorSpec) -> tuple[api.ParameterPrior, ...]:
        retained = self._without_prior(name)
        return (*retained, api.ParameterPrior(name, spec))

    def _without_prior(self, name: str) -> tuple[api.ParameterPrior, ...]:
        dataset_id = self._require_active_dataset_id()
        priors = self._dataset(dataset_id).parameter_priors
        return tuple(value for value in priors if value.name != name)

    def _table_setting_changed(self, item: object) -> None:
        # Column 0 carries the freedom gear as its check state, so a toggle there is a
        # setting change exactly like an edited bound.
        if item.column() not in (0, 1, 2, 3):
            return
        row = item.row()
        try:
            name, values, freedom = self._read_row(row)
        except (KeyError, ValueError) as error:
            self._refresh()
            self.status_label.setText(str(error))
            return
        problem = bounds_problem(*values)
        if problem is not None:
            # Keep the user's entry on screen and point at the offending bound
            # instead of silently reverting the whole row on a fixable typo.
            self._mark_row_invalid(row, problem)
            return
        # Boundary is consistent; submit normally. The resulting project_changed
        # signal triggers _refresh() which rebuilds the table, clearing any stale
        # red highlights from earlier edits automatically.
        try:
            self.set_display_parameter(
                name,
                initial=values[0],
                lower=values[1],
                upper=values[2],
                freedom=freedom,
            )
        except (KeyError, ValueError) as error:
            self._refresh()
            self.status_label.setText(str(error))

    def _read_row(self, row: int) -> tuple[str, tuple[float, float, float], api.ParameterFreedom]:
        # The name cell holds both the identity and the freedom gear: it is the only
        # cell a caption row populates, so requiring the numeric cells is still what
        # keeps a caption from being read as a parameter.
        name_item = self.parameter_table.item(row, 0)
        value_items = tuple(self.parameter_table.item(row, column) for column in VALUE_COLUMNS)
        if name_item is None or any(value is None for value in value_items):
            raise ValueError("parameter row is incomplete")
        name = str(name_item.data(Qt.ItemDataRole.UserRole))
        initial, lower, upper = (self.parameter_table.entered_value(value) for value in value_items)
        return name, (initial, lower, upper), FREEDOM_BY_CHECK_STATE[name_item.checkState()]

    def _mark_row_invalid(self, row: int, problem: str) -> None:
        for column in VALUE_COLUMNS:
            cell = self.parameter_table.item(row, column)
            if cell is not None:
                cell.setBackground(INVALID_CELL_BRUSH)
                cell.setToolTip(problem)
        self.status_label.setText(problem)

    def _refresh(self, *_args) -> None:
        blocker = QSignalBlocker(self.expert_toggle)
        self.expert_toggle.setChecked(self.expert_mode)
        del blocker
        dataset_id = self.document.active_dataset_id
        if dataset_id is None or self._dataset(dataset_id).structure is None:
            self._clear_projection("当前数据集尚未定义结构")
            return
        try:
            self._definitions = api.describe_parameters(
                self.document.project,
                dataset_id,
            )
        except (OSError, ValueError) as error:
            self._clear_projection(str(error))
            return
        self.parameter_table.load(
            self._definitions,
            expert_mode=self.expert_mode,
            captions=self._component_captions(dataset_id),
            freedom=self._freedom_map(dataset_id),
            intervals=self._range_intervals(dataset_id),
            densities=bulk_densities(self._dataset(dataset_id).structure),
        )
        # The row count is legible from the table itself, so the status line
        # stays empty here and is reserved for validation problems and the
        # outcome of a reset.
        self.status_label.clear()
        self.definitions_changed.emit(self._definitions)

    def _freedom_map(self, dataset_id: str) -> dict[str, api.ParameterFreedom]:
        """已持久化 setting 的那些参数当前各在哪一档。

        声明上只有 ``locked``，两态装不下「仅范围」；档位只存在 setting 里，所以要有
        setting 的按 setting 报，没 setting 的不进这份映射，交给表按声明回落。
        """
        return {setting.name: setting.freedom for setting in self._dataset(dataset_id).parameter_settings}

    def _range_intervals(self, dataset_id: str) -> dict[str, tuple[float, float]]:
        """「仅范围」各参数自己的区间（存储单位）。

        这一档编译出来的声明保留的是硬边界，区间只在 setting 里；表照声明画会把硬边界当成
        区间显示，切档时整行读回就把它写进 setting。
        """
        return {
            setting.name: (setting.lower, setting.upper)
            for setting in self._dataset(dataset_id).parameter_settings
            if setting.freedom is api.ParameterFreedom.RANGE_ONLY
        }

    def _component_captions(self, dataset_id: str) -> dict[str, str]:
        return component_captions(self._dataset(dataset_id).structure)

    def _clear_projection(self, message: str) -> None:
        self._definitions = ()
        self.parameter_table.clear_parameters()
        self.status_label.setText(message)
        self.definitions_changed.emit(self._definitions)

    def _definition(self, name: str) -> api.ParameterDefinition:
        matches = tuple(value for value in self._definitions if value.name == name)
        if len(matches) != 1:
            raise KeyError(f"unknown parameter: {name}")
        return matches[0]

    def _require_active_dataset_id(self) -> str:
        dataset_id = self.document.active_dataset_id
        if dataset_id is None:
            raise ValueError("an active dataset is required")
        return dataset_id

    def _dataset(self, dataset_id: str) -> api.DatasetProject:
        matches = tuple(dataset for dataset in self.document.project.datasets if dataset.dataset_id == dataset_id)
        if len(matches) != 1:
            raise KeyError(f"unknown dataset: {dataset_id}")
        return matches[0]

    def _with_setting(
        self,
        existing: tuple[api.ParameterSetting, ...],
        setting: api.ParameterSetting,
    ) -> tuple[api.ParameterSetting, ...]:
        retained = tuple(value for value in existing if value.name != setting.name)
        position = next(
            (index for index, value in enumerate(existing) if value.name == setting.name),
            len(retained),
        )
        values = list(retained)
        values.insert(position, setting)
        return tuple(values)
