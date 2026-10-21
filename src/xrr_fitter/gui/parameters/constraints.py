"""API-backed expression constraint editor and its builder dialog."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api
from xrr_fitter.gui.accessibility import localize_standard_buttons
from xrr_fitter.gui.document import ProjectDocument
from xrr_fitter.gui.parameters.constraint_text import (
    OPERATOR_TEXTS,
    OPERATOR_TOOLTIP,
    Labels,
    constraint_error_text,
    expression_item,
    expression_text,
    parameter_labels,
    reference_label,
)

# The binary operators the builder can reduce with; leaf operands ("ref",
# "const") are pushed directly and are never selected as an operator.
BINARY_OPERATORS = ("add", "sub", "mul", "div", "pow")


def _command_button(text: str, name: str, tooltip: str, slot: Callable[[], None]) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName(name)
    button.setAccessibleName(text)
    button.setToolTip(tooltip)
    button.clicked.connect(slot)
    return button


def _reference_choices(combo: QComboBox, references: Sequence[api.ParameterReference], labels: Labels) -> None:
    for reference in references:
        combo.addItem(reference_label(reference, labels), reference)
        # 两个数据集里同名的层读起来一样，原始标识留在悬停提示里供分辨。
        combo.setItemData(combo.count() - 1, reference_label(reference), Qt.ItemDataRole.ToolTipRole)


class ConstraintEditor(QWidget):
    """Validate and commit complete constraint-rule sequences atomically."""

    constraints_changed = Signal(tuple)

    def __init__(self, document: ProjectDocument) -> None:
        super().__init__()
        self.document = document
        self.setObjectName("constraintEditor")
        self.tree = QTreeWidget()
        self.tree.setObjectName("constraintTree")
        self.tree.setAccessibleName("表达式约束列表")
        self.tree.setHeaderLabels(("目标参数", "表达式"))
        self.error_label = QLabel()
        self.error_label.setObjectName("constraintValidationError")
        self.error_label.setAccessibleName("约束校验错误")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.add_button = QPushButton("添加约束")
        self.add_button.setObjectName("constraintAddRuleButton")
        self.add_button.setAccessibleName("添加参数约束")
        self.add_button.setToolTip("创建一条新的表达式约束")
        self.add_button.clicked.connect(self._add_rule)
        self.delete_button = QPushButton("删除约束")
        self.delete_button.setObjectName("constraintDeleteRuleButton")
        self.delete_button.setAccessibleName("删除参数约束")
        self.delete_button.setToolTip("删除当前选中的表达式约束")
        self.delete_button.clicked.connect(self._delete_selected_rule)
        commands = QHBoxLayout()
        commands.addWidget(self.add_button)
        commands.addWidget(self.delete_button)
        commands.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(commands)
        layout.addWidget(self.tree)
        layout.addWidget(self.error_label)
        self._labels: tuple[object, dict[api.ParameterReference, str]] | None = None
        document.project_changed.connect(self._render)
        self._render()

    @property
    def rules(self) -> tuple[api.ConstraintRule, ...]:
        return self.document.project.constraint_rules

    def apply_rules(self, rules) -> bool:
        """Commit ``rules``; a refusal is said on this page in Chinese and re-raised as is."""
        try:
            return self._publish(tuple(rules))
        except (TypeError, ValueError) as error:
            self.error_label.setText(constraint_error_text(error, self.labels()))
            self.error_label.show()
            raise

    def _publish(self, values: tuple[api.ConstraintRule, ...]) -> bool:
        current = self.document.project
        validated = api.validate_constraint_rules(current, values)
        updated = api.set_constraint_rules(current, validated)
        self.error_label.hide()
        if updated is current:
            return False
        self.document.replace_project(updated)
        self.constraints_changed.emit(updated.constraint_rules)
        return True

    def labels(self) -> dict[api.ParameterReference, str]:
        """每个参数引用的显示名，按决定显示名的那几项缓存。

        ``describe_parameters`` 要读源文件，而列表在每次 ``project_changed`` 都重画；显示名
        只随结构、仪器、拟合配置与数据集名变，改一个初值不该再把每个数据集读一遍。
        """
        project = self.document.project
        key = (
            project.fit_config,
            tuple(
                (dataset.dataset_id, dataset.display_name, dataset.source_sha256, dataset.structure, dataset.instrument)
                for dataset in project.datasets
            ),
        )
        if self._labels is None or self._labels[0] != key:
            self._labels = (key, parameter_labels(project))
        return self._labels[1]

    def remove_rule(self, target: api.ParameterReference) -> bool:
        retained = tuple(rule for rule in self.rules if rule.target != target)
        if len(retained) == len(self.rules):
            return False
        return self.apply_rules(retained)

    def eligible_targets(self, dataset_id: str) -> tuple[api.ParameterReference, ...]:
        """References a new rule may drive: continuous, free, and unclaimed.

        Mirrors ``SharingEditor.eligible_names`` (declared and not locked) and
        additionally drops integer parameters and parameters some rule already
        drives. Discrete topology cannot carry an analytic constraint tangent.
        """
        driven = {rule.target.parameter_name for rule in self.rules if rule.target.dataset_id == dataset_id}
        return tuple(
            api.ParameterReference(dataset_id, definition.name)
            for definition in api.describe_parameters(self.document.project, dataset_id)
            if not definition.locked
            and not definition.integer
            and not definition.constrained
            and definition.name not in driven
        )

    def eligible_sources(self, dataset_id: str) -> tuple[api.ParameterReference, ...]:
        """References usable as independent variables: any non-integer parameter.

        Integer parameters are excluded because the stage-split validator
        forbids them from acting as a constraint's independent variable.
        """
        return tuple(
            api.ParameterReference(dataset_id, definition.name)
            for definition in api.describe_parameters(self.document.project, dataset_id)
            if not definition.integer
        )

    def error_text(self) -> str:
        return self.error_label.text()

    def _add_rule(self) -> None:
        targets = tuple(
            reference
            for dataset in self.document.project.datasets
            for reference in self.eligible_targets(dataset.dataset_id)
        )
        sources = tuple(
            reference
            for dataset in self.document.project.datasets
            for reference in self.eligible_sources(dataset.dataset_id)
        )
        if not targets or not sources:
            self.error_label.setText("没有可用的目标参数或自变量参数")
            self.error_label.show()
            return
        committed = False

        def commit_rule(rule: api.ConstraintRule) -> None:
            nonlocal committed
            # 拒绝由对话框自己说：写进这一页的话，读者取消对话框之后页面上还挂着那句话。
            self._publish((*self.rules, rule))
            committed = True

        dialog = ConstraintDialog(
            targets,
            sources,
            self,
            commit_rule=commit_rule,
            labels=self.labels(),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if committed:
            return
        try:
            self.apply_rules((*self.rules, dialog.rule()))
        except (TypeError, ValueError):
            return

    def _delete_selected_rule(self) -> None:
        selected = self.tree.currentItem()
        if selected is None:
            self.error_label.setText("请先选择要删除的约束")
            self.error_label.show()
            return
        while selected.parent() is not None:
            selected = selected.parent()
        target = selected.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(target, api.ParameterReference):
            try:
                self.remove_rule(target)
            except (TypeError, ValueError):
                return

    def _render(self, *_args) -> None:
        self.tree.clear()
        if not self.rules:
            return
        labels = self.labels()
        for rule in self.rules:
            parent = QTreeWidgetItem((reference_label(rule.target, labels), expression_text(rule.expression, labels)))
            parent.setData(0, Qt.ItemDataRole.UserRole, rule.target)
            # 原始标识留在悬停提示里：两个数据集里同名的层读起来一样。
            parent.setToolTip(0, reference_label(rule.target))
            parent.setToolTip(1, expression_text(rule.expression))
            for operand in rule.expression.operands:
                parent.addChild(expression_item(operand, labels))
            self.tree.addTopLevelItem(parent)
        self.tree.expandAll()


class ConstraintDialog(QDialog):
    """Build one constraint rule incrementally and commit it atomically."""

    def __init__(
        self,
        targets: Sequence[api.ParameterReference],
        sources: Sequence[api.ParameterReference],
        parent: QWidget | None = None,
        *,
        commit_rule: Callable[[api.ConstraintRule], None] | None = None,
        labels: Labels | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("constraintDialog")
        self.setWindowTitle("添加参数约束")
        self.setAccessibleName("添加参数约束")
        self.setModal(True)
        self._commit_rule = commit_rule
        self._labels: Labels = {} if labels is None else labels
        self._rule: api.ConstraintRule | None = None
        self._stack: list[api.ConstraintNode] = []
        # 每一步之前的整栈快照：撤销一次运算要把它吃掉的两个操作数原样吐回来。
        self._history: list[tuple[api.ConstraintNode, ...]] = []
        self._build_inputs(targets, sources)
        self.add_constant_button = _command_button(
            "添加常数", "constraintAddConstantButton", "将当前常数压入表达式栈", self._push_constant
        )
        self.add_parameter_button = _command_button(
            "添加参数", "constraintAddParameterButton", "将选中的自变量参数压入表达式栈", self._push_parameter
        )
        self.add_operator_button = _command_button(
            "添加运算符", "constraintAddOperatorButton", "弹出栈顶两个操作数并压入其运算结果", self._push_operator
        )
        self.undo_button = _command_button("撤销一步", "constraintUndoButton", "撤回上一次压栈或运算", self._undo)
        self.clear_button = _command_button("清空", "constraintClearButton", "清空表达式栈（可撤销）", self._clear)
        self.expression_tree = QTreeWidget()
        self.expression_tree.setObjectName("constraintExpressionTree")
        self.expression_tree.setAccessibleName("表达式栈")
        self.expression_tree.setToolTip("当前构建中的表达式栈")
        self.expression_tree.setHeaderLabels(("类型", "内容"))
        self.error_label = QLabel()
        self.error_label.setObjectName("constraintDialogError")
        self.error_label.setAccessibleName("约束构建错误")
        self.error_label.setToolTip("表达式构建或校验失败时显示的原因")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.setObjectName("constraintDialogButtons")
        localize_standard_buttons(self.buttons, confirm="确定")
        self.buttons.accepted.connect(self._accept_fields)
        self.buttons.rejected.connect(self.reject)
        self._arrange()
        self._render_stack()

    def _build_inputs(
        self,
        targets: Sequence[api.ParameterReference],
        sources: Sequence[api.ParameterReference],
    ) -> None:
        self.target_input = QComboBox()
        self.target_input.setObjectName("constraintTargetInput")
        self.target_input.setAccessibleName("目标参数")
        self.target_input.setToolTip("约束驱动的目标参数")
        _reference_choices(self.target_input, targets, self._labels)
        self.source_input = QComboBox()
        self.source_input.setObjectName("constraintSourceInput")
        self.source_input.setAccessibleName("自变量参数")
        self.source_input.setToolTip("作为表达式操作数的自变量参数")
        _reference_choices(self.source_input, sources, self._labels)
        self.operator_input = QComboBox()
        self.operator_input.setObjectName("constraintOperatorInput")
        self.operator_input.setAccessibleName("运算符")
        self.operator_input.setToolTip(OPERATOR_TOOLTIP)
        for operator in BINARY_OPERATORS:
            self.operator_input.addItem(OPERATOR_TEXTS[operator], operator)
        self.constant_input = QDoubleSpinBox()
        self.constant_input.setObjectName("constraintConstantInput")
        self.constant_input.setAccessibleName("常数值")
        # 常数按内部单位参与运算：厚度与粗糙度是 Å，相对密度是比值。
        self.constant_input.setToolTip("要压入表达式栈的常数；厚度、粗糙度按 Å 计（1 nm = 10 Å），相对密度是比值")
        self.constant_input.setDecimals(8)
        self.constant_input.setRange(-1e6, 1e6)

    def _arrange(self) -> None:
        form = QFormLayout()
        form.addRow("目标参数", self.target_input)
        form.addRow("自变量参数", self.source_input)
        form.addRow("常数值", self.constant_input)
        form.addRow("运算符", self.operator_input)
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.add_constant_button)
        buttons_row.addWidget(self.add_parameter_button)
        buttons_row.addWidget(self.add_operator_button)
        buttons_row.addStretch(1)
        buttons_row.addWidget(self.undo_button)
        buttons_row.addWidget(self.clear_button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons_row)
        layout.addWidget(self.expression_tree)
        layout.addWidget(self.error_label)
        layout.addWidget(self.buttons)

    def _push_constant(self) -> None:
        try:
            node = api.ConstraintNode("const", value=self.constant_input.value())
        except (TypeError, ValueError) as error:
            self._show_error(error)
            return
        self._replace_top(0, node)

    def _push_parameter(self) -> None:
        reference = self.source_input.currentData()
        if reference is None:
            self._show_error(ValueError("没有可用的自变量参数"))
            return
        self._replace_top(0, api.ConstraintNode("ref", reference=reference))

    def _push_operator(self) -> None:
        if len(self._stack) < 2:
            self._show_error(ValueError("运算符需要两个操作数"))
            return
        operator = self.operator_input.currentData()
        left, right = self._stack[-2], self._stack[-1]
        try:
            node = api.ConstraintNode(operator, operands=(left, right))
        except (TypeError, ValueError) as error:
            self._show_error(error)
            return
        self._replace_top(2, node)

    def _replace_top(self, count: int, node: api.ConstraintNode) -> None:
        """栈顶 ``count`` 个节点换成 ``node``，换之前记一份快照供撤销。"""
        self._history.append(tuple(self._stack))
        self._stack[len(self._stack) - count :] = [node]
        self.error_label.hide()
        self._render_stack()

    def _undo(self) -> None:
        if not self._history:
            return
        self._stack = list(self._history.pop())
        self.error_label.hide()
        self._render_stack()

    def _clear(self) -> None:
        if not self._stack:
            return
        self._history.append(tuple(self._stack))
        self._stack = []
        self.error_label.hide()
        self._render_stack()

    def _build_rule(self) -> api.ConstraintRule:
        if not self._stack:
            raise ValueError("表达式不能为空")
        if len(self._stack) != 1:
            raise ValueError("表达式尚未归约为单个节点")
        target = self.target_input.currentData()
        if target is None:
            raise ValueError("必须选择目标参数")
        return api.ConstraintRule(target, self._stack[0])

    def _accept_fields(self) -> None:
        try:
            candidate = self._build_rule()
            if self._commit_rule is not None:
                self._commit_rule(candidate)
        except (TypeError, ValueError) as error:
            self._show_error(error)
            return
        self._rule = candidate
        self.error_label.hide()
        self.accept()

    def _show_error(self, error: Exception) -> None:
        self.error_label.setText(constraint_error_text(error, self._labels))
        self.error_label.show()

    def _render_stack(self) -> None:
        self.expression_tree.clear()
        for node in self._stack:
            self.expression_tree.addTopLevelItem(expression_item(node, self._labels))
        self.expression_tree.expandAll()
        self.undo_button.setEnabled(bool(self._history))
        self.clear_button.setEnabled(bool(self._stack))

    def rule(self) -> api.ConstraintRule:
        if self._rule is None:
            raise LookupError("constraint has not been confirmed")
        return self._rule
