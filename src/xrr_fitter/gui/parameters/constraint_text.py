"""约束构建器与约束列表上屏的文字：运算符、参数名、表达式，以及拒绝原因的中文说法。

模型层与服务层的拒绝原文保持英文、带 ``数据集::参数`` 的内部键，便于日志与持久化排查；
这里只在 GUI 一侧把它们翻成读者的话，原文认不出来就照原样上屏——编一个原因比露出原文更糟。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from PySide6.QtWidgets import QTreeWidgetItem

import xrr_fitter.api as api

#: 参数引用 → 屏上那个名字。没有登记的引用退回 ``数据集:参数``，至少还认得出是哪一个。
Labels = Mapping[api.ParameterReference, str]

OPERATOR_SYMBOLS = {"add": "+", "sub": "−", "mul": "×", "div": "÷", "pow": "^"}
OPERATOR_TEXTS = {"add": "+ 加", "sub": "− 减", "mul": "× 乘", "div": "÷ 除", "pow": "^ 乘方"}
OPERATOR_TOOLTIP = "对栈顶两个操作数施加的二元运算：先压入的在左，后压入的在右（例如 a − b、a ÷ b、a ^ b）"

SHARED_PREFIX = "parameters are driven by both a sharing and a constraint rule: "
CYCLE_PREFIX = "constraint rules form a dependency cycle: "
EXACT_TEXTS = {
    "constraint targets must be unique": "约束目标重复：一个参数只能由一条约束驱动，请先删除已有的那条",
    "constraint rule references a missing dataset": "约束引用了不存在的数据集",
    "pow exponent must be a constant": "乘方的指数必须是常数：先压入底数，再压入一个常数作指数",
    "constraint expression nesting exceeds the maximum depth": "表达式嵌套太深（最多 64 层）",
}
PREFIX_TEMPLATES = (
    ("constraint references unknown parameter: ", "约束引用了不存在的参数：{}"),
    ("constraint integer target is unsupported: ", "{} 是整数参数，不能作为约束目标"),
    ("constraint independent variable must not be integer: ", "{} 是整数参数，不能作为约束的自变量"),
    ("constraint target must not also have a parameter prior: ", "{} 已设先验，不能同时由约束驱动；请先清除先验"),
)
SELF_REFERENCE = re.compile(r"constraint target (?P<target>\S+) must not appear in its own expression")
ROUGHNESS = re.compile(
    r"non-roughness constraint target (?P<target>\S+) cannot depend on roughness parameter (?P<source>\S+)"
)


def reference_label(reference: api.ParameterReference, labels: Labels | None = None) -> str:
    known = None if labels is None else labels.get(reference)
    return known or f"{reference.dataset_id}:{reference.parameter_name}"


def reference_key(reference: api.ParameterReference) -> str:
    """拒绝原文里引用的写法（``model.parameters._reference_key``）。"""
    return f"{reference.dataset_id}::{reference.parameter_name}"


def parameter_labels(project: api.XrrProject) -> dict[api.ParameterReference, str]:
    """每个数据集里每个参数的显示名；不止一个数据集时前面带上数据集名。

    读不出声明的数据集（源文件挪走、结构不成立）就不登记，它的引用退回原始标识：
    读失败本身由参数页报出来，这里只是名字说得没那么好听。
    """
    several = len(project.datasets) > 1
    labels: dict[api.ParameterReference, str] = {}
    for dataset in project.datasets:
        if dataset.structure is None:
            continue
        try:
            definitions = api.describe_parameters(project, dataset.dataset_id)
        except (OSError, ValueError):
            continue
        prefix = f"{dataset.display_name or dataset.dataset_id} · " if several else ""
        for definition in definitions:
            labels[api.ParameterReference(dataset.dataset_id, definition.name)] = prefix + definition.display_name
    return labels


def expression_text(node: api.ConstraintNode, labels: Labels | None = None) -> str:
    """把表达式读成一行中缀式，例如 ``a-Si 厚度 × 1.1``；嵌套的运算加括号。"""
    if node.op == "ref":
        return reference_label(node.reference, labels)
    if node.op == "const":
        return f"{node.value:g}"
    if node.op not in OPERATOR_SYMBOLS:
        # 一元函数（漂移生成的 sin/cos）
        return f"{node.op}({expression_text(node.operands[0], labels)})"
    left, right = (_operand_text(operand, labels) for operand in node.operands)
    return f"{left} {OPERATOR_SYMBOLS[node.op]} {right}"


def _operand_text(node: api.ConstraintNode, labels: Labels | None) -> str:
    text = expression_text(node, labels)
    return text if node.op in ("ref", "const") else f"({text})"


def expression_item(node: api.ConstraintNode, labels: Labels | None = None) -> QTreeWidgetItem:
    """表达式树的一行：类型 + 内容；运算行读整条子式，子行逐个列出操作数。"""
    if node.op == "ref":
        item = QTreeWidgetItem(("参数", reference_label(node.reference, labels)))
        item.setToolTip(1, reference_label(node.reference))
        return item
    if node.op == "const":
        return QTreeWidgetItem(("常数", expression_text(node)))
    item = QTreeWidgetItem(("运算", expression_text(node, labels)))
    for operand in node.operands:
        item.addChild(expression_item(operand, labels))
    return item


def constraint_error_text(error: BaseException, labels: Labels | None = None) -> str:
    """把一次约束拒绝说成中文；原文里的 ``数据集::参数`` 换成 ``labels`` 里的名字。"""
    message = str(error)
    names = {reference_key(reference): label for reference, label in (labels or {}).items()}
    for translate in TRANSLATORS:
        text = translate(message, names)
        if text is not None:
            return text
    return EXACT_TEXTS.get(message, message)


def _name(key: str, names: Mapping[str, str]) -> str:
    known = names.get(key.strip())
    return key.strip() if known is None else f"「{known}」"


def _shared(message: str, names: Mapping[str, str]) -> str | None:
    if not message.startswith(SHARED_PREFIX):
        return None
    keys = message[len(SHARED_PREFIX) :].split(",")
    return f"{'、'.join(_name(key, names) for key in keys)} 已在共享组里，不能再由约束驱动；请先在「共享」页移除它"


def _cycle(message: str, names: Mapping[str, str]) -> str | None:
    if not message.startswith(CYCLE_PREFIX):
        return None
    path = " → ".join(_name(key, names) for key in message[len(CYCLE_PREFIX) :].split("->"))
    return f"约束之间形成了循环依赖：{path}"


def _prefixed(message: str, names: Mapping[str, str]) -> str | None:
    for prefix, template in PREFIX_TEMPLATES:
        if message.startswith(prefix):
            return template.format(_name(message[len(prefix) :], names))
    return None


def _patterned(message: str, names: Mapping[str, str]) -> str | None:
    found = SELF_REFERENCE.fullmatch(message)
    if found is not None:
        return f"{_name(found['target'], names)} 不能出现在驱动它自己的表达式里"
    found = ROUGHNESS.fullmatch(message)
    if found is not None:
        # 粗糙度的上限取决于已解出的厚度，求值时排在最后一轮；别的量读不到它的值。
        target, source = _name(found["target"], names), _name(found["source"], names)
        return f"{target} 不是粗糙度，不能依赖粗糙度参数 {source}：粗糙度要等其他量解出后才确定"
    return None


TRANSLATORS: tuple[Callable[[str, Mapping[str, str]], str | None], ...] = (_shared, _cycle, _prefixed, _patterned)
