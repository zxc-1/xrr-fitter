"""Chinese presentation texts for public API status and error values.

The public API deliberately reports machine-oriented English messages.  This
module owns the user-facing Chinese projection: known messages translate to
guidance text, unknown messages pass through verbatim so tests and expert
diagnostics never lose information.
"""

from __future__ import annotations

import re


READY_TEXT = "已就绪，可以开始拟合"

READINESS_TEXTS = {
    "ready": READY_TEXT,
    "project has no datasets": "项目中还没有数据集，请先导入 XRR 数据",
    "source validation failed": "数据源校验失败，请检查源文件后重试",
}

STRUCTURE_PATTERN = re.compile(r"^dataset (?P<dataset>.+) has no structure$")

ERROR_TITLES = {
    "RuntimeError": "运行失败",
    "ValueError": "输入无效",
    "TypeError": "输入无效",
    "KeyError": "输入无效",
    "OSError": "文件读写失败",
    "TimeoutError": "操作超时",
}


def readiness_text(message: str) -> str:
    """Project one preflight readiness message into user-facing Chinese."""
    known = READINESS_TEXTS.get(message)
    if known is not None:
        return known
    match = STRUCTURE_PATTERN.match(message)
    if match is not None:
        dataset = match.group("dataset")
        return f"数据集 {dataset} 尚未定义结构，请先在左侧初始化样品结构"
    return message


def operation_error_text(error: object) -> str:
    """Render an OperationError without leaking exception class names."""
    title = ERROR_TITLES.get(str(error.exception_type), "操作失败")
    return f"{title}：{error.message}"
