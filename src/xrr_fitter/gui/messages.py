"""Chinese presentation texts for public API status and error values.

The public API deliberately reports machine-oriented English messages.  This
module owns the user-facing Chinese projection: known messages translate to
guidance text.

未登记的诊断串不再照原样上屏。``preflight_fit`` 的兜底是 ``str(error)``，fit 编译栈里任意一个
异常都能走到那里——那是个开放集合，逐条登记永远追不上，而底栏第一段在设计稿五帧里一律是
两到八个汉字。所以那一档报分类结论、原文写进日志：屏上守住那一段的语言与长度，专家要的原文
仍取得回。本身已是中文的消息（拟合面板自己构造的那几句）照原样穿透，它不需要翻译。
"""

from __future__ import annotations

import logging
import re

LOG = logging.getLogger(__name__)

# 设计稿帧① 底栏第一段（HTML 474 行）就这两个字。前面那颗 8px 圆点已经把「可以开始了」
# 说完，后面还要排判定与 J / χ²ᵥ 两段读数；写成一句邀请，30px 高的底栏光第一段就占掉近三
# 分之一宽度。四帧底栏的第一段一律是这个长度（引导模式 / 结构已修改（未保存） / 拟合进行中
# / MCMC 采样中）。
READY_TEXT = "就绪"

READINESS_TEXTS = {
    "ready": READY_TEXT,
    "project has no datasets": "项目中还没有数据集，请先导入 XRR 数据",
    "source validation failed": "数据源校验失败，请检查源文件后重试",
    "Gaussian fitting requires known intensity sigma at every selected point": (
        "Gaussian 拟合需要每个拟合点的有效标准差 sigma，请设置强度不确定度列"
    ),
    "Poisson fitting requires finite nonnegative integer raw counts": (
        "Poisson 拟合需要原始非负整数 counts，请确认不是归一化强度或计数率"
    ),
    "automatic fit requires a measurement preset": ("自动拟合需要测量预设，请通过导入对话框设置光路与仪器"),
    # 拖窄拟合范围最容易撞上的那一条。``fit/problem.py`` 抛的 ``ValueError`` 只说「没就绪」，
    # 而读者要知道的是把范围放回去——掩码里剩不下 30 点，拟合就无从谈起。
    "current fit mask is not fit-ready": "当前拟合范围内的可用点不足，请放宽拟合范围",
    # 自动拟合只认还没跑完的那几个数据集（``AUTOMATIC_RUNNABLE``）；一个都不剩时说的是
    # 「没有待运行的」，而不是「没有数据集」——项目里可能满是已经收敛的结果。
    "no runnable automatic datasets": "没有待运行的自动拟合数据集，请检查数据集的自动化状态",
    "automatic fit does not support cross-dataset constraints": (
        "自动拟合不支持跨数据集约束，请改用手动拟合或先移除约束"
    ),
}

# 没登记过的英文诊断落在这一档。分类结论说到「拟合条件未满足」为止：再具体就是替一个不认识
# 的异常编造成因，而底栏那一段没有第二行可以放原文。
UNKNOWN_READINESS_TEXT = "拟合条件未满足，请检查数据与参数设置"

STRUCTURE_PATTERN = re.compile(r"^dataset (?P<dataset>.+) has no structure$")

# Readiness answers "can a fit start", which stays true while one is already
# running -- and then reads as an invitation the disabled start button refuses.
# A live operation therefore takes over that line, naming which of the three
# controllers holds the window so nobody hunts for stage progress that an MCMC
# run or an export is never going to produce.
# 措辞照设计稿底栏抄：帧④ 是「拟合进行中」，帧⑤ 是「MCMC 采样中」——采样那句不带
# 「进行」，因为它后面还跟着 split-R̂ 与 ESS 两个读数，短一点才容得下。
RUNNING_TEXTS = {
    "fit": "拟合进行中",
    "mcmc": "MCMC 采样中",
    "external": "后台操作进行中",
}

ERROR_TITLES = {
    "RuntimeError": "运行失败",
    "ValueError": "输入无效",
    "TypeError": "输入无效",
    "KeyError": "输入无效",
    "OSError": "文件读写失败",
    "TimeoutError": "操作超时",
}

# Actionable next steps keyed by exception type. A bare error message names what
# went wrong but leaves the user stuck; naming a concrete recovery move turns a
# dead end into a next step. Types without a documented recovery fall through to
# no suffix rather than inventing generic advice.
ERROR_ADVICE = {
    "OSError": "请确认数据源文件仍然存在且可读，必要时重新链接数据源后再试。",
    "TimeoutError": "可尝试缩小拟合角度范围或减少参数数量，再重新开始拟合。",
    "ValueError": "请检查参数上下限与初值是否自洽，以及结构是否完整。",
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
    if _has_han(message):
        return message
    LOG.debug("unmapped readiness message reached the status bar: %s", message)
    return UNKNOWN_READINESS_TEXT


def _has_han(text: str) -> bool:
    """``text`` 里有没有汉字——分「这句话是给读者看的」还是「这是机器串」的那道线。

    比「含不含拉丁字母」稳：译文里本来就有 ``XRR``、``χ²ᵥ`` 这样的记号，按字母判会把它们
    误判成机器串。
    """
    return any("一" <= character <= "鿿" for character in text)


def running_text(kind: str) -> str:
    """Name the live operation that makes the readiness line moot."""
    return RUNNING_TEXTS.get(kind, "操作进行中")


def operation_error_text(error: object) -> str:
    """Render an OperationError with a concrete recovery hint when known."""
    exception_type = str(error.exception_type)
    title = ERROR_TITLES.get(exception_type, "操作失败")
    advice = ERROR_ADVICE.get(exception_type)
    base = f"{title}：{error.message}"
    return f"{base}\n建议：{advice}" if advice else base
