"""Presentation labels for declared noise modes and stored residual evidence."""

from __future__ import annotations

NOISE_MODE_LABELS = {
    "robust_log": "稳健对数（探索）",
    "gaussian": "Gaussian（已知标准差）",
    "poisson": "Poisson（原始整数计数）",
}

NOISE_MODE_REQUIREMENTS = {
    "robust_log": "探索模式：稳健对数损失；损失支持区间不等同于已校准的置信区间。",
    "gaussian": "需要显式强度标准差 sigma 列，且 sigma 有限、严格大于 0；零和负强度仍是有效观测。",
    "poisson": "选择此模式即声明强度列为原始非负整数 counts，不是归一化强度、计数率或背景扣除值；不得平滑，保留零计数。",
}


# 状态栏、绘图卡说明这种一行读数里用的短名：括号里的说明留给下拉框和结果卡。
NOISE_MODE_SHORT_LABELS = {
    "robust_log": "稳健对数",
    "gaussian": "Gaussian",
    "poisson": "Poisson",
}

# 汇总读数里的 J 只给三位有效数字。左栏、数据集卡、进度、状态栏、图卡说明原先各写各的
# （``.2f``、``:g``、``.4g``），同一个 J 于是在左栏读 0.01、在状态栏读 0.007593；稳健对数
# 的 J 常在 1e-2 以下，``.2f`` 把它读成 0.00。要逐位比较的地方（候选列表、判定卡）另有
# 六位有效数字那一档，见 ``results.candidate_row.OBJECTIVE_SIGNIFICANT``。
OBJECTIVE_SUMMARY_SIGNIFICANT = 3


def objective_text(value: float) -> str:
    """The objective as every one-line summary writes it.

    千以上照整数写：高斯的 J 是 χ²/N，坏拟合能到几千，``1.23e+03`` 在一行小字里读不出来。
    """
    if abs(value) >= 10**OBJECTIVE_SUMMARY_SIGNIFICANT:
        return f"{value:.0f}"
    return f"{value:.{OBJECTIVE_SUMMARY_SIGNIFICANT}g}"


def residual_label(candidate: object) -> str:
    """Keep the evidence's residual identity and units beside the plotted data."""
    return f"加权残差（{candidate.residual_name} / {candidate.residual_unit}）"


def candidate_mode_lines(candidate: object) -> list[str]:
    return [
        f"结果模式：{NOISE_MODE_LABELS[candidate.noise_model]}",
        f"残差：{candidate.residual_name} / {candidate.residual_unit}",
    ]
