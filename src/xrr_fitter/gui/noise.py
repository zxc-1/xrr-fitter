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


def residual_label(candidate: object) -> str:
    """Keep the evidence's residual identity and units beside the plotted data."""
    return f"加权残差（{candidate.residual_name} / {candidate.residual_unit}）"


def candidate_mode_lines(candidate: object) -> list[str]:
    return [
        f"结果模式：{NOISE_MODE_LABELS[candidate.noise_model]}",
        f"残差：{candidate.residual_name} / {candidate.residual_unit}",
    ]
