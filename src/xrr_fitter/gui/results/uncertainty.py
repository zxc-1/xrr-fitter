"""Candidate-owned uncertainty, classification, and warning projection.

The project stores all scientific values in their domain units.  This module
only formats copies for display: physical length evidence is shown in nm,
MCMC diagnostics retain an explicit unavailable state, and every warning keeps
the candidate identity that owns it.  No NumPy operation is needed at the GUI
boundary because reports already expose immutable iterable arrays.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import xrr_fitter.api as api

CLASSIFICATION_LABELS = {
    "bootstrap_failure_rate": "Bootstrap 失败率超过阈值",
    "boundary_hit": "参数触及边界",
    "distinct_equivalent_clusters": "存在相互分离的近等价候选簇",
    "insufficient_cluster_support": "最佳聚类支持不足",
    "invalid_candidate_evidence": "候选证据无效",
    "missing_candidate_clusters": "缺少候选聚类证据",
    "nevot_croce_applicability_exceeded": "Nevot-Croce 适用范围超限",
    "no_active_candidates": "无活动候选",
    "primary_profile_open": "主要参数 profile 区间双侧开放",
    "profile_interval_open": "profile 区间未闭合",
    "profile_path_merge_failed": "profile 路径合并失败",
    "strong_correlation": "参数强相关",
    "systematic_residual": "检出系统性残差",
    "two_seed_cluster_support": "最佳聚类仅有两个种子支持",
}

DIAGNOSTIC_LABELS = {
    "gauss_hermite_unconverged": "Gauss-Hermite 积分未收敛",
    "ideal_reflectivity_above_one": "理想反射率超过 1",
    "nevot_croce_applicability_exceeded": "Nevot-Croce 适用范围超限",
    "surface_thin_layer_residual": "疑似表面薄层残差",
    "suspected_diffuse_background": "疑似漫散射背景",
    "suspected_unmodeled_footprint": "疑似未建模的足迹效应",
}

# Stage warnings arrive as stable machine codes, unlike the prose some stages
# already emit. Rendering the code alone left a Chinese surface showing raw
# identifiers, so a known code gains an explanation while keeping the code for
# users matching against logs.
FIT_WARNING_LABELS = {
    "fringe_count_screen_disabled": "条纹计数筛选已关闭",
    "stage_a_all_candidates_rejected": "阶段 A 的候选解全部被剔除",
    "stage_a_fringe_candidate_rejected": "部分候选解因条纹特征不符被剔除",
    "stage_a_invalid_candidate_evaluation": "部分候选解的评估结果无效",
    "stage_a_physical_candidate_rejected": "候选解因不满足物理约束被剔除",
}

LENGTH_SUFFIXES = (
    "thickness_a",
    "period_a",
    "roughness_a",
    "microslab_max_a",
)

DISPLAY_UNITS = (
    ("_deg", "°"),
    ("absolute_sigma_a_inv", "Å⁻¹"),
    ("sld_real_a2", "Å⁻²"),
    ("sld_imag_a2", "Å⁻²"),
    ("linear_background_per_a_inv", "Å"),
)


def _joined(values: object) -> str:
    return "、".join(str(value) for value in values)


def _display_value(name: str, value: float) -> float:
    return value / 10.0 if name.endswith(LENGTH_SUFFIXES) else value


def _display_unit(name: str) -> str:
    if name.endswith(LENGTH_SUFFIXES):
        return "nm"
    for suffix, unit in DISPLAY_UNITS:
        if name.endswith(suffix):
            return unit
    return "core unit"


def _interval_text(interval: object) -> str:
    name, lower, upper = interval
    low = _display_value(name, float(lower))
    high = _display_value(name, float(upper))
    return f"{name} [{low:g}, {high:g}] {_display_unit(name)}"


def _profile_text(profile: object) -> str:
    lower = "闭合" if profile.lower_closed else "开放"
    upper = "闭合" if profile.upper_closed else "开放"
    return f"{profile.name}（下侧{lower}，上侧{upper}）"


def _report_lines(report: object) -> list[str]:
    boundaries = _joined(report.boundary_hits) or "无"
    correlations = (
        _joined(f"{left}/{right}={value:.3g}" for left, right, value in report.strong_correlations) or "无强相关"
    )
    profiles = _joined(_profile_text(profile) for profile in report.profiles)
    intervals = _joined(_interval_text(item) for item in report.bootstrap_intervals)
    lines = [
        f"Bootstrap 失败率：{report.bootstrap_failure_rate:.3g}",
        f"边界命中：{boundaries}",
        f"强相关：{correlations}",
        f"profile 区间：{profiles or '不可用'}",
        f"bootstrap 区间：{intervals or '不可用'}",
        f"系统性残差：{'是' if report.systematic_residual else '否'}",
        f"残差 ACF：{'是' if report.residual_autocorrelation else '否'}",
    ]
    return lines


def _finite_values(values: object) -> tuple[float, ...] | None:
    try:
        converted = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if not converted or not all(isfinite(value) for value in converted):
        return None
    return converted


def _metric_extreme(
    values: object,
    choose: Callable[[tuple[float, ...]], float],
) -> str:
    finite = _finite_values(values)
    return "不可用" if finite is None else f"{choose(finite):g}"


def _acceptance_text(values: object) -> str:
    finite = _finite_values(values)
    if finite is None:
        return "不可用"
    return f"{min(finite):g}–{max(finite):g}"


def _linear_quantile(values: tuple[float, ...], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _sample_row(row: object, width: int) -> tuple[float, ...] | None:
    try:
        converted = tuple(float(value) for value in row)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(converted) != width or not all(isfinite(value) for value in converted):
        return None
    return converted


def _sample_columns(report: object) -> tuple[tuple[float, ...], ...] | None:
    names = tuple(report.parameter_names)
    rows = tuple(_sample_row(row, len(names)) for row in report.samples_physical)
    if not names or not rows or any(row is None for row in rows):
        return None
    return tuple(tuple(row[index] for row in rows) for index in range(len(names)))


def _quantile_lines(report: object) -> list[str]:
    columns = _sample_columns(report)
    if columns is None:
        return ["后验分位数：不可用"]
    lines = ["后验分位数（P16 / P50 / P84）："]
    for name, column in zip(report.parameter_names, columns, strict=True):
        ordered = tuple(sorted(column))
        quantiles = (_linear_quantile(ordered, value) for value in (0.16, 0.5, 0.84))
        values = " / ".join(f"{_display_value(name, value):g}" for value in quantiles)
        lines.append(f"{name}：{values} {_display_unit(name)}")
    return lines


def _mcmc_lines(report: object, candidate_id: str) -> list[str]:
    mcmc = report.mcmc
    if mcmc is None:
        return []
    owner = mcmc.candidate_id
    if owner is None:
        return [f"存在未归属的 MCMC 证据；不能将其作为当前候选 {candidate_id} 的证据"]
    if owner != candidate_id:
        return [f"当前候选 {candidate_id} 暂无 MCMC 证据；现有证据属于 {owner}"]
    lines = [
        "MCMC：目标函数伪后验",
        f"接受率范围：{_acceptance_text(mcmc.acceptance_fraction)}",
        f"最大 split-Rhat：{_metric_extreme(mcmc.split_rhat, max)}",
        f"最小 ESS：{_metric_extreme(mcmc.effective_sample_size, min)}",
        f"MCMC 边界命中：{_joined(mcmc.boundary_hits) or '无'}",
    ]
    lines.extend(_quantile_lines(mcmc))
    lines.extend(f"MCMC 警告：{warning}" for warning in mcmc.warnings)
    return lines


def _classification_lines(result: object) -> list[str]:
    return [f"分类证据：{CLASSIFICATION_LABELS.get(code, code)}（{code}）" for code in result.classification_evidence]


def classification_summary(result: object) -> str:
    """Condense the confidence reasons into one hover-sized line.

    The confidence badge sits far from the evidence panel, so a user reading a
    red "不可信" badge has to hunt downward for the cause. Surfacing the same
    reasons as a badge tooltip answers "why" exactly where the question is asked.
    Returns an empty string when no reasons exist, so the caller can fall back to
    the bare classification name rather than assert a meaning the data lacks.
    """
    reasons = [CLASSIFICATION_LABELS.get(code, code) for code in result.classification_evidence]
    return "；".join(reasons)


def _diagnostic_text(diagnostic: object) -> str:
    code = diagnostic.code
    message = diagnostic.message
    label = DIAGNOSTIC_LABELS.get(code)
    if label is None:
        return f"{code}: {message}"
    return f"{label}（{code}: {message}）"


def _owned_warning(owner: str | None, text: str) -> str:
    return text if owner is None else f"{owner}: {text}"


def _fit_warning_text(warning: object) -> str:
    """Explain a known stage warning code, passing anything else through.

    Some stages already emit Chinese prose, so only exact code matches are
    translated; an unrecognised value is never reworded, because losing an
    unmapped warning would hide evidence the user needs.
    """
    text = str(warning)
    label = FIT_WARNING_LABELS.get(text)
    return text if label is None else f"{label}（{text}）"


def _warning_lines(result: object) -> tuple[str, ...]:
    lines = [_fit_warning_text(warning) for warning in result.warnings]
    for candidate in result.candidates:
        lines.extend(
            _owned_warning(candidate.candidate_id, _diagnostic_text(diagnostic)) for diagnostic in candidate.diagnostics
        )
    report = result.uncertainty
    if report is None:
        return tuple(lines)
    lines.extend(_owned_warning(report.candidate_id, _diagnostic_text(diagnostic)) for diagnostic in report.diagnostics)
    if report.mcmc is not None:
        lines.extend(_owned_warning(report.mcmc.candidate_id, f"MCMC: {warning}") for warning in report.mcmc.warnings)
    return tuple(dict.fromkeys(lines))


def _evidence_lines(result: object, candidate_id: str | None) -> list[str]:
    lines = ["可信度仅针对当前结构模型", *_classification_lines(result)]
    if candidate_id is None:
        return [*lines, "尚未选择候选解"]
    report = result.uncertainty
    if report is None:
        return [*lines, f"当前候选 {candidate_id} 暂无不确定度证据"]
    owner = report.candidate_id
    if owner is None:
        return [*lines, f"存在未归属的不确定度证据；不能将其作为当前候选 {candidate_id} 的证据"]
    if owner != candidate_id:
        return [
            *lines,
            f"当前候选 {candidate_id} 暂无不确定度证据",
            f"现有证据属于 {owner}",
        ]
    lines.extend([f"不确定度证据候选：{owner}", *_report_lines(report)])
    lines.extend(_mcmc_lines(report, candidate_id))
    return lines


def _spin(
    object_name: str,
    accessible_name: str,
    minimum: int,
    maximum: int,
) -> QSpinBox:
    widget = QSpinBox()
    widget.setObjectName(object_name)
    widget.setAccessibleName(accessible_name)
    widget.setRange(minimum, maximum)
    return widget


class McmcControls(QGroupBox):
    """Own MCMC input widgets and their operation-state projection."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("专家 MCMC", parent)
        self._configured_candidate_id: str | None = None
        self._free_count = 0
        self.setObjectName("expertMcmcGroup")
        self.walkers = _spin("mcmcWalkers", "MCMC walkers 数", 2, 100000)
        self.walkers.setSingleStep(2)
        self.burn_in = _spin("mcmcBurnIn", "MCMC burn-in 步数", 0, 10000000)
        self.production = _spin(
            "mcmcProduction",
            "MCMC production 步数",
            1,
            10000000,
        )
        self.thin = _spin("mcmcThin", "MCMC thinning 间隔", 1, 1000000)
        self.recommend_button = QPushButton("推荐配置")
        self.recommend_button.setObjectName("mcmcRecommendButton")
        self.recommend_button.setToolTip("根据当前候选的自由参数数量填入推荐采样参数")
        self.recommend_button.clicked.connect(self.apply_recommended)
        self.run_button = QPushButton("运行 MCMC")
        self.run_button.setObjectName("mcmcButton")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("cancelMcmcButton")
        self.force_button = QPushButton("强制停止")
        self.force_button.setObjectName("forceStopMcmcButton")
        buttons = QHBoxLayout()
        buttons.addWidget(self.recommend_button)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.force_button)
        form = QGridLayout(self)
        form.setContentsMargins(6, 2, 6, 2)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        for row, values in enumerate(
            (
                (("walkers", self.walkers), ("burn-in", self.burn_in)),
                (("production", self.production), ("thin", self.thin)),
            )
        ):
            for column, (text, control) in enumerate(values):
                label = QLabel(text)
                label.setBuddy(control)
                form.addWidget(label, row, column * 2)
                form.addWidget(control, row, column * 2 + 1)
        form.addLayout(buttons, 2, 0, 1, 4)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        form.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

    def configure(self, candidate_id: str | None, free_count: int) -> None:
        self._free_count = free_count
        if candidate_id == self._configured_candidate_id:
            return
        self._configured_candidate_id = candidate_id
        if free_count == 0:
            return
        self._fill_recommended(free_count)

    def apply_recommended(self) -> bool:
        """Reset the inputs to the standard config for the current candidate.

        ``configure`` short-circuits when the candidate is unchanged, so a user
        who has hand-edited the fields cannot recover the defaults by reselecting
        the same candidate. This button gives them an unconditional way back.
        """
        if self._free_count == 0:
            return False
        self._fill_recommended(self._free_count)
        return True

    def _fill_recommended(self, free_count: int) -> None:
        config = api.McmcConfig.standard(free_count)
        self.walkers.setValue(config.walkers)
        self.burn_in.setValue(config.burn_in)
        self.production.setValue(config.production_steps)
        self.thin.setValue(config.thin)

    def config(self) -> api.McmcConfig:
        return api.McmcConfig(
            walkers=self.walkers.value(),
            burn_in=self.burn_in.value(),
            production_steps=self.production.value(),
            thin=self.thin.value(),
        )

    def validated_config(self, free_count: int) -> api.McmcConfig:
        config = self.config()
        if config.walkers < 2 * free_count + 2:
            raise ValueError("walkers must be even and at least 2*nfree+2")
        if config.production_steps < 4 or config.thin >= config.production_steps:
            raise ValueError("invalid MCMC step configuration")
        return config

    def set_operation_state(self, *, running: bool, ready: bool) -> None:
        self.run_button.setEnabled(ready and not running)
        self.cancel_button.setEnabled(running)
        self.force_button.setEnabled(running)
        self.recommend_button.setEnabled(not running)
        for widget in (self.walkers, self.burn_in, self.production, self.thin):
            widget.setEnabled(not running)


class UncertaintyView(QWidget):
    """Show only evidence owned by the currently inspected candidate."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._warnings: tuple[str, ...] = ()
        self.setObjectName("uncertaintyView")
        self.evidence = QPlainTextEdit()
        self.evidence.setObjectName("uncertaintyEvidence")
        self.evidence.setAccessibleName("候选解不确定度证据")
        self.evidence.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.evidence)

    def clear_evidence(self, message: str) -> None:
        self.evidence.setPlainText(message)

    def clear_result(self, message: str) -> None:
        self._warnings = ()
        self.clear_evidence(message)

    def set_result(self, result: object, candidate_id: str | None) -> None:
        self._warnings = _warning_lines(result)
        self.evidence.setPlainText("\n".join(_evidence_lines(result, candidate_id)))

    def text(self) -> str:
        return self.evidence.toPlainText()

    def warning_texts(self) -> tuple[str, ...]:
        return self._warnings
