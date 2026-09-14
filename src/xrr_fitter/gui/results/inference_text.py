"""Format saved inference provenance without deriving new statistical claims."""

from __future__ import annotations

INTERVAL_LABELS = {
    "loss_support": "损失支持（探索）",
    "likelihood_ratio": "似然比区间",
    "percentile_bootstrap": "百分位 Bootstrap 区间",
    "exploratory_bootstrap": "Bootstrap 样本（探索）",
    "unavailable": "不可用",
}


def interval_metadata(evidence: object) -> str:
    kind = evidence.interval_kind
    level = evidence.confidence_level
    confidence = "未校准/不可用" if level is None else f"{level:.0%}"
    parts = [
        f"类型：{INTERVAL_LABELS[kind]}（{kind}）",
        f"置信水平：{confidence}",
        f"方法：{evidence.method}",
    ]
    if evidence.unavailable_reason is not None:
        parts.append(f"不可用原因：{evidence.unavailable_reason}")
    return "；".join(parts)


def compact_interval_metadata(evidence: object) -> tuple[str, ...]:
    """Use separate short lines for a plot key; the full report remains explicit."""
    level = evidence.confidence_level
    confidence = "探索/不可用" if level is None else f"{level:.0%}"
    parts = (f"{evidence.interval_kind} · {confidence}", evidence.method)
    if evidence.unavailable_reason is not None:
        parts += (evidence.unavailable_reason,)
    return parts


def bootstrap_metadata(report: object) -> list[str]:
    evidence = report.bootstrap_evidence
    if evidence is None:
        return ["Bootstrap：未执行"]
    lines = [
        f"Bootstrap：已执行，成功样本 {evidence.successful_samples}/{evidence.attempted_count}",
        f"Bootstrap 失败率：{evidence.failure_rate:.3g}",
        f"Bootstrap {interval_metadata(evidence)}",
    ]
    if evidence.diagnostic_unavailable_reason is not None:
        lines.append(f"Bootstrap 诊断资格不可用原因：{evidence.diagnostic_unavailable_reason}")
    return lines


def covariance_metadata(report: object) -> list[str]:
    evidence = report.covariance_evidence
    if evidence is None:
        return ["协方差：未执行/不可用"]
    available = "可用" if evidence.matrix is not None else "不可用"
    parts = [f"协方差：{available}", f"方法：{evidence.method}", f"秩：{evidence.rank}/{len(evidence.names)}"]
    if evidence.unavailable_reason is not None:
        parts.append(f"不可用原因：{evidence.unavailable_reason}")
    if evidence.unidentifiable_names:
        parts.append(f"不可辨识参数：{'、'.join(evidence.unidentifiable_names)}")
    return ["；".join(parts)]


def diagnostic_state(value: bool | None) -> str:
    return {True: "是", False: "否", None: "未执行/不可用"}[value]


def _member_residual_line(evidence: object) -> str:
    owner = evidence.dataset_id or "当前数据集"
    state = "已执行" if evidence.executed else "未执行/不可用"
    parts = [f"{owner} 残差诊断：{state}", f"点数：{evidence.point_count}"]
    if evidence.executed:
        parts.extend(
            (
                f"系统性残差：{diagnostic_state(evidence.systematic)}",
                f"ACF：{diagnostic_state(evidence.autocorrelation)}",
            )
        )
    else:
        parts.append(f"原因：{evidence.unavailable_reason}")
    return "；".join(parts)


def residual_diagnostic_lines(report: object) -> list[str]:
    return [
        f"系统性残差：{diagnostic_state(report.systematic_residual)}",
        f"残差 ACF：{diagnostic_state(report.residual_autocorrelation)}",
        *(line for evidence in report.member_residuals for line in _member_residual_lines(evidence)),
    ]


def _raw_residual_lines(evidence: object) -> list[str]:
    lines = []
    if evidence.raw_systematic is not None or evidence.raw_autocorrelation is not None:
        lines.append(
            f"原始筛查：系统性残差：{diagnostic_state(evidence.raw_systematic)}；"
            f"ACF：{diagnostic_state(evidence.raw_autocorrelation)}"
        )
    lines.extend(f"原始提示（advisory）：{item.code}：{item.message}" for item in evidence.advisories)
    return lines


def _calibration_statistics(evidence: object) -> list[str]:
    calibration = evidence.calibration
    lines = []
    for item in calibration.statistics:
        if item.dataset_id != evidence.dataset_id:
            continue
        text = f"检测器 {item.kind}：observed={item.observed:.6g}"
        if item.adjusted_p_value is not None:
            text += f"；族调整 p={item.adjusted_p_value:.6g}"
        lines.append(text)
    return lines


def _calibration_lines(evidence: object) -> list[str]:
    calibration = evidence.calibration
    if calibration is None:
        if evidence.raw_systematic is False and evidence.raw_autocorrelation is False:
            return ["诊断校准：未触发校准"]
        return ["诊断校准：校准未执行"]
    state = "校准不可用"
    if calibration.status == "available":
        state = "校准拒绝" if calibration.rejected else "校准未拒绝"
    lines = [
        f"Poisson 诊断族：{state}",
        f"校准方法：{calibration.method}；重拟合策略：{calibration.refit_policy}",
        f"诊断 MC 成功/尝试：{calibration.successful_count}/{calibration.attempted_count}；"
        f"计划 B={calibration.sample_count}；refit nfev={calibration.refit_nfev}",
    ]
    if calibration.p_value is not None:
        lines.append(
            f"诊断族 p={calibration.p_value:.6g}；alpha={calibration.alpha:g}；"
            f"尾计数={calibration.tail_count}；并列计数={calibration.tie_count}；"
            f"分辨率={calibration.resolution:g}（不是误差条）"
        )
    if calibration.unavailable_reason is not None:
        lines.append(f"校准不可用原因：{calibration.unavailable_reason}")
    lines.extend(f"重拟合失败 [{index}]：{reason}" for index, reason in calibration.failure_reasons)
    return [*lines, *_calibration_statistics(evidence)]


def _member_residual_lines(evidence: object) -> list[str]:
    return [_member_residual_line(evidence), *_raw_residual_lines(evidence), *_calibration_lines(evidence)]
