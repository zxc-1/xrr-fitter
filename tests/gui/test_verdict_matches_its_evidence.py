"""判定卡说的话要与它下面的证据对得上。

替身照基准工程 ``single_robust_log`` 的实测搭：判定「可用但相关」，可强相关是空的、协方差因最优解
贴边而不可用；理由行漏出 ``covariance_unavailable`` 原码；自助只跑了 8 次、服务层自己已判样本不足，
徽章却是绿的「✓ 自助收敛 100%」；±1σ 整列「不可用」不说为什么；「⚠ 2 处边界命中」不点名。
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import QLabel
from tests.support.bootstrap_cases import bootstrap_evidence
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate

import xrr_fitter.api as api
from xrr_fitter.model.inference import CovarianceEvidence

NAMES = ("component.0.thickness_a", "instrument.background", "instrument.relative_sigma")
DISPLAY = ("ox 厚度", "本底", "相对分辨率 σq/q")
HITS = ("instrument.background", "instrument.relative_sigma")
BENCHMARK_EVIDENCE = ("boundary_hit", "profile_interval_open", "covariance_unavailable")
MACHINE_CODE = re.compile(r"\b[a-z]+_[a-z_]+\b")


def _definitions() -> tuple[api.ParameterDefinition, ...]:
    return tuple(
        api.ParameterDefinition(name, display, "", name.split(".")[0], 1.0, 0.5, 1.5, "linear", False)
        for name, display in zip(NAMES, DISPLAY, strict=True)
    )


def _report(*, candidate_id: str | None = "candidate-0") -> api.UncertaintyReport:
    """协方差不可用（贴边）、自助 8 次、两处边界命中、没有强相关。"""
    bootstrap = bootstrap_evidence(tuple((name, 0.9, 1.1) for name in NAMES), attempted_count=8)
    return api.UncertaintyReport(
        correlation_names=NAMES,
        correlation_matrix=np.eye(len(NAMES)),
        covariance_evidence=CovarianceEvidence(NAMES, None, "robust_sandwich_iid", 2, (), "boundary_solution"),
        profiles=(),
        bootstrap_intervals=bootstrap.intervals,
        bootstrap_failure_rate=bootstrap.failure_rate,
        boundary_hits=HITS,
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate_id,
        bootstrap_performed=True,
        bootstrap_evidence=bootstrap,
    )


def _sigma_report(*, candidate_id: str | None = "candidate-0") -> api.UncertaintyReport:
    """一份协方差可用、逐参数 σ 齐全的报告。"""
    return api.UncertaintyReport(
        correlation_names=NAMES,
        correlation_matrix=np.eye(len(NAMES)),
        covariance_evidence=CovarianceEvidence(NAMES, np.eye(len(NAMES)), "gaussian_known_sigma", len(NAMES)),
        parameter_sigma=np.ones(len(NAMES)),
        profiles=(),
        bootstrap_intervals=(),
        bootstrap_failure_rate=0.0,
        boundary_hits=(),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate_id,
    )


def _result(
    report: api.UncertaintyReport,
    evidence: tuple[str, ...] = BENCHMARK_EVIDENCE,
    candidates: tuple = (),
) -> api.FitResult:
    return replace(
        final_fit_result(*candidates),
        parameter_definitions=_definitions(),
        uncertainty=report,
        confidence=api.ConfidenceClass.CORRELATED,
        classification_evidence=evidence,
    )


def _panel(qtbot, result: api.FitResult):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    project = api.XrrProject.new((dataset_project(result=result),), master_seed=1201)
    project = replace(project, base_directory="/private/tmp")
    panel = ResultsPanel(ProjectDocument(api.select_active_dataset(project, "curve")))
    qtbot.addWidget(panel)
    panel.show()
    return panel


def _badge_kinds(panel) -> dict[str, str]:
    return {
        badge.text(): badge.property("statusKind")
        for badge in panel.verdict_evidence.badges()
        if badge.isVisibleTo(panel)
    }


def test_a_correlated_verdict_without_a_strong_correlation_says_what_it_is_about(qtbot) -> None:
    """「可用但相关」是分类器的兜底档；强相关不在原因里时，理由行先把这件事说破。

    否则判定写「相关」，紧挨着的徽章写「相关不可用」，读者会认定其中一处错了。
    """
    panel = _panel(qtbot, _result(_report()))
    reason = panel.confidence_reason.text()

    assert reason.startswith("原因（非强相关）："), reason
    assert "协方差不可用" in reason
    assert not MACHINE_CODE.search(reason), reason


def test_a_correlated_verdict_about_correlation_needs_no_preamble() -> None:
    from xrr_fitter.gui.results.uncertainty import classification_summary

    assert classification_summary(_result(_report(), ("strong_correlation",))) == "参数强相关"


def test_a_skipped_stage_reason_reads_in_chinese() -> None:
    """跳过阶段的那条证据是模型拼的英文句子（``FitResult.from_incomplete_search``），不是码。"""
    from xrr_fitter.gui.results.uncertainty import classification_summary

    result = replace(
        final_fit_result(),
        confidence=api.ConfidenceClass.UNTRUSTED,
        classification_evidence=("incomplete search: skipped stages B, E",),
    )

    assert classification_summary(result) == "搜索未完成：跳过了阶段 B、E"


def test_every_code_the_classifier_emits_reads_in_chinese() -> None:
    """分类器能报的每一个原因码都登记过中文；没登记的会以原码上屏。"""
    from xrr_fitter.gui.results.uncertainty import CLASSIFICATION_LABELS

    emitted = {
        "bootstrap_failure_rate",
        "boundary_hit",
        "covariance_unavailable",
        "distinct_equivalent_clusters",
        "insufficient_cluster_support",
        "invalid_candidate_evidence",
        "missing_candidate_clusters",
        "nevot_croce_applicability_exceeded",
        "no_active_candidates",
        "primary_profile_open",
        "profile_interval_open",
        "profile_path_merge_failed",
        "residual_diagnostics_not_executed",
        "strong_correlation",
        "systematic_residual",
        "two_seed_cluster_support",
    }
    assert emitted - set(CLASSIFICATION_LABELS) == set()


def test_too_few_bootstrap_samples_are_not_reported_as_convergence(qtbot) -> None:
    """8 次重采样、0 次失败，失败率确实是 0——可服务层已判它不够给区间，徽章不能打绿勾。"""
    kinds = _badge_kinds(_panel(qtbot, _result(_report())))

    assert not any("自助收敛" in text for text in kinds), kinds
    badge = next(text for text in kinds if "自助" in text)
    assert "8" in badge and kinds[badge] == "warn", kinds


def test_the_bootstrap_count_row_warns_when_the_service_found_too_few_samples() -> None:
    from xrr_fitter.gui.results.inspector import bootstrap_readings

    assert bootstrap_readings(_report())[0] == ("8", "warn")


def test_the_boundary_badge_names_the_parameters_it_counts(qtbot) -> None:
    """「⚠ 2 处边界命中」只给了个数；读者要的是哪两个，名字照相关矩阵那套短名。"""
    from xrr_fitter.gui.plots.parameter_labels import short_labels

    result = _result(_report())
    panel = _panel(qtbot, result)
    badge = next(label for label in panel.verdict_evidence.badges() if "边界命中" in label.text())

    for name in short_labels(HITS, result.parameter_definitions):
        assert name in badge.text(), badge.text()
    assert all(display in badge.toolTip() for display in DISPLAY[1:]), badge.toolTip()


def test_an_unavailable_sigma_column_says_why(qtbot) -> None:
    panel = _panel(qtbot, _result(_report()))
    note = panel.findChild(QLabel, "resultSigmaNote")

    assert note is not None and note.isVisibleTo(panel)
    assert note.text().startswith("±1σ 不可用：") and "参数边界" in note.text(), note.text()


def test_sigma_owned_by_another_candidate_is_withheld_and_named(qtbot) -> None:
    """σ 跟着报告归属走：别的候选解的 σ 摆在这一行的值旁边，读起来就是这个值的误差。"""
    owned_elsewhere = _result(
        _sigma_report(candidate_id="candidate-9"),
        candidates=(fit_candidate("candidate-0"), fit_candidate("candidate-9", objective=2.0)),
    )
    panel = _panel(qtbot, owned_elsewhere)
    assert panel.selected_candidate_id() == "candidate-0"
    table = panel.result_values
    cells = {table.item(row, 2).text() for row in range(table.rowCount()) if table.item(row, 2) is not None}
    note = panel.findChild(QLabel, "resultSigmaNote")

    assert cells and cells <= {"不可用", "锁定"}, cells
    assert note.isVisibleTo(panel) and "candidate-9" in note.text(), note.text()


def test_an_owned_sigma_column_carries_no_note(qtbot) -> None:
    panel = _panel(qtbot, _result(_sigma_report()))
    note = panel.findChild(QLabel, "resultSigmaNote")

    assert note is not None and not note.isVisibleTo(panel)
