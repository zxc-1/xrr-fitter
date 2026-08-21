"""Result-workflow contracts at the immutable project/Qt boundary.

The panel may inspect every retained candidate, but only an eligible candidate
may become the persisted API selection.  Archived evidence therefore requires
an explicit confirmation and invalid evidence never mutates the project.

Uncertainty and MCMC records are candidate-owned.  A visible row must not make
evidence from another (or an unowned legacy record) appear applicable.  MCMC
completion is also bound to the exact project identity used to start the job;
otherwise an asynchronous result could revive invalidated data or discard a
newer selection.

These tests deliberately exercise direct methods and Qt signal paths.  The
former prove API routing and pre-publication failure atomicity, while the latter
prove that invalid user input cannot escape through the event-loop exception
handler.

Automatic result tables add a second ownership boundary. Point roles, quality
status, fitted layer values, and population uniformity must come from the same
completed import batch. The assertions intentionally inspect both table models
and public API calls so presentation changes cannot silently weaken persistence
rules or attach statistics to excluded points.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QTableWidget
from tests.support.model_cases import (
    dataset_project,
    final_fit_result,
    fit_candidate,
    project,
)

import xrr_fitter.api as api
from xrr_fitter.model.instrument import PhysicsDiagnostic


def _uncertainty(candidate_id: str = "candidate-a") -> api.UncertaintyReport:
    return api.UncertaintyReport(
        correlation_names=("scale",),
        correlation_matrix=np.array([[1.0]]),
        profiles=(),
        bootstrap_intervals=(("scale", 0.8, 1.2),),
        bootstrap_failure_rate=0.125,
        boundary_hits=("scale",),
        strong_correlations=(),
        systematic_residual=False,
        diagnostics=(),
        candidate_id=candidate_id,
    )


def _two_candidate_result(*, with_uncertainty: bool = True):
    first = replace(
        fit_candidate("candidate-a", 0.2),
        ranking_objective=0.4,
        unit_vector=np.zeros(17),
    )
    second = replace(
        fit_candidate("candidate-b", 0.3),
        ranking_objective=0.8,
        unit_vector=np.zeros(3),
    )
    result = final_fit_result(first, second)
    if with_uncertainty:
        result = replace(result, uncertainty=_uncertainty())
    return result


def _mcmc_report(**changes):
    values = {
        "config": api.McmcConfig(walkers=4, burn_in=2, production_steps=4),
        "child_seed": 7,
        "parameter_names": ("component.0.thickness_a", "instrument.scale"),
        "samples_physical": np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]]),
        "log_probability": np.zeros(4),
        "acceptance_fraction": np.array([0.2, 0.6, 0.4, 0.8]),
        "split_rhat": np.array([1.05, 1.12]),
        "effective_sample_size": np.array([120.0, 80.0]),
        "boundary_hits": ("component.0.thickness_a",),
        "warnings": ("chain warning",),
        "candidate_id": "candidate-a",
    }
    values.update(changes)
    return api.McmcReport(**values)


def _project_with_result(result=None, *, selected: str | None = None, expert=False):
    """Build result state only through supported immutable API transitions.

    Candidate selection and expert mode remain optional so each panel test can
    declare the exact persisted ownership state it intends to render.
    """
    value = project(dataset_project(result=result))
    value = replace(value, base_directory="/private/tmp")
    value = api.select_active_dataset(value, "curve")
    if selected is not None:
        value = api.select_candidate(value, "curve", selected)
    if expert:
        value = api.set_expert_mode(value, True)
    return value


def _panel(qtbot, value):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    panel = ResultsPanel(ProjectDocument(value))
    qtbot.addWidget(panel)
    return panel


def _candidate_audit_result():
    best = replace(
        fit_candidate("candidate-best", 0.2),
        ranking_objective=0.4,
    )
    archived = replace(
        fit_candidate("candidate-archived", 0.1),
        ranking_objective=0.1,
        stop_reason="early_eliminated",
    )
    invalid = replace(
        fit_candidate("candidate-invalid", 0.3),
        valid=False,
        objective=float("inf"),
        ranking_objective=3.0,
        stop_reason="invalid_model",
    )
    return final_fit_result(best, archived, invalid)


def test_candidate_rows_show_recommendation_and_both_objectives(qtbot) -> None:
    panel = _panel(qtbot, _project_with_result(_candidate_audit_result()))

    assert panel.confidence_text() == "可信"
    assert panel.candidate_count() == 3
    assert "candidate-best" in panel.candidate_text(0)
    assert "局部目标值 J=0.2" in panel.candidate_text(0)
    assert "全局排序目标值 J=0.4" in panel.candidate_text(0)
    assert "推荐" in panel.candidate_text(0)


def test_candidate_rows_keep_archived_and_invalid_inspection_evidence(qtbot) -> None:
    panel = _panel(qtbot, _project_with_result(_candidate_audit_result()))

    assert "仅供检查" in panel.candidate_text(1)
    assert "早期淘汰" in panel.candidate_text(1)
    assert "仅供检查" in panel.candidate_text(2)
    assert "无效" in panel.candidate_text(2)


def test_invalid_candidate_is_inspected_without_persisting_selection(qtbot) -> None:
    panel = _panel(qtbot, _project_with_result(_candidate_audit_result()))
    before = panel.document.project

    panel.candidates.setCurrentRow(2)

    assert panel.selected_candidate_id() == "candidate-invalid"
    assert panel.document.project is before
    assert panel.document.project.ui_state.selected_candidate_ids == ()


def test_arrow_key_navigation_inspects_next_candidate(qtbot) -> None:
    # Moving the selection with the keyboard must reach the same inspection path
    # as a mouse click, so a keyboard-only user can walk candidates and watch the
    # evidence panel follow. The list advertises this affordance so it is
    # discoverable rather than a hidden Qt default.
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    panel = _panel(qtbot, _project_with_result(_two_candidate_result()))
    inspected: list[str] = []
    panel.candidates.candidate_requested.connect(inspected.append)
    panel.candidates.setCurrentRow(0)
    panel.candidates.setFocus()
    inspected.clear()

    QTest.keyClick(panel.candidates, Qt.Key.Key_Down)

    assert panel.selected_candidate_id() == "candidate-b"
    assert inspected == ["candidate-b"]
    assert "方向键" in panel.candidates.accessibleDescription()


@pytest.mark.parametrize("confirmed", (False, True))
def test_archived_candidate_requires_confirmation_before_persisting(
    qtbot,
    monkeypatch,
    confirmed,
) -> None:
    panel = _panel(qtbot, _project_with_result(_candidate_audit_result()))
    before = panel.document.project
    monkeypatch.setattr(
        panel,
        "_confirm_archived_candidate",
        lambda _candidate: confirmed,
        raising=False,
    )

    panel.candidates.setCurrentRow(1)

    assert panel.selected_candidate_id() == "candidate-archived"
    expected = (("curve", "candidate-archived"),) if confirmed else ()
    assert panel.document.project.ui_state.selected_candidate_ids == expected
    if not confirmed:
        assert panel.document.project is before


def test_candidate_selection_uses_public_api_returned_project(
    qtbot,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result()))
    original = panel.document.project
    updated = api.set_expert_mode(
        api.select_candidate(original, "curve", "candidate-b"),
        True,
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "select_candidate",
        lambda *args: (calls.append(args), updated)[1],
    )
    selected: list[str] = []
    panel.candidate_selected.connect(selected.append)

    changed = panel.select_candidate("candidate-b")

    assert changed is True
    assert calls == [(original, "curve", "candidate-b")]
    assert panel.document.project is updated
    assert panel.selected_candidate_id() == "candidate-b"
    assert selected == ["candidate-b"]


def test_candidate_selection_failure_is_atomic(qtbot, monkeypatch) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result()))
    before_project = panel.document.project
    before_rows = tuple(panel.candidate_text(index) for index in range(panel.candidate_count()))
    before_selected = panel.selected_candidate_id()
    selected: list[str] = []
    panel.candidate_selected.connect(selected.append)
    monkeypatch.setattr(
        api,
        "select_candidate",
        lambda *_args: (_ for _ in ()).throw(ValueError("selection rejected")),
    )

    with pytest.raises(ValueError, match="selection rejected"):
        panel.select_candidate("candidate-b")

    assert panel.document.project is before_project
    assert panel.selected_candidate_id() == before_selected
    assert tuple(panel.candidate_text(index) for index in range(panel.candidate_count())) == before_rows
    assert selected == []


def test_panel_restores_persisted_candidate_or_projects_best_candidate(qtbot) -> None:
    result = _two_candidate_result()
    persisted = _panel(
        qtbot,
        _project_with_result(result, selected="candidate-b"),
    )
    defaulted = _panel(qtbot, _project_with_result(result))

    assert persisted.selected_candidate_id() == "candidate-b"
    assert persisted.recommended_candidate_id() == "candidate-a"
    assert defaulted.selected_candidate_id() == "candidate-a"
    assert defaulted.recommended_candidate_id() == "candidate-a"
    assert defaulted.document.project.ui_state.selected_candidate_ids == ()


def test_result_invalidation_clears_candidate_and_uncertainty_views(qtbot) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result()))

    panel.document.replace_project(_project_with_result())

    assert panel.candidate_count() == 0
    assert panel.selected_candidate_id() is None
    assert "尚无拟合结果" in panel.uncertainty_text()


def test_clear_results_uses_public_api_and_adopts_returned_project(
    qtbot,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result()))
    original = panel.document.project
    updated = api.clear_fit_results(original, ("curve",))
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "clear_fit_results",
        lambda *args: (calls.append(args), updated)[1],
    )
    cleared: list[tuple[str, ...]] = []
    panel.results_cleared.connect(cleared.append)

    changed = panel.clear_results()

    assert changed is True
    assert calls == [(original, ("curve",))]
    assert panel.document.project is updated
    assert panel.candidate_count() == 0
    assert cleared == [("curve",)]


def test_uncertainty_projection_only_shows_current_candidate_evidence(qtbot) -> None:
    result = _two_candidate_result()
    panel = _panel(
        qtbot,
        _project_with_result(result, selected="candidate-b"),
    )

    other_text = panel.uncertainty_text()
    assert "当前候选 candidate-b 暂无不确定度证据" in other_text
    assert "现有证据属于 candidate-a" in other_text
    assert "边界命中（可疑）：scale" not in other_text

    panel.select_candidate("candidate-a")

    assert "边界命中（可疑）：scale" in panel.uncertainty_text()


@pytest.mark.parametrize(
    ("code", "label"),
    (
        ("bootstrap_failure_rate", "Bootstrap 失败率超过阈值"),
        ("boundary_hit", "参数触及边界"),
        ("distinct_equivalent_clusters", "存在相互分离的近等价候选簇"),
        ("insufficient_cluster_support", "最佳聚类支持不足"),
        ("invalid_candidate_evidence", "候选证据无效"),
        ("missing_candidate_clusters", "缺少候选聚类证据"),
        ("nevot_croce_applicability_exceeded", "Nevot-Croce 适用范围超限"),
        ("no_active_candidates", "无活动候选"),
        ("primary_profile_open", "主要参数 profile 区间双侧开放"),
        ("profile_interval_open", "profile 区间未闭合"),
        ("profile_path_merge_failed", "profile 路径合并失败"),
        ("strong_correlation", "参数强相关"),
        ("systematic_residual", "检出系统性残差"),
        ("two_seed_cluster_support", "最佳聚类仅有两个种子支持"),
    ),
)
def test_classification_reason_codes_have_chinese_labels_and_keep_codes(
    qtbot,
    code,
    label,
) -> None:
    result = replace(_two_candidate_result(), classification_evidence=(code,))
    panel = _panel(qtbot, _project_with_result(result))

    assert label in panel.uncertainty_text()
    assert f"（{code}）" in panel.uncertainty_text()


def test_uncertainty_formats_length_intervals_and_residual_statuses(qtbot) -> None:
    report = replace(
        _uncertainty(),
        bootstrap_intervals=(("component.0.thickness_a", 95.0, 108.0),),
        systematic_residual=False,
        residual_autocorrelation=False,
    )
    result = replace(_two_candidate_result(), uncertainty=report)
    panel = _panel(qtbot, _project_with_result(result))

    text = panel.uncertainty_text()
    assert "component.0.thickness_a [9.5, 10.8] nm" in text
    assert "系统性残差：否" in text
    assert "残差 ACF：否" in text
    assert "可信度仅针对当前结构模型" in text


def test_mcmc_evidence_uses_fixed_label_metrics_and_physical_quantiles(qtbot) -> None:
    report = replace(_uncertainty(), mcmc=_mcmc_report(label="untrusted label"))
    result = replace(_two_candidate_result(), uncertainty=report)
    panel = _panel(qtbot, _project_with_result(result))

    text = panel.uncertainty_text()
    assert "MCMC：目标函数伪后验" in text
    assert "untrusted label" not in text
    assert "接受率范围：0.2–0.8" in text
    assert "最大 split-Rhat：1.12" in text
    assert "最小 ESS：80" in text
    assert "component.0.thickness_a：1.48 / 2.5 / 3.52 nm" in text
    assert "MCMC 警告：chain warning" in text


def test_nonfinite_mcmc_diagnostics_are_explicitly_unavailable(qtbot) -> None:
    mcmc = _mcmc_report(
        acceptance_fraction=np.array([0.2, np.nan, 0.4, np.inf]),
        split_rhat=np.array([np.nan, 1.12]),
        effective_sample_size=np.array([np.inf, 80.0]),
    )
    result = replace(_two_candidate_result(), uncertainty=replace(_uncertainty(), mcmc=mcmc))
    panel = _panel(qtbot, _project_with_result(result))

    text = panel.uncertainty_text()
    assert "接受率范围：不可用" in text
    assert "最大 split-Rhat：不可用" in text
    assert "最小 ESS：不可用" in text


def test_result_warnings_keep_candidate_and_evidence_owner_ids(qtbot) -> None:
    result = _two_candidate_result()
    local = PhysicsDiagnostic("candidate_diagnostic", "local evidence")
    second = replace(result.candidates[1], diagnostics=(local,))
    report = replace(
        _uncertainty(),
        diagnostics=(PhysicsDiagnostic("report_diagnostic", "report evidence"),),
        mcmc=_mcmc_report(),
    )
    result = replace(
        result,
        candidates=(result.candidates[0], second),
        warnings=("fit warning",),
        uncertainty=report,
    )
    panel = _panel(qtbot, _project_with_result(result))

    warnings = panel.warning_texts()
    assert "fit warning" in warnings
    assert "candidate-b: candidate_diagnostic: local evidence" in warnings
    assert "candidate-a: report_diagnostic: report evidence" in warnings
    assert "candidate-a: MCMC: chain warning" in warnings


def test_known_fit_warning_codes_are_shown_in_chinese(qtbot) -> None:
    """A stage warning code is machine-oriented; the panel must explain it.

    ``result.warnings`` carries stable codes from the fit stages, and the panel
    was rendering them verbatim, so a Chinese-language surface showed raw
    identifiers like ``stage_a_fringe_candidate_rejected``. The code is retained
    alongside the explanation because expert users match it against logs.
    """
    result = _two_candidate_result()
    result = replace(
        result,
        warnings=(
            "stage_a_physical_candidate_rejected",
            "stage_a_fringe_candidate_rejected",
        ),
    )
    panel = _panel(qtbot, _project_with_result(result))

    warnings = panel.warning_texts()
    assert "候选解因不满足物理约束被剔除（stage_a_physical_candidate_rejected）" in warnings
    assert "部分候选解因条纹特征不符被剔除（stage_a_fringe_candidate_rejected）" in warnings


def test_unknown_fit_warning_codes_pass_through_verbatim(qtbot) -> None:
    """An untranslated warning must never be swallowed or renamed."""
    result = _two_candidate_result()
    result = replace(result, warnings=("some_future_warning_code",))
    panel = _panel(qtbot, _project_with_result(result))

    assert "some_future_warning_code" in panel.warning_texts()


def test_confidence_uses_redundant_marker_and_semantic_status_kind(qtbot) -> None:
    """The badge must take its colour from the theme, not an inline sheet.

    The four hardcoded hex values were the light-theme colours, so on a dark
    desktop the badge stayed light green while every control around it had
    switched.  They also reached the screen through ``setStyleSheet``, which
    outranks the application sheet and so cannot follow the palette at all.
    """
    result = _two_candidate_result()
    multiple = replace(result, confidence=type(result.confidence).MULTIPLE)
    panel = _panel(qtbot, _project_with_result(multiple))

    assert panel.confidence_text() == "多解"
    assert panel.confidence_marker.text() == "▲"
    assert panel.confidence_marker.accessibleDescription() == "多解"
    assert panel.confidence_label.property("statusKind") == "warn"
    assert panel.confidence_marker.property("statusKind") == "warn"
    assert panel.confidence_label.styleSheet() == ""
    assert panel.confidence_marker.styleSheet() == ""


@pytest.mark.parametrize(
    ("level", "kind"),
    [("TRUSTED", "ok"), ("CORRELATED", "info"), ("MULTIPLE", "warn"), ("UNTRUSTED", "error")],
)
def test_every_confidence_level_reaches_a_distinct_painted_kind(qtbot, level: str, kind: str) -> None:
    """Four levels over three status colours is why `info` exists.

    Collapsing "可用但相关" onto either neighbour would erase the distinction
    the classification is published to make.
    """
    result = _two_candidate_result()
    confidence = getattr(type(result.confidence), level)
    panel = _panel(qtbot, _project_with_result(replace(result, confidence=confidence)))

    assert panel.confidence_label.property("statusKind") == kind


def test_mcmc_configuration_tracks_selected_candidate_free_dimension(qtbot) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result(), expert=True))

    assert panel.mcmc_config() == api.McmcConfig.standard(17)

    panel.select_candidate("candidate-b")

    assert panel.mcmc_config() == api.McmcConfig.standard(3)


def test_unowned_uncertainty_remains_inspection_only(qtbot) -> None:
    report = replace(_uncertainty(), candidate_id=None)
    result = replace(_two_candidate_result(), uncertainty=report)
    panel = _panel(qtbot, _project_with_result(result, expert=True))

    assert "未归属" in panel.uncertainty_text()
    assert "边界命中（可疑）：scale" not in panel.uncertainty_text()
    assert panel.mcmc_button.isEnabled() is False


def test_invalid_mcmc_config_from_button_is_reported_without_slot_exception(
    qtbot,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result(), expert=True))
    panel.walkers.setValue(5)
    starts: list[object] = []
    monkeypatch.setattr(panel.controller, "start_mcmc", starts.append)

    with qtbot.capture_exceptions() as exceptions:
        panel.mcmc_button.click()

    assert exceptions == []
    assert starts == []
    assert "walkers" in panel.status_text()


def test_mcmc_runs_through_fit_controller_and_adopts_completed_project(
    qtbot,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result(), expert=True))
    original = panel.document.project
    expected = api.McmcConfig.standard(17)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        panel.controller,
        "start_mcmc",
        lambda *args: (calls.append(args), True)[1],
    )
    completed: list[object] = []
    panel.mcmc_completed.connect(completed.append)

    assert panel.start_mcmc() is True
    assert calls == [(original, "curve", "candidate-a", expected)]

    updated = api.set_expert_mode(original, False)
    panel.controller.mcmc_finished.emit(updated)

    assert panel.document.project is updated
    assert completed == [updated]


def test_mcmc_completion_rejects_stale_source_project(
    qtbot,
    monkeypatch,
) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result(), expert=True))
    source = panel.document.project
    monkeypatch.setattr(panel.controller, "start_mcmc", lambda *_args: True)
    failures: list[api.OperationError] = []
    panel.operation_failed.connect(failures.append)
    panel.start_mcmc()
    concurrent = api.select_candidate(source, "curve", "candidate-b")
    panel.document.replace_project(concurrent)

    panel.controller.mcmc_finished.emit(api.set_expert_mode(source, False))

    assert panel.document.project is concurrent
    assert len(failures) == 1
    assert "stale" in failures[0].message


def test_running_mcmc_locks_result_mutations(qtbot) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result(), expert=True))

    panel.controller.running_changed.emit(True)

    assert panel.candidates.isEnabled() is False
    assert panel.clear_button.isEnabled() is False
    assert panel.mcmc_button.isEnabled() is False
    assert panel.cancel_button.isEnabled() is True


@pytest.mark.parametrize(
    ("signal_name", "payload", "status_fragment"),
    (
        ("cancelled", "requested", "requested"),
        (
            "failed",
            api.OperationError("ValueError", "bad chain", "traceback text"),
            "bad chain",
        ),
    ),
    ids=("cancelled", "failed"),
)
def test_mcmc_nonresult_terminal_events_preserve_project(
    qtbot,
    signal_name,
    payload,
    status_fragment,
) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result(), expert=True))
    before = panel.document.project

    getattr(panel.controller, signal_name).emit(payload)

    assert panel.document.project is before
    assert status_fragment in panel.status_text()


def test_results_package_initializer_is_empty() -> None:
    root = Path(__file__).resolve().parents[2]

    assert (root / "src/xrr_fitter/gui/results/__init__.py").read_bytes() == b""


def test_automatic_result_tables_render_point_layers_and_uniformity(
    qtbot,
    monkeypatch,
) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel
    from xrr_fitter.model.automation import (
        AutomaticDatasetSummary,
        AutomaticLayerResult,
        AutomaticResultSummary,
        AutomaticStatus,
        LayerUniformitySummary,
    )

    layer = AutomaticLayerResult(
        "point-1",
        0,
        "Zr",
        120.0,
        4.0,
        3.1e-5,
        2.0e-7,
        0.92,
        6.52,
        1.03,
        6.72,
        None,
    )
    summary = AutomaticResultSummary(
        "batch-9",
        (AutomaticDatasetSummary("point-1", AutomaticStatus.PASSED, True, None, (layer,)),),
        (LayerUniformitySummary("group-1", 0, "Zr", 2, 121.0, 120.0, 122.0, 1.0, 0.83, 1.65),),
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        api,
        "summarize_automatic_results",
        lambda *args: (calls.append(args), summary)[1],
        raising=False,
    )
    document = ProjectDocument(api.new_project())
    panel = ResultsPanel(document)
    qtbot.addWidget(panel)

    points = panel.findChild(QTableWidget, "automaticPointLayerTable")
    uniformity = panel.findChild(QTableWidget, "automaticUniformityTable")
    assert calls == [(document.project,)]
    assert points.columnCount() == 11
    assert points.rowCount() == 1
    assert points.item(0, 0).text() == "point-1"
    assert points.item(0, 1).text() == "通过"
    assert points.item(0, 3).text() == "Zr"
    assert uniformity.columnCount() == 9
    assert uniformity.rowCount() == 1
    assert uniformity.item(0, 0).text() == "group-1"


@pytest.mark.parametrize(
    ("status", "label"),
    (
        ("passed", "通过"),
        ("refining", "精修中"),
        ("review", "需复核"),
        ("failed", "失败"),
    ),
)
def test_automatic_status_labels_are_exact(status, label) -> None:
    from xrr_fitter.gui.results.automatic import automatic_status_text

    assert automatic_status_text(status) == label


def test_confidence_badge_tooltip_explains_classification_reasons(qtbot) -> None:
    # A downgraded result carries machine-readable reasons; the badge must
    # surface their Chinese translation on hover so "why" is answered in place.
    result = replace(
        _two_candidate_result(),
        confidence=type(_two_candidate_result().confidence).CORRELATED,
        classification_evidence=("strong_correlation", "boundary_hit"),
    )
    panel = _panel(qtbot, _project_with_result(result))

    tooltip = panel.confidence_label.toolTip()
    assert "参数强相关" in tooltip
    assert "参数触及边界" in tooltip
    # The accessible description carries the same reasons for screen readers.
    description = panel.confidence_marker.accessibleDescription()
    assert "可用但相关" in description
    assert "参数强相关" in description


def test_mcmc_recommend_button_restores_standard_config_after_hand_edits(qtbot) -> None:
    from PySide6.QtWidgets import QPushButton

    panel = _panel(qtbot, _project_with_result(_two_candidate_result(), expert=True))
    group = panel.mcmc_group
    recommended = api.McmcConfig.standard(17)  # candidate-a has 17 free params

    # The user hand-edits away from the recommended walkers count.
    group.walkers.setValue(recommended.walkers + 4)
    assert group.config().walkers != recommended.walkers

    # Reselecting the same candidate would short-circuit; the button does not.
    button = group.findChild(QPushButton, "mcmcRecommendButton")
    button.click()

    assert group.config() == recommended


def test_mcmc_recommend_button_disabled_while_running(qtbot) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result(), expert=True))
    group = panel.mcmc_group

    group.set_operation_state(running=True, ready=False)
    assert group.recommend_button.isEnabled() is False

    group.set_operation_state(running=False, ready=True)
    assert group.recommend_button.isEnabled() is True
