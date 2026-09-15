"""Status bar contract: readiness, quality, metrics, stage, position, batch mode."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

import xrr_fitter.api as api


def _write_curve(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{0.05 + index * 0.02:.6f} {1000.0 / (index + 1):.12g}" for index in range(64)) + "\n",
        encoding="utf-8",
    )
    return path


def _span_text(window, name: str) -> str:
    """一段状态栏读出来是什么样：说明文字与取值按排版顺序连起来。

    设计稿的一段是 ``活动数据集：<b>aSi_ML_25C</b>``——说明文字与取值分属两个标签，
    但读者看见的是一句话。断言写在这句话上，就不必替实现记住「冒号归哪个标签」，
    换了拆法也不会误报。
    """
    from xrr_fitter.gui.status_bar import StatusSpan

    span = window.findChild(StatusSpan, name)
    assert span is not None, name
    layout = span.layout()
    pieces = []
    for index in range(layout.count()):
        label = layout.itemAt(index).widget()
        if isinstance(label, QLabel):
            pieces.append(label.text())
    return "".join(pieces)


def _fitted_window(qtbot, tmp_path, confidence, objective):
    """A window whose one dataset already carries a persisted fit result."""
    from dataclasses import replace

    import numpy as np
    from tests.support.model_cases import fit_candidate, fit_result

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(
        api.new_project(),
        _write_curve(tmp_path / "q.xy"),
        api.InstrumentSpec(),
    )
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("SiO2", "SiO2", 2.20), 40.0),),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    project = api.set_structure(project, "q", structure)
    size = 64
    candidate = replace(
        fit_candidate("candidate-a", objective),
        qz_a_inv=np.linspace(0.015, 0.25, size),
        model_normalized=np.geomspace(0.9, 2e-5, size),
        log_residuals_decades=np.full(size, 0.1),
        residuals=np.full(size, 0.1),
        weighted_residuals=np.zeros(size),
    )
    result = api.FitResult.from_search(
        fit_result(candidate),
        confidence=confidence,
        uncertainty=None,
        classification_evidence=(),
    )
    project = replace(project, datasets=(replace(project.datasets[0], last_valid_result=result),))
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.show()
    return window


class _LiveJob:
    """The least a controller accepts as a live operation."""

    def cancel(self) -> None:
        return None

    def force_stop(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_readiness_is_reported_once_across_status_bar_and_fit_panel(qtbot, tmp_path) -> None:
    """The same readiness verdict must not occupy two labels at once."""
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.new_project()
    project = api.add_dataset(project, _write_curve(tmp_path / "e.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("SiO2", "SiO2", 2.20), 40.0),),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    project = api.set_structure(project, "e", structure)
    project = api.set_expert_mode(project, True)
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.show()

    bar_text = window.findChild(QLabel, "fitReadinessStatus").text()
    assert "就绪" in bar_text
    assert window.fit_panel.status_text() != bar_text


def test_status_bar_reports_the_quality_of_the_fit_that_already_ran(qtbot, tmp_path) -> None:
    """Readiness answers "can I start", never "was the last answer any good"."""
    window = _fitted_window(qtbot, tmp_path, api.ConfidenceClass.UNTRUSTED, 12.5)

    quality = window.findChild(QLabel, "fitQualityStatus")
    metrics = window.findChild(QLabel, "fitMetricsStatus")
    assert quality is not None and metrics is not None
    assert quality.text() == "不可信"
    assert _span_text(window, "statusVerdictSpan") == "拟合结果：■不可信"
    assert metrics.text().startswith("J = 12.5")
    assert "就绪" in window.findChild(QLabel, "fitReadinessStatus").text()


def test_quality_status_carries_the_semantic_kind_matching_its_verdict(qtbot, tmp_path) -> None:
    """An untrusted result must not be painted in the same colour as a trusted one."""
    untrusted = _fitted_window(qtbot, tmp_path / "a", api.ConfidenceClass.UNTRUSTED, 12.5)
    trusted = _fitted_window(qtbot, tmp_path / "b", api.ConfidenceClass.TRUSTED, 0.02)

    assert untrusted.findChild(QLabel, "fitQualityStatus").property("statusKind") == "error"
    assert trusted.findChild(QLabel, "fitQualityStatus").property("statusKind") == "ok"


def test_quality_status_stays_empty_until_a_fit_has_run(qtbot) -> None:
    """With no result there is no verdict, so the label claims nothing."""
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    assert window.findChild(QLabel, "fitQualityStatus").text() == ""


def test_quality_dot_encodes_the_verdict_as_a_shape_not_only_a_colour(qtbot, tmp_path) -> None:
    """四个判定各有颜色，字形是给看不出颜色的人留的第二条线索。"""
    from xrr_fitter.gui.theme import CONFIDENCE_COMPARISON_KINDS, CONFIDENCE_GLYPHS

    correlated = _fitted_window(qtbot, tmp_path / "a", api.ConfidenceClass.CORRELATED, 0.3)
    multiple = _fitted_window(qtbot, tmp_path / "b", api.ConfidenceClass.MULTIPLE, 0.4)

    correlated_dot = correlated.findChild(QLabel, "fitQualityDot")
    multiple_dot = multiple.findChild(QLabel, "fitQualityDot")

    assert correlated_dot.property("statusKind") == CONFIDENCE_COMPARISON_KINDS["可用但相关"] == "info"
    assert multiple_dot.property("statusKind") == CONFIDENCE_COMPARISON_KINDS["多解"] == "warn"
    assert correlated_dot.text() == CONFIDENCE_GLYPHS["可用但相关"]
    assert multiple_dot.text() == CONFIDENCE_GLYPHS["多解"]


def test_quality_dot_glyph_is_sourced_from_theme(qtbot, tmp_path, monkeypatch) -> None:
    """The bar reads theme's shapes rather than keeping literals of its own."""
    import xrr_fitter.gui.theme as theme

    monkeypatch.setitem(theme.CONFIDENCE_GLYPHS, "可信", "✦")
    window = _fitted_window(qtbot, tmp_path, api.ConfidenceClass.TRUSTED, 0.02)

    assert window.findChild(QLabel, "fitQualityDot").text() == "✦"


def test_the_ready_line_reads_the_two_characters_the_design_prints(qtbot, tmp_path) -> None:
    """设计稿帧① 底栏第一段是 ``就绪``（HTML 474 行），不是一句邀请。"""
    from xrr_fitter.gui import messages

    window = _fitted_window(qtbot, tmp_path, api.ConfidenceClass.TRUSTED, 0.02)
    label = window.findChild(QLabel, "fitReadinessStatus")

    assert label.text() == "就绪"
    assert messages.READY_TEXT == "就绪"


def test_a_narrowed_fit_range_does_not_put_english_in_the_status_bar(qtbot, tmp_path) -> None:
    """把拟合范围拖窄到掩码里剩不下 30 点，底栏第一段照原样写出了一句英文。

    这条路一点也不偏门：范围滑块就在数据面板上，拖过头 ``fit/problem.py`` 便抛
    ``ValueError("current fit mask is not fit-ready")``，preflight 拿 ``str(error)`` 当消息，
    投影层没登记过它就把原文送上屏。设计稿五帧的这一段一律是中文短句（``就绪`` / ``引导模式``
    / ``结构已修改（未保存）`` / ``拟合进行中`` / ``MCMC 采样中``），30px 高的底栏也放不下一句
    英文诊断——它还要和右边的判定、J / χ²ᵥ 两段读数分同一行。

    这一段的 tooltip 归无障碍描述（``拟合就绪状态``），所以原文没有第二行可以退：译文自己得把
    「怎么办」说到位。
    """
    import numpy as np

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(api.new_project(), _write_curve(tmp_path / "n.xy"), api.InstrumentSpec())
    structure = api.StructureSpec(
        api.MaterialSpec("Air", None, None, 0.0j),
        (api.LayerSpec("film", api.MaterialSpec("SiO2", "SiO2", 2.20), 40.0, roughness_a=3.0),),
        api.MaterialSpec("Si", "Si", 2.329),
    )
    project = api.set_structure(project, "n", structure)
    mask = np.zeros(64, dtype=bool)
    mask[:10] = True
    project = api.set_fit_mask(project, "n", mask)
    assert api.preflight_fit(project).message == "current fit mask is not fit-ready", "这条路已经换了消息"

    window = MainWindow(ProjectDocument(api.select_active_dataset(project, "n")))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.show()
    qtbot.wait(1)

    text = window.findChild(QLabel, "fitReadinessStatus").text()
    assert "范围" in text, text
    assert not any("a" <= character.lower() <= "z" for character in text), text


def test_readiness_line_announces_a_live_run_instead_of_inviting_another(qtbot, tmp_path) -> None:
    """A running fit must not leave the bar reading 就绪."""
    from xrr_fitter.gui import messages
    from xrr_fitter.gui.operation_state import refresh_operation_state

    window = _fitted_window(qtbot, tmp_path, api.ConfidenceClass.TRUSTED, 0.02)
    label = window.findChild(QLabel, "fitReadinessStatus")
    dot = window.findChild(QLabel, "fitReadinessDot")
    assert label.text() == messages.READY_TEXT

    window.fit_panel.controller._job = _LiveJob()
    refresh_operation_state(window)

    assert label.text() == "拟合进行中"
    assert dot.property("statusKind") == "accent", "设计稿帧④ 那颗圆点是 --accent"

    window.fit_panel.controller._job = None
    refresh_operation_state(window)
    assert label.text() == messages.READY_TEXT
    assert dot.property("statusKind") == "ok"


def test_readiness_line_names_which_operation_holds_the_window(qtbot, tmp_path) -> None:
    """MCMC sampling is not a fit, so the line may not report it as one."""
    from xrr_fitter.gui.operation_state import refresh_operation_state

    window = _fitted_window(qtbot, tmp_path, api.ConfidenceClass.TRUSTED, 0.02)
    window.result_panel.controller._job = _LiveJob()
    refresh_operation_state(window)
    text = window.findChild(QLabel, "fitReadinessStatus").text()
    window.result_panel.controller._job = None

    assert "MCMC" in text
    assert "拟合进行中" not in text


def test_status_bar_carries_the_running_stage_and_position(qtbot, tmp_path) -> None:
    """Frame ④ puts 阶段 4 / 9 and 进度 620 / 1000 in the bar, not only in the card."""
    from xrr_fitter.gui.operation_state import refresh_operation_state
    from xrr_fitter.gui.status_bar import StatusSpan

    window = _fitted_window(qtbot, tmp_path, api.ConfidenceClass.TRUSTED, 0.02)
    stage_span = window.findChild(StatusSpan, "statusStageSpan")
    position = window.findChild(QLabel, "fitPositionStatus")
    assert stage_span is not None and position is not None
    assert stage_span.isVisibleTo(window) is False
    assert position.text() == ""

    window.fit_panel.controller.progress_changed.emit(api.FitProgress("q", "B", 5, 10, 1.0, "search"))

    assert _span_text(window, "statusStageSpan") == "阶段 2 / 9"
    assert position.text() == "进度 230 / 1000"

    refresh_operation_state(window)
    assert position.text() == ""
    assert stage_span.isVisibleTo(window) is False


def test_the_status_bar_names_the_batch_mode(qtbot) -> None:
    """设计稿帧④ 的状态栏有「模式：联合批量」这一段。

    批量模式决定整屏参数表读作共享还是独立，是一次运行最要紧的前提；命令栏那对
    独立/联合按钮在引导模式下不露面，于是这个前提在屏幕上没有任何地方说得出来。

    说明文字「模式：」住在段里，取值是它右边那个加粗标签——设计稿写的是
    ``模式：<b>联合批量</b>``，所以断言读整段。
    """
    from dataclasses import replace

    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    label = window.findChild(QLabel, "batchModeStatus")
    assert label is not None
    assert _span_text(window, "statusBatchSpan") == "模式：独立批量"

    window.document.replace_project(replace(window.document.project, batch_mode="joint"))

    assert _span_text(window, "statusBatchSpan") == "模式：联合批量"
