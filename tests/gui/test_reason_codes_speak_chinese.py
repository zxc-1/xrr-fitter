"""服务层的原因码上屏时先说中文，码本身留在括号里。

审计实测：「相关不可用：rank_deficient」「不可用原因：boundary_solution」「已取消：requested」
直接把机器码摆在中文界面上。分类证据早就是「中文（码）」的写法（``CLASSIFICATION_LABELS``），
这里把同一个写法推广到不可用原因与停止原因：已登记的码给一句解释，未登记的码仍然原样可查。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import xrr_fitter.api as api


@pytest.mark.parametrize(
    ("code", "fragment"),
    (
        ("rank_deficient", "信息矩阵不满秩"),
        ("boundary_solution", "边界"),
        ("residual_diagnostics_not_executed", "残差诊断未执行"),
        ("insufficient_successful_samples", "成功样本不足"),
    ),
)
def test_known_reason_codes_gain_a_chinese_explanation(code, fragment) -> None:
    from xrr_fitter.gui import messages

    text = messages.reason_text(code)

    assert fragment in text
    assert f"（{code}）" in text


def test_an_unregistered_code_is_kept_verbatim() -> None:
    from xrr_fitter.gui import messages

    assert "some_new_code" in messages.reason_text("some_new_code")


def test_correlation_unavailability_speaks_chinese() -> None:
    from xrr_fitter.gui.results.inference_text import correlation_unavailable_reason

    evidence = SimpleNamespace(matrix=None, unavailable_reason="rank_deficient")
    report = SimpleNamespace(covariance_evidence=evidence)

    assert correlation_unavailable_reason(report).startswith("信息矩阵不满秩")


def test_a_cancelled_fit_says_why_in_chinese(qtbot, tmp_path) -> None:
    from tests.gui.test_fit_progress import _project

    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.fitting.panel import FitPanel

    panel = FitPanel(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(panel)

    panel.controller.cancelled.emit("requested")

    assert "requested" not in panel.status_text()
    assert "按请求停止" in panel.status_text()


def test_a_force_stopped_chain_says_so_in_chinese(qtbot) -> None:
    from tests.gui.test_results import _panel, _project_with_result, _two_candidate_result

    panel = _panel(qtbot, _project_with_result(_two_candidate_result(), expert=True))

    panel.controller.cancelled.emit("force_stop")

    assert "force_stop" not in panel.status_text()
    assert "强制" in panel.status_text()
    assert isinstance(panel.document.project, api.XrrProject)
