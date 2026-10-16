"""拟合卡上「本次」那一行由窗口按项目写出，项目一变就跟着改。

卡片会念路线（``FitPanel.set_route``），路线由主窗口算（``fit_route``）；但没有谁把两头接起来，
那一行就一直藏着。批处理方式换成联合，路线跟着换，卡片也得跟着换，否则念的是上一条路线。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel
from tests.gui.test_expert_views import _project, _write_curve

import xrr_fitter.api as api


def _route_line(window) -> QLabel:
    line = window.fit_panel.findChild(QLabel, "fitRouteLine")
    assert line is not None
    return line


def _expected(window) -> str:
    title, detail = window.fit_route()
    return f"本次：{title} — {detail}"


def test_the_fit_card_names_the_route_the_window_works_out(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)

    assert not _route_line(window).isHidden()
    assert _route_line(window).text() == _expected(window)


def test_the_route_line_follows_a_switch_that_changes_the_route(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    project = api.add_dataset(_project(tmp_path), _write_curve(tmp_path / "second.xy"), api.InstrumentSpec())
    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    before = window.fit_route()

    assert window.fit_panel.set_batch_mode("joint")

    assert window.fit_route() != before, "前提：联合拟合是另一条路线"
    assert _route_line(window).text() == _expected(window)
