"""掩膜模式下点一次排除、再点一次放回。

审计实测：图上的掩膜点击只会把点设成「排除」（``_plot_point_requested`` 恒传 ``False``），
点错了没有办法在图上放回，只能撤销或去别处改。这里钉住：同一个点点两下回到原样。
"""

from __future__ import annotations

from tests.gui.control_operation_driver import setup_project


def test_a_second_click_puts_the_point_back(qtbot, tmp_path) -> None:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(setup_project(tmp_path)))
    qtbot.addWidget(window)
    index = window.document.project.datasets[0].fit_mask.index(True)
    window.plot_panel.set_interaction_mode("mask")

    window.plot_panel.request_point_mask(index)
    assert window.document.project.datasets[0].fit_mask[index] is False

    window.plot_panel.request_point_mask(index)
    assert window.document.project.datasets[0].fit_mask[index] is True
