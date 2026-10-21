"""坐标读数用发出它的那张卡自己的轴名，而不是此刻选中的那一页的。

审计实测：残差卡钉在反射率页下面，指针停在残差卡上时读数却借用当前 tab 的轴名——选着 qz⁴·R 页就写成
「散射矢量 qz … · qz⁴R …」；选着分析页时干脆什么都不写。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel
from tests.gui.plot_support import _panel, _result
from tests.support.model_cases import prepared_data


def _readout(panel) -> str:
    return panel.findChild(QLabel, "plotCursorReadout").text()


def _fitted_panel(qtbot):
    data = prepared_data(size=4)
    return _panel(qtbot, data=data, result=_result(data))


def test_the_readout_names_the_axes_of_the_pane_under_the_pointer(qtbot) -> None:
    panel = _fitted_panel(qtbot)
    panel.select_view("qz4")
    residual = panel.view("residual")
    xlabel, ylabel = residual.axis_labels()

    residual.cursor_moved.emit(1.5, 0.25)

    assert _readout(panel) == f"{xlabel} 1.5 · {ylabel} 0.25"


def test_the_residual_card_reads_out_while_an_analysis_page_is_selected(qtbot) -> None:
    panel = _fitted_panel(qtbot)
    panel.select_view("candidates")
    residual = panel.view("residual")
    xlabel, ylabel = residual.axis_labels()

    residual.cursor_moved.emit(1.5, 0.25)

    assert _readout(panel) == f"{xlabel} 1.5 · {ylabel} 0.25"
