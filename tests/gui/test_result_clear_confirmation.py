"""「清除结果」先问一句，说清只清当前数据集、能撤销，清完再报一声。

审计：点一下就清，没有确认也没有反馈（拟合菜单那条注释说有确认框，实际没有）；文案也不说只清
当前这一个数据集。Ctrl+Z 能撤销，可屏幕上没有任何地方告诉读者。
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox
from tests.gui.test_results import _panel, _project_with_result, _two_candidate_result


def _ask(monkeypatch, answer):
    asked: list[str] = []

    def question(_parent, _title, text, *_args):
        asked.append(text)
        return answer

    monkeypatch.setattr(QMessageBox, "question", question)
    return asked


def test_declining_the_confirmation_keeps_the_result(qtbot, monkeypatch) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result()))
    before = panel.document.project
    asked = _ask(monkeypatch, QMessageBox.StandardButton.No)

    panel.clear_button.click()

    assert len(asked) == 1
    assert "curve" in asked[0] and "其他数据集" in asked[0] and "撤销" in asked[0]
    assert panel.document.project is before


def test_confirming_clears_only_the_current_dataset_and_says_so(qtbot, monkeypatch) -> None:
    panel = _panel(qtbot, _project_with_result(_two_candidate_result()))
    _ask(monkeypatch, QMessageBox.StandardButton.Yes)

    panel.clear_button.click()

    assert panel.document.project.datasets[0].last_valid_result is None
    assert "已清除 curve 的拟合结果" in panel.status_text()
    assert "撤销" in panel.status_text()
    assert "当前数据集" in panel.clear_button.toolTip()
