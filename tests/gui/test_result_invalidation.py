"""Edits that throw fit work away say so; switches that throw all of it away ask first.

Switching the noise model or the batch mode clears every dataset's result and
checkpoint, and both used to do it on one click without a word. Any other edit that
drops a result (a layer, a range, a mask) now leaves a status-bar line naming how
many datasets lost theirs and where to get them back. Undoing back to what is on
disk also reads as clean again instead of always as unsaved.
"""

from __future__ import annotations

import pytest
from tests.gui.test_export_dialog import _fitted_project

import xrr_fitter.api as api
from xrr_fitter.gui.project import dialogs


@pytest.fixture
def drop_prompts(monkeypatch):
    asked: list[tuple[str, int]] = []
    answer = {"value": True}

    def confirm(_parent, change: str, count: int) -> bool:
        asked.append((change, count))
        return answer["value"]

    monkeypatch.setattr(dialogs, "confirm_dropping_results", confirm, raising=False)
    return asked, answer


def _window(qtbot, project):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.show()
    return window


def _results(window) -> list[bool]:
    return [dataset.last_valid_result is not None for dataset in window.document.project.datasets]


def _other_noise_index(selector) -> int:
    return next(index for index in range(selector.count()) if index != selector.currentIndex())


def test_declining_a_noise_switch_keeps_the_mode_and_the_results(qtbot, tmp_path, drop_prompts) -> None:
    window = _window(qtbot, _fitted_project(tmp_path))
    asked, answer = drop_prompts
    answer["value"] = False
    selector = window.fit_panel.noise_selector
    mode = selector.currentData()
    project = window.document.project

    selector.setCurrentIndex(_other_noise_index(selector))

    assert asked == [("切换噪声模式", 1)]
    assert selector.currentData() == mode
    assert window.document.project is project
    assert window.document.is_dirty is False


def test_accepting_a_noise_switch_clears_and_says_how_to_undo(qtbot, tmp_path, drop_prompts) -> None:
    window = _window(qtbot, _fitted_project(tmp_path))
    asked, _answer = drop_prompts
    selector = window.fit_panel.noise_selector

    selector.setCurrentIndex(_other_noise_index(selector))

    assert asked == [("切换噪声模式", 1)]
    assert window.document.project.fit_config.noise_model == selector.currentData()
    assert _results(window) == [False]
    notice = window.statusBar().currentMessage()
    assert "1 个数据集" in notice and "撤销" in notice


def test_a_noise_switch_without_fit_work_asks_nothing(qtbot, tmp_path, drop_prompts) -> None:
    window = _window(qtbot, api.clear_fit_results(_fitted_project(tmp_path), ("curve",)))
    asked, _answer = drop_prompts
    selector = window.fit_panel.noise_selector

    selector.setCurrentIndex(_other_noise_index(selector))

    assert asked == []
    assert window.document.project.fit_config.noise_model == selector.currentData()


@pytest.mark.parametrize("accept", (False, True))
def test_a_batch_switch_over_fitted_datasets_asks_first(qtbot, tmp_path, drop_prompts, accept) -> None:
    window = _window(qtbot, _fitted_project(tmp_path, datasets=2))
    asked, answer = drop_prompts
    answer["value"] = accept
    joint = window.batch_mode_segment.buttons()["joint"]

    joint.click()

    assert asked == [("切换批量模式", 2)]
    assert window.document.project.batch_mode == ("joint" if accept else "independent")
    assert _results(window) == ([False, False] if accept else [True, True])
    # 段控件跟着项目走：拒绝之后亮着的仍是「独立」。
    assert window.batch_mode_segment.value() == window.document.project.batch_mode


def test_an_edit_that_drops_a_result_leaves_an_undo_notice(qtbot, tmp_path) -> None:
    window = _window(qtbot, _fitted_project(tmp_path))
    window.statusBar().clearMessage()

    window.document.replace_project(api.clear_fit_results(window.document.project, ("curve",)))

    notice = window.statusBar().currentMessage()
    assert "1 个数据集" in notice and "撤销" in notice


def test_a_finished_run_replacing_its_result_is_not_a_loss(qtbot, tmp_path) -> None:
    window = _window(qtbot, _fitted_project(tmp_path))
    window.statusBar().clearMessage()

    # 运行把旧结果换掉（或先写一个不带结果的检查点）是它的本分，不是读者丢了东西。
    window.document.adopt_operation_result(api.clear_fit_results(window.document.project, ("curve",)))

    assert window.statusBar().currentMessage() == ""


def test_undo_back_to_the_saved_project_reads_clean(qtbot, tmp_path) -> None:
    window = _window(qtbot, api.new_project())
    path = tmp_path / "saved.xrrproj.json"
    window.save_project(path)
    document = window.document

    document.replace_project(api.set_expert_mode(document.project, True))
    assert document.is_dirty
    document.undo()
    clean_after_undo = document.is_dirty
    document.redo()
    dirty_after_redo = document.is_dirty
    document.undo()

    assert (clean_after_undo, dirty_after_redo, document.is_dirty) == (False, True, False)
    assert window.isWindowModified() is False
