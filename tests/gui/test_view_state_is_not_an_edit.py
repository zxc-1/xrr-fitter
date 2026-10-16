"""翻页、换数据集、拖分隔条是「看哪里」，不是「改了什么」。

审计实测：每切一次 tab，文档就置脏、撤销栈多一条、redo 被清空，撤销还把读者拽回改动那一刻
停着的那一页。这里钉住四件事：看哪里的变化不置脏、不入撤销栈、不清 redo；真正的编辑被撤销
或重做时，读者此刻停在哪一页保持不动；后台运行中途写回的检查点不把页面拽回运行开始时那一页，
一次运行在撤销栈里只占一格；文档层不接受夹带内容改动的「视图更新」。
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.gui.control_operation_driver import setup_project

import xrr_fitter.api as api


def _window(qtbot, project):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def _settled(window, tmp_path):
    """存一次盘：之后的脏标记与撤销栈只记录测试自己做的动作。"""
    window.save_project(tmp_path / "view.xrrproj.json")
    assert window.document.is_dirty is False
    assert window.document.can_undo is False


def test_switching_the_plot_page_is_not_an_edit(qtbot, tmp_path) -> None:
    window = _window(qtbot, setup_project(tmp_path, datasets=2))
    _settled(window, tmp_path)

    window.plot_panel.select_view("raw")

    assert window.document.project.ui_state.plot_tab_index == 1
    assert window.document.is_dirty is False
    assert window.document.can_undo is False


def test_switching_the_active_dataset_is_not_an_edit(qtbot, tmp_path) -> None:
    window = _window(qtbot, setup_project(tmp_path, datasets=2))
    _settled(window, tmp_path)

    window.select_active_dataset("curve1")

    assert window.document.active_dataset_id == "curve1"
    assert window.document.is_dirty is False
    assert window.document.can_undo is False


def test_looking_around_after_an_undo_keeps_the_redo(qtbot, tmp_path) -> None:
    window = _window(qtbot, setup_project(tmp_path, datasets=2))
    _settled(window, tmp_path)
    assert window.fit_panel.set_batch_mode("joint") is True
    assert window.document.undo() is True

    window.plot_panel.select_view("raw")
    window.select_active_dataset("curve1")

    assert window.document.can_redo is True
    assert window.document.redo() is True
    assert window.document.project.batch_mode == "joint"


def test_undo_reverts_the_edit_without_moving_the_reader(qtbot, tmp_path) -> None:
    window = _window(qtbot, setup_project(tmp_path, datasets=2))
    _settled(window, tmp_path)
    assert window.fit_panel.set_batch_mode("joint") is True
    window.plot_panel.select_view("raw")

    assert window.document.undo() is True

    assert window.document.project.batch_mode == "independent"
    assert window.document.project.ui_state.plot_tab_index == 1
    assert window.plot_panel.current_view_key() == "raw"


def test_a_page_change_still_reaches_the_saved_file(qtbot, tmp_path) -> None:
    window = _window(qtbot, setup_project(tmp_path, datasets=2))
    _settled(window, tmp_path)
    window.plot_panel.select_view("raw")

    target = tmp_path / "again.xrrproj.json"
    window.save_project(target)

    assert api.load_project(target).ui_state.plot_tab_index == 1


def test_running_checkpoints_keep_the_page_and_share_one_undo_step(qtbot, tmp_path) -> None:
    window = _window(qtbot, setup_project(tmp_path, datasets=2))
    _settled(window, tmp_path)
    started = window.document.project
    window.plot_panel.select_view("raw")
    # 运行开始那一刻的项目停在第 0 页；检查点是它的后代，内容变了、视图没跟上读者。
    first = api.set_fit_config(started, replace(started.fit_config, noise_model="gaussian"))
    second = api.set_fit_config(started, replace(started.fit_config, noise_model="poisson"))

    window.fit_panel.controller.checkpoint_ready.emit(first)
    window.fit_panel.controller.checkpoint_ready.emit(second)

    assert window.document.project.fit_config.noise_model == "poisson"
    assert window.document.project.ui_state.plot_tab_index == 1
    assert window.document.is_dirty is True
    assert window.document.undo() is True
    assert window.document.project.fit_config.noise_model == started.fit_config.noise_model
    assert window.document.can_undo is False


def _project_without_files() -> api.XrrProject:
    from tests.support.model_cases import dataset_project

    value = api.XrrProject.new((dataset_project("curve"), dataset_project("other")), master_seed=1201)
    value = replace(value, base_directory="/private/tmp")
    return api.select_active_dataset(value, "curve")


def test_the_document_refuses_a_view_update_that_carries_an_edit() -> None:
    from xrr_fitter.gui.document import ProjectDocument

    project = _project_without_files()
    document = ProjectDocument(project)
    edited = api.set_batch_mode(project, "joint")

    with pytest.raises(ValueError, match="view"):
        document.apply_view(edited)

    assert document.project is project
