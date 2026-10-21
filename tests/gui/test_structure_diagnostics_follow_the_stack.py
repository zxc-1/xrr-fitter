"""结构诊断只在有膜层可诊断时谈膜层。

审计实测：项目里一层膜都没有时，右栏「结构诊断」仍写着「厚度 d 与密度 ρ 可能相关」——一句静态
文案，说的是一个还不存在的东西。这里钉住：没有膜层时换成一句说明还没有可诊断的东西；建了膜层
之后换回相关提示。
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel
from tests.gui.control_operation_driver import setup_project

import xrr_fitter.api as api


def _hint(qtbot, project) -> tuple[object, QLabel]:
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(project))
    qtbot.addWidget(window)
    return window, window.findChild(QLabel, "structureCorrelationHint")


def test_no_layers_means_nothing_to_diagnose_yet(qtbot, tmp_path) -> None:
    _window, hint = _hint(qtbot, setup_project(tmp_path, layers=0))

    assert "厚度 d 与密度 ρ" not in hint.text()
    assert "尚无膜层" in hint.text()


def test_the_hint_returns_once_a_layer_exists(qtbot, tmp_path) -> None:
    window, hint = _hint(qtbot, setup_project(tmp_path, layers=0))
    (tmp_path / "with-layer").mkdir()
    structured = setup_project(tmp_path / "with-layer", layers=1)

    dataset_id = window.document.active_dataset_id
    structure = structured.datasets[0].structure
    window.document.replace_project(api.set_structure(window.document.project, dataset_id, structure))

    assert "厚度 d 与密度 ρ 可能相关" in hint.text()
