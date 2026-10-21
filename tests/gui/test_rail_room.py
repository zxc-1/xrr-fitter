"""结果步上的左栏：数据集列表不被管线挤到一两行（F19），一栏里只有一套步骤序号（F47）。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QLabel, QScrollArea, QToolButton, QWidget

import xrr_fitter.api as api

AIR = api.MaterialSpec("Air", None, None, 0.0j)
SI = api.MaterialSpec("Si", "Si", 2.329)
SIO2 = api.MaterialSpec("SiO2", "SiO2", 2.20)
SIZE = 64


def _project(tmp_path: Path, datasets: int = 5):
    from tests.support.model_cases import fit_candidate, fit_result

    project = api.new_project()
    for index in range(datasets):
        path = tmp_path / f"curve{index}.xy"
        path.write_text(
            "\n".join(f"{0.05 + i * 0.02:.6f} {1000.0 / (i + 1):.12g}" for i in range(SIZE)) + "\n", encoding="utf-8"
        )
        project = api.add_dataset(project, path, api.InstrumentSpec())
    first = project.datasets[0].dataset_id
    layer = api.LayerSpec("film", SIO2, 40.0, roughness_a=3.0)
    project = api.set_structure(project, first, api.StructureSpec(AIR, (layer,), SI))
    candidate = replace(
        fit_candidate("candidate-a", 1.83),
        qz_a_inv=np.linspace(0.015, 0.25, SIZE),
        model_normalized=np.geomspace(0.9, 2e-5, SIZE),
        log_residuals_decades=np.full(SIZE, 0.1),
        residuals=np.full(SIZE, 0.1),
        weighted_residuals=np.zeros(SIZE),
    )
    result = api.FitResult.from_search(
        fit_result(candidate), confidence=api.ConfidenceClass.TRUSTED, uncertainty=None, classification_evidence=()
    )
    datasets = (replace(project.datasets[0], last_valid_result=result), *project.datasets[1:])
    return api.select_active_dataset(replace(project, datasets=datasets), first)


def _window(qtbot, tmp_path, size):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.main_window import MainWindow

    window = MainWindow(ProjectDocument(_project(tmp_path)))
    qtbot.addWidget(window)
    window.set_guidance_visible(False)
    window.resize(*size)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(20)
    assert window.pipeline_nav.methods_visible()
    return window


def _bottom_in(widget, point: QPoint, scroll: QScrollArea) -> int:
    return widget.mapTo(scroll.viewport(), point).y()


@pytest.mark.parametrize("size", ((1280, 800), (1272, 693)))
def test_three_datasets_and_the_six_steps_stay_on_screen_on_the_result_step(qtbot, tmp_path, size) -> None:
    window = _window(qtbot, tmp_path, size)
    data_scroll = window.nav_column.findChild(QScrollArea, "dataPanelScroll")
    pipe_scroll = window.nav_column.findChild(QScrollArea, "pipelineNavScroll")
    tree = window.data_panel.tree
    third = tree.visualRect(tree.model().index(2, 0))

    assert _bottom_in(tree.viewport(), third.bottomLeft(), data_scroll) <= data_scroll.viewport().height(), (
        data_scroll.height()
    )
    last_step = window.pipeline_nav.step_rows()[-1]
    assert _bottom_in(last_step, last_step.rect().bottomLeft(), pipe_scroll) <= pipe_scroll.viewport().height()


def test_the_uncertainty_methods_fold_away(qtbot, tmp_path) -> None:
    window = _window(qtbot, tmp_path, (1280, 800))
    nav = window.pipeline_nav
    toggle = nav.findChild(QToolButton, "uncertaintyMethodToggle")
    assert toggle is not None and toggle.accessibleName()
    before = nav.sizeHint().height()

    toggle.click()

    assert not any(label.isVisibleTo(nav) for label in nav.method_labels)
    assert nav.method_heading.isVisibleTo(nav), "抬头和开关留着，否则折起来就展不回去"
    assert nav.sizeHint().height() < before
    toggle.click()
    assert all(label.isVisibleTo(nav) for label in nav.method_labels)


def _markers(host: QWidget, *, outside: QWidget | None = None) -> list[str]:
    """``host`` 里每枚步骤圆点上的字；给了 ``outside`` 就跳过它里面的那些。"""
    return [
        label.text()
        for label in host.findChildren(QLabel)
        if label.property("stepDot") and (outside is None or not outside.isAncestorOf(label))
    ]


def test_one_rail_counts_its_steps_once(qtbot, tmp_path) -> None:
    """F47：结果步上六步地图和四种方法同在一栏，两段都数 1、2、3……读者分不清「3」是第三步还是
    第三种方法。序号只归六步；四种方法是第 6 步展开出来的子项，记作 a–d。"""
    window = _window(qtbot, tmp_path, (1280, 800))
    methods = window.pipeline_nav.findChild(QWidget, "uncertaintyMethodNav")

    assert _markers(methods) == ["a", "b", "c", "d"]
    assert _markers(window.pipeline_nav, outside=methods)[-1] == "6"
