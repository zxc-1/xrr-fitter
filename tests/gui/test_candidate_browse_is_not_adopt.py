"""浏览候选与采用候选是两件事：只有「采用」才改项目。

原先点一行（或按方向键走到一行）就 ``api.select_candidate``：导出换了解、文件变脏、撤销栈多一格，
而读者只是想看看 B 长什么样。现在点行只是看；「采用此候选」那一枚才改项目。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from tests.support.model_cases import dataset_project, final_fit_result, fit_candidate, project

import xrr_fitter.api as api
from xrr_fitter.model.analysis import FitResult
from xrr_fitter.model.fitting import FitSearchResult


def _result():
    """推荐解排在输入的最后——求解器按阶段产出，最好的那个往往是最后一个 E 阶段候选。"""
    candidates = (
        replace(fit_candidate("E-0", 0.9), ranking_objective=0.9),
        replace(fit_candidate("E-1", 0.5), ranking_objective=0.5),
        replace(fit_candidate("E-2", 0.2), ranking_objective=0.2),
    )
    search = FitSearchResult(
        parameter_definitions=(),
        candidates=candidates,
        best_index=2,
        warnings=(),
        child_seeds=(101,),
        stage_summaries=(),
        region_labels=np.zeros(4, dtype=int),
        region_weights=np.ones(4),
    )
    return FitResult.from_search(search, confidence=api.ConfidenceClass.TRUSTED, uncertainty=None)


def _panel(qtbot, result=None):
    from xrr_fitter.gui.document import ProjectDocument
    from xrr_fitter.gui.results.panel import ResultsPanel

    value = replace(
        project(dataset_project(result=_result() if result is None else result)), base_directory="/private/tmp"
    )
    document = ProjectDocument(api.select_active_dataset(value, "curve"))
    panel = ResultsPanel(document)
    qtbot.addWidget(panel)
    panel.show()
    return panel


def _row_of(panel, candidate_id: str) -> int:
    from PySide6.QtCore import Qt

    return next(
        row
        for row in range(panel.candidates.count())
        if panel.candidates.item(row).data(Qt.ItemDataRole.UserRole) == candidate_id
    )


def test_browsing_a_candidate_changes_neither_the_project_nor_its_dirty_flag(qtbot) -> None:
    panel = _panel(qtbot)
    document = panel.document
    before = document.project

    panel.candidates.setCurrentRow(_row_of(panel, "E-0"))

    assert panel.selected_candidate_id() == "E-0"
    assert document.project is before
    assert document.is_dirty is False
    assert document.can_undo is False


def test_adopting_the_viewed_candidate_is_the_one_edit(qtbot) -> None:
    panel = _panel(qtbot)
    document = panel.document
    adopted: list[str] = []
    panel.candidate_selected.connect(adopted.append)
    panel.candidates.setCurrentRow(_row_of(panel, "E-1"))

    panel.adopt_button.click()

    assert document.project.ui_state.selected_candidate_ids == (("curve", "E-1"),)
    assert document.is_dirty is True
    assert adopted == ["E-1"]


def test_the_adopt_button_says_why_it_is_greyed(qtbot) -> None:
    invalid = replace(fit_candidate("E-bad", 0.1), valid=False, objective=float("inf"), stop_reason="max_nfev")
    panel = _panel(qtbot, final_fit_result(fit_candidate("E-0", 0.2), invalid))

    assert panel.adopt_button.isEnabled() is False
    assert "已采用" in panel.adopt_button.toolTip()

    panel.candidates.setCurrentRow(_row_of(panel, "E-bad"))

    assert panel.adopt_button.isEnabled() is False
    assert "不能采用" in panel.adopt_button.toolTip()


def test_the_recommended_candidate_is_listed_first(qtbot) -> None:
    from xrr_fitter.gui.results.candidate_row import CANDIDATE_ROW_ROLE

    panel = _panel(qtbot)

    assert panel.recommended_candidate_id() == "E-2"
    assert _row_of(panel, "E-2") == 0
    assert panel.candidates.item(0).data(CANDIDATE_ROW_ROLE).title == "候选 A"
    assert [_row_of(panel, value) for value in ("E-1", "E-0")] == [1, 2]


def test_only_the_graded_row_carries_a_verdict_and_the_rest_read_as_ungraded(qtbot) -> None:
    """「多解」是一个判定：说的是还有别的解跟它一样好。没被判过的行不能一律这么写。"""
    from xrr_fitter.gui.results.candidate_row import CANDIDATE_ROW_ROLE

    panel = _panel(qtbot)
    states = {panel.candidates.item(row).data(CANDIDATE_ROW_ROLE).state for row in range(1, panel.candidates.count())}

    assert panel.candidates.item(0).data(CANDIDATE_ROW_ROLE).state == "可信"
    assert states == {"未判定"}
