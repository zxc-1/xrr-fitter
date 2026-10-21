"""Seeds that converged to the same solution say so instead of reading as rivals.

A single-layer fit keeps eleven candidates; three C seeds and three D seeds land on
the same parameters to within 1e-6 of each bound span, and the list read them as
six alternatives. Rows are kept (each seed is still selectable and named by its own
ID on hover) and the later twin names the earlier one. Equal J with different
parameters is not a twin: four E seeds share J to nine digits while one parameter
moves across a tenth of its span -- that parameter is unidentified, and folding the
rows would hide exactly that.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.support.model_cases import fit_candidate


def _candidate(candidate_id: str, objective: float, position: float):
    return replace(fit_candidate(candidate_id, objective), unit_vector=np.array([position]))


def _rows(qtbot, candidates):
    from xrr_fitter.gui.results.candidate_row import CANDIDATE_ROW_ROLE
    from xrr_fitter.gui.results.candidates import CandidateList

    widget = CandidateList()
    qtbot.addWidget(widget)
    widget.load(candidates, selected_id=candidates[0].candidate_id, recommended_id=candidates[0].candidate_id)
    return [(widget.item(row).data(CANDIDATE_ROW_ROLE), widget.item(row).toolTip()) for row in range(widget.count())]


def test_a_seed_that_landed_on_an_earlier_solution_names_it(qtbot) -> None:
    rows = _rows(
        qtbot,
        (
            _candidate("E-2", 0.5, 0.30),
            _candidate("C-0-0", 1.6785, 0.70),
            _candidate("C-0-1", 1.6785, 0.70 + 2e-9),
            _candidate("C-0-2", 1.6785, 0.70 - 3e-9),
        ),
    )

    assert [row.title for row, _tip in rows] == ["候选 A", "候选 B", "候选 C", "候选 D"]
    assert [row.twin for row, _tip in rows] == ["", "", "同候选 B", "同候选 B"]
    # 悬停仍写求解器 ID：哪条种子收敛到了哪一个解，对照日志时要查得到。
    assert "C-0-1" in rows[2][1] and "C-0-0" in rows[2][1]


def test_equal_objectives_with_different_parameters_are_not_twins(qtbot) -> None:
    rows = _rows(
        qtbot,
        (
            _candidate("E-2", 0.0075873, 0.30),
            _candidate("E-0", 0.0075873, 0.40),
        ),
    )

    assert [row.twin for row, _tip in rows] == ["", ""]


def _failed(candidate_id: str):
    # 无效求值只用 +inf 作哨兵（``_validate_candidate_objective``），valid 与 J 得一起换。
    return replace(_candidate(candidate_id, 1.0, 0.70), valid=False, objective=float("inf"), stop_reason="invalid")


def _parameterless(candidate_id: str):
    return replace(_failed(candidate_id), parameters=(), unit_vector=np.array([]))


def _eliminated(candidate_id: str):
    return replace(_candidate(candidate_id, 2.5, 0.70), stop_reason="early_eliminated")


@pytest.mark.parametrize("unconverged", (_failed, _parameterless, _eliminated), ids=("inf", "empty", "eliminated"))
def test_seeds_that_never_converged_are_not_twins(qtbot, unconverged) -> None:
    # isclose(inf, inf) 为真、空参数逐维比较也全过：没收敛的种子凑在一起，不能读成「收敛到同一个解」。
    rows = _rows(qtbot, (_candidate("E-2", 0.5, 0.30), unconverged("C-0-0"), unconverged("C-0-1")))

    assert [row.twin for row, _tip in rows] == ["", "", ""]
    assert all("收敛到同一个解" not in tip for _row, tip in rows)
